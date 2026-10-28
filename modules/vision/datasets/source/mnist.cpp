/**************************************************************************/
/* mnist.cpp                                                              */
/**************************************************************************/
/*                          This file is part of:                         */
/*                                 SushiAI                                */
/*                 https://github.com/SushiSystems/SushiAI                */
/*                         https://sushisystems.io                        */
/**************************************************************************/
/* Copyright (c) 2026-present Mustafa Garip & Sushi Systems               */
/*                                                                        */
/* Licensed under the PolyForm Noncommercial License 1.0.0 (the           */
/* "License"); you may not use this file except in compliance with the    */
/* License. You may obtain a copy of the License at                       */
/*                                                                        */
/*     https://polyformproject.org/licenses/noncommercial/1.0.0           */
/*                                                                        */
/* Noncommercial use is free. Commercial use requires a separate licence  */
/* from Sushi Systems; see COMMERCIAL.md. The software is provided        */
/* "as is", without warranty of any kind.                                 */
/**************************************************************************/

/**
 * @file mnist.cpp
 * @brief Implements the MNIST dataset loader and its configuration factory.
 * @author Mustafa Garip
 */

#include <SushiAI/vision/datasets/mnist.hpp>

#include <algorithm>
#include <fstream>
#include <iterator>
#include <string>
#include <system_error>
#include <utility>
#include <vector>
#include <SushiRuntime/core/span.hpp>
#include <SushiLog/text/text_format.hpp>
#include <SushiLog/text/value_text.hpp>
#include <SushiAI/seam/seam.hpp>
#include <SushiAI/shape/shape.hpp>
#include <SushiAI/data/input.hpp>
#include <SushiAI/data/materialised_dataset.hpp>
#include <SushiAI/data/roles.hpp>
#include <SushiAI/data/target.hpp>

namespace SushiAI::Vision::Datasets
{
    namespace
    {
        /** @brief Lists every key a "mnist" dataset accepts. */
        constexpr std::string_view K_MNIST_KEYS[] = {
            "directory", "split", "layout", "batch_size", "max_samples",
        };

        /** @brief Names the "split" value that selects the training files. */
        constexpr std::string_view K_MNIST_SPLIT_TRAIN = "train";

        /** @brief Names the "split" value that selects the test files. */
        constexpr std::string_view K_MNIST_SPLIT_TEST = "test";

        /** @brief Names the "layout" value that yields one row of pixels per sample. */
        constexpr std::string_view K_MNIST_LAYOUT_FLAT = "flat";

        /** @brief Names the "layout" value that yields one image per sample. */
        constexpr std::string_view K_MNIST_LAYOUT_IMAGE = "image";

        /** @brief Names the IDX image file of the training split. */
        constexpr std::string_view K_MNIST_TRAIN_IMAGES = "train-images-idx3-ubyte";

        /** @brief Names the IDX label file of the training split. */
        constexpr std::string_view K_MNIST_TRAIN_LABELS = "train-labels-idx1-ubyte";

        /** @brief Names the IDX image file of the test split. */
        constexpr std::string_view K_MNIST_TEST_IMAGES = "t10k-images-idx3-ubyte";

        /** @brief Names the IDX label file of the test split. */
        constexpr std::string_view K_MNIST_TEST_LABELS = "t10k-labels-idx1-ubyte";

        /** @brief Reads a big-endian 32-bit integer, as the IDX format stores them. */
        std::uint32_t read_be32(std::istream& in)
        {
            unsigned char bytes[4] = {0, 0, 0, 0};
            in.read(reinterpret_cast<char*>(bytes), 4);

            return (static_cast<std::uint32_t>(bytes[0]) << 24)
                 | (static_cast<std::uint32_t>(bytes[1]) << 16)
                 | (static_cast<std::uint32_t>(bytes[2]) << 8)
                 | static_cast<std::uint32_t>(bytes[3]);
        }

        /** @brief The IDX image file for a split. */
        std::filesystem::path image_path(const MNISTOptions& options)
        {
            return options.directory /
                   (options.test_split ? K_MNIST_TEST_IMAGES : K_MNIST_TRAIN_IMAGES);
        }

        /** @brief The IDX label file for a split. */
        std::filesystem::path label_path(const MNISTOptions& options)
        {
            return options.directory /
                   (options.test_split ? K_MNIST_TEST_LABELS : K_MNIST_TRAIN_LABELS);
        }

        /**
         * @brief Reads the optional "split" key of a "mnist" dataset.
         * @return True for the test split, false for the training split, and the
         *         MNISTOptions default for an absent key; InvalidDescription for a value
         *         that names neither split.
         */
        [[nodiscard]] Result<bool> is_test_split(const Config::Value& attributes,
                                                 const std::string& prefix)
        {
            const Config::Value spelled = Config::Value::string(std::string(
                MNISTOptions().test_split ? K_MNIST_SPLIT_TEST : K_MNIST_SPLIT_TRAIN));
            SA_TRY_ASSIGN(const std::string& split,
                          attributes.at_or("split", spelled).as_string(prefix + " 'split'"));

            SA_FAIL_IF(split != K_MNIST_SPLIT_TRAIN && split != K_MNIST_SPLIT_TEST,
                       StatusCode::InvalidDescription,
                       "{} 'split' is '{}'; it must be '{}' or '{}'",
                       prefix, split, K_MNIST_SPLIT_TRAIN, K_MNIST_SPLIT_TEST);

            return split == K_MNIST_SPLIT_TEST;
        }

        /**
         * @brief Reads the optional "layout" key of a "mnist" dataset.
         * @return Flat or Image, the MNISTOptions default for an absent key; or
         *         InvalidDescription for a value that names neither layout.
         */
        [[nodiscard]] Result<MNISTLayout> read_layout(const Config::Value& attributes,
                                                      const std::string& prefix)
        {
            const bool image_default = MNISTOptions().layout == MNISTLayout::Image;
            const Config::Value spelled = Config::Value::string(std::string(
                image_default ? K_MNIST_LAYOUT_IMAGE : K_MNIST_LAYOUT_FLAT));
            SA_TRY_ASSIGN(const std::string layout,
                          attributes.at_or("layout", spelled).as_string(prefix + " 'layout'"));

            SA_FAIL_IF(layout != K_MNIST_LAYOUT_FLAT && layout != K_MNIST_LAYOUT_IMAGE,
                       StatusCode::InvalidDescription,
                       "{} 'layout' is '{}'; it must be '{}' or '{}'",
                       prefix, layout, K_MNIST_LAYOUT_FLAT, K_MNIST_LAYOUT_IMAGE);

            return layout == K_MNIST_LAYOUT_IMAGE ? MNISTLayout::Image : MNISTLayout::Flat;
        }

        /**
         * @brief Reads the MNIST split that @p options names into @p dataset.
         * @param dataset Receives the batches; unchanged when the read fails.
         * @return Success, FileNotFound when a file is absent, InvalidFile for a file that
         *         is not the split, InvalidDescription for a batch size that is not
         *         positive, or the failure of an allocation.
         */
        [[nodiscard]] Status read_mnist(SushiBLAS::Engine& engine,
                                        const MNISTOptions& options,
                                        std::unique_ptr<Data::Dataset>& dataset)
        {
            SA_FAIL_IF(!mnist_available(options), StatusCode::FileNotFound,
                       "{}", mnist_expected_paths(options));

            SA_FAIL_IF(options.batch_size <= 0, StatusCode::InvalidDescription,
                       "MNIST needs a positive batch size; got {}", options.batch_size);

            std::ifstream images(image_path(options), std::ios::binary);
            std::ifstream labels(label_path(options), std::ios::binary);

            SA_FAIL_IF(!images || !labels, StatusCode::InvalidFile,
                       "MNIST files exist but could not be opened ({})",
                       mnist_expected_paths(options));

            const std::uint32_t image_magic = read_be32(images);
            const std::uint32_t image_count = read_be32(images);
            const std::uint32_t image_rows = read_be32(images);
            const std::uint32_t image_cols = read_be32(images);

            const std::uint32_t label_magic = read_be32(labels);
            const std::uint32_t label_count = read_be32(labels);

            SA_FAIL_IF(image_magic != 0x00000803u, StatusCode::InvalidFile,
                       "{} is not an IDX3 image file (magic 0x{})",
                       image_path(options).string(), SushiLog::hex(image_magic, 8));
            SA_FAIL_IF(label_magic != 0x00000801u, StatusCode::InvalidFile,
                       "{} is not an IDX1 label file (magic 0x{})",
                       label_path(options).string(), SushiLog::hex(label_magic, 8));
            SA_FAIL_IF(image_count != label_count, StatusCode::InvalidFile,
                       "MNIST holds {} images but {} labels", image_count, label_count);
            SA_FAIL_IF(image_rows == 0 || image_cols == 0, StatusCode::InvalidFile,
                       "MNIST images are {}x{}; both extents must be positive",
                       image_rows, image_cols);

            const int64_t pixels = static_cast<int64_t>(image_rows) * image_cols;
            const Shape sample = options.layout == MNISTLayout::Image
                ? Shape{static_cast<int64_t>(image_rows), static_cast<int64_t>(image_cols), 1}
                : Shape{pixels};
            const Data::InputDescription input{sample, DType::Float32};
            const int64_t available = static_cast<int64_t>(image_count);
            const int64_t wanted = options.max_samples > 0
                                 ? std::min(available, options.max_samples)
                                 : available;

            const int64_t rows = options.batch_size;
            const int64_t batch_total = wanted / rows;
            SA_FAIL_IF(batch_total <= 0, StatusCode::InvalidFile,
                       "MNIST has {} usable samples, fewer than one batch of {}", wanted, rows);

            const int64_t classes = 10;
            std::vector<Data::TargetDescription> targets{Data::TargetDescription{
                std::string(Data::K_ROLE_LABELS), Shape{}, DType::Int32, classes}};

            std::vector<unsigned char> raw(static_cast<std::size_t>(rows * pixels));
            std::vector<unsigned char> raw_labels(static_cast<std::size_t>(rows));

            std::vector<Data::Batch> batches;
            batches.reserve(static_cast<std::size_t>(batch_total));

            for (int64_t b = 0; b < batch_total; ++b)
            {
                images.read(reinterpret_cast<char*>(raw.data()),
                            static_cast<std::streamsize>(raw.size()));
                labels.read(reinterpret_cast<char*>(raw_labels.data()),
                            static_cast<std::streamsize>(raw_labels.size()));

                SA_FAIL_IF(!images || !labels, StatusCode::InvalidFile,
                           "MNIST file ended after {} of {} batches; the download is truncated",
                           b, batch_total);

                SA_TRY_ASSIGN(Data::Batch batch,
                              Data::allocate_batch(engine, rows, input, targets));

                float* pixel_data = batch.inputs.data_as<float>();
                for (std::size_t i = 0; i < raw.size(); ++i)
                    pixel_data[i] = static_cast<float>(raw[i]) / 255.0f;

                int32_t* label_data = batch.targets.front().values.data_as<int32_t>();
                for (int64_t r = 0; r < rows; ++r)
                {
                    const int32_t label =
                        static_cast<int32_t>(raw_labels[static_cast<std::size_t>(r)]);
                    SA_FAIL_IF(label < 0 || label >= classes, StatusCode::InvalidFile,
                               "MNIST label {} is out of range", label);
                    label_data[r] = label;
                }

                batches.push_back(std::move(batch));
            }

            dataset = std::make_unique<Data::MaterialisedDataset>(
                std::move(batches), rows, input, std::move(targets), "mnist");
            return Status();
        }
    } // namespace

    bool mnist_available(const MNISTOptions& options) noexcept
    {
        std::error_code ignored;
        return std::filesystem::is_regular_file(image_path(options), ignored)
            && std::filesystem::is_regular_file(label_path(options), ignored);
    }

    std::string mnist_expected_paths(const MNISTOptions& options) noexcept
    {
        return SushiLog::format_text("images: {}, labels: {}",
                                       image_path(options).string(),
                                       label_path(options).string());
    }

    Result<std::unique_ptr<Data::Dataset>> try_load_mnist(SushiBLAS::Engine& engine,
                                                          const MNISTOptions& options) noexcept
    {
        std::unique_ptr<Data::Dataset> dataset;
        SA_TRY(Detail::guard(Detail::SeamPhase::Allocate, [&]()
        {
            return read_mnist(engine, options, dataset);
        }));
        return std::move(dataset);
    }

    Result<std::unique_ptr<Data::Dataset>> make_mnist(SushiBLAS::Engine& engine,
                                                      const Config::Value& attributes,
                                                      std::string_view context) noexcept
    {
        SA_TRY(attributes.reject_unknown_keys(
            context,
            SushiRuntime::span<const std::string_view>(K_MNIST_KEYS, std::size(K_MNIST_KEYS))));

        MNISTOptions options;
        const std::string prefix(context);

        SA_TRY_ASSIGN(const Config::Value& stated_directory, attributes.at("directory", context));
        SA_TRY_ASSIGN(options.directory, stated_directory.as_string(prefix + " 'directory'"));
        SA_TRY_ASSIGN(options.test_split, is_test_split(attributes, prefix));
        SA_TRY_ASSIGN(options.layout, read_layout(attributes, prefix));

        const Config::Value batch_size = Config::Value::integer(options.batch_size);
        SA_TRY_ASSIGN(options.batch_size,
                      attributes.at_or("batch_size", batch_size)
                          .as_int(prefix + " 'batch_size'"));
        const Config::Value max_samples = Config::Value::integer(options.max_samples);
        SA_TRY_ASSIGN(options.max_samples,
                      attributes.at_or("max_samples", max_samples)
                          .as_int(prefix + " 'max_samples'"));

        Result<std::unique_ptr<Data::Dataset>> loaded = try_load_mnist(engine, options);

        // A FileNotFound message holds the expected paths; the prefix names the directory.
        if (loaded.status().code() == StatusCode::FileNotFound)
            return SA_ANNOTATE(loaded.status(),
                               "{} 'directory' '{}' does not hold the MNIST {} split. A "
                               "complete directory holds {}, {}, {} and {}",
                               prefix, options.directory.string(),
                               options.test_split ? K_MNIST_SPLIT_TEST : K_MNIST_SPLIT_TRAIN,
                               K_MNIST_TRAIN_IMAGES, K_MNIST_TRAIN_LABELS,
                               K_MNIST_TEST_IMAGES, K_MNIST_TEST_LABELS);

        return loaded;
    }

} // namespace SushiAI::Vision::Datasets
