/**************************************************************************/
/* dataset.cpp                                                            */
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
 * @file dataset.cpp
 * @brief Implements dataset loaders and synthetic batch generators.
 * @author Mustafa Garip
 */

#include <SushiAI/data/dataset.hpp>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <random>
#include <string>
#include <utility>
#include <SushiRuntime/core/span.hpp>
#include <SushiBLAS/core/format.hpp>
#include <SushiBLAS/engine/blas.hpp>
#include <SushiBLAS/engine/math/random.hpp>
#include <SushiAI/config/registry.hpp>
#include <SushiAI/core/allocation.hpp>
#include <SushiAI/core/seam.hpp>

namespace SushiAI
{
    namespace Data
    {
        namespace
        {
            /** @brief Every key a "rings" dataset accepts. */
            constexpr std::string_view K_RINGS_KEYS[] = {
                "batch_size", "batches", "classes", "distractors",
                "ring_width", "position_noise", "seed",
            };

            /** @brief Lists every key a "mnist" dataset accepts. */
            constexpr std::string_view K_MNIST_KEYS[] = {
                "directory", "split", "batch_size", "max_samples",
            };

            /** @brief Names the "split" value that selects the training files. */
            constexpr std::string_view K_MNIST_SPLIT_TRAIN = "train";

            /** @brief Names the "split" value that selects the test files. */
            constexpr std::string_view K_MNIST_SPLIT_TEST = "test";

            /** @brief Names the IDX image file of the training split. */
            constexpr std::string_view K_MNIST_TRAIN_IMAGES = "train-images-idx3-ubyte";

            /** @brief Names the IDX label file of the training split. */
            constexpr std::string_view K_MNIST_TRAIN_LABELS = "train-labels-idx1-ubyte";

            /** @brief Names the IDX image file of the test split. */
            constexpr std::string_view K_MNIST_TEST_IMAGES = "t10k-images-idx3-ubyte";

            /** @brief Names the IDX label file of the test split. */
            constexpr std::string_view K_MNIST_TEST_LABELS = "t10k-labels-idx1-ubyte";

            /** @brief Wraps a literal key list as a span, for reject_unknown_keys. */
            template<std::size_t N>
            SushiRuntime::span<const std::string_view> as_span(const std::string_view (&keys)[N])
            {
                return SushiRuntime::span<const std::string_view>(keys, N);
            }

            /**
             * @brief Reads an optional whole number from a dataset's attributes.
             * @param attributes The dataset's keys.
             * @param key Which one.
             * @param prefix What to call this dataset in a diagnostic.
             * @param fallback Returned when @p key is absent.
             * @return The value, or @p fallback.
             */
            int64_t read_int(const Config::Value& attributes, std::string_view key,
                             const std::string& prefix, int64_t fallback)
            {
                const Config::Value spelled = Config::Value::integer(fallback);
                return attributes.at_or(key, spelled)
                                 .as_int(prefix + " '" + std::string(key) + "'");
            }

            /**
             * @brief Reads an optional real from a dataset's attributes.
             * @param attributes The dataset's keys.
             * @param key Which one.
             * @param prefix What to call this dataset in a diagnostic.
             * @param fallback Returned when @p key is absent.
             * @return The value, or @p fallback.
             */
            double read_real(const Config::Value& attributes, std::string_view key,
                             const std::string& prefix, double fallback)
            {
                const Config::Value spelled = Config::Value::real(fallback);
                return attributes.at_or(key, spelled)
                                 .as_real(prefix + " '" + std::string(key) + "'");
            }

            /**
             * @brief Allocates the tensors of one batch and sizes its label list.
             * @param rows Samples in the batch.
             * @param features Width of the input matrix.
             * @param classes Width of the target matrix.
             * @return The batch, its contents unset, or the failure of an allocation.
             */
            [[nodiscard]] Result<Batch> allocate_batch(SushiBLAS::Engine& engine, int64_t rows,
                                                       int64_t features, int64_t classes)
            {
                const int64_t sample_dims[2] = {rows, features};
                const int64_t target_dims[2] = {rows, classes};

                Batch batch;
                SA_TRY_ASSIGN(batch.inputs,
                              allocate_tensor(engine,
                                              SushiRuntime::span<const int64_t>(sample_dims, 2)));
                SA_TRY_ASSIGN(batch.targets,
                              allocate_tensor(engine,
                                              SushiRuntime::span<const int64_t>(target_dims, 2)));
                SA_TRY(Detail::guard(Detail::SeamPhase::Allocate, [&]()
                {
                    batch.labels.resize(static_cast<std::size_t>(rows));
                    return Status();
                }));

                return std::move(batch);
            }

            /** @brief Writes a one-hot row pattern straight into shared USM. */
            void write_one_hot(SushiBLAS::Tensor& targets,
                               const std::vector<int32_t>& labels,
                               int64_t classes)
            {
                float* data = targets.data_as<float>();
                std::fill(data, data + targets.num_elements, 0.0f);

                for (std::size_t row = 0; row < labels.size(); ++row)
                    data[static_cast<int64_t>(row) * classes + labels[row]] = 1.0f;
            }

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
            std::filesystem::path image_path(const MnistOptions& options)
            {
                return options.directory /
                       (options.test_split ? K_MNIST_TEST_IMAGES : K_MNIST_TRAIN_IMAGES);
            }

            /** @brief The IDX label file for a split. */
            std::filesystem::path label_path(const MnistOptions& options)
            {
                return options.directory /
                       (options.test_split ? K_MNIST_TEST_LABELS : K_MNIST_TRAIN_LABELS);
            }

            /**
             * @brief Reads the optional "split" key of a "mnist" dataset.
             * @return True for the test split, false for the training split, and the
             *         MnistOptions default for an absent key; InvalidDescription for a value
             *         that names neither split.
             */
            [[nodiscard]] Result<bool> is_test_split(const Config::Value& attributes,
                                                     const std::string& prefix)
            {
                const Config::Value spelled = Config::Value::string(std::string(
                    MnistOptions().test_split ? K_MNIST_SPLIT_TEST : K_MNIST_SPLIT_TRAIN));
                const std::string& split =
                    attributes.at_or("split", spelled).as_string(prefix + " 'split'");

                SA_FAIL_IF(split != K_MNIST_SPLIT_TRAIN && split != K_MNIST_SPLIT_TEST,
                           StatusCode::InvalidDescription,
                           "{} 'split' is '{}'; it must be '{}' or '{}'",
                           prefix, split, K_MNIST_SPLIT_TRAIN, K_MNIST_SPLIT_TEST);

                return split == K_MNIST_SPLIT_TEST;
            }

            /**
             * @class MaterialisedDataset
             * @brief A dataset whose batches are already on the device.
             */
            class MaterialisedDataset final : public Dataset
            {
                public:
                    MaterialisedDataset(std::vector<Batch> batches, int64_t batch_size,
                                        int64_t features, int64_t classes, std::string name)
                        : batches_(std::move(batches)), batch_size_(batch_size),
                          features_(features), classes_(classes), name_(std::move(name)) {}

                    int64_t batch_size() const noexcept override { return batch_size_; }
                    int64_t input_dim() const noexcept override { return features_; }
                    int64_t num_classes() const noexcept override { return classes_; }
                    std::size_t batch_count() const noexcept override { return batches_.size(); }
                    std::string_view name() const noexcept override { return name_; }

                    const Batch& batch(std::size_t index) const override
                    {
                        SA_FATAL_IF(index >= batches_.size(),
                                    "Batch {} is out of range; the dataset has {}",
                                    index, batches_.size());
                        return batches_[index];
                    }

                private:
                    std::vector<Batch> batches_;
                    int64_t batch_size_;
                    int64_t features_;
                    int64_t classes_;
                    std::string name_;
            };
        }

        SyntheticClassification::SyntheticClassification(SyntheticOptions options,
                                                         std::vector<Batch> batches) noexcept
            : options_(options), batches_(std::move(batches))
        {

        }

        Result<std::unique_ptr<SyntheticClassification>> SyntheticClassification::make(
            SushiBLAS::Engine& engine,
            SyntheticOptions options)
        {
            SA_FAIL_IF(options.batch_size <= 0 || options.batches <= 0
                       || options.features <= 0 || options.classes <= 0,
                       StatusCode::InvalidDescription,
                       "SyntheticClassification needs positive extents; got batch={} batches={} "
                       "features={} classes={}",
                       options.batch_size, options.batches, options.features, options.classes);

            SushiBLAS::RandomOps random(engine);
            SushiBLAS::BLASOps blas(engine);

            const int64_t rows = options.batch_size;
            const int64_t features = options.features;
            const int64_t classes = options.classes;

            // Class means are shared across all batches to form a consistent task.
            const int64_t mean_dims[2] = {classes, features};
            SA_TRY_ASSIGN(SushiBLAS::Tensor means,
                          allocate_tensor(engine, SushiRuntime::span<const int64_t>(mean_dims, 2)));

            SA_TRY(Detail::guard(Detail::SeamPhase::Record, [&]()
            {
                random.set_seed(options.seed);
                random.normal(means, 0.0, 1.0);
                return Status();
            }));
            SA_TRY(Detail::run_recorded(engine));

            const int64_t noise_dims[2] = {rows, features};
            SA_TRY_ASSIGN(SushiBLAS::Tensor noise,
                          allocate_tensor(engine,
                                          SushiRuntime::span<const int64_t>(noise_dims, 2)));

            std::vector<Batch> batches;
            SA_TRY(Detail::guard(Detail::SeamPhase::Allocate, [&]() -> Status
            {
                batches.reserve(static_cast<std::size_t>(options.batches));

                int64_t running = 0;
                for (int64_t b = 0; b < options.batches; ++b)
                {
                    SA_TRY_ASSIGN(Batch batch, allocate_batch(engine, rows, features, classes));

                    for (int64_t r = 0; r < rows; ++r)
                        batch.labels[static_cast<std::size_t>(r)] =
                            static_cast<int32_t>((running++) % classes);

                    write_one_hot(batch.targets, batch.labels, classes);

                    // Matrix multiplication by one-hot targets gathers each sample's class mean.
                    SA_TRY(Detail::guard(Detail::SeamPhase::Record, [&]()
                    {
                        blas.gemm(batch.targets, means, batch.inputs);
                        random.normal(noise, 0.0, options.noise);
                        blas.axpy(1.0f, noise, batch.inputs);
                        return Status();
                    }));
                    SA_TRY(Detail::run_recorded(engine));

                    batches.push_back(std::move(batch));
                }
                return Status();
            }));

            return Detail::guard_value(Detail::SeamPhase::Allocate, [&]()
            {
                return std::unique_ptr<SyntheticClassification>(
                    new SyntheticClassification(options, std::move(batches)));
            });
        }

        const Batch& SyntheticClassification::batch(std::size_t index) const
        {
            SA_FATAL_IF(index >= batches_.size(),
                        "Batch {} is out of range; the dataset has {}", index, batches_.size());
            return batches_[index];
        }

        bool mnist_available(const MnistOptions& options)
        {
            std::error_code ignored;
            return std::filesystem::is_regular_file(image_path(options), ignored)
                && std::filesystem::is_regular_file(label_path(options), ignored);
        }

        std::string mnist_expected_paths(const MnistOptions& options)
        {
            return SushiBLAS::Core::format("images: {}, labels: {}",
                                           image_path(options).string(),
                                           label_path(options).string());
        }

        namespace
        {
            /**
             * @brief Reads the MNIST split that @p options names into @p dataset.
             * @param dataset Receives the batches; unchanged when the read fails.
             * @return Success, FileNotFound when a file is absent, InvalidFile for a file that
             *         is not the split, InvalidDescription for a batch size that is not
             *         positive, or the failure of an allocation.
             */
            [[nodiscard]] Status read_mnist(SushiBLAS::Engine& engine,
                                            const MnistOptions& options,
                                            std::unique_ptr<Dataset>& dataset)
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
                           "{} is not an IDX3 image file (magic 0x{:08x})",
                           image_path(options).string(), image_magic);
                SA_FAIL_IF(label_magic != 0x00000801u, StatusCode::InvalidFile,
                           "{} is not an IDX1 label file (magic 0x{:08x})",
                           label_path(options).string(), label_magic);
                SA_FAIL_IF(image_count != label_count, StatusCode::InvalidFile,
                           "MNIST holds {} images but {} labels", image_count, label_count);

                const int64_t pixels = static_cast<int64_t>(image_rows) * image_cols;
                const int64_t available = static_cast<int64_t>(image_count);
                const int64_t wanted = options.max_samples > 0
                                     ? std::min(available, options.max_samples)
                                     : available;

                const int64_t rows = options.batch_size;
                const int64_t batch_total = wanted / rows;
                SA_FAIL_IF(batch_total <= 0, StatusCode::InvalidFile,
                           "MNIST has {} usable samples, fewer than one batch of {}", wanted, rows);

                const int64_t classes = 10;

                std::vector<unsigned char> raw(static_cast<std::size_t>(rows * pixels));
                std::vector<unsigned char> raw_labels(static_cast<std::size_t>(rows));

                std::vector<Batch> batches;
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

                    SA_TRY_ASSIGN(Batch batch, allocate_batch(engine, rows, pixels, classes));

                    float* pixel_data = batch.inputs.data_as<float>();
                    for (std::size_t i = 0; i < raw.size(); ++i)
                        pixel_data[i] = static_cast<float>(raw[i]) / 255.0f;

                    for (int64_t r = 0; r < rows; ++r)
                    {
                        const int32_t label =
                            static_cast<int32_t>(raw_labels[static_cast<std::size_t>(r)]);
                        SA_FAIL_IF(label < 0 || label >= classes, StatusCode::InvalidFile,
                                   "MNIST label {} is out of range", label);
                        batch.labels[static_cast<std::size_t>(r)] = label;
                    }

                    write_one_hot(batch.targets, batch.labels, classes);
                    batches.push_back(std::move(batch));
                }

                dataset = std::make_unique<MaterialisedDataset>(std::move(batches), rows, pixels,
                                                                classes, "mnist");
                return Status();
            }
        }

        Result<std::unique_ptr<Dataset>> try_load_mnist(SushiBLAS::Engine& engine,
                                                        const MnistOptions& options)
        {
            std::unique_ptr<Dataset> dataset;
            SA_TRY(Detail::guard(Detail::SeamPhase::Allocate, [&]()
            {
                return read_mnist(engine, options, dataset);
            }));
            return std::move(dataset);
        }

        Rings::Rings(RingsOptions options, std::vector<Batch> batches) noexcept
            : options_(options), batches_(std::move(batches))
        {

        }

        Result<std::unique_ptr<Rings>> Rings::make(SushiBLAS::Engine& engine, RingsOptions options)
        {
            SA_FAIL_IF(options.batch_size <= 0 || options.batches <= 0 || options.classes <= 0,
                       StatusCode::InvalidDescription,
                       "Rings needs a positive batch size, batch count and class count; "
                       "got {}, {} and {}",
                       options.batch_size, options.batches, options.classes);
            SA_FAIL_IF(options.distractor_features < 0, StatusCode::InvalidDescription,
                       "Rings needs a non-negative distractor count, got {}",
                       options.distractor_features);
            SA_FAIL_IF(options.ring_width <= 0.0, StatusCode::InvalidDescription,
                       "Rings needs a positive ring width, got {}", options.ring_width);
            SA_FAIL_IF(options.position_noise < 0.0, StatusCode::InvalidDescription,
                       "Rings needs a non-negative position noise, got {}",
                       options.position_noise);

            const int64_t features = 2 + options.distractor_features;
            const int64_t classes = options.classes;
            const int64_t rows = options.batch_size;

            constexpr double K_TWO_PI = 6.283185307179586476925286766559;

            // Host RNG ensures reproducible sequence generation from the seed alone.
            std::mt19937_64 rng(options.seed);
            std::uniform_real_distribution<double> theta_dist(0.0, K_TWO_PI);
            std::uniform_real_distribution<double> band_dist(0.0, options.ring_width);
            std::normal_distribution<double> jitter(0.0, options.position_noise);
            std::normal_distribution<double> distractor(0.0, 1.0);

            std::vector<Batch> batches;
            SA_TRY(Detail::guard(Detail::SeamPhase::Allocate, [&]() -> Status
            {
                batches.reserve(static_cast<std::size_t>(options.batches));

                int64_t running = 0;
                for (int64_t b = 0; b < options.batches; ++b)
                {
                    SA_TRY_ASSIGN(Batch batch, allocate_batch(engine, rows, features, classes));

                    float* x = batch.inputs.data_as<float>();

                    for (int64_t r = 0; r < rows; ++r)
                    {
                        // Round-robin class assignment balances every batch evenly.
                        const int64_t label = (running++) % classes;
                        batch.labels[static_cast<std::size_t>(r)] = static_cast<int32_t>(label);

                        const double radius = static_cast<double>(label) + band_dist(rng);
                        const double theta = theta_dist(rng);

                        float* row = x + r * features;
                        row[0] = static_cast<float>(radius * std::cos(theta) + jitter(rng));
                        row[1] = static_cast<float>(radius * std::sin(theta) + jitter(rng));
                        for (int64_t f = 0; f < options.distractor_features; ++f)
                            row[2 + f] = static_cast<float>(distractor(rng));
                    }

                    write_one_hot(batch.targets, batch.labels, classes);
                    batches.push_back(std::move(batch));
                }
                return Status();
            }));

            return Detail::guard_value(Detail::SeamPhase::Allocate, [&]()
            {
                return std::unique_ptr<Rings>(new Rings(options, std::move(batches)));
            });
        }

        const Batch& Rings::batch(std::size_t index) const
        {
            SA_FATAL_IF(index >= batches_.size(),
                        "Batch {} is out of range; the dataset has {}", index, batches_.size());
            return batches_[index];
        }

        Result<std::unique_ptr<Dataset>> make_rings(SushiBLAS::Engine& engine,
                                                    const Config::Value& attributes,
                                                    std::string_view context)
        {
            attributes.reject_unknown_keys(context, as_span(K_RINGS_KEYS));

            RingsOptions options;
            const std::string prefix(context);

            options.batch_size = read_int(attributes, "batch_size", prefix, options.batch_size);
            options.batches = read_int(attributes, "batches", prefix, options.batches);
            options.classes = read_int(attributes, "classes", prefix, options.classes);
            options.distractor_features =
                read_int(attributes, "distractors", prefix, options.distractor_features);
            options.ring_width = read_real(attributes, "ring_width", prefix, options.ring_width);
            options.position_noise =
                read_real(attributes, "position_noise", prefix, options.position_noise);

            // Reinterpreting the signed integer preserves arbitrary 64-bit seed patterns.
            options.seed = static_cast<std::uint64_t>(
                read_int(attributes, "seed", prefix, static_cast<int64_t>(options.seed)));

            SA_TRY_ASSIGN(std::unique_ptr<Dataset> dataset, Rings::make(engine, options));
            return std::move(dataset);
        }

        Result<std::unique_ptr<Dataset>> make_mnist(SushiBLAS::Engine& engine,
                                                    const Config::Value& attributes,
                                                    std::string_view context)
        {
            attributes.reject_unknown_keys(context, as_span(K_MNIST_KEYS));

            MnistOptions options;
            const std::string prefix(context);

            options.directory = attributes.at("directory", context)
                                          .as_string(prefix + " 'directory'");
            SA_TRY_ASSIGN(options.test_split, is_test_split(attributes, prefix));
            options.batch_size = read_int(attributes, "batch_size", prefix, options.batch_size);
            options.max_samples = read_int(attributes, "max_samples", prefix, options.max_samples);

            Result<std::unique_ptr<Dataset>> loaded = try_load_mnist(engine, options);

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

        Result<std::unique_ptr<Dataset>> make_dataset(SushiBLAS::Engine& engine,
                                                      std::string_view type,
                                                      const Config::Value& attributes,
                                                      std::string_view context)
        {
            return Config::find_entry(K_DATASET_REGISTRY, type, "dataset type")
                       .make(engine, attributes, context);
        }

    } // namespace Data
} // namespace SushiAI
