/**************************************************************************/
/* dataset.cpp                                                            */
/**************************************************************************/
/*                          This file is part of:                         */
/*                                 SushiAI                                */
/*                https://github.com/SushiSystems/SushiAI                 */
/*                         https://sushisystems.io                        */
/**************************************************************************/
/* Copyright (c) 2026-present Mustafa Garip & Sushi Systems               */
/*                                                                        */
/* Licensed under the Apache License, Version 2.0 (the "License");        */
/* you may not use this file except in compliance with the License.       */
/* You may obtain a copy of the License at                                */
/*                                                                        */
/*     http://www.apache.org/licenses/LICENSE-2.0                         */
/*                                                                        */
/* Unless required by applicable law or agreed to in writing, software    */
/* distributed under the License is distributed on an "AS IS" BASIS,      */
/* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or        */
/* implied. See the License for the specific language governing           */
/* permissions and limitations under the License.                         */
/**************************************************************************/

#include <algorithm>
#include <cstring>
#include <fstream>
#include <SushiRuntime/core/span.hpp>
#include <SushiBLAS/core/format.hpp>
#include <SushiAI/core/allocation.hpp>
#include <SushiAI/data/dataset.hpp>

namespace SushiAI
{
    namespace Data
    {
        namespace
        {
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
                       (options.test_split ? "t10k-images-idx3-ubyte" : "train-images-idx3-ubyte");
            }

            /** @brief The IDX label file for a split. */
            std::filesystem::path label_path(const MnistOptions& options)
            {
                return options.directory /
                       (options.test_split ? "t10k-labels-idx1-ubyte" : "train-labels-idx1-ubyte");
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
                        SA_THROW_IF(index >= batches_.size(),
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

        // =====================================================================
        // SyntheticClassification
        // =====================================================================

        SyntheticClassification::SyntheticClassification(SushiBLAS::Engine& engine,
                                                         SyntheticOptions options)
            : options_(options)
        {
            SA_THROW_IF(options_.batch_size <= 0 || options_.batches <= 0
                        || options_.features <= 0 || options_.classes <= 0,
                        "SyntheticClassification needs positive extents; got batch={} batches={} "
                        "features={} classes={}",
                        options_.batch_size, options_.batches, options_.features, options_.classes);

            engine.set_seed(options_.seed);

            const int64_t rows = options_.batch_size;
            const int64_t features = options_.features;
            const int64_t classes = options_.classes;

            // One random mean per class, shared by every batch. Drawing it once
            // is what makes the problem a single consistent classification task
            // rather than a fresh one per batch.
            const int64_t mean_dims[2] = {classes, features};
            SushiBLAS::Tensor means = allocate_tensor(engine, SushiRuntime::span<const int64_t>(mean_dims, 2));
            engine.random().normal(means, 0.0, 1.0);
            engine.execute();
            engine.get_context().wait_all();

            const int64_t sample_dims[2] = {rows, features};
            const int64_t target_dims[2] = {rows, classes};

            SushiBLAS::Tensor noise = allocate_tensor(engine, SushiRuntime::span<const int64_t>(sample_dims, 2));

            batches_.reserve(static_cast<std::size_t>(options_.batches));

            int64_t running = 0;
            for (int64_t b = 0; b < options_.batches; ++b)
            {
                Batch batch;
                batch.inputs = allocate_tensor(engine, SushiRuntime::span<const int64_t>(sample_dims, 2));
                batch.targets = allocate_tensor(engine, SushiRuntime::span<const int64_t>(target_dims, 2));
                batch.labels.resize(static_cast<std::size_t>(rows));

                for (int64_t r = 0; r < rows; ++r)
                    batch.labels[static_cast<std::size_t>(r)] =
                        static_cast<int32_t>((running++) % classes);

                write_one_hot(batch.targets, batch.labels, classes);

                // X = Y * M picks each sample's class mean; the one-hot matrix
                // doubles as the gather SushiBLAS does not have.
                engine.blas().gemm(batch.targets, means, batch.inputs, false, false, 1.0, 0.0);
                engine.random().normal(noise, 0.0, options_.noise);
                engine.blas().axpy(1.0f, noise, batch.inputs);

                engine.execute();
                engine.get_context().wait_all();

                batches_.push_back(std::move(batch));
            }
        }

        const Batch& SyntheticClassification::batch(std::size_t index) const
        {
            SA_THROW_IF(index >= batches_.size(),
                        "Batch {} is out of range; the dataset has {}", index, batches_.size());
            return batches_[index];
        }

        // =====================================================================
        // MNIST
        // =====================================================================

        bool mnist_available(const MnistOptions& options)
        {
            std::error_code ignored;
            return std::filesystem::is_regular_file(image_path(options), ignored)
                && std::filesystem::is_regular_file(label_path(options), ignored);
        }

        std::string mnist_expected_paths(const MnistOptions& options)
        {
            return SushiBLAS::Core::format("images: {}, labels: {}",
                                           image_path(options).string(), label_path(options).string());
        }

        std::unique_ptr<Dataset> try_load_mnist(SushiBLAS::Engine& engine,
                                                const MnistOptions& options)
        {
            if (!mnist_available(options)) return nullptr;

            SA_THROW_IF(options.batch_size <= 0,
                        "MNIST needs a positive batch size; got {}", options.batch_size);

            std::ifstream images(image_path(options), std::ios::binary);
            std::ifstream labels(label_path(options), std::ios::binary);

            SA_THROW_IF(!images || !labels,
                        "MNIST files exist but could not be opened ({})",
                        mnist_expected_paths(options));

            const std::uint32_t image_magic = read_be32(images);
            const std::uint32_t image_count = read_be32(images);
            const std::uint32_t image_rows = read_be32(images);
            const std::uint32_t image_cols = read_be32(images);

            const std::uint32_t label_magic = read_be32(labels);
            const std::uint32_t label_count = read_be32(labels);

            SA_THROW_IF(image_magic != 0x00000803u,
                        "{} is not an IDX3 image file (magic 0x{:08x})",
                        image_path(options).string(), image_magic);
            SA_THROW_IF(label_magic != 0x00000801u,
                        "{} is not an IDX1 label file (magic 0x{:08x})",
                        label_path(options).string(), label_magic);
            SA_THROW_IF(image_count != label_count,
                        "MNIST holds {} images but {} labels", image_count, label_count);

            const int64_t pixels = static_cast<int64_t>(image_rows) * image_cols;
            const int64_t available = static_cast<int64_t>(image_count);
            const int64_t wanted = options.max_samples > 0
                                 ? std::min(available, options.max_samples)
                                 : available;

            const int64_t rows = options.batch_size;
            const int64_t batch_total = wanted / rows;
            SA_THROW_IF(batch_total <= 0,
                        "MNIST has {} usable samples, fewer than one batch of {}", wanted, rows);

            const int64_t classes = 10;
            const int64_t sample_dims[2] = {rows, pixels};
            const int64_t target_dims[2] = {rows, classes};

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

                SA_THROW_IF(!images || !labels,
                            "MNIST file ended after {} of {} batches; the download is truncated",
                            b, batch_total);

                Batch batch;
                batch.inputs = allocate_tensor(engine, SushiRuntime::span<const int64_t>(sample_dims, 2));
                batch.targets = allocate_tensor(engine, SushiRuntime::span<const int64_t>(target_dims, 2));
                batch.labels.resize(static_cast<std::size_t>(rows));

                float* pixel_data = batch.inputs.data_as<float>();
                for (std::size_t i = 0; i < raw.size(); ++i)
                    pixel_data[i] = static_cast<float>(raw[i]) / 255.0f;

                for (int64_t r = 0; r < rows; ++r)
                {
                    const int32_t label = static_cast<int32_t>(raw_labels[static_cast<std::size_t>(r)]);
                    SA_THROW_IF(label < 0 || label >= classes,
                                "MNIST label {} is out of range", label);
                    batch.labels[static_cast<std::size_t>(r)] = label;
                }

                write_one_hot(batch.targets, batch.labels, classes);
                batches.push_back(std::move(batch));
            }

            return std::make_unique<MaterialisedDataset>(std::move(batches), rows, pixels,
                                                         classes, "mnist");
        }

    } // namespace Data
} // namespace SushiAI
