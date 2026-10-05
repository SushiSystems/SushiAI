/****************************************************************/
/* dataset.hpp                                                  */
/* SushiAI - https://github.com/SushiSystems/SushiAI            */
/* Copyright (c) 2026 Sushi Systems                             */
/* Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.    */
/* Commercial use requires a licence from Sushi Systems.        */
/****************************************************************/

/**
 * @file dataset.hpp
 * @brief Declares dataset interfaces and implementations for model training.
 * @author Mustafa Garip
 */

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <string_view>
#include <vector>
#include <SushiBLAS/engine.hpp>
#include <SushiAI/config/value.hpp>
#include <SushiAI/core/common.hpp>

namespace SushiAI
{
    namespace Data
    {
        /**
         * @struct Batch
         * @brief One minibatch, resident on the device.
         */
        struct Batch
        {
            /** @brief An [N, D] matrix of samples. */
            SushiBLAS::Tensor inputs;

            /** @brief An [N, C] one-hot target matrix. */
            SushiBLAS::Tensor targets;

            /**
             * @brief The class index per sample, on the host.
             *
             * Kept host-side because accuracy is a metric, not training maths:
             * scoring it needs a per-row argmax, SushiBLAS's argmax is
             * whole-tensor, and adding a device reduction for a diagnostic
             * would be the wrong trade.
             */
            std::vector<int32_t> labels;
        };

        /**
         * @class Dataset
         * @brief The seam every training loop reads through.
         */
        class Dataset
        {
            public:
                virtual ~Dataset() = default;

                /** @brief Samples per batch. */
                virtual int64_t batch_size() const noexcept = 0;

                /** @brief Features per sample. */
                virtual int64_t input_dim() const noexcept = 0;

                /** @brief How many classes the targets span. */
                virtual int64_t num_classes() const noexcept = 0;

                /** @brief How many batches an epoch holds. */
                virtual std::size_t batch_count() const noexcept = 0;

                /**
                 * @brief One batch.
                 * @param index Which batch, in [0, batch_count()).
                 * @return The batch; owned by the dataset.
                 * @throws Error If @p index is out of range.
                 */
                virtual const Batch& batch(std::size_t index) const = 0;

                /** @brief A short name for a run report. */
                virtual std::string_view name() const noexcept = 0;
        };

        /**
         * @struct SyntheticOptions
         * @brief How to shape the generated classification problem.
         */
        struct SyntheticOptions
        {
            /** @brief Samples per batch. */
            int64_t batch_size = 32;

            /** @brief How many batches to generate. */
            int64_t batches = 16;

            /** @brief Features per sample. */
            int64_t features = 16;

            /** @brief How many classes. */
            int64_t classes = 4;

            /**
             * @brief Noise standard deviation, relative to unit class means.
             *
             * Below about 0.7 the classes stay comfortably separable, so a
             * correct trainer reaches near-perfect accuracy and a test can
             * assert it.
             */
            double noise = 0.35;

            /** @brief The Philox seed; the same seed gives the same data. */
            std::uint64_t seed = 0x0D15EA5E'0BADF00Dull;
        };

        /**
         * @class SyntheticClassification
         * @brief Generates deterministic, linearly separable multi-class batches.
         *
         * Samples are assigned to classes round-robin with Gaussian noise added to
         * class means.
         */
        class SyntheticClassification final : public Dataset
        {
            public:
                /**
                 * @brief Generates the whole dataset and waits for it.
                 * @param engine The engine that allocates and fills the tensors.
                 * @param options The problem's shape.
                 * @throws Error If any extent in @p options is not positive.
                 */
                SyntheticClassification(SushiBLAS::Engine& engine, SyntheticOptions options = {});

                int64_t batch_size() const noexcept override { return options_.batch_size; }
                int64_t input_dim() const noexcept override { return options_.features; }
                int64_t num_classes() const noexcept override { return options_.classes; }
                std::size_t batch_count() const noexcept override { return batches_.size(); }
                const Batch& batch(std::size_t index) const override;
                std::string_view name() const noexcept override { return "synthetic-classification"; }

            private:
                SyntheticOptions options_;
                std::vector<Batch> batches_;
        };

        /**
         * @struct RingsOptions
         * @brief How to shape the concentric-rings problem.
         */
        struct RingsOptions
        {
            /** @brief Samples per batch. */
            int64_t batch_size = 64;

            /** @brief How many batches to generate. */
            int64_t batches = 64;

            /** @brief How many classes; one per ring. */
            int64_t classes = 4;

            /** @brief Extra pure-Gaussian features carrying no signal at all. */
            int64_t distractor_features = 4;

            /** @brief Ring c spans radius [c, c + ring_width). */
            double ring_width = 1.0;

            /** @brief Gaussian jitter added to x and y. */
            double position_noise = 0.12;

            /** @brief The seed; the same seed gives the same data. */
            std::uint64_t seed = 0x0D15EA5E'0BADF00Dull;
        };

        /**
         * @class Rings
         * @brief Generates concentric rings labelled by ring index.
         *
         * Decision boundary is quadratic in inputs, making it linearly non-separable.
         * Distractor features contain pure Gaussian noise.
         */
        class Rings final : public Dataset
        {
            public:
                /**
                 * @brief Generates the whole dataset.
                 * @param engine The engine that allocates the tensors.
                 * @param options The problem's shape.
                 * @throws Error If any extent in @p options is not positive, or
                 *         the distractor count is negative.
                 */
                Rings(SushiBLAS::Engine& engine, RingsOptions options = {});

                int64_t batch_size() const noexcept override { return options_.batch_size; }
                int64_t input_dim() const noexcept override { return 2 + options_.distractor_features; }
                int64_t num_classes() const noexcept override { return options_.classes; }
                std::size_t batch_count() const noexcept override { return batches_.size(); }
                const Batch& batch(std::size_t index) const override;
                std::string_view name() const noexcept override { return "rings"; }

            private:
                RingsOptions options_;
                std::vector<Batch> batches_;
        };

        /**
         * @struct MnistOptions
         * @brief Where the IDX files are and how much of them to use.
         */
        struct MnistOptions
        {
            /** @brief The directory holding the four standard IDX files. */
            std::filesystem::path directory = "data/mnist";

            /** @brief Samples per batch. */
            int64_t batch_size = 64;

            /** @brief Cap on how many samples to load; 0 means all of them. */
            int64_t max_samples = 4096;

            /** @brief Load the test split rather than the training split. */
            bool test_split = false;
        };

        /**
         * @brief Whether the IDX files for @p options are present and readable.
         * @param options Where to look and which split.
         * @return True when both the image and label files exist.
         */
        bool mnist_available(const MnistOptions& options);

        /**
         * @brief The path pair @c mnist_available checks, for a diagnostic message.
         * @param options Where to look and which split.
         * @return "images: <path>, labels: <path>".
         */
        std::string mnist_expected_paths(const MnistOptions& options);

        /**
         * @brief Loads MNIST from IDX files, returning nullptr if absent.
         * @param engine Engine allocating device tensors.
         * @param options Directory, split, and sample count limits.
         * @return Loaded dataset or nullptr when files are missing.
         * @throws Error If an existing IDX file is malformed.
         */
        std::unique_ptr<Dataset> try_load_mnist(SushiBLAS::Engine& engine,
                                                const MnistOptions& options = {});

        /**
         * @brief Builds one dataset from its configuration attributes.
         * @param engine The engine that allocates and fills the tensors.
         * @param attributes The dataset's own keys; the factory rejects any it does not know.
         * @param context What to call this dataset in a diagnostic.
         * @return The dataset.
         */
        using DatasetFactory = std::unique_ptr<Dataset> (*)(SushiBLAS::Engine& engine,
                                                            const Config::Value& attributes,
                                                            std::string_view context);

        /**
         * @struct DatasetEntry
         * @brief One row of @ref DATASET_REGISTRY.
         */
        struct DatasetEntry
        {
            /** @brief The "type" a configuration file spells. */
            std::string_view type;

            /** @brief What builds it. */
            DatasetFactory make;
        };

        /**
         * @brief Builds a @ref Rings dataset from configuration attributes.
         * @param attributes Dataset parameters dictionary.
         * @param context Diagnostic context string.
         * @return Constructed dataset instance.
         * @throws Error If an attribute key is unknown or value is invalid.
         */
        std::unique_ptr<Dataset> make_rings(SushiBLAS::Engine& engine,
                                            const Config::Value& attributes,
                                            std::string_view context);

        /**
         * @brief Every dataset type a configuration file may name.
         *
         * A file-scope constant table; nothing self-registers. See
         * config/registry.hpp for why that is forced rather than preferred.
         */
        inline constexpr std::array<DatasetEntry, 1> DATASET_REGISTRY = {
            DatasetEntry{"rings", &make_rings},
        };

        /**
         * @brief Builds the dataset named @p type from the registry.
         * @param type Registry key identifying dataset implementation.
         * @param attributes Dataset configuration attributes.
         * @return Constructed dataset instance.
         * @throws Error If @p type is not registered.
         */
        std::unique_ptr<Dataset> make_dataset(SushiBLAS::Engine& engine, std::string_view type,
                                              const Config::Value& attributes,
                                              std::string_view context);

    } // namespace Data
} // namespace SushiAI
