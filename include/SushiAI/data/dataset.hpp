/**************************************************************************/
/* dataset.hpp                                                            */
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

#pragma once

/**
 * @file dataset.hpp
 * @brief Where a training loop's batches come from.
 *
 * Batches are materialised on the device once, up front, and handed out by
 * index. That is the right shape for this stage: it keeps the training loop
 * free of host-to-device transfer, keeps a step's cost attributable to the
 * step, and makes a run reproducible without a data-loader thread to
 * synchronise. A streaming loader belongs behind the same @c Dataset seam once
 * there is a dataset that does not fit.
 *
 * Every batch owns its own tensors, each with its own Storage - see
 * lowering.hpp for why an offset view into one big buffer would be silently
 * mis-read by almost every SushiBLAS kernel.
 *
 * @section synthetic Why a synthetic dataset is the test gate
 *
 * @c SyntheticClassification generates a linearly separable multi-class
 * problem from the engine's own seeded Philox stream. It needs no files, no
 * network and no fixture, so "does this library actually train" is a
 * hermetic test rather than a manual check - and its separability means a
 * correct implementation must reach near-perfect accuracy, so the assertion
 * can be tight instead of "the loss went down a bit".
 *
 * @warning Reproducibility is a property of a @e fresh Engine, not of the
 *          seed alone. SushiBLAS's @c Engine::set_seed() sets the seed but
 *          leaves the engine's monotonically increasing RNG offset counter
 *          untouched, and every RNG kernel mixes that counter into the Philox
 *          stream. Two generators built back to back on one engine therefore
 *          produce different data from the same seed. Build a fresh Engine
 *          when you need byte-identical data twice.
 */

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <string_view>
#include <vector>
#include <SushiBLAS/engine.hpp>
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
         * @brief A deterministic, seeded, linearly separable multi-class problem.
         *
         * Each class gets a random mean vector; a sample is its class mean plus
         * Gaussian noise. Both come from @c engine.random(), so the data is
         * generated where it will be consumed and the only host work is the
         * deterministic one-hot pattern - which is a fixed integer layout, not
         * randomness, and so needs no device kernel to be reproducible.
         *
         * Samples are assigned to classes round-robin, so every batch is
         * exactly balanced and accuracy needs no class weighting to read
         * correctly.
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
         * @brief Loads MNIST from IDX files, or returns null when they are absent.
         *
         * Absence is not an error: MNIST is 11 MB of files this repository does
         * not ship, so the demo and the tests fall back to the synthetic
         * dataset and say so. A file that is present but malformed @e is an
         * error, because that is a broken download rather than an absent one.
         *
         * @param engine The engine that allocates and fills the tensors.
         * @param options Where to look, which split, and how much.
         * @return The loaded dataset, or nullptr when the files are not there.
         * @throws Error If a file exists but its IDX header or size is wrong.
         */
        std::unique_ptr<Dataset> try_load_mnist(SushiBLAS::Engine& engine,
                                                const MnistOptions& options = {});

    } // namespace Data
} // namespace SushiAI
