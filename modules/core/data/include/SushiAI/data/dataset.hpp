/**************************************************************************/
/* dataset.hpp                                                            */
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
 * @file dataset.hpp
 * @brief Declares dataset interfaces and implementations for model training.
 * @author Mustafa Garip
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <vector>
#include <SushiBLAS/engine/engine.hpp>
#include <SushiAI/config/value.hpp>
#include <SushiAI/core/common.hpp>
#include <SushiAI/core/result.hpp>
#include <SushiAI/core/shape.hpp>

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
            /** @brief An [N, sample...] tensor, where sample is Dataset::sample_shape(). */
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

                /** @brief Returns the shape of one sample, batch axis left out. */
                virtual Shape sample_shape() const = 0;

                /** @brief How many classes the targets span. */
                virtual int64_t num_classes() const noexcept = 0;

                /** @brief How many batches an epoch holds. */
                virtual std::size_t batch_count() const noexcept = 0;

                /**
                 * @brief One batch.
                 * @param index Which batch, in [0, batch_count()).
                 * @return The batch; owned by the dataset.
                 * @pre @p index is in range; one that is not stops the process.
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
                 * @return The dataset, or InvalidDescription when an extent in @p options is
                 *         not positive, or the failure of an allocation or of the fill.
                 */
                [[nodiscard]] static Result<std::unique_ptr<SyntheticClassification>> make(
                    SushiBLAS::Engine& engine,
                    SyntheticOptions options = {});

                int64_t batch_size() const noexcept override { return options_.batch_size; }
                int64_t input_dim() const noexcept override { return options_.features; }
                Shape sample_shape() const override { return Shape{options_.features}; }
                int64_t num_classes() const noexcept override { return options_.classes; }
                std::size_t batch_count() const noexcept override { return batches_.size(); }
                const Batch& batch(std::size_t index) const override;
                std::string_view name() const noexcept override
                {
                    return "synthetic-classification";
                }

            private:
                /** @brief Holds @p batches, which were generated from @p options. */
                SyntheticClassification(SyntheticOptions options,
                                        std::vector<Batch> batches) noexcept;

                /** @brief Holds the shape the batches were generated from. */
                SyntheticOptions options_;

                /** @brief Holds every batch of the epoch, in order. */
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
                 * @return The dataset, or InvalidDescription when @p options holds an extent,
                 *         a width or a noise out of range, or the failure of an allocation.
                 */
                [[nodiscard]] static Result<std::unique_ptr<Rings>> make(
                    SushiBLAS::Engine& engine,
                    RingsOptions options = {});

                int64_t batch_size() const noexcept override { return options_.batch_size; }
                int64_t input_dim() const noexcept override
                {
                    return 2 + options_.distractor_features;
                }
                Shape sample_shape() const override { return Shape{input_dim()}; }
                int64_t num_classes() const noexcept override { return options_.classes; }
                std::size_t batch_count() const noexcept override { return batches_.size(); }
                const Batch& batch(std::size_t index) const override;
                std::string_view name() const noexcept override { return "rings"; }

            private:
                /** @brief Holds @p batches, which were generated from @p options. */
                Rings(RingsOptions options, std::vector<Batch> batches) noexcept;

                /** @brief Holds the shape the batches were generated from. */
                RingsOptions options_;

                /** @brief Holds every batch of the epoch, in order. */
                std::vector<Batch> batches_;
        };

        /**
         * @brief Builds a @ref Rings dataset from configuration attributes.
         * @param attributes Dataset parameters dictionary.
         * @param context Diagnostic context string.
         * @return The dataset, never null, or the failure Rings::make returns.
         */
        [[nodiscard]] Result<std::unique_ptr<Dataset>> make_rings(SushiBLAS::Engine& engine,
                                                                  const Config::Value& attributes,
                                                                  std::string_view context);

    } // namespace Data
} // namespace SushiAI
