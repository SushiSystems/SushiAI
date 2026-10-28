/**************************************************************************/
/* free_batch_decoder.hpp                                                 */
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
 * @file free_batch_decoder.hpp
 * @brief Declares the decode of one batch of anchor-free head outputs into boxes and scores.
 * @author Mustafa Garip
 */

#pragma once

#include <cstdint>
#include <vector>

#include <SushiRuntime/core/span.hpp>

#include <SushiAI/failure/result.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/vision/detect/free/anchors.hpp>
#include <SushiAI/vision/training/free_model_facts.hpp>

namespace SushiAI::Vision::Training
{
    /** @brief Holds the one-to-many outputs of one level, as the step publishes them. */
    struct FreeLevelLogits
    {
        SushiRuntime::span<const float> boxes;  /**< `[B, H, W, 4 * reg_max]` bin logits. */
        SushiRuntime::span<const float> scores; /**< `[B, H, W, C]` class logits. */
    };

    /**
     * @brief Decodes the outputs of every level of one batch into the rows the suppression
     *        reads: `[B, A, 4]` centre x, centre y, width, height in input pixels and
     *        `[B, A, C]` class probabilities, levels in order, row-major in a level.
     */
    class FreeBatchDecoder final
    {
        public:
            /**
             * @brief Prepares the anchors of @p facts for batches of @p batch images.
             * @return The decoder, or InvalidDescription when @p facts holds no level, a level
             *         without cells or stride, a reg_max or a class count below 1, or @p batch is
             *         below 1.
             */
            [[nodiscard]] static Result<FreeBatchDecoder> make(const FreeModelFacts& facts,
                                                               std::int64_t batch) noexcept;

            /** @brief Returns the anchors of one image over every level, A. */
            [[nodiscard]] std::int64_t anchors() const noexcept;

            /**
             * @brief Writes the boxes and the class probabilities of one batch.
             * @param levels One entry per level of the facts, finest first.
             * @param boxes Receives `[B, A, 4]` centre x, centre y, width, height in pixels.
             * @param scores Receives `[B, A, C]`, the sigmoid of each class logit.
             * @return InvalidDescription when a span holds other extents, or for a NaN or an
             *         infinity in a level, naming the level and the index.
             */
            [[nodiscard]] Status decode(SushiRuntime::span<const FreeLevelLogits> levels,
                                        SushiRuntime::span<float> boxes,
                                        SushiRuntime::span<float> scores) noexcept;

        private:
            /** @brief Stores the extents of checked @p facts for batches of @p batch images. */
            FreeBatchDecoder(const FreeModelFacts& facts, std::int64_t batch);

            /** @brief Builds the anchors and sizes the working arrays. */
            [[nodiscard]] Status prepare_anchors();

            /** @brief Bins per box side. */
            std::int64_t reg_max_ = 0;

            /** @brief Class logits per anchor. */
            std::int64_t classes_ = 0;

            /** @brief Images per batch, B. */
            std::int64_t batch_ = 0;

            /** @brief Anchors of one image over every level, A. */
            std::int64_t anchors_ = 0;

            /** @brief Extents and stride of each level, finest first. */
            std::vector<Detect::Free::FeatureLevel> levels_;

            /** @brief `[A, 2]` cell centres in grid units. */
            std::vector<float> points_;

            /** @brief `[A]` stride of each anchor. */
            std::vector<float> strides_;

            /** @brief `[B, A, 4 * reg_max]` bin logits joined over the levels. */
            std::vector<float> bin_logits_;

            /** @brief `[B, A, 4]` expected distances in grid units. */
            std::vector<float> distances_;
    };
} // namespace SushiAI::Vision::Training
