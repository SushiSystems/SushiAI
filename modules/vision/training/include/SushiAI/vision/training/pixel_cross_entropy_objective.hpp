/**************************************************************************/
/* pixel_cross_entropy_objective.hpp                                      */
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
 * @file pixel_cross_entropy_objective.hpp
 * @brief Declares the per-pixel cross entropy: the mean over the pixels a mask does not ignore.
 * @author Mustafa Garip
 */

#pragma once

#include <cstdint>
#include <memory>
#include <string_view>
#include <SushiAI/config/value.hpp>
#include <SushiAI/core/common.hpp>
#include <SushiAI/core/result.hpp>
#include <SushiAI/core/status.hpp>
#include <SushiAI/data/dataset.hpp>
#include <SushiAI/graph/builder.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/graph/tensor_pool.hpp>
#include <SushiAI/nn/model_outputs.hpp>
#include <SushiAI/train/metrics.hpp>
#include <SushiAI/train/objective.hpp>
#include <SushiAI/train/objective_context.hpp>
#include <SushiAI/train/step_view.hpp>
#include <SushiAI/vision/segment/ignore_index.hpp>
#include <SushiAI/vision/training/pixel_extents.hpp>

namespace SushiAI::Vision::Training
{
    /** @brief Holds the "type" a training file states for the per-pixel cross entropy. */
    inline constexpr std::string_view K_PIXEL_CROSS_ENTROPY_TYPE = "pixel_cross_entropy";

    /** @brief Holds the role of the INT32 [R] leaf of log-probability indices. */
    inline constexpr std::string_view K_PIXEL_TARGETS_LEAF = "targets";

    /** @brief Holds the role of the [R, 1] leaf of pixel weights. */
    inline constexpr std::string_view K_PIXEL_WEIGHTS_LEAF = "weights";

    /**
     * @brief Computes the mean cross entropy of [N, H, W, C] logits against an [N, H, W] mask,
     *        over the pixels that are not the ignore index.
     *
     * The loss reads one log-probability per pixel through K_GATHER_ROWS, so no one-hot
     * leaf is held. README.md states what the two leaves cost.
     */
    class PixelCrossEntropyObjective final : public Train::Objective
    {
        public:
            /**
             * @brief Builds the objective over the logits @p extents describe.
             * @param dtype The element type of the weight leaf, float32 or float64.
             * @return The objective, or InvalidDescription for an extent below 1, for
             *         N * H * W * C above what INT32 indexes, for an ignore index that names
             *         a class, or for another element type.
             */
            [[nodiscard]] static Result<std::unique_ptr<PixelCrossEntropyObjective>> make(
                const PixelExtents& extents,
                std::int32_t ignore_index = Segment::K_DEFAULT_IGNORE_INDEX,
                DType dtype = DType::Float32);

            /** @brief Declares the two leaves and requires the INT32 [H, W] mask role. */
            void declare(const Graph::GraphBuilder& builder,
                         Train::ObjectiveLeaves& leaves) const override;

            /**
             * @brief Emits emit_pixel_cross_entropy() over the model's only output.
             * @pre @p outputs holds one NHWC value of the extents make() was given; anything
             *      else stops the process.
             */
            Graph::ValueID loss(const Graph::GraphBuilder& builder, const Graph::Graph& graph,
                                const NN::ModelOutputs& outputs,
                                const Train::ObjectiveLeaves& leaves) const override;

            /**
             * @brief Writes both leaves from the mask role with fill_pixel_targets().
             * @return InvalidDescription for a batch with no mask role, or what the fill
             *         refuses.
             */
            [[nodiscard]] Status feed(const Data::Batch& batch,
                                      const Train::ObjectiveLeaves& leaves,
                                      Graph::TensorPool& pool) const override;

            /** @brief Returns true: a batch alone fills the leaves. */
            [[nodiscard]] bool is_fed_by_batch() const noexcept override { return true; }

            /** @brief Appends "mean_iou" and "pixel_accuracy" with measure_segmentation(). */
            [[nodiscard]] Status measure(const Train::StepView& step,
                                         Train::Metrics& metrics) const override;

            /** @brief Returns "pixel_cross_entropy". */
            std::string_view name() const noexcept override
            {
                return K_PIXEL_CROSS_ENTROPY_TYPE;
            }

            /** @brief Returns the extents of the logits. */
            [[nodiscard]] const PixelExtents& extents() const noexcept { return extents_; }

            /** @brief Returns the mask value of a pixel that takes no part in the loss. */
            [[nodiscard]] std::int32_t ignore_index() const noexcept { return ignore_index_; }

        private:
            /** @brief Stores what make() has checked. */
            PixelCrossEntropyObjective(const PixelExtents& extents, std::int32_t ignore_index,
                                       DType dtype) noexcept;

            /// @brief The extents of the logits.
            PixelExtents extents_{};

            /// @brief The mask value of a pixel that takes no part in the loss.
            std::int32_t ignore_index_ = Segment::K_DEFAULT_IGNORE_INDEX;

            /// @brief The element type of the weight leaf.
            DType dtype_ = DType::Float32;
    };

    /**
     * @brief Builds a per-pixel cross entropy from the one output of @p given.
     * @param attributes The objective's own keys; it reads "ignore_index" alone.
     * @param context What to call this objective in a diagnostic.
     * @return The objective, or InvalidDescription for a key, for a model that does not
     *         publish one [N, H, W, C] output at the size of its input, for data of more
     *         classes than C, or for what PixelCrossEntropyObjective::make refuses.
     */
    [[nodiscard]] Result<std::unique_ptr<Train::Objective>> make_pixel_cross_entropy_objective(
        const Train::ObjectiveContext& given, const Config::Value& attributes,
        std::string_view context);

} // namespace SushiAI::Vision::Training
