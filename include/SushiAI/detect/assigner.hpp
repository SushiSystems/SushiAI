/**************************************************************************/
/* assigner.hpp                                                           */
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
 * @file assigner.hpp
 * @brief Declares TaskAlignedAssigner, which gives ground-truth boxes to anchors on the host.
 * @author Mustafa Garip
 * @see include/SushiAI/detect/README.md
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include <SushiRuntime/core/span.hpp>

#include <SushiAI/detect/assigner_config.hpp>
#include <SushiAI/detect/assignment.hpp>

namespace SushiAI
{
    namespace Detect
    {
        /**
         * @brief Holds the predictions and the ground truth of one assigner call.
         * @see include/SushiAI/detect/README.md
         */
        struct AssignerInputs
        {
            std::int64_t batch = 0;     /**< Images, B. */
            std::int64_t anchors = 0;   /**< Anchors over all levels, A. */
            std::int64_t max_boxes = 0; /**< Ground-truth rows per image, M; may be 0. */

            SushiRuntime::span<const float> scores;           /**< `[B, A, C]` probabilities. */
            SushiRuntime::span<const float> boxes;            /**< `[B, A, 4]` pixel x1 y1 x2 y2. */
            SushiRuntime::span<const float> anchor_points;    /**< `[A, 2]` x then y, pixels. */
            SushiRuntime::span<const std::int64_t> gt_labels; /**< `[B, M]` class indices. */
            SushiRuntime::span<const float> gt_boxes;         /**< `[B, M, 4]` pixel x1 y1 x2 y2. */
            SushiRuntime::span<const std::uint8_t> gt_valid;  /**< `[B, M]`; 0 in a padding row. */
        };

        /**
         * @brief Assigns ground-truth boxes to anchors by the task-aligned metric.
         * @see include/SushiAI/detect/README.md
         */
        class TaskAlignedAssigner final
        {
            public:
                /** @brief Stores @p config after checking its counts, stride_val and eps. */
                explicit TaskAlignedAssigner(const AssignerConfig& config);

                /**
                 * @brief Writes the targets of every anchor of @p inputs into @p assignment.
                 * @pre No input value is NaN.
                 * @see include/SushiAI/detect/README.md
                 */
                void assign(const AssignerInputs& inputs, Assignment& assignment);

                /** @brief Returns the settings this assigner was built with. */
                [[nodiscard]] const AssignerConfig& config() const noexcept
                {
                    return config_;
                }

            private:
                /** @brief Throws unless the arrays and the labels of @p inputs fit its shape. */
                void check_inputs(const AssignerInputs& inputs) const;

                /** @brief Fills the metric, the overlap and the first top-k mask of one image. */
                void select_candidates(const AssignerInputs& inputs, std::size_t image);

                /** @brief Leaves each anchor in the mask of at most one box. */
                void resolve_conflicts(const AssignerInputs& inputs);

                /** @brief Leaves each box of @p image its topk2 held anchors of largest metric. */
                void keep_second_top_k(const AssignerInputs& inputs, std::size_t image);

                /** @brief Writes one image's targets from the final mask. */
                void write_targets(const AssignerInputs& inputs,
                                   std::size_t image,
                                   Assignment& assignment);

                /// @brief Holds the settings of this assigner.
                AssignerConfig config_;

                /// @brief Holds the alignment metric of one image, `[M, A]`; zero elsewhere.
                std::vector<float> metric_;

                /// @brief Holds the clamped complete IoU of one image, `[M, A]`; zero elsewhere.
                std::vector<float> overlap_;

                /// @brief Holds the positive mask of one image, `[M, A]`.
                std::vector<std::uint8_t> mask_;

                /// @brief Holds the number of boxes whose mask has each anchor of one image, `[A]`.
                std::vector<std::int32_t> holders_;

                /// @brief Holds the largest masked metric of each box of one image, `[M]`.
                std::vector<float> peak_metric_;

                /// @brief Holds the largest masked overlap of each box of one image, `[M]`.
                std::vector<float> peak_overlap_;

                /// @brief Holds one row of masked metric values, `[A]`.
                std::vector<float> row_;

                /// @brief Holds the indices a top-k selection returns.
                std::vector<std::int64_t> selected_;
        };

    } // namespace Detect
} // namespace SushiAI
