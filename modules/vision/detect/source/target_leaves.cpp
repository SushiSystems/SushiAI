/**************************************************************************/
/* target_leaves.cpp                                                      */
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
 * @file target_leaves.cpp
 * @brief Implements the full-shape arrays a masked detection loss reads from an assignment.
 * @author Mustafa Garip
 * @see include/SushiAI/vision/detect/README.md
 */

#include <SushiAI/vision/detect/target_leaves.hpp>

#include <algorithm>
#include <cstddef>

#include <SushiAI/core/common.hpp>
#include <SushiAI/vision/detect/anchors.hpp>
#include <SushiAI/vision/detect/box_coding.hpp>
#include <SushiAI/vision/detect/box_format.hpp>

namespace SushiAI
{
    namespace Vision::Detect
    {
        namespace
        {
            /** @brief Holds the factor that averages the four distance errors of a row. */
            constexpr float K_SIDE_MEAN = 0.25f;

            /** @brief Throws unless the assignment and the geometry describe the same anchors. */
            void check_leaf_inputs(const Assignment& assignment, const TargetLeafInputs& inputs)
            {
                SA_THROW_IF(assignment.batch <= 0 || assignment.anchors <= 0 ||
                                assignment.num_classes <= 0,
                            "target leaves need an assignment with positive extents; got batch "
                            "{}, {} anchors and {} classes",
                            assignment.batch, assignment.anchors, assignment.num_classes);

                const std::size_t anchors = static_cast<std::size_t>(assignment.anchors);
                const std::size_t rows = static_cast<std::size_t>(assignment.batch) * anchors;
                const std::size_t classes = static_cast<std::size_t>(assignment.num_classes);
                SA_THROW_IF(assignment.foreground_mask.size() != rows ||
                                assignment.target_boxes.size() != rows * K_BOX_VALUES ||
                                assignment.target_scores.size() != rows * classes,
                            "the assignment's arrays do not match its extents: {} mask, {} box "
                            "and {} score values for {} rows of {} classes",
                            assignment.foreground_mask.size(), assignment.target_boxes.size(),
                            assignment.target_scores.size(), rows, classes);
                SA_THROW_IF(inputs.anchor_points.size() != anchors * K_POINT_VALUES ||
                                inputs.strides.size() != anchors,
                            "target leaves need [{}, 2] anchor points and [{}] strides; got {} "
                            "and {} values",
                            anchors, anchors, inputs.anchor_points.size(), inputs.strides.size());
                SA_THROW_IF(!(inputs.image_height > 0.0f) || !(inputs.image_width > 0.0f),
                            "target leaves need a positive image size; got {} by {}",
                            inputs.image_height, inputs.image_width);
            }

            /** @brief Writes each row's score sum and returns the sum over all rows. */
            double sum_weights(const Assignment& assignment, std::vector<float>& weight)
            {
                const std::size_t classes = static_cast<std::size_t>(assignment.num_classes);
                double total = 0.0;
                for (std::size_t row = 0; row < weight.size(); ++row)
                {
                    const float* scores = assignment.target_scores.data() + row * classes;
                    float sum = 0.0f;
                    for (std::size_t column = 0; column < classes; ++column)
                        sum += scores[column];
                    weight[row] = sum;
                    total += static_cast<double>(sum);
                }
                return total;
            }
        } // namespace

        void build_target_leaves(const Assignment& assignment,
                                 const TargetLeafInputs& inputs,
                                 TargetLeaves& leaves)
        {
            check_leaf_inputs(assignment, inputs);

            const std::size_t anchors = static_cast<std::size_t>(assignment.anchors);
            const std::size_t rows = static_cast<std::size_t>(assignment.batch) * anchors;
            leaves.box_mask.assign(rows * K_BOX_VALUES, 0.0f);
            leaves.weight.assign(rows, 0.0f);
            leaves.normalized_weight.assign(rows, 0.0f);
            leaves.target_boxes.resize(rows * K_BOX_VALUES);
            leaves.target_distances.assign(rows * K_BOX_VALUES, 0.0f);
            leaves.l1_scale.assign(rows * K_BOX_VALUES, 0.0f);
            leaves.safe_boxes.resize(rows * K_BOX_VALUES);

            const float total = static_cast<float>(sum_weights(assignment, leaves.weight));
            const float scores_sum = std::max(total, K_MIN_TARGET_SCORES_SUM);
            leaves.target_scores_sum[0] = scores_sum;

            for (std::size_t row = 0; row < rows; ++row)
            {
                const std::size_t at = row * K_BOX_VALUES;
                std::copy(K_SAFE_BOX.begin(), K_SAFE_BOX.end(), leaves.safe_boxes.begin() + at);
                std::copy(K_SAFE_BOX.begin(), K_SAFE_BOX.end(), leaves.target_boxes.begin() + at);
                if (assignment.foreground_mask[row] == 0)
                    continue;

                const std::size_t anchor = row % anchors;
                const float stride = inputs.strides[anchor];
                const SushiRuntime::span<float> box(leaves.target_boxes.data() + at, K_BOX_VALUES);
                const float share = leaves.weight[row] / scores_sum;
                const float across = stride / inputs.image_width * K_SIDE_MEAN * share;
                const float down = stride / inputs.image_height * K_SIDE_MEAN * share;

                leaves.normalized_weight[row] = share;
                std::fill_n(leaves.box_mask.begin() + at, K_BOX_VALUES, 1.0f);
                for (std::size_t column = 0; column < K_BOX_VALUES; ++column)
                    box[column] = assignment.target_boxes[at + column] / stride;
                encode_distances(
                    inputs.anchor_points.subspan(anchor * K_POINT_VALUES, K_POINT_VALUES), box,
                    SushiRuntime::span<float>(leaves.target_distances.data() + at, K_BOX_VALUES));
                leaves.l1_scale[at] = across;
                leaves.l1_scale[at + 1] = down;
                leaves.l1_scale[at + 2] = across;
                leaves.l1_scale[at + 3] = down;
            }
        }

    } // namespace Vision::Detect
} // namespace SushiAI
