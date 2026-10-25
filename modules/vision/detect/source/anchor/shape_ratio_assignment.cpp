/**************************************************************************/
/* shape_ratio_assignment.cpp                                             */
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
 * @file shape_ratio_assignment.cpp
 * @brief Implements the assignment of truths to every anchor within a side ratio.
 * @author Mustafa Garip
 * @see modules/vision/detect/README.md
 */

#include <SushiAI/vision/detect/anchor/shape_ratio_assignment.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <tuple>

#include <SushiAI/vision/detect/anchor/anchor_size.hpp>
#include <SushiAI/vision/detect/anchor/head_truth.hpp>
#include <SushiAI/vision/detect/float_order.hpp>

#include "truth_box.hpp"

namespace SushiAI
{
    namespace Vision::Detect::Anchor
    {
        namespace
        {
            /** @brief Holds one assignment before collisions on its slot are resolved. */
            struct Candidate
            {
                TargetAssignment assignment; /**< The assignment as the truth would write it. */
                float score = 0.0f;          /**< shape_iou of the truth and the slot's anchor. */
                std::int64_t truth = 0;      /**< Index of the truth row, for ties. */
            };

            /** @brief Holds one cell offset, in units of the neighbour offset. */
            struct CellOffset
            {
                float x = 0.0f; /**< Subtracted from the centre along the columns. */
                float y = 0.0f; /**< Subtracted from the centre along the rows. */
            };

            /** @brief Returns the fractional part of a value that is not negative. */
            float fraction_of(float value) noexcept
            {
                return value - std::floor(value);
            }

            /** @brief Reports whether @p truth fits @p anchor within the side ratio @p limit. */
            bool fits_ratio(float truth_width, float truth_height, float anchor_width,
                            float anchor_height, float limit) noexcept
            {
                const float ratio_w = truth_width / anchor_width;
                const float ratio_h = truth_height / anchor_height;
                const float worst = std::max(std::max(ratio_w, 1.0f / ratio_w),
                                             std::max(ratio_h, 1.0f / ratio_h));
                return worst < limit;
            }

            /** @brief Returns @p value truncated to a cell index and clamped to [0, extent). */
            std::int64_t clamp_cell(float value, std::int64_t extent) noexcept
            {
                const std::int64_t cell = static_cast<std::int64_t>(value);
                return std::min(std::max(cell, std::int64_t{0}), extent - 1);
            }

            /**
             * @brief Returns the offsets of the cells a centre takes: its own first, then the
             *        left, top, right and bottom neighbours it lies near, in that order.
             */
            std::vector<CellOffset> select_offsets(float grid_x, float grid_y, float width,
                                                   float height, float offset)
            {
                std::vector<CellOffset> offsets{{0.0f, 0.0f}};
                const float inverse_x = width - grid_x;
                const float inverse_y = height - grid_y;
                if (fraction_of(grid_x) < offset && grid_x > 1.0f)
                    offsets.push_back({offset, 0.0f});
                if (fraction_of(grid_y) < offset && grid_y > 1.0f)
                    offsets.push_back({0.0f, offset});
                if (fraction_of(inverse_x) < offset && inverse_x > 1.0f)
                    offsets.push_back({-offset, 0.0f});
                if (fraction_of(inverse_y) < offset && inverse_y > 1.0f)
                    offsets.push_back({0.0f, -offset});
                return offsets;
            }

            /** @brief Reports whether @p left comes before @p right in the result's order. */
            bool precedes(const TargetAssignment& left, const TargetAssignment& right) noexcept
            {
                return std::tie(left.image, left.slot, left.row, left.col) <
                       std::tie(right.image, right.slot, right.row, right.col);
            }

            /** @brief Reports whether two assignments name the same slot of the same cell. */
            bool same_place(const TargetAssignment& left, const TargetAssignment& right) noexcept
            {
                return !precedes(left, right) && !precedes(right, left);
            }
        } // namespace

        Status validate_shape_ratio_options(const ShapeRatioOptions& options)
        {
            SA_FAIL_IF(!std::isfinite(options.ratio_limit) || !(options.ratio_limit > 1.0f),
                       StatusCode::InvalidDescription,
                       "'ratio_limit' must be finite and above 1; got {}", options.ratio_limit);
            SA_FAIL_IF(!std::isfinite(options.neighbour_offset) ||
                           options.neighbour_offset < 0.0f || options.neighbour_offset > 0.5f,
                       StatusCode::InvalidDescription,
                       "'neighbour_offset' must be finite and in [0, 0.5]; got {}",
                       options.neighbour_offset);
            return Status();
        }

        ShapeRatioAssignment::ShapeRatioAssignment(const ShapeRatioOptions& options) noexcept
            : options_(options)
        {

        }

        Status ShapeRatioAssignment::assign(const AssignmentInput& input,
                                            std::vector<TargetAssignment>& result) const
        {
            const AnchorLayout& layout = *input.layout;
            const GroundTruthBatch& truth = *input.truth;
            const std::int64_t heads = static_cast<std::int64_t>(layout.head_anchors.size());
            SA_FAIL_IF(input.head_index < 0 || input.head_index >= heads,
                       StatusCode::InvalidDescription, "head {} is not one of the layout's {}",
                       input.head_index, heads);
            SA_FAIL_IF(input.shape.height < 1 || input.shape.width < 1,
                       StatusCode::InvalidDescription, "a head grid of {} by {} holds no cell",
                       input.shape.height, input.shape.width);
            SA_TRY(validate_head_truth(truth, input.shape.batch, input.classes, input.network));

            const std::vector<std::int64_t>& mask =
                layout.head_anchors[static_cast<std::size_t>(input.head_index)];
            const float width = static_cast<float>(input.shape.width);
            const float height = static_cast<float>(input.shape.height);
            const std::size_t rows = static_cast<std::size_t>(truth.max_boxes);

            std::vector<Candidate> candidates;
            for (std::size_t index = 0; index < truth.valid.size(); ++index)
            {
                if (truth.valid[index] == 0)
                    continue;

                const Detail::CentreBox box =
                    Detail::read_truth_box(truth.boxes.data() + index * 4, input.network);
                const float grid_x = box.x * width;
                const float grid_y = box.y * height;
                const float grid_w = box.width * width;
                const float grid_h = box.height * height;
                const std::vector<CellOffset> offsets = select_offsets(
                    grid_x, grid_y, width, height, options_.neighbour_offset);

                for (std::size_t slot = 0; slot < mask.size(); ++slot)
                {
                    const AnchorSize anchor = layout.anchors[static_cast<std::size_t>(mask[slot])];
                    if (!fits_ratio(grid_w, grid_h, anchor.width * width, anchor.height * height,
                                    options_.ratio_limit))
                        continue;

                    const float score = shape_iou(anchor, AnchorSize{box.width, box.height});
                    for (const CellOffset& offset : offsets)
                    {
                        Candidate candidate;
                        TargetAssignment& entry = candidate.assignment;
                        entry.image = static_cast<std::int64_t>(index / rows);
                        entry.slot = static_cast<std::int64_t>(slot);
                        entry.col = clamp_cell(grid_x - offset.x, input.shape.width);
                        entry.row = clamp_cell(grid_y - offset.y, input.shape.height);
                        entry.box_x = grid_x - static_cast<float>(entry.col);
                        entry.box_y = grid_y - static_cast<float>(entry.row);
                        entry.box_width = grid_w;
                        entry.box_height = grid_h;
                        entry.label = truth.labels[index];
                        entry.truth_row = static_cast<std::int64_t>(index % rows);
                        candidate.score = score;
                        candidate.truth = static_cast<std::int64_t>(index);
                        candidates.push_back(candidate);
                    }
                }
            }

            // The higher shape_iou wins a shared slot; the lower truth index breaks a tie.
            std::sort(candidates.begin(), candidates.end(),
                      [](const Candidate& left, const Candidate& right)
                      {
                          if (precedes(left.assignment, right.assignment))
                              return true;
                          if (precedes(right.assignment, left.assignment))
                              return false;
                          if (ranks_above(left.score, right.score))
                              return true;
                          if (ranks_above(right.score, left.score))
                              return false;
                          return left.truth < right.truth;
                      });

            result.clear();
            for (const Candidate& candidate : candidates)
            {
                if (!result.empty() && same_place(result.back(), candidate.assignment))
                    continue;
                result.push_back(candidate.assignment);
            }
            return Status();
        }

    } // namespace Vision::Detect::Anchor
} // namespace SushiAI
