/**************************************************************************/
/* free_detection_loss_host.cpp                                           */
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

/**
 * @file free_detection_loss_host.cpp
 * @brief Implements a host evaluation of the anchor-free detection loss on double arrays.
 * @author Mustafa Garip
 */

#include <SushiAI/vision/training/free_detection_loss_host.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>

#include <SushiAI/failure/result.hpp>
#include <SushiAI/failure/status.hpp>

namespace SushiAI::Vision::Training
{
    namespace
    {
        /** @brief Names the values of a box: four sides or four corners. */
        constexpr std::size_t K_BOX_VALUES = 4;

        /** @brief Names the values of an anchor point: x then y. */
        constexpr std::size_t K_POINT_VALUES = 2;

        /** @brief Holds the epsilon of every CIoU guard, as the graph has it. */
        constexpr double K_EPSILON = 1e-7;

        /** @brief Holds 4 / pi^2, the factor of CIoU's aspect term. */
        constexpr double K_ASPECT_FACTOR = 0.40528473456935108578;

        /** @brief Holds one box, x1 y1 x2 y2. */
        using Box = std::array<double, K_BOX_VALUES>;

        /** @brief Returns log(1 + exp(x)) without overflow. */
        double softplus(double x)
        {
            return std::max(x, 0.0) + std::log1p(std::exp(-std::abs(x)));
        }

        /** @brief Returns the largest of @p bins logits. */
        double largest(const double* logits, std::size_t bins)
        {
            return *std::max_element(logits, logits + bins);
        }

        /** @brief Returns log(sum(exp(logits))) of @p bins logits. */
        double log_sum_exp(const double* logits, std::size_t bins)
        {
            const double top = largest(logits, bins);
            double sum = 0.0;
            for (std::size_t bin = 0; bin < bins; ++bin)
                sum += std::exp(logits[bin] - top);
            return top + std::log(sum);
        }

        /** @brief Returns the expected bin index of @p bins logits under their softmax. */
        double expected_bin(const double* logits, std::size_t bins)
        {
            const double normaliser = log_sum_exp(logits, bins);
            double expected = 0.0;
            for (std::size_t bin = 0; bin < bins; ++bin)
                expected += static_cast<double>(bin) * std::exp(logits[bin] - normaliser);
            return expected;
        }

        /** @brief Returns the CIoU of @p p against @p t, the graph's formula in double. */
        double complete_iou(const Box& p, const Box& t)
        {
            const double w1 = p[2] - p[0];
            const double h1 = p[3] - p[1] + K_EPSILON;
            const double w2 = t[2] - t[0];
            const double h2 = t[3] - t[1] + K_EPSILON;

            const double inter_w = std::max(0.0, std::min(p[2], t[2]) - std::max(p[0], t[0]));
            const double inter_h = std::max(0.0, std::min(p[3], t[3]) - std::max(p[1], t[1]));
            const double inter = inter_w * inter_h;
            const double iou = inter / (w1 * h1 + w2 * h2 - inter + K_EPSILON);

            const double cw = std::max(p[2], t[2]) - std::min(p[0], t[0]);
            const double ch = std::max(p[3], t[3]) - std::min(p[1], t[1]);
            const double c2 = cw * cw + ch * ch + K_EPSILON;
            const double dx = t[0] + t[2] - p[0] - p[2];
            const double dy = t[1] + t[3] - p[1] - p[3];
            const double rho2 = (dx * dx + dy * dy) * 0.25;

            const double angle = std::atan(w2 / h2) - std::atan(w1 / h1);
            const double v = K_ASPECT_FACTOR * angle * angle;
            const double alpha = v / (v - iou + (1.0 + K_EPSILON));
            return iou - (rho2 / c2 + v * alpha);
        }

        /** @brief Refuses @p values unless it holds @p rows times @p per_row; names it @p what. */
        Status require_size(SushiRuntime::span<const double> values, std::size_t rows,
                            std::size_t per_row, const char* what)
        {
            SA_FAIL_IF(values.size() != rows * per_row, StatusCode::InvalidDescription,
                       "the {} hold {} values; {} rows of {} need {}", what, values.size(),
                       rows, per_row, rows * per_row);
            return Status();
        }

        /** @brief Adds the terms of one level to @p terms, or refuses a level of wrong sizes. */
        Status add_level(const FreeLevelLossArrays& level, std::size_t bins, std::size_t classes,
                         const FreeLossGains& gains, FreeLossTerms& terms)
        {
            const std::size_t rows = level.scores.size() / classes;
            const std::size_t box_channels = K_BOX_VALUES * bins;
            SA_TRY(require_size(level.boxes, rows, box_channels, "box outputs"));
            SA_TRY(require_size(level.scores, rows, classes, "scores"));
            SA_TRY(require_size(level.target_scores, rows, classes, "target scores"));
            SA_TRY(require_size(level.box_mask, rows, K_BOX_VALUES, "box mask"));
            SA_TRY(require_size(level.normalized_weight, rows, 1, "normalized weights"));
            SA_TRY(require_size(level.target_boxes, rows, K_BOX_VALUES, "target boxes"));
            SA_TRY(require_size(level.safe_boxes, rows, K_BOX_VALUES, "safe boxes"));
            SA_TRY(require_size(level.anchor_points, rows, K_POINT_VALUES, "anchor points"));
            if (bins > 1)
            {
                SA_TRY(require_size(level.distribution_weights, rows, box_channels,
                                    "distribution weights"));
            }
            if (gains.l1_gain > 0.0f)
            {
                SA_TRY(require_size(level.target_distances, rows, K_BOX_VALUES,
                                    "target distances"));
                SA_TRY(require_size(level.l1_scale, rows, K_BOX_VALUES, "L1 scales"));
            }

            for (std::size_t row = 0; row < rows; ++row)
            {
                for (std::size_t c = 0; c < classes; ++c)
                {
                    const double x = level.scores[row * classes + c];
                    terms.classes += softplus(x) - x * level.target_scores[row * classes + c];
                }

                const double* outputs = level.boxes.data() + row * box_channels;
                Box distances{};
                for (std::size_t side = 0; side < K_BOX_VALUES; ++side)
                {
                    const double* side_bins = outputs + side * bins;
                    distances[side] = bins == 1 ? side_bins[0] : expected_bin(side_bins, bins);
                    if (bins > 1)
                    {
                        const double normaliser = log_sum_exp(side_bins, bins);
                        for (std::size_t bin = 0; bin < bins; ++bin)
                            terms.distribution -=
                                (side_bins[bin] - normaliser) *
                                level.distribution_weights[row * box_channels + side * bins + bin];
                    }
                }

                const double x = level.anchor_points[row * K_POINT_VALUES];
                const double y = level.anchor_points[row * K_POINT_VALUES + 1];
                const Box decoded = {x - distances[0], y - distances[1], x + distances[2],
                                     y + distances[3]};
                Box predicted{};
                Box target{};
                for (std::size_t c = 0; c < K_BOX_VALUES; ++c)
                {
                    const std::size_t at = row * K_BOX_VALUES + c;
                    predicted[c] = level.box_mask[at] != 0.0 ? decoded[c] : level.safe_boxes[at];
                    target[c] = level.target_boxes[at];
                }
                terms.box += (1.0 - complete_iou(predicted, target)) * level.normalized_weight[row];

                if (gains.l1_gain > 0.0f)
                {
                    for (std::size_t c = 0; c < K_BOX_VALUES; ++c)
                    {
                        const std::size_t at = row * K_BOX_VALUES + c;
                        terms.l1 += std::abs(distances[c] - level.target_distances[at]) *
                                    level.l1_scale[at];
                    }
                }
            }

            return Status();
        }
    } // namespace

    Result<FreeLossTerms> evaluate_free_detection_loss(
        SushiRuntime::span<const FreeLevelLossArrays> levels, std::int64_t reg_max,
        std::int64_t classes, std::int64_t batch, double target_scores_sum,
        const FreeLossGains& gains)
    {
        SA_FAIL_IF(reg_max < 1 || classes < 1 || batch < 1 || !(target_scores_sum > 0.0),
                   StatusCode::InvalidDescription,
                   "the host loss reads reg_max {}, {} classes, a batch of {} and a score sum "
                   "of {}; each is positive",
                   reg_max, classes, batch, target_scores_sum);

        FreeLossTerms terms;
        for (const FreeLevelLossArrays& level : levels)
        {
            SA_TRY(add_level(level, static_cast<std::size_t>(reg_max),
                             static_cast<std::size_t>(classes), gains, terms));
        }
        terms.classes /= target_scores_sum;

        const double images = static_cast<double>(batch);
        terms.total = images * (static_cast<double>(gains.box_gain) * terms.box +
                                static_cast<double>(gains.class_gain) * terms.classes +
                                static_cast<double>(gains.distribution_gain) * terms.distribution +
                                static_cast<double>(gains.l1_gain) * terms.l1);
        return terms;
    }

} // namespace SushiAI::Vision::Training
