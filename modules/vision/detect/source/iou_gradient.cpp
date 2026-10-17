/**************************************************************************/
/* iou_gradient.cpp                                                       */
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
 * @file iou_gradient.cpp
 * @brief Implements the two gradients of complete IoU over centre-and-size boxes in double.
 * @author Mustafa Garip
 * @see docs/agent/2026_10_09_DARKNET_YOLO/REPORT.md
 */

#include <SushiAI/vision/detect/iou_gradient.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>

#include <SushiAI/failure/check.hpp>
#include <SushiAI/vision/detect/box_format.hpp>

namespace SushiAI
{
    namespace Vision::Detect
    {
        namespace
        {
            /** @brief Holds pi in double. */
            constexpr double K_PI = 3.141592653589793;

            /** @brief Holds 8 / pi squared, the factor in the slope of the aspect penalty. */
            constexpr double K_ASPECT_SLOPE = 8.0 / (K_PI * K_PI);

            /** @brief Holds the four edges of one centre-and-size box. */
            struct Edges
            {
                double left = 0.0;   /**< Centre minus half the width. */
                double right = 0.0;  /**< Centre plus half the width. */
                double top = 0.0;    /**< Centre minus half the height. */
                double bottom = 0.0; /**< Centre plus half the height. */
            };

            /** @brief Holds every term of one pair of boxes that both gradients read. */
            struct BoxTerms
            {
                double width = 0.0;        /**< Width of the predicted box. */
                double height = 0.0;       /**< Height of the predicted box. */
                double slope_left = 0.0;   /**< Derivative of the IoU by the left edge. */
                double slope_right = 0.0;  /**< Derivative of the IoU by the right edge. */
                double slope_top = 0.0;    /**< Derivative of the IoU by the top edge. */
                double slope_bottom = 0.0; /**< Derivative of the IoU by the bottom edge. */
                double shift_x = 0.0;      /**< Truth centre minus predicted centre, across. */
                double shift_y = 0.0;      /**< Truth centre minus predicted centre, down. */
                double distance = 0.0;     /**< Squared distance of the two centres. */
                double hull_width = 0.0;   /**< Width of the smallest box enclosing both. */
                double hull_height = 0.0;  /**< Height of the smallest box enclosing both. */
                double diagonal = 0.0;     /**< Squared diagonal of that enclosing box. */
                double reach_x = 0.0;      /**< 1 past the right of the truth, -1 past its left. */
                double reach_y = 0.0;      /**< 1 past the bottom of the truth, -1 past its top. */
                double spread_x = 0.0;     /**< Count of the side edges of the truth it passes. */
                double spread_y = 0.0;     /**< Count of the top and bottom edges it passes. */
                double angle_gap = 0.0;    /**< atan(width / height) of the truth minus its own. */
                double trade_off = 0.0;    /**< Weight of the aspect penalty in the measure. */
            };

            /** @brief Holds the signature every form of the gradient is written in. */
            using GradientFunction = void (*)(const BoxTerms& terms, float* gradient) noexcept;

            /** @brief Holds one form and the function that writes it. */
            struct GradientRow
            {
                BoxGradient form = BoxGradient::Analytic; /**< The form the row serves. */
                GradientFunction write = nullptr;         /**< Writes four components. */
            };

            /** @brief Returns the edges of a box given as centre and size. */
            Edges measure_edges(const float* box) noexcept
            {
                const double half_width = static_cast<double>(box[2]) / 2.0;
                const double half_height = static_cast<double>(box[3]) / 2.0;

                Edges edges;
                edges.left = static_cast<double>(box[0]) - half_width;
                edges.right = static_cast<double>(box[0]) + half_width;
                edges.top = static_cast<double>(box[1]) - half_height;
                edges.bottom = static_cast<double>(box[1]) + half_height;
                return edges;
            }

            /**
             * @brief Returns the derivative of intersection / union by one predicted edge.
             * @param area_slope Derivative of the area of the predicted box by that edge.
             * @param shared_slope Derivative of the intersection by that edge.
             * @pre @p joined is positive.
             */
            double edge_slope(double area_slope, double shared_slope, double intersection,
                              double joined) noexcept
            {
                return (joined * shared_slope - intersection * (area_slope - shared_slope)) /
                       (joined * joined);
            }

            /** @brief Computes the shared terms of a predicted box and its truth. */
            BoxTerms measure_terms(const float* box, const float* target) noexcept
            {
                const Edges own = measure_edges(box);
                const Edges truth = measure_edges(target);

                BoxTerms terms;
                terms.width = static_cast<double>(box[2]);
                terms.height = static_cast<double>(box[3]);

                const double across =
                    std::min(own.right, truth.right) - std::max(own.left, truth.left);
                const double down =
                    std::min(own.bottom, truth.bottom) - std::max(own.top, truth.top);
                const bool overlapping = across > 0.0 && down > 0.0;
                const double intersection = overlapping ? across * down : 0.0;
                const double joined =
                    (own.bottom - own.top) * (own.right - own.left) +
                    (truth.bottom - truth.top) * (truth.right - truth.left) - intersection;
                const double iou = joined > 0.0 ? intersection / joined : 0.0;

                if (overlapping && joined > 0.0)
                {
                    terms.slope_left =
                        edge_slope(-terms.height, own.left > truth.left ? -down : 0.0,
                                   intersection, joined);
                    terms.slope_right =
                        edge_slope(terms.height, own.right < truth.right ? down : 0.0,
                                   intersection, joined);
                    terms.slope_top =
                        edge_slope(-terms.width, own.top > truth.top ? -across : 0.0,
                                   intersection, joined);
                    terms.slope_bottom =
                        edge_slope(terms.width, own.bottom < truth.bottom ? across : 0.0,
                                   intersection, joined);
                }

                terms.shift_x = static_cast<double>(target[0]) - static_cast<double>(box[0]);
                terms.shift_y = static_cast<double>(target[1]) - static_cast<double>(box[1]);
                terms.distance = terms.shift_x * terms.shift_x + terms.shift_y * terms.shift_y;
                terms.hull_width =
                    std::max(own.right, truth.right) - std::min(own.left, truth.left);
                terms.hull_height =
                    std::max(own.bottom, truth.bottom) - std::min(own.top, truth.top);
                terms.diagonal = terms.hull_width * terms.hull_width +
                                 terms.hull_height * terms.hull_height;

                const double past_right = own.right > truth.right ? 1.0 : 0.0;
                const double past_left = own.left < truth.left ? 1.0 : 0.0;
                const double past_bottom = own.bottom > truth.bottom ? 1.0 : 0.0;
                const double past_top = own.top < truth.top ? 1.0 : 0.0;
                terms.reach_x = past_right - past_left;
                terms.reach_y = past_bottom - past_top;
                terms.spread_x = past_right + past_left;
                terms.spread_y = past_bottom + past_top;

                terms.angle_gap =
                    std::atan(static_cast<double>(target[2]) / static_cast<double>(target[3])) -
                    std::atan(terms.width / terms.height);
                const double aspect = 4.0 * terms.angle_gap * terms.angle_gap / (K_PI * K_PI);
                terms.trade_off = aspect / (1.0 - iou + aspect + K_ASPECT_TRADE_OFF_EPS);
                return terms;
            }

            /**
             * @brief Writes the derivative of the measure by centre and size.
             * @pre The diagonal of @p terms is positive.
             */
            void write_analytic(const BoxTerms& terms, float* gradient) noexcept
            {
                const double square = terms.diagonal * terms.diagonal;
                const double kept = 1.0 - terms.trade_off * terms.trade_off;
                const double aspect = terms.trade_off * (2.0 - terms.trade_off) * K_ASPECT_SLOPE *
                                      terms.angle_gap /
                                      (terms.width * terms.width + terms.height * terms.height);

                const double x = kept * (terms.slope_left + terms.slope_right) +
                                 (2.0 * terms.shift_x * terms.diagonal +
                                  2.0 * terms.hull_width * terms.reach_x * terms.distance) /
                                     square;
                const double y = kept * (terms.slope_top + terms.slope_bottom) +
                                 (2.0 * terms.shift_y * terms.diagonal +
                                  2.0 * terms.hull_height * terms.reach_y * terms.distance) /
                                     square;
                const double width = kept * (terms.slope_right - terms.slope_left) / 2.0 +
                                     terms.hull_width * terms.spread_x * terms.distance / square +
                                     aspect * terms.height;
                const double height = kept * (terms.slope_bottom - terms.slope_top) / 2.0 +
                                      terms.hull_height * terms.spread_y * terms.distance / square -
                                      aspect * terms.width;

                gradient[0] = static_cast<float>(x);
                gradient[1] = static_cast<float>(y);
                gradient[2] = static_cast<float>(width);
                gradient[3] = static_cast<float>(height);
            }

            /**
             * @brief Writes the gradient section 11 of the report states.
             * @pre The diagonal of @p terms is positive.
             */
            void write_darknet_compatible(const BoxTerms& terms, float* gradient) noexcept
            {
                const double square = terms.diagonal * terms.diagonal;
                const double aspect = terms.trade_off * K_ASPECT_SLOPE * terms.angle_gap;

                const double x = terms.slope_left + terms.slope_right +
                                 (2.0 * terms.shift_x * terms.diagonal -
                                  2.0 * terms.hull_width * terms.reach_x * terms.distance) /
                                     square;
                const double y = terms.slope_top + terms.slope_bottom +
                                 (2.0 * terms.shift_y * terms.diagonal -
                                  2.0 * terms.hull_height * terms.reach_y * terms.distance) /
                                     square;
                const double width = terms.slope_right - terms.slope_left +
                                     terms.hull_width * terms.spread_x * terms.distance / square +
                                     aspect * terms.height;
                const double height = terms.slope_bottom - terms.slope_top +
                                      terms.hull_height * terms.spread_y * terms.distance / square -
                                      aspect * terms.width;

                gradient[0] = static_cast<float>(x);
                gradient[1] = static_cast<float>(y);
                gradient[2] = static_cast<float>(width);
                gradient[3] = static_cast<float>(height);
            }

            /** @brief Holds the function of every form. */
            constexpr GradientRow K_GRADIENT_ROWS[] = {
                {BoxGradient::Analytic, &write_analytic},
                {BoxGradient::DarknetCompatible, &write_darknet_compatible},
            };
        } // namespace

        void complete_iou_gradient(SushiRuntime::span<const float> box,
                                   SushiRuntime::span<const float> target, BoxGradient form,
                                   SushiRuntime::span<float> gradient) noexcept
        {
            SA_FATAL_IF(box.size() != K_BOX_VALUES || target.size() != K_BOX_VALUES ||
                            gradient.size() != K_BOX_VALUES,
                        "complete_iou_gradient needs three spans of four values; got {}, {} "
                        "and {}",
                        box.size(), target.size(), gradient.size());

            GradientFunction write = nullptr;
            for (const GradientRow& row : K_GRADIENT_ROWS)
            {
                if (row.form == form)
                    write = row.write;
            }
            SA_FATAL_IF(write == nullptr, "no complete IoU gradient is defined for form {}",
                        static_cast<int>(form));

            const BoxTerms terms = measure_terms(box.data(), target.data());
            if (!(terms.diagonal > 0.0))
            {
                std::fill(gradient.begin(), gradient.end(), 0.0f);
                return;
            }
            write(terms, gradient.data());
        }

    } // namespace Vision::Detect
} // namespace SushiAI
