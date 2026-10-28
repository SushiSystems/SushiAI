/**************************************************************************/
/* free_detection_loss.cpp                                                */
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
 * @file free_detection_loss.cpp
 * @brief Implements the loss graph of an anchor-free detector's one-to-many outputs.
 * @author Mustafa Garip
 */

#include <SushiAI/vision/training/free_detection_loss.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <string_view>

#include <SushiAI/dtype/dtype.hpp>
#include <SushiAI/failure/check.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/shape/shape.hpp>
#include <SushiAI/vision/training/distribution_expectation_graph.hpp>
#include <SushiAI/vision/training/distribution_focal_loss.hpp>

namespace SushiAI::Vision::Training
{
    namespace
    {
        using Graph::Traced;
        using Graph::ValueID;

        /** @brief Names the rank of an NHWC port or leaf. */
        constexpr std::int32_t K_RANK = 4;

        /** @brief Names the values of a box: four sides or four corners. */
        constexpr std::int64_t K_BOX_VALUES = 4;

        /** @brief Names the values of an anchor point: x then y. */
        constexpr std::int64_t K_POINT_VALUES = 2;

        /** @brief Holds the epsilon of every CIoU guard, as Ultralytics' bbox_iou has it. */
        constexpr double K_EPSILON = 1e-7;

        /** @brief Holds 4 / pi^2, the factor of CIoU's aspect term. */
        constexpr double K_ASPECT_FACTOR = 0.40528473456935108578;

        /** @brief Holds a quarter, the factor of CIoU's centre distance. */
        constexpr double K_QUARTER = 0.25;

        /** @brief Holds the four `[B, H, W, 1]` corner columns of a box value. */
        struct Corners
        {
            Traced x1; /**< Left edge. */
            Traced y1; /**< Top edge. */
            Traced x2; /**< Right edge. */
            Traced y2; /**< Bottom edge. */
        };

        /** @brief Stops the process unless @p id is @p expected of @p dtype; names it @p what. */
        void require(const Graph::Graph& graph, ValueID id, const Shape& expected, DType dtype,
                     std::string_view what)
        {
            SA_FATAL_IF(id == Graph::K_NO_VALUE, "The free detection loss has no {}", what);
            const Graph::Value& held = graph.value(id);
            SA_FATAL_IF(held.shape != expected || held.dtype != dtype,
                        "The free detection loss reads {} as {} {}; got {} {}", what,
                        dtype_name(dtype), expected.to_string(), dtype_name(held.dtype),
                        held.shape.to_string());
        }

        /** @brief Returns @p shape with its channels replaced by @p channels. */
        Shape with_channels(const Shape& shape, std::int64_t channels)
        {
            return Shape{shape[0], shape[1], shape[2], channels};
        }

        /** @brief Stops the process unless every value of @p level fits @p facts and @p dtype. */
        void require_level(const Graph::Graph& graph, const FreeModelFacts& facts,
                           const FreeLevelLossValues& level, const FreeLossGains& gains,
                           DType dtype)
        {
            SA_FATAL_IF(level.boxes == Graph::K_NO_VALUE, "The free detection loss has no boxes");
            const Shape shape = graph.value(level.boxes).shape;
            SA_FATAL_IF(shape.rank() != K_RANK, "The box port is {}; the loss reads rank 4",
                        shape.to_string());

            require(graph, level.boxes, with_channels(shape, K_BOX_VALUES * facts.reg_max),
                    dtype, "the box port");
            require(graph, level.scores, with_channels(shape, facts.classes), dtype,
                    "the score port");
            require(graph, level.target_scores, with_channels(shape, facts.classes), dtype,
                    "the target scores");
            require(graph, level.box_mask, with_channels(shape, K_BOX_VALUES), dtype,
                    "the box mask");
            require(graph, level.normalized_weight, with_channels(shape, 1), dtype,
                    "the normalized weight");
            require(graph, level.target_boxes, with_channels(shape, K_BOX_VALUES), dtype,
                    "the target boxes");
            require(graph, level.safe_boxes, with_channels(shape, K_BOX_VALUES), dtype,
                    "the safe boxes");
            require(graph, level.anchor_points, with_channels(shape, K_POINT_VALUES), dtype,
                    "the anchor points");
            if (gains.l1_gain > 0.0f)
            {
                require(graph, level.target_distances, with_channels(shape, K_BOX_VALUES), dtype,
                        "the target distances");
                require(graph, level.l1_scale, with_channels(shape, K_BOX_VALUES), dtype,
                        "the L1 scale");
            }
            if (facts.reg_max > 1)
                require(graph, level.distribution_weights,
                        with_channels(shape, K_BOX_VALUES * facts.reg_max), dtype,
                        "the distribution weights");
        }

        /** @brief Returns the sum of every element of @p x as a `[1]` value. */
        Traced sum_all(const Graph::GraphBuilder& builder, const Graph::Graph& graph,
                       const Traced& traced)
        {
            SA_TRY_ASSIGN(const ValueID x, traced);
            const std::int64_t count = graph.value(x).shape.num_elements();
            return builder.sum_rows(builder.reshape(x, Shape{count, 1}));
        }

        /** @brief Returns `max(0, x)` with no gradient at exactly 0, as torch's clamp_(0). */
        Traced clamp_at_zero(const Graph::GraphBuilder& builder, const Traced& x)
        {
            return builder.maximum(builder.constant_like(x, 0.0), x);
        }

        /** @brief Returns `x * x`. */
        Traced square(const Graph::GraphBuilder& builder, const Traced& x)
        {
            return builder.mul(x, x);
        }

        /** @brief Returns `x + value`, the constant shaped like @p x. */
        Traced add_constant(const Graph::GraphBuilder& builder, const Traced& x, double value)
        {
            return builder.add(x, builder.constant_like(x, value));
        }

        /** @brief Splits a `[B, H, W, 4]` box value into its corner columns. */
        Corners split_corners(const Graph::GraphBuilder& builder, const Traced& box)
        {
            return Corners{builder.slice_channels(box, 0, 1), builder.slice_channels(box, 1, 1),
                           builder.slice_channels(box, 2, 1), builder.slice_channels(box, 3, 1)};
        }

        /** @brief Returns the `[B, H, W, 4]` x1 y1 x2 y2 of distances around anchor points. */
        Traced decode_boxes(const Graph::GraphBuilder& builder, const Traced& distances,
                            ValueID points)
        {
            const Traced x = builder.slice_channels(points, 0, 1);
            const Traced y = builder.slice_channels(points, 1, 1);
            SA_TRY_ASSIGN(const ValueID left,
                          builder.sub(x, builder.slice_channels(distances, 0, 1)));
            SA_TRY_ASSIGN(const ValueID top,
                          builder.sub(y, builder.slice_channels(distances, 1, 1)));
            SA_TRY_ASSIGN(const ValueID right,
                          builder.add(x, builder.slice_channels(distances, 2, 1)));
            SA_TRY_ASSIGN(const ValueID bottom,
                          builder.add(y, builder.slice_channels(distances, 3, 1)));
            const std::array<ValueID, K_BOX_VALUES> corners = {left, top, right, bottom};
            return builder.concat_channels(
                SushiRuntime::span<const ValueID>(corners.data(), corners.size()));
        }

        /**
         * @brief Returns the `[B, H, W, 1]` CIoU of @p predicted against @p target.
         * @pre Every row of both is a box of positive height, so no division reads a zero.
         */
        Traced complete_iou(const Graph::GraphBuilder& builder, const Traced& predicted,
                            ValueID target)
        {
            const Corners p = split_corners(builder, predicted);
            const Corners t = split_corners(builder, target);

            const Traced w1 = builder.sub(p.x2, p.x1);
            const Traced h1 = add_constant(builder, builder.sub(p.y2, p.y1), K_EPSILON);
            const Traced w2 = builder.sub(t.x2, t.x1);
            const Traced h2 = add_constant(builder, builder.sub(t.y2, t.y1), K_EPSILON);

            const Traced inter_w = clamp_at_zero(
                builder, builder.sub(builder.minimum(p.x2, t.x2), builder.maximum(p.x1, t.x1)));
            const Traced inter_h = clamp_at_zero(
                builder, builder.sub(builder.minimum(p.y2, t.y2), builder.maximum(p.y1, t.y1)));
            const Traced inter = builder.mul(inter_w, inter_h);
            const Traced area = builder.add(builder.mul(w1, h1), builder.mul(w2, h2));
            const Traced union_area = add_constant(builder, builder.sub(area, inter), K_EPSILON);
            const Traced iou = builder.div(inter, union_area);

            const Traced cw =
                builder.sub(builder.maximum(p.x2, t.x2), builder.minimum(p.x1, t.x1));
            const Traced ch =
                builder.sub(builder.maximum(p.y2, t.y2), builder.minimum(p.y1, t.y1));
            const Traced c2 = add_constant(
                builder, builder.add(square(builder, cw), square(builder, ch)), K_EPSILON);
            const Traced dx =
                builder.sub(builder.add(t.x1, t.x2), builder.add(p.x1, p.x2));
            const Traced dy =
                builder.sub(builder.add(t.y1, t.y2), builder.add(p.y1, p.y2));
            const Traced rho2 = builder.scale(
                builder.add(square(builder, dx), square(builder, dy)), K_QUARTER);

            const Traced angle = builder.sub(builder.atan(builder.div(w2, h2)),
                                             builder.atan(builder.div(w1, h1)));
            const Traced v = builder.scale(square(builder, angle), K_ASPECT_FACTOR);
            const Traced alpha = builder.stop_gradient(builder.div(
                v, add_constant(builder, builder.sub(v, iou), 1.0 + K_EPSILON)));

            const Traced penalty =
                builder.add(builder.div(rho2, c2), builder.mul(v, alpha));
            return builder.sub(iou, penalty);
        }

        /** @brief Returns `a + b`, or @p b alone when @p a is K_NO_VALUE. */
        Traced accumulate_term(const Graph::GraphBuilder& builder, ValueID a, const Traced& b)
        {
            return a == Graph::K_NO_VALUE ? b : builder.add(a, b);
        }

        /** @brief Returns @p term times @p factor, added to @p total when it holds a value. */
        Traced add_gained(const Graph::GraphBuilder& builder, ValueID total, ValueID term,
                          double factor)
        {
            return accumulate_term(builder, total, builder.scale(term, factor));
        }
    } // namespace

    Result<FreeLossTermIDs> emit_free_detection_loss(const Graph::GraphBuilder& builder,
                                                     const Graph::Graph& graph,
                                                     const FreeModelFacts& facts,
                                                     const FreeLossValues& values,
                                                     const FreeLossGains& gains) noexcept
    {
        SA_FATAL_IF(!validate_free_loss_gains(gains).is_ok(),
                    "The free detection loss reads gains validate_free_loss_gains() refuses");
        SA_FATAL_IF(facts.reg_max < 1 || facts.classes < 1 || facts.levels.empty() ||
                        values.levels.size() != facts.levels.size(),
                    "The free detection loss reads {} levels of reg_max {} and {} classes; got "
                    "{} levels",
                    facts.levels.size(), facts.reg_max, facts.classes, values.levels.size());
        SA_FATAL_IF(values.levels[0].boxes == Graph::K_NO_VALUE,
                    "The free detection loss has no boxes");

        // By value: the emissions below can reallocate the value vector.
        const DType dtype = graph.value(values.levels[0].boxes).dtype;
        const std::int64_t batch = graph.value(values.levels[0].boxes).shape[0];
        require(graph, values.target_scores_sum, Shape{1}, dtype, "the target scores sum");
        if (facts.reg_max > 1)
            require(graph, values.projection, Shape{facts.reg_max, 1}, dtype, "the projection");
        for (const FreeLevelLossValues& level : values.levels)
        {
            require_level(graph, facts, level, gains, dtype);
            SA_FATAL_IF(graph.value(level.boxes).shape[0] != batch,
                        "The free detection loss reads one batch of {} over every level", batch);
        }

        FreeLossTermIDs terms;
        ValueID class_sum = Graph::K_NO_VALUE;
        for (const FreeLevelLossValues& level : values.levels)
        {
            const Traced cross_entropy = builder.sub(builder.softplus(level.scores),
                                                     builder.mul(level.scores,
                                                                 level.target_scores));
            SA_TRY_ASSIGN(class_sum, accumulate_term(builder, class_sum,
                                                     sum_all(builder, graph, cross_entropy)));

            const Traced distances =
                emit_distribution_expectation(builder, graph, level.boxes, values.projection);
            const Traced guarded = builder.select(
                level.box_mask, decode_boxes(builder, distances, level.anchor_points),
                level.safe_boxes);
            const Traced ciou = complete_iou(builder, guarded, level.target_boxes);
            const Traced miss = builder.sub(builder.constant_like(ciou, 1.0), ciou);
            const Traced box_error = builder.mul(miss, level.normalized_weight);
            SA_TRY_ASSIGN(terms.box, accumulate_term(builder, terms.box,
                                                     sum_all(builder, graph, box_error)));

            if (facts.reg_max > 1)
            {
                SA_TRY_ASSIGN(terms.distribution,
                              accumulate_term(builder, terms.distribution,
                                              emit_distribution_focal_loss(
                                                  builder, graph, level.boxes,
                                                  level.distribution_weights)));
            }

            if (gains.l1_gain > 0.0f)
            {
                const Traced error = builder.mul(
                    builder.abs(builder.sub(distances, level.target_distances)), level.l1_scale);
                SA_TRY_ASSIGN(terms.l1, accumulate_term(builder, terms.l1,
                                                        sum_all(builder, graph, error)));
            }
        }
        SA_TRY_ASSIGN(terms.classes, builder.div(class_sum, values.target_scores_sum));

        const double images = static_cast<double>(batch);
        ValueID total = Graph::K_NO_VALUE;
        SA_TRY_ASSIGN(total, add_gained(builder, total, terms.box, gains.box_gain * images));
        SA_TRY_ASSIGN(total,
                      add_gained(builder, total, terms.classes, gains.class_gain * images));
        if (terms.distribution != Graph::K_NO_VALUE)
        {
            SA_TRY_ASSIGN(total, add_gained(builder, total, terms.distribution,
                                            gains.distribution_gain * images));
        }
        if (terms.l1 != Graph::K_NO_VALUE)
        {
            SA_TRY_ASSIGN(total, add_gained(builder, total, terms.l1, gains.l1_gain * images));
        }
        terms.total = total;
        return terms;
    }

} // namespace SushiAI::Vision::Training
