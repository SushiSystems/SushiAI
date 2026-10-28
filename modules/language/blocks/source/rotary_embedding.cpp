/**************************************************************************/
/* rotary_embedding.cpp                                                   */
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
 * @file rotary_embedding.cpp
 * @brief Implements RotaryEmbedding's two traces and its width rule.
 * @author Mustafa Garip
 */

#include <SushiAI/language/blocks/rotary_embedding.hpp>

#include <cmath>
#include <cstddef>
#include <string_view>
#include <utility>
#include <vector>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/failure/check.hpp>
#include <SushiAI/dtype/dtype.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/graph/builder.hpp>
#include <SushiAI/graph/traced.hpp>
#include <SushiAI/language/blocks/incremental.hpp>
#include <SushiAI/language/blocks/rotary_positions.hpp>
#include "sequence.hpp"
#include "slice.hpp"

namespace SushiAI
{
    namespace Language::Blocks
    {
        namespace
        {
            /** @brief Names this block in a diagnostic. */
            constexpr std::string_view K_BLOCK = "RotaryEmbedding";

            /** @brief Holds the channels that turn together. */
            constexpr int64_t K_PAIR = 2;

            /** @brief Holds the factor that gives the sine of the opposite angle. */
            constexpr double K_NEGATED = -1.0;

            /** @brief Holds the [rows, pairs] cosines and sines the rows of a trace turn by. */
            struct Angles
            {
                Graph::ValueID cosine = Graph::K_NO_VALUE; /**< @brief cos of each angle. */
                Graph::ValueID sine = Graph::K_NO_VALUE;   /**< @brief sin of each angle. */
            };

            /** @brief Holds the extents of the value a trace turns. */
            struct Extents
            {
                int64_t batch = 0;  /**< @brief N, the number of sequences. */
                int64_t rows = 0;   /**< @brief The positions the value holds. */
                int64_t width = 0;  /**< @brief The channels of a position. */
                int64_t groups = 0; /**< @brief The parts of the width that turn alike. */
            };

            /** @brief Returns the element type of the host tables that serve @p dtype. */
            [[nodiscard]] constexpr DType table_type(DType dtype) noexcept
            {
                return dtype == DType::Float64 ? DType::Float64 : DType::Float32;
            }

            /**
             * @brief Returns @p value, converted when its element type is not @p dtype.
             * @return The value, or the failure @p value holds or the cast returns.
             */
            [[nodiscard]] Graph::Traced as_type(const Graph::GraphBuilder& builder,
                                                const Graph::Traced& value, DType dtype)
            {
                SA_TRY_ASSIGN(const Graph::ValueID id, value);
                if (builder.graph().value(id).dtype == dtype)
                    return id;

                return builder.cast(id, dtype);
            }

            /**
             * @brief Joins @p first and @p second, one after the other, once per group, and
             *        returns the [rows * width] vector that weighs every channel of a row.
             * @param first The [1, rows, 1, pairs] weights of the first half of a group.
             * @param second The weights of its second half, shaped alike.
             * @return The vector, or the refusal of an emitter.
             */
            [[nodiscard]] Graph::Traced spread(const Graph::GraphBuilder& builder,
                                               Graph::ValueID first, Graph::ValueID second,
                                               const Extents& e)
            {
                std::vector<Graph::ValueID> halves;
                halves.reserve(static_cast<std::size_t>(K_PAIR * e.groups));
                for (int64_t group = 0; group < e.groups; ++group)
                {
                    halves.push_back(first);
                    halves.push_back(second);
                }

                const Graph::Traced joined = builder.concat_channels(
                    SushiRuntime::span<const Graph::ValueID>(halves.data(), halves.size()));

                return builder.reshape(joined, Shape{e.rows * e.width});
            }

            /**
             * @brief Turns every pair of @p x: x * cos + swapped(x) * (-sin, sin), where
             *        swapped() exchanges the two halves of each group.
             * @param x A [batch, rows, width] value.
             * @return The value turned, shaped as @p x is, or the refusal of an emitter.
             */
            [[nodiscard]] Result<Graph::ValueID> turn(const Graph::GraphBuilder& builder,
                                                      Graph::ValueID x, const Angles& angles,
                                                      const Extents& e)
            {
                const int64_t group = e.width / e.groups;
                const int64_t pairs = group / K_PAIR;
                const int64_t flat = e.rows * e.width;

                // By value: every emission below grows the value table.
                const DType dtype = builder.graph().value(x).dtype;
                const Shape by_row{1, e.rows, 1, pairs};
                SA_TRY_ASSIGN(const Graph::ValueID cosine,
                              as_type(builder, builder.reshape(angles.cosine, by_row), dtype));
                SA_TRY_ASSIGN(const Graph::ValueID sine,
                              as_type(builder, builder.reshape(angles.sine, by_row), dtype));

                const Graph::Traced kept_weight = spread(builder, cosine, cosine, e);
                SA_TRY_ASSIGN(const Graph::ValueID negated_sine, builder.scale(sine, K_NEGATED));
                const Graph::Traced crossed_weight = spread(builder, negated_sine, sine, e);
                const Graph::ValueID no_shift = builder.zeros(Shape{flat}, dtype);

                // The channel emitters read rank 4, so a group stands as the channel axis.
                SA_TRY_ASSIGN(const Graph::ValueID grouped,
                              builder.reshape(x, Shape{e.batch, e.rows, e.groups, group}));
                SA_TRY_ASSIGN(const Graph::ValueID second_half,
                              builder.slice_channels(grouped, pairs, pairs));
                SA_TRY_ASSIGN(const Graph::ValueID first_half,
                              builder.slice_channels(grouped, 0, pairs));
                const Graph::ValueID halves[] = {second_half, first_half};
                const Graph::Traced swapped = builder.concat_channels(
                    SushiRuntime::span<const Graph::ValueID>(halves, K_PAIR));

                // A weight per position and channel is a weight per channel of [N, 1, 1, T * C].
                const Shape by_sequence{e.batch, 1, 1, flat};
                const Graph::Traced kept = builder.channel_affine(
                    builder.reshape(x, by_sequence), kept_weight, no_shift);
                const Graph::Traced crossed = builder.channel_affine(
                    builder.reshape(swapped, by_sequence), crossed_weight, no_shift);
                SA_TRY_ASSIGN(const Graph::ValueID turned,
                              builder.reshape(builder.add(kept, crossed),
                                              Shape{e.batch, e.rows, e.width}));

                return turned;
            }

            /** @brief Declares the cosines and the sines of positions 0 to rows - 1. */
            [[nodiscard]] Result<Angles> declare_tables(const NN::Trace& trace, int64_t rows,
                                                        int64_t pairs, DType dtype)
            {
                const std::size_t count = static_cast<std::size_t>(rows * pairs);
                std::vector<double> cosines(count);
                std::vector<double> sines(count);

                for (int64_t row = 0; row < rows; ++row)
                {
                    for (int64_t pair = 0; pair < pairs; ++pair)
                    {
                        const double angle = rotary_angle(row, pair, pairs);
                        const std::size_t at = static_cast<std::size_t>(row * pairs + pair);
                        cosines[at] = std::cos(angle);
                        sines[at] = std::sin(angle);
                    }
                }

                Angles angles;
                SA_TRY_ASSIGN(angles.cosine,
                              trace.host_table(Shape{rows, pairs}, K_ROTARY_COSINE_LEAF, dtype,
                                               std::move(cosines)));
                SA_TRY_ASSIGN(angles.sine,
                              trace.host_table(Shape{rows, pairs}, K_ROTARY_SINE_LEAF, dtype,
                                               std::move(sines)));

                return angles;
            }

            /** @brief Traces the turn of every row, or returns why @p x or @p block misfits. */
            [[nodiscard]] Result<Graph::ValueID> trace_rows(const RotaryEmbedding& block,
                                                            const NN::Trace& trace,
                                                            Graph::ValueID x)
            {
                const Graph::GraphBuilder& builder = trace.builder();
                SA_TRY(check_rotary_widths(block.width, block.groups));
                SA_TRY_ASSIGN(const Shape sequence,
                              Detail::checked_sequence(builder, x, block.width, K_BLOCK));

                // By value: a declaration below grows the value table.
                const DType dtype = builder.graph().value(x).dtype;
                const int64_t pairs = block.width / block.groups / K_PAIR;
                SA_TRY_ASSIGN(const Angles angles,
                              declare_tables(trace, sequence[1], pairs, table_type(dtype)));

                const Extents extents{sequence[0], sequence[1], block.width, block.groups};
                return turn(builder, x, angles, extents);
            }

            /** @brief Traces the turn of one fed position, or returns what does not fit. */
            [[nodiscard]] Result<Graph::ValueID> trace_row(const RotaryEmbedding& block,
                                                           const NN::Trace& trace,
                                                           Graph::ValueID x, const Shape& sample)
            {
                const Graph::GraphBuilder& builder = trace.builder();
                SA_TRY(check_rotary_widths(block.width, block.groups));
                SA_TRY(Detail::check_slice(builder, x, block.width, K_BLOCK));
                SA_TRY(Detail::window_of(sample, block.width, K_BLOCK));

                // By value: a declaration below grows the value table.
                const DType dtype = builder.graph().value(x).dtype;
                const Shape row{1, block.width / block.groups / K_PAIR};
                Angles angles;
                SA_TRY_ASSIGN(angles.cosine, trace.feed(K_ROTARY_COSINE_LEAF, row, dtype));
                SA_TRY_ASSIGN(angles.sine, trace.feed(K_ROTARY_SINE_LEAF, row, dtype));

                const Extents extents{1, 1, block.width, block.groups};
                return turn(builder, x, angles, extents);
            }
        } // namespace

        RotaryEmbedding::RotaryEmbedding(int64_t width_in, int64_t groups_in) noexcept
            : width(width_in), groups(groups_in)
        {

        }

        Graph::ValueID RotaryEmbedding::forward(const NN::Trace& trace, Graph::ValueID x)
        {
            return trace.value_or_fail(trace_rows(*this, trace, x), x);
        }

        Graph::ValueID RotaryEmbedding::forward_incremental(const NN::Trace& trace,
                                                            Graph::ValueID x,
                                                            const Shape& sample)
        {
            return trace.value_or_fail(trace_row(*this, trace, x, sample), x);
        }

        Result<Shape> RotaryEmbedding::output_shape(const Shape& input) const
        {
            return input;
        }

        Status check_rotary_widths(int64_t width, int64_t groups)
        {
            SA_FAIL_IF(width <= 0 || groups <= 0 || width % groups != 0 ||
                           (width / groups) % K_PAIR != 0,
                       StatusCode::InvalidDescription,
                       "RotaryEmbedding turns channels in pairs, so it needs a positive width "
                       "that its group count cuts into parts of an even size; got width={} "
                       "groups={}",
                       width, groups);

            return Status();
        }

    } // namespace Language::Blocks
} // namespace SushiAI
