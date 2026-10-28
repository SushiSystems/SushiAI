/**************************************************************************/
/* position_embedding.cpp                                                 */
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
 * @file position_embedding.cpp
 * @brief Implements PositionEmbedding's trace, its shape rule and its node factory.
 * @author Mustafa Garip
 */

#include <SushiAI/language/blocks/position_embedding.hpp>

#include <SushiAI/failure/check.hpp>
#include <SushiAI/dtype/dtype.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/graph/builder.hpp>
#include <SushiAI/graph/traced.hpp>
#include "block_attributes.hpp"
#include "sequence.hpp"
#include "slice.hpp"

namespace SushiAI
{
    namespace Language::Blocks
    {
        namespace
        {
            /** @brief Lists every key a "position_embedding" row accepts. */
            constexpr std::string_view K_POSITION_KEYS[] = {Detail::K_LENGTH_KEY};

            /**
             * @brief Returns the first @p count elements of the table as a rank-1 value.
             * @param table The [length, dim] leaf; its rows lie one after another.
             * @param count A multiple of dim, at most length * dim.
             * @return The [count] value, or the refusal of an emitter.
             */
            [[nodiscard]] Graph::Traced leading_elements(const Graph::GraphBuilder& builder,
                                                         Graph::ValueID table, int64_t total,
                                                         int64_t count)
            {
                if (count == total)
                    return builder.reshape(table, Shape{total});

                // The channel emitters read rank 4, so the table is cut as one row of channels.
                const Graph::Traced wide = builder.reshape(table, Shape{1, 1, 1, total});
                return builder.reshape(builder.slice_channels(wide, 0, count), Shape{count});
            }

            /** @brief Returns InvalidDescription unless the table of @p block has extents. */
            [[nodiscard]] Status check_extents(const PositionEmbedding& block)
            {
                SA_FAIL_IF(block.length <= 0 || block.dim <= 0, StatusCode::InvalidDescription,
                           "PositionEmbedding needs positive extents; got length={} dim={}",
                           block.length, block.dim);

                return Status();
            }

            /** @brief Returns InvalidDescription when the table has no row for @p positions. */
            [[nodiscard]] Status check_length(const PositionEmbedding& block, int64_t positions)
            {
                SA_FAIL_IF(positions > block.length, StatusCode::InvalidDescription,
                           "PositionEmbedding({}, {}) was given a sequence of {} positions; its "
                           "table has {} rows",
                           block.length, block.dim, positions, block.length);

                return Status();
            }

            /** @brief Declares the [length, dim] table of @p block in the builder's graph. */
            [[nodiscard]] Result<Graph::ValueID> declare_table(PositionEmbedding& block,
                                                               const Graph::GraphBuilder& builder,
                                                               DType dtype)
            {
                return block.weight.declare(
                    builder, Shape{block.length, block.dim}, NN::ParameterRole::Embedding,
                    block.weight_init, block.dim, block.dim, "weight", dtype);
            }

            /** @brief Traces the sum, or returns why @p x or @p block does not fit. */
            [[nodiscard]] Result<Graph::ValueID> trace_positions(PositionEmbedding& block,
                                                                 const NN::Trace& trace,
                                                                 Graph::ValueID x)
            {
                const Graph::GraphBuilder& builder = trace.builder();
                SA_TRY(check_extents(block));
                SA_TRY_ASSIGN(const Shape sequence,
                              Detail::checked_sequence(builder, x, block.dim,
                                                       "PositionEmbedding"));
                SA_TRY(check_length(block, sequence[1]));

                // By value: declaring the table grows the value table.
                const DType dtype = builder.graph().value(x).dtype;
                SA_TRY_ASSIGN(const Graph::ValueID table, declare_table(block, builder, dtype));

                const int64_t read = sequence[1] * block.dim;
                const Graph::Traced offsets =
                    leading_elements(builder, table, block.length * block.dim, read);
                const Graph::Traced rows = builder.reshape(x, Shape{sequence[0], read});
                SA_TRY_ASSIGN(const Graph::ValueID sum,
                              builder.reshape(builder.bias_add(rows, offsets), sequence));

                return sum;
            }

            /** @brief Traces the slice plus the row of the fed position, or says what refuses. */
            [[nodiscard]] Result<Graph::ValueID> trace_position(PositionEmbedding& block,
                                                                const NN::Trace& trace,
                                                                Graph::ValueID x,
                                                                const Shape& sample)
            {
                const Graph::GraphBuilder& builder = trace.builder();
                SA_TRY(check_extents(block));
                SA_TRY(Detail::check_slice(builder, x, block.dim, "PositionEmbedding"));
                SA_TRY_ASSIGN(const int64_t window,
                              Detail::window_of(sample, block.dim, "PositionEmbedding"));
                SA_TRY(check_length(block, window));
                SA_TRY_ASSIGN(const Graph::ValueID position, Detail::position_leaf(trace));

                // By value: declaring the table grows the value table.
                const DType dtype = builder.graph().value(x).dtype;
                SA_TRY_ASSIGN(const Graph::ValueID table, declare_table(block, builder, dtype));

                // The index is [1, 1], so the row comes back as the [1, 1, dim] the slice is.
                const Graph::Traced row = builder.gather_rows(table, position);
                SA_TRY(SA_ANNOTATE(row.status(), "PositionEmbedding({}, {})", block.length,
                                   block.dim));
                SA_TRY_ASSIGN(const Graph::ValueID sum, builder.add(x, row));

                return sum;
            }
        } // namespace

        PositionEmbedding::PositionEmbedding(int64_t length_in, int64_t dim_in) noexcept
            : length(length_in), dim(dim_in)
        {

        }

        Graph::ValueID PositionEmbedding::forward(const NN::Trace& trace, Graph::ValueID x) noexcept
        {
            return trace.value_or_fail(trace_positions(*this, trace, x), x);
        }

        Graph::ValueID PositionEmbedding::forward_incremental(const NN::Trace& trace,
                                                              Graph::ValueID x,
                                                              const Shape& sample) noexcept
        {
            return trace.value_or_fail(trace_position(*this, trace, x, sample), x);
        }

        Result<Shape> PositionEmbedding::output_shape(const Shape& input) const noexcept
        {
            return input;
        }

        Result<std::unique_ptr<NN::Node>> make_position_embedding_node(
            NN::ShapeSpan inputs, const Config::Value& attributes, std::string_view context) noexcept
        {
            SA_TRY(Detail::reject_unknown_keys(attributes, Detail::KeySpan(K_POSITION_KEYS),
                                               context));
            SA_TRY_ASSIGN(const Shape input, Detail::read_sequence(inputs, context));
            SA_TRY_ASSIGN(const int64_t length,
                          Detail::read_positive_int_or(attributes, Detail::K_LENGTH_KEY, context,
                                                       input[0]));
            SA_FAIL_IF(length < input[0], StatusCode::InvalidDescription,
                       "{} '{}' is {}, but the row reads sequences of {} positions; the table "
                       "needs a row for each",
                       context, Detail::K_LENGTH_KEY, length, input[0]);

            return Detail::wrap_block(PositionEmbedding(length, input[1]), input);
        }

    } // namespace Language::Blocks
} // namespace SushiAI
