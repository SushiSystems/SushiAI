/**************************************************************************/
/* causal_self_attention.cpp                                              */
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
 * @file causal_self_attention.cpp
 * @brief Implements CausalSelfAttention's trace, its shape rule and its node factory.
 * @author Mustafa Garip
 */

#include <SushiAI/language/blocks/causal_self_attention.hpp>

#include <cmath>
#include <cstddef>
#include <utility>
#include <vector>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/graph/builder.hpp>
#include <SushiAI/graph/traced.hpp>
#include <SushiAI/language/blocks/incremental.hpp>
#include <SushiAI/language/blocks/rotary_embedding.hpp>
#include <SushiAI/nn/throwing_bridge.hpp>
#include "block_attributes.hpp"
#include "positions_attribute.hpp"
#include "sequence.hpp"
#include "slice.hpp"

namespace SushiAI
{
    namespace Language::Blocks
    {
        namespace
        {
            /** @brief Holds how many projections qkv joins: queries, keys and values. */
            constexpr int64_t K_PROJECTIONS = 3;

            /** @brief Holds how many projections rotary positions turn: queries and keys. */
            constexpr int64_t K_TURNED = 2;

            /** @brief Lists every key a "causal_self_attention" row accepts. */
            constexpr std::string_view K_ATTENTION_KEYS[] = {
                Detail::K_HEADS_KEY, Detail::K_CAUSAL_KEY, Detail::K_POSITIONS_KEY};

            /** @brief Holds what the per-head trace reads beside the projected value. */
            struct HeadGeometry
            {
                int64_t batch = 0;    /**< @brief N, the number of sequences. */
                int64_t length = 0;   /**< @brief T, the number of positions. */
                int64_t dim = 0;      /**< @brief The width of the block. */
                int64_t heads = 0;    /**< @brief The head count. */
                AttentionMask mask{}; /**< @brief The positions a position may read. */
            };

            /** @brief Names the two [N, T, 1, C] maps the heads are cut from. */
            struct Projections
            {
                /** @brief Holds the queries in channels 0 to dim - 1, then the keys. */
                Graph::ValueID queries_keys = Graph::K_NO_VALUE;

                /** @brief Holds the values in channels 2 * dim to 3 * dim - 1. */
                Graph::ValueID values = Graph::K_NO_VALUE;
            };

            /**
             * @brief Returns the maps the heads of @p block read: @p map for both, or the
             *        queries and the keys of @p map turned when the positions are rotary.
             * @param map The [batch, length, 1, 3 * dim] projections of qkv.
             * @param trace_turn Maps a RotaryEmbedding and a [batch, length, 2 * dim] value
             *        to the id of its result.
             */
            template<typename TraceTurn>
            [[nodiscard]] Result<Projections> place(const CausalSelfAttention& block,
                                                    const NN::Trace& trace, Graph::ValueID map,
                                                    int64_t batch, int64_t length,
                                                    TraceTurn&& trace_turn)
            {
                if (block.positions == AttentionPositions::Learned)
                    return Projections{map, map};

                const Graph::GraphBuilder& builder = trace.builder();
                const int64_t width = K_TURNED * block.dim;
                SA_TRY_ASSIGN(const Graph::ValueID joined, NN::Detail::bridge_throwing([&]()
                {
                    return builder.reshape(builder.slice_channels(map, 0, width),
                                           Shape{batch, length, width});
                }));

                RotaryEmbedding rotary(width, K_TURNED * block.heads);
                const Graph::ValueID turned = std::forward<TraceTurn>(trace_turn)(rotary, joined);
                SA_TRY(trace.status());

                SA_TRY_ASSIGN(const Graph::ValueID turned_map, NN::Detail::bridge_throwing([&]()
                {
                    return builder.reshape(turned, Shape{batch, length, 1, width});
                }));

                return Projections{turned_map, map};
            }

            /**
             * @brief Traces every head over @p projections and joins the results.
             * @param projections The queries, the keys and the values, each split into heads
             *        of dim / heads channels.
             * @return The [N, T, dim] value of the heads joined; an emitter's Error escapes.
             */
            [[nodiscard]] Graph::ValueID attend(const Graph::GraphBuilder& builder,
                                                const Projections& projections,
                                                const HeadGeometry& g)
            {
                const int64_t head_dim = g.dim / g.heads;
                const double scale = 1.0 / std::sqrt(static_cast<double>(head_dim));
                const Shape tokens{g.batch, g.length, head_dim};
                const Shape token_map{g.batch, g.length, 1, head_dim};

                std::vector<Graph::ValueID> attended;
                attended.reserve(static_cast<std::size_t>(g.heads));
                for (int64_t head = 0; head < g.heads; ++head)
                {
                    const int64_t offset = head * head_dim;
                    const Graph::ValueID q = builder.reshape(
                        builder.slice_channels(projections.queries_keys, offset, head_dim),
                        tokens);
                    const Graph::ValueID k = builder.reshape(
                        builder.slice_channels(projections.queries_keys, g.dim + offset,
                                               head_dim),
                        tokens);
                    const Graph::ValueID v = builder.reshape(
                        builder.slice_channels(projections.values, 2 * g.dim + offset,
                                               head_dim),
                        tokens);

                    const Graph::ValueID scores =
                        builder.scale(builder.matmul(q, k, false, true), scale);

                    // scores is a floating [N, T, T], so a refusal here is fatal in value().
                    const Graph::ValueID readable = g.mask == AttentionMask::Causal
                        ? builder.causal_mask(scores, K_MASKED_ATTENTION_SCORE).value()
                        : scores;

                    const Graph::ValueID weighed = builder.matmul(builder.softmax(readable), v);
                    attended.push_back(builder.reshape(weighed, token_map));
                }

                const Graph::ValueID joined = builder.concat_channels(
                    SushiRuntime::span<const Graph::ValueID>(attended.data(), attended.size()));

                return builder.reshape(joined, Shape{g.batch, g.length, g.dim});
            }

            /** @brief Traces the block, or returns why @p x or @p block does not fit. */
            [[nodiscard]] Result<Graph::ValueID> trace_attention(CausalSelfAttention& block,
                                                                 const NN::Trace& trace,
                                                                 Graph::ValueID x)
            {
                const Graph::GraphBuilder& builder = trace.builder();
                SA_TRY(check_attention_widths(block.dim, block.heads));
                SA_TRY(check_attention_positions(block.dim, block.heads, block.positions));
                SA_TRY_ASSIGN(const Shape sequence,
                              Detail::checked_sequence(builder, x, block.dim,
                                                       "CausalSelfAttention"));

                SA_TRY_ASSIGN(const Graph::ValueID mixed,
                              Detail::over_rows(builder, x, sequence, K_PROJECTIONS * block.dim,
                                                [&](Graph::ValueID rows)
                {
                    return NN::trace_layer(block.qkv, trace, rows);
                }));

                // The channel emitters read rank 4, so the sequence stands as a [N, T, 1, C] map.
                SA_TRY_ASSIGN(const Graph::ValueID map, NN::Detail::bridge_throwing([&]()
                {
                    return builder.reshape(mixed, Shape{sequence[0], sequence[1], 1,
                                                        K_PROJECTIONS * block.dim});
                }));
                SA_TRY_ASSIGN(const Projections projections,
                              place(block, trace, map, sequence[0], sequence[1],
                                    [&trace](RotaryEmbedding& rotary, Graph::ValueID value)
                {
                    return NN::trace_layer(rotary, trace, value);
                }));

                const HeadGeometry geometry{sequence[0], sequence[1], block.dim, block.heads,
                                            block.mask};
                SA_TRY_ASSIGN(const Graph::ValueID joined, NN::Detail::bridge_throwing([&]()
                {
                    return attend(builder, projections, geometry);
                }));

                return Detail::over_rows(builder, joined, sequence, block.dim,
                                         [&](Graph::ValueID rows)
                {
                    return NN::trace_layer(block.proj, trace, rows);
                });
            }

            /** @brief Holds the leaves one position of a cached attention reads. */
            struct CacheLeaves
            {
                Graph::ValueID row = Graph::K_NO_VALUE;    /**< @brief The [1, 1] row index. */
                Graph::ValueID mask = Graph::K_NO_VALUE;   /**< @brief The [1, T] addend. */
                Graph::ValueID keys = Graph::K_NO_VALUE;   /**< @brief The carried [T, dim]. */
                Graph::ValueID values = Graph::K_NO_VALUE; /**< @brief The carried [T, dim]. */

                /** @brief Holds 1 in the written row of a [T, dim] ring, or K_NO_VALUE. */
                Graph::ValueID replaced = Graph::K_NO_VALUE;
            };

            /**
             * @brief Declares the row leaf of a ring and the [T, dim] value that marks the row.
             * @return The failure of the leaf or of the scatter.
             */
            [[nodiscard]] Status declare_ring_row(const NN::Trace& trace, int64_t window,
                                                  int64_t dim, DType dtype, CacheLeaves& leaves)
            {
                const Graph::GraphBuilder& builder = trace.builder();
                SA_TRY_ASSIGN(leaves.row,
                              trace.feed(K_CACHE_SLOT_LEAF, Shape{1, 1}, DType::Int32));
                SA_TRY_ASSIGN(const Graph::ValueID ones, NN::Detail::bridge_throwing([&]()
                {
                    return builder.constant(Shape{1, 1, dim}, 1.0, dtype);
                }));

                const Graph::Traced marked = builder.scatter_add_rows(ones, leaves.row, window);
                SA_TRY(marked.status());
                leaves.replaced = marked.value();

                return Status();
            }

            /**
             * @brief Declares the fed leaves and the two carried leaves of one block.
             * @param positions Rotary writes row K_CACHE_SLOT_LEAF over what it holds;
             *        Learned adds row K_POSITION_LEAF to zeros.
             */
            [[nodiscard]] Result<CacheLeaves> declare_cache(const NN::Trace& trace,
                                                            AttentionPositions positions,
                                                            int64_t window, int64_t dim,
                                                            DType dtype)
            {
                CacheLeaves leaves;
                if (positions == AttentionPositions::Rotary)
                {
                    SA_TRY(declare_ring_row(trace, window, dim, dtype, leaves));
                }
                else
                {
                    SA_TRY_ASSIGN(leaves.row, Detail::position_leaf(trace));
                }

                SA_TRY_ASSIGN(leaves.mask,
                              trace.feed(K_ATTENTION_MASK_LEAF, Shape{1, window}, dtype));
                SA_TRY_ASSIGN(leaves.keys,
                              trace.carried_leaf(Shape{window, dim}, "cached_keys", dtype));
                SA_TRY_ASSIGN(leaves.values,
                              trace.carried_leaf(Shape{window, dim}, "cached_values", dtype));

                return leaves;
            }

            /**
             * @brief Returns @p cached with @p row in the row the leaf of @p leaves names:
             *        put in place of what the row holds in a ring, added to it otherwise.
             * @param cached A carried [T, dim] leaf; outside a ring its row holds zeros.
             * @param row The [1, 1, 1, dim] slice of the projection.
             * @return The [T, dim] value the leaf takes after the run, or the scatter's refusal.
             */
            [[nodiscard]] Result<Graph::ValueID> write_row(const NN::Trace& trace,
                                                           Graph::ValueID cached,
                                                           Graph::ValueID row,
                                                           const CacheLeaves& leaves,
                                                           int64_t window, int64_t dim)
            {
                const Graph::GraphBuilder& builder = trace.builder();
                SA_TRY_ASSIGN(const Graph::ValueID rows, NN::Detail::bridge_throwing([&]()
                {
                    return builder.reshape(row, Shape{1, 1, dim});
                }));

                const Graph::Traced placed = builder.scatter_add_rows(rows, leaves.row, window);
                SA_TRY(placed.status());
                SA_TRY_ASSIGN(const Graph::ValueID written, NN::Detail::bridge_throwing([&]()
                {
                    if (leaves.replaced == Graph::K_NO_VALUE)
                        return builder.add(cached, placed.value());

                    return builder.select(leaves.replaced, placed.value(), cached);
                }));
                SA_TRY(trace.carry(cached, written));

                return written;
            }

            /**
             * @brief Attends the query of one position to every cached row, head by head.
             * @param query The [1, 1, 1, dim] queries of the position.
             * @param keys The [T, dim] keys, the position's own row written.
             * @param values The [T, dim] values, the position's own row written.
             * @param mask The [1, T] value added to the scores of every head.
             * @return The [1, dim] value of the heads joined; an emitter's Error escapes.
             */
            [[nodiscard]] Graph::ValueID attend_cached(const Graph::GraphBuilder& builder,
                                                       Graph::ValueID query, Graph::ValueID keys,
                                                       Graph::ValueID values,
                                                       Graph::ValueID mask,
                                                       const HeadGeometry& g)
            {
                const int64_t head_dim = g.dim / g.heads;
                const double scale = 1.0 / std::sqrt(static_cast<double>(head_dim));
                const Shape cached{g.length, head_dim};

                // The channel emitters read rank 4, so the cache stands as a [1, T, 1, C] map.
                const Graph::ValueID key_map = builder.reshape(keys, Shape{1, g.length, 1, g.dim});
                const Graph::ValueID value_map =
                    builder.reshape(values, Shape{1, g.length, 1, g.dim});

                std::vector<Graph::ValueID> attended;
                attended.reserve(static_cast<std::size_t>(g.heads));
                for (int64_t head = 0; head < g.heads; ++head)
                {
                    const int64_t offset = head * head_dim;
                    const Graph::ValueID q = builder.reshape(
                        builder.slice_channels(query, offset, head_dim), Shape{1, head_dim});
                    const Graph::ValueID k = builder.reshape(
                        builder.slice_channels(key_map, offset, head_dim), cached);
                    const Graph::ValueID v = builder.reshape(
                        builder.slice_channels(value_map, offset, head_dim), cached);

                    const Graph::ValueID scores =
                        builder.scale(builder.matmul(q, k, false, true), scale);
                    const Graph::ValueID readable = builder.add(scores, mask);
                    const Graph::ValueID weighed = builder.matmul(builder.softmax(readable), v);
                    attended.push_back(builder.reshape(weighed, Shape{1, 1, 1, head_dim}));
                }

                const Graph::ValueID joined = builder.concat_channels(
                    SushiRuntime::span<const Graph::ValueID>(attended.data(), attended.size()));

                return builder.reshape(joined, Shape{1, g.dim});
            }

            /** @brief Traces one position of the block, or returns what does not fit. */
            [[nodiscard]] Result<Graph::ValueID> trace_cached_attention(
                CausalSelfAttention& block, const NN::Trace& trace, Graph::ValueID x,
                const Shape& sample)
            {
                const Graph::GraphBuilder& builder = trace.builder();
                SA_TRY(check_attention_widths(block.dim, block.heads));
                SA_TRY(check_attention_positions(block.dim, block.heads, block.positions));
                SA_FAIL_IF(block.mask != AttentionMask::Causal, StatusCode::InvalidDescription,
                           "CausalSelfAttention({}, {}) has no mask, so a position reads the "
                           "positions after it and the block has no incremental form",
                           block.dim, block.heads);
                SA_TRY(Detail::check_slice(builder, x, block.dim, "CausalSelfAttention"));
                SA_TRY_ASSIGN(const int64_t window,
                              Detail::window_of(sample, block.dim, "CausalSelfAttention"));

                // By value: a declaration below grows the value table.
                const DType dtype = builder.graph().value(x).dtype;
                SA_TRY_ASSIGN(const CacheLeaves leaves,
                              declare_cache(trace, block.positions, window, block.dim, dtype));

                SA_TRY_ASSIGN(const Graph::ValueID mixed, NN::Detail::bridge_throwing([&]()
                {
                    const Graph::ValueID rows = builder.reshape(x, Shape{1, block.dim});
                    return builder.reshape(NN::trace_layer(block.qkv, trace, rows),
                                           Shape{1, 1, 1, K_PROJECTIONS * block.dim});
                }));
                SA_TRY(trace.status());

                const Shape turned_sample{window, K_TURNED * block.dim};
                SA_TRY_ASSIGN(const Projections placed,
                              place(block, trace, mixed, 1, 1,
                                    [&](RotaryEmbedding& rotary, Graph::ValueID value)
                {
                    return NN::trace_layer_incremental(rotary, trace, value, turned_sample);
                }));

                Graph::ValueID projections[K_PROJECTIONS] = {};
                SA_TRY(NN::Detail::bridge_throwing([&]()
                {
                    projections[0] = builder.slice_channels(placed.queries_keys, 0, block.dim);
                    projections[1] =
                        builder.slice_channels(placed.queries_keys, block.dim, block.dim);
                    projections[2] =
                        builder.slice_channels(placed.values, 2 * block.dim, block.dim);

                    return true;
                }));

                SA_TRY_ASSIGN(const Graph::ValueID keys,
                              write_row(trace, leaves.keys, projections[1], leaves, window,
                                        block.dim));
                SA_TRY_ASSIGN(const Graph::ValueID values,
                              write_row(trace, leaves.values, projections[2], leaves, window,
                                        block.dim));

                const HeadGeometry geometry{1, window, block.dim, block.heads, block.mask};
                return NN::Detail::bridge_throwing([&]()
                {
                    const Graph::ValueID joined = attend_cached(builder, projections[0], keys,
                                                                values, leaves.mask, geometry);

                    return builder.reshape(NN::trace_layer(block.proj, trace, joined),
                                           Shape{1, 1, block.dim});
                });
            }
        } // namespace

        CausalSelfAttention::CausalSelfAttention(int64_t dim_in, int64_t heads_in,
                                                 AttentionMask mask_in,
                                                 AttentionPositions positions_in) noexcept
            : dim(dim_in), heads(heads_in), mask(mask_in), positions(positions_in)
        {
            qkv.in_features = dim;
            qkv.out_features = K_PROJECTIONS * dim;
            proj.in_features = dim;
            proj.out_features = dim;
        }

        Graph::ValueID CausalSelfAttention::forward(const NN::Trace& trace, Graph::ValueID x)
        {
            return trace.value_or_fail(trace_attention(*this, trace, x), x);
        }

        Graph::ValueID CausalSelfAttention::forward_incremental(const NN::Trace& trace,
                                                                Graph::ValueID x,
                                                                const Shape& sample)
        {
            return trace.value_or_fail(trace_cached_attention(*this, trace, x, sample), x);
        }

        Shape CausalSelfAttention::output_shape(const Shape& input) const
        {
            return input;
        }

        Status check_attention_widths(int64_t dim, int64_t heads)
        {
            SA_FAIL_IF(dim <= 0 || heads <= 0 || dim % heads != 0,
                       StatusCode::InvalidDescription,
                       "CausalSelfAttention needs a positive dim that its head count divides; "
                       "got dim={} heads={}",
                       dim, heads);

            return Status();
        }

        Status check_attention_positions(int64_t dim, int64_t heads,
                                         AttentionPositions positions)
        {
            SA_FAIL_IF(positions == AttentionPositions::Rotary &&
                           !check_rotary_widths(dim, heads).is_ok(),
                       StatusCode::InvalidDescription,
                       "CausalSelfAttention({}, {}) states rotary positions, which turn the "
                       "channels of a head in pairs, but a head has {} channels",
                       dim, heads, dim / heads);

            return Status();
        }

        Result<std::unique_ptr<NN::Node>> make_causal_self_attention_node(
            NN::ShapeSpan inputs, const Config::Value& attributes, std::string_view context)
        {
            SA_TRY(Detail::reject_unknown_keys(attributes, Detail::KeySpan(K_ATTENTION_KEYS),
                                               context));
            SA_TRY_ASSIGN(const Shape input, Detail::read_sequence(inputs, context));
            SA_TRY_ASSIGN(const int64_t heads,
                          Detail::read_positive_int(attributes, Detail::K_HEADS_KEY, context));
            SA_TRY_ASSIGN(const bool causal,
                          Detail::read_bool_or(attributes, Detail::K_CAUSAL_KEY, context,
                                               Detail::K_DEFAULT_CAUSAL));
            SA_TRY_ASSIGN(const AttentionPositions positions,
                          Detail::read_positions(attributes, context));
            SA_TRY(SA_ANNOTATE(check_attention_widths(input[1], heads), "{}", context));
            SA_TRY(SA_ANNOTATE(check_attention_positions(input[1], heads, positions), "{}",
                               context));

            const AttentionMask mask = causal ? AttentionMask::Causal : AttentionMask::None;
            return Detail::wrap_block(CausalSelfAttention(input[1], heads, mask, positions),
                                      input);
        }

    } // namespace Language::Blocks
} // namespace SushiAI
