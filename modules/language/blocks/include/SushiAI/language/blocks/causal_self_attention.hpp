/**************************************************************************/
/* causal_self_attention.hpp                                              */
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
 * @file causal_self_attention.hpp
 * @brief Declares CausalSelfAttention, the multi-head self-attention over a token sequence.
 * @author Mustafa Garip
 */

#pragma once

#include <cstdint>
#include <memory>
#include <string_view>
#include <tuple>
#include <SushiAI/config/value.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/shape/shape.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/nn/layers.hpp>
#include <SushiAI/nn/module.hpp>
#include <SushiAI/nn/node.hpp>
#include <SushiAI/nn/trace.hpp>

namespace SushiAI
{
    namespace Language::Blocks
    {
        /** @brief Names which positions an attention may read. */
        enum class AttentionMask : std::uint8_t
        {
            Causal, /**< Position t reads positions 0 to t. */
            None,   /**< Every position reads every position, as an encoder does. */
        };

        /** @brief Names where an attention takes the order of the positions from. */
        enum class AttentionPositions : std::uint8_t
        {
            Learned, /**< From its input, which a position table has been added to. */
            Rotary,  /**< From a turn of every query and key by its position. */
        };

        /** @brief Attends every position of a [N, T, dim] sequence to the others, head by head. */
        struct CausalSelfAttention
        {
            /** @brief Holds the width of the input, which is that of the result. */
            int64_t dim = 0;

            /** @brief Holds the number of heads dim is split into. */
            int64_t heads = 0;

            /** @brief Names the positions a position may read. */
            AttentionMask mask = AttentionMask::Causal;

            /** @brief Names where the order of the positions comes from. */
            AttentionPositions positions = AttentionPositions::Learned;

            /** @brief Holds the projection to 3 * dim: every query, every key, every value. */
            NN::Linear qkv{};

            /** @brief Holds the projection over the heads joined. */
            NN::Linear proj{};

            /** @brief Builds a block of no width, which no input fits. */
            CausalSelfAttention() = default;

            /**
             * @brief Builds the two projections over @p dim; forward() checks the widths.
             * @param positions Rotary needs a head of an even number of channels.
             */
            CausalSelfAttention(
                int64_t dim, int64_t heads, AttentionMask mask,
                AttentionPositions positions = AttentionPositions::Learned) noexcept;

            /** @brief Lists the two children that hold state, under "qkv" and "proj". */
            static constexpr auto fields()
            {
                return std::tuple{
                    NN::child("qkv", &CausalSelfAttention::qkv),
                    NN::child("proj", &CausalSelfAttention::proj),
                };
            }

            /**
             * @brief Traces qkv, the turn of the queries and the keys when the positions are
             *        rotary, then per head the scaled query-key product, the mask, the softmax
             *        and the product with the values, then proj over the heads joined.
             * @param x A [N, T, dim] value of a real floating type.
             * @return The [N, T, dim] result's id, or @p x after trace.fail() when @p x or
             *         this block's description does not fit.
             */
            Graph::ValueID forward(const NN::Trace& trace, Graph::ValueID x);

            /**
             * @brief Traces one position against the keys and values the step carries. The row
             *        it writes is K_POSITION_LEAF, or K_CACHE_SLOT_LEAF for rotary positions.
             * @param x The [1, 1, dim] slice of one sequence, of a real floating type.
             * @param sample The per-sample [T, dim] shape of the ordinary form.
             * @return The [1, 1, dim] result's id, or @p x after trace.fail().
             */
            Graph::ValueID forward_incremental(const NN::Trace& trace, Graph::ValueID x,
                                               const Shape& sample);

            /** @brief Returns the per-sample shape forward() produces, which is @p input. */
            Shape output_shape(const Shape& input) const;
        };

        static_assert(NN::TraceModule<CausalSelfAttention>,
                      "CausalSelfAttention must satisfy the TraceModule contract");
        static_assert(NN::IncrementalModule<CausalSelfAttention>,
                      "CausalSelfAttention must have an incremental form");

        /** @brief Returns InvalidDescription unless @p heads is positive and divides @p dim. */
        [[nodiscard]] Status check_attention_widths(int64_t dim, int64_t heads);

        /**
         * @brief Returns InvalidDescription for rotary positions over a head of an odd
         *        number of channels.
         * @pre check_attention_widths() accepted @p dim and @p heads.
         */
        [[nodiscard]] Status check_attention_positions(int64_t dim, int64_t heads,
                                                       AttentionPositions positions);

        /**
         * @brief Builds a CausalSelfAttention node from "heads", with optional "causal", true
         *        when absent, and "positions", "learned" when absent.
         * @param inputs One per-sample [T, C] shape; C becomes dim.
         * @return The node, or InvalidDescription when @p inputs or a key does not fit.
         */
        [[nodiscard]] Result<std::unique_ptr<NN::Node>> make_causal_self_attention_node(
            NN::ShapeSpan inputs, const Config::Value& attributes, std::string_view context);

    } // namespace Language::Blocks
} // namespace SushiAI
