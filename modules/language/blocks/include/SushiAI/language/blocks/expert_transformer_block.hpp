/**************************************************************************/
/* expert_transformer_block.hpp                                           */
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
 * @file expert_transformer_block.hpp
 * @brief Declares ExpertTransformerBlock, a transformer block whose feed-forward is a mixture
 *        of experts.
 * @author Mustafa Garip
 */

#pragma once

#include <cstdint>
#include <tuple>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/shape/shape.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/language/blocks/causal_self_attention.hpp>
#include <SushiAI/language/blocks/expert_trace.hpp>
#include <SushiAI/language/blocks/moe_feed_forward.hpp>
#include <SushiAI/nn/layer_norm.hpp>
#include <SushiAI/nn/module.hpp>
#include <SushiAI/nn/trace.hpp>

namespace SushiAI
{
    namespace Language::Blocks
    {
        /** @brief Adds attention over the normalised input, then its experts over that sum. */
        struct ExpertTransformerBlock
        {
            /** @brief Holds the normalisation the attention reads. */
            NN::LayerNorm norm1{};

            /** @brief Holds the attention whose result is added to the input. */
            CausalSelfAttention attn{};

            /** @brief Holds the normalisation the experts read. */
            NN::LayerNorm norm2{};

            /** @brief Holds the experts whose mixture is added to the first sum. */
            MoEFeedForward ff{};

            /** @brief Builds a block of no width, which no input fits. */
            ExpertTransformerBlock() = default;

            /**
             * @brief Builds the two normalisations and the attention over the width of
             *        @p feed_forward, which the block takes; trace_experts() checks the widths.
             * @param mask Causal for a decoder, None for an encoder.
             * @param epsilon The epsilon of both normalisations.
             * @param positions What the attention takes; Rotary needs dim / heads even.
             */
            ExpertTransformerBlock(
                int64_t heads, MoEFeedForward feed_forward, AttentionMask mask, double epsilon,
                AttentionPositions positions = AttentionPositions::Learned) noexcept;

            /** @brief Lists the four children, under "norm1", "attn", "norm2" and "ff". */
            static constexpr auto fields() noexcept
            {
                return std::tuple{
                    NN::child("norm1", &ExpertTransformerBlock::norm1),
                    NN::child("attn", &ExpertTransformerBlock::attn),
                    NN::child("norm2", &ExpertTransformerBlock::norm2),
                    NN::child("ff", &ExpertTransformerBlock::ff),
                };
            }

            /**
             * @brief Traces x + attn(norm1(x)), then that sum plus the mixture over norm2 of it.
             * @param x A [N, T, dim] value of a real floating type.
             * @return The result with the usage and the balance of the experts, or the
             *         failure of a child; a child may have recorded it on @p trace as well.
             */
            [[nodiscard]] Result<ExpertTrace> trace_experts(const NN::Trace& trace,
                                                            Graph::ValueID x) noexcept;

            /**
             * @brief Traces the incremental form of each child in the order trace_experts() has.
             * @param x The [1, 1, dim] slice of one sequence, of a real floating type.
             * @param sample The per-sample [T, dim] shape of the ordinary form.
             */
            [[nodiscard]] Result<ExpertTrace> trace_experts_incremental(const NN::Trace& trace,
                                                                        Graph::ValueID x,
                                                                        const Shape& sample) noexcept;

            /**
             * @brief Traces the block and returns its result alone.
             * @return The [N, T, dim] result's id, or @p x once a child has failed the trace.
             */
            Graph::ValueID forward(const NN::Trace& trace, Graph::ValueID x) noexcept;

            /**
             * @brief Traces the incremental form and returns its result alone.
             * @return The [1, 1, dim] result's id, or @p x once a child has failed the trace.
             */
            Graph::ValueID forward_incremental(const NN::Trace& trace, Graph::ValueID x,
                                               const Shape& sample) noexcept;

            /** @brief Returns the per-sample shape forward() produces, which is @p input. */
            Result<Shape> output_shape(const Shape& input) const noexcept;
        };

        static_assert(NN::TraceModule<ExpertTransformerBlock>,
                      "ExpertTransformerBlock must satisfy the TraceModule contract");
        static_assert(NN::IncrementalModule<ExpertTransformerBlock>,
                      "ExpertTransformerBlock must have an incremental form");

    } // namespace Language::Blocks
} // namespace SushiAI
