/**************************************************************************/
/* text_encoder.hpp                                                       */
/**************************************************************************/
/*                          This file is part of:                         */
/*                                 SushiAI                                */
/*                 https://github.com/SushiSystems/SushiAI                */
/*                         https://sushisystems.io                        */
/**************************************************************************/
/* Copyright (c) 2026-present Mustafa Garip & Sushi Systems               */
/*                                                                        */
/* Licensed under the Apache License, Version 2.0 (the "License"); you    */
/* may not use this file except in compliance with the License. You may   */
/* obtain a copy of the License at                                        */
/*                                                                        */
/*     https://www.apache.org/licenses/LICENSE-2.0                        */
/*                                                                        */
/* Unless required by applicable law or agreed to in writing, software    */
/* distributed under the License is distributed on an "AS IS" BASIS,      */
/* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or        */
/* implied. See the License for the specific language governing           */
/* permissions and limitations under the License.                         */
/**************************************************************************/

/**
 * @file text_encoder.hpp
 * @brief Declares TextEncoder, the token embedding, the learned positions, the unmasked
 *        transformer blocks and the pool that turn a padded text into one vector.
 * @author Mustafa Garip
 */

#pragma once

#include <cstdint>
#include <memory>
#include <string_view>
#include <tuple>
#include <SushiAI/config/value.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/language/blocks/position_embedding.hpp>
#include <SushiAI/language/blocks/sequence_pool.hpp>
#include <SushiAI/language/blocks/transformer_block.hpp>
#include <SushiAI/nn/embedding.hpp>
#include <SushiAI/nn/module.hpp>
#include <SushiAI/nn/module_list.hpp>
#include <SushiAI/nn/node.hpp>
#include <SushiAI/nn/trace.hpp>
#include <SushiAI/shape/shape.hpp>

namespace SushiAI
{
    namespace Language::Blocks
    {
        /** @brief Encodes a padded [N, T] text into [N, dim] and reads no position past its length. */
        struct TextEncoder
        {
            /** @brief Holds the table that maps a token to a [dim] row. */
            NN::Embedding embedding{};

            /** @brief Holds the learned table added to each position. */
            PositionEmbedding positions{};

            /** @brief Holds the blocks in the order they read, each without the causal mask. */
            NN::ModuleList<TransformerBlock> blocks{};

            /** @brief Holds the mean over the positions each text holds. */
            SequencePool pool{};

            /** @brief Builds an encoder of no width, which no input fits. */
            TextEncoder() = default;

            /**
             * @brief Builds the children; forward() checks the extents.
             * @param vocabulary The row count of the token table.
             * @param length The row count of the position table, the longest text it serves.
             * @param hidden The width between the two linear layers of each feed-forward.
             * @param block_count How many transformer blocks follow the positions.
             * @param epsilon The epsilon of the normalisations of every block.
             */
            TextEncoder(int64_t vocabulary, int64_t dim, int64_t length, int64_t heads,
                        int64_t hidden, int64_t block_count, double epsilon) noexcept;

            /** @brief Lists the three children that hold state, under "embedding", "positions"
             *         and "blocks". */
            static constexpr auto fields() noexcept
            {
                return std::tuple{
                    NN::child("embedding", &TextEncoder::embedding),
                    NN::child("positions", &TextEncoder::positions),
                    NN::child("blocks", &TextEncoder::blocks),
                };
            }

            /**
             * @brief Traces the token table, the positions, each block, then the pool.
             * @param tokens An [N, T] INT32 value, T at most the rows of the position table.
             * @param lengths N real floating counts, each at most T: the tokens of each text
             *        before its padding. A token at or past its text's length changes nothing.
             * @return The [N, dim] result's id, or @p tokens after trace.fail() when an
             *         argument or this encoder's description does not fit.
             */
            Graph::ValueID forward(const NN::Trace& trace, Graph::ValueID tokens,
                                   Graph::ValueID lengths) noexcept;

            /** @brief Returns the per-sample [dim] shape forward() produces for a [T] input. */
            Result<Shape> output_shape(const Shape& input) const noexcept;
        };

        /**
         * @brief Builds a TextEncoder node from "vocabulary", "dim", "heads" and "blocks", with
         *        optional "hidden", four times "dim" when absent, "length", T when absent, and
         *        "ln_epsilon".
         * @param inputs Two per-sample shapes: the [T] tokens, then the [1] length.
         * @return A node of two inputs and the one port "output" of shape [dim], or
         *         InvalidDescription when @p inputs or a key does not fit.
         */
        [[nodiscard]] Result<std::unique_ptr<NN::Node>> make_text_encoder_node(
            NN::ShapeSpan inputs, const Config::Value& attributes, std::string_view context) noexcept;

    } // namespace Language::Blocks
} // namespace SushiAI
