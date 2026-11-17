/**************************************************************************/
/* text_encoder.cpp                                                       */
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
 * @file text_encoder.cpp
 * @brief Implements TextEncoder's trace, its shape rule and its node factory.
 * @author Mustafa Garip
 */

#include <SushiAI/language/blocks/text_encoder.hpp>

#include <cstddef>
#include <string>
#include <utility>
#include <vector>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/failure/check.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/language/blocks/causal_self_attention.hpp>
#include <SushiAI/seam/seam.hpp>
#include "block_attributes.hpp"
#include "text_encoder_attributes.hpp"

namespace SushiAI
{
    namespace Language::Blocks
    {
        namespace
        {
            /** @brief Holds the rank of the [T] tokens a text encoder row reads. */
            constexpr int32_t K_TOKEN_RANK = 1;

            /** @brief Holds the extent of the per-sample length input: one count. */
            constexpr int64_t K_LENGTH_EXTENT = 1;

            /** @brief Holds how many values a text encoder node reads: the tokens, the lengths. */
            constexpr std::size_t K_INPUT_COUNT = 2;

            /** @brief Traces the encoder, or returns what stopped it. */
            [[nodiscard]] Result<Graph::ValueID> trace_encoder(TextEncoder& encoder,
                                                               const NN::Trace& trace,
                                                               Graph::ValueID tokens,
                                                               Graph::ValueID lengths)
            {
                Graph::ValueID state = NN::trace_layer(encoder.embedding, trace, tokens);
                SA_TRY(trace.status());
                state = NN::trace_layer(encoder.positions, trace, state);
                SA_TRY(trace.status());

                for (TransformerBlock& block : encoder.blocks)
                {
                    state = block.forward_with_lengths(trace, state, lengths);
                    SA_TRY(trace.status());
                }

                state = encoder.pool.forward_with_lengths(trace, state, lengths);
                SA_TRY(trace.status());

                return state;
            }

            /** @brief Presents a TextEncoder as a node of two inputs and one port. */
            class TextEncoderNode final : public NN::Node
            {
                public:
                    /**
                     * @brief Wraps @p encoder and declares its one port.
                     * @return The node, or OutOfMemory.
                     */
                    [[nodiscard]] static Result<std::unique_ptr<NN::Node>> create(
                        TextEncoder encoder, int64_t dim)
                    {
                        return SushiAI::Detail::guard_value(
                            SushiAI::Detail::SeamPhase::Allocate, [&]()
                        {
                            return std::unique_ptr<NN::Node>(
                                new TextEncoderNode(std::move(encoder), dim));
                        });
                    }

                    /**
                     * @brief Traces the encoder and appends its [N, dim] result.
                     * @return InvalidDescription unless @p inputs holds the tokens and the
                     *         lengths, which the encoder accepts; @p outputs is unchanged then.
                     */
                    [[nodiscard]] Status forward(const NN::Trace& trace, NN::ValueSpan inputs,
                                                 std::vector<Graph::ValueID>& outputs) override
                    {
                        SA_TRY(NN::Detail::require_input_count(inputs, K_INPUT_COUNT));
                        const Graph::ValueID encoded =
                            encoder_.forward(trace, inputs[0], inputs[1]);

                        // A failure the encoder recorded sits on the trace; the id is meaningless.
                        SA_TRY(trace.status());

                        outputs.push_back(encoded);
                        return Status();
                    }

                    /** @brief Walks the encoder's state in the order NN::walk_state() yields. */
                    void walk_state(std::string_view prefix,
                                    const NN::StateVisitor& visitor) override
                    {
                        NN::walk_state(encoder_, prefix, visitor);
                    }

                    /** @brief Returns two, the tokens and the lengths. */
                    [[nodiscard]] std::size_t input_count() const noexcept override
                    {
                        return K_INPUT_COUNT;
                    }

                    /** @brief Lists the single port, named NN::K_SINGLE_PORT_NAME. */
                    [[nodiscard]] SushiRuntime::span<const NN::Port> ports() const noexcept override
                    {
                        return SushiRuntime::span<const NN::Port>(&port_, 1);
                    }

                private:
                    /** @brief Binds the encoder and the shape of its port. */
                    TextEncoderNode(TextEncoder encoder, int64_t dim)
                        : encoder_(std::move(encoder)),
                          port_{std::string(NN::K_SINGLE_PORT_NAME), Shape{dim}}
                    {

                    }

                    /** @brief Holds the wrapped encoder. */
                    TextEncoder encoder_;

                    /** @brief Holds the one port forward() appends to. */
                    NN::Port port_;
            };

            /**
             * @brief Returns the [T] token shape and checks the length shape beside it.
             * @return InvalidDescription unless @p inputs holds a rank-1 shape, then [1].
             */
            [[nodiscard]] Result<Shape> read_text_inputs(NN::ShapeSpan inputs,
                                                         std::string_view context)
            {
                SA_FAIL_IF(inputs.size() != K_INPUT_COUNT, StatusCode::InvalidDescription,
                           "{} reads the tokens and the lengths, but it was given {} inputs",
                           context, inputs.size());
                SA_FAIL_IF(inputs[0].rank() != K_TOKEN_RANK, StatusCode::InvalidDescription,
                           "{} was given tokens of {}, but it reads a rank-1 [T] input", context,
                           inputs[0].to_string());
                SA_FAIL_IF(inputs[1] != (Shape{K_LENGTH_EXTENT}), StatusCode::InvalidDescription,
                           "{} was given lengths of {}, but it reads one length for each text, "
                           "[1]",
                           context, inputs[1].to_string());

                return inputs[0];
            }
        } // namespace

        TextEncoder::TextEncoder(int64_t vocabulary, int64_t dim, int64_t length, int64_t heads,
                                 int64_t hidden, int64_t block_count, double epsilon) noexcept
            : embedding{vocabulary, dim}, positions(length, dim)
        {
            for (int64_t i = 0; i < block_count; ++i)
                blocks.push_back(TransformerBlock(dim, heads, hidden, AttentionMask::None, epsilon));
        }

        Graph::ValueID TextEncoder::forward(const NN::Trace& trace, Graph::ValueID tokens,
                                            Graph::ValueID lengths) noexcept
        {
            return trace.value_or_fail(trace_encoder(*this, trace, tokens, lengths), tokens);
        }

        Result<Shape> TextEncoder::output_shape(const Shape& input) const noexcept
        {
            SA_FAIL_IF(input.rank() != K_TOKEN_RANK, StatusCode::InvalidDescription,
                       "TextEncoder reads a per-sample [T] input; got {}", input.to_string());

            return Shape{embedding.embedding_dim};
        }

        namespace Detail
        {
            Result<TextEncoder> read_text_encoder(const Config::Value& attributes,
                                                  int64_t positions,
                                                  std::string_view context) noexcept
            {
                SA_TRY_ASSIGN(const int64_t vocabulary,
                              read_positive_int(attributes, K_VOCABULARY_KEY, context));
                SA_TRY_ASSIGN(const int64_t dim,
                              read_positive_int(attributes, K_DIM_KEY, context));
                SA_TRY_ASSIGN(const int64_t heads,
                              read_positive_int(attributes, K_HEADS_KEY, context));
                SA_TRY_ASSIGN(const int64_t block_count,
                              read_positive_int(attributes, K_BLOCKS_KEY, context));
                SA_TRY_ASSIGN(const int64_t hidden,
                              read_positive_int_or(attributes, K_HIDDEN_KEY, context,
                                                   K_DEFAULT_HIDDEN_RATIO * dim));
                SA_TRY_ASSIGN(const int64_t length,
                              read_positive_int_or(attributes, K_LENGTH_KEY, context,
                                                   positions));
                SA_TRY_ASSIGN(const double epsilon, read_layer_norm_epsilon(attributes, context));
                SA_TRY(SA_ANNOTATE(check_attention_widths(dim, heads), "{}", context));
                SA_FAIL_IF(length < positions, StatusCode::InvalidDescription,
                           "{} states '{}' as {}, but its texts hold {} tokens", context,
                           K_LENGTH_KEY, length, positions);

                return TextEncoder(vocabulary, dim, length, heads, hidden, block_count, epsilon);
            }
        } // namespace Detail

        Result<std::unique_ptr<NN::Node>> make_text_encoder_node(
            NN::ShapeSpan inputs, const Config::Value& attributes, std::string_view context) noexcept
        {
            SA_TRY(Detail::reject_unknown_keys(attributes, Detail::keys_of(Detail::K_ENCODER_KEYS),
                                               context));
            SA_TRY_ASSIGN(const Shape tokens, read_text_inputs(inputs, context));
            SA_TRY_ASSIGN(TextEncoder encoder,
                          Detail::read_text_encoder(attributes, tokens[0], context));
            const int64_t dim = encoder.embedding.embedding_dim;

            return TextEncoderNode::create(std::move(encoder), dim);
        }

    } // namespace Language::Blocks
} // namespace SushiAI
