/**************************************************************************/
/* option_head.cpp                                                        */
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
 * @file option_head.cpp
 * @brief Implements OptionHead's encoding, its score and its shape rule.
 * @author Mustafa Garip
 */

#include <SushiAI/language/blocks/option_head.hpp>

#include <cstddef>
#include <cstdint>
#include <SushiAI/dtype/dtype.hpp>
#include <SushiAI/failure/check.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/graph/builder.hpp>
#include <SushiAI/graph/traced.hpp>
#include "key_padding.hpp"

namespace SushiAI
{
    namespace Language::Blocks
    {
        namespace
        {
            /** @brief Names the rank of the [N, K, T] tokens and of the [N, K, dim] states. */
            constexpr int32_t K_OPTION_RANK = 3;

            /** @brief Names the rank of the [N, dim] question state. */
            constexpr int32_t K_QUESTION_RANK = 2;

            /** @brief Names the rank of a per-sample [K, T] input. */
            constexpr int32_t K_SAMPLE_OPTION_RANK = 2;

            /** @brief Holds the count an option needs in the valid mask to be held. */
            constexpr double K_HELD_COUNT = 1.0;

            /** @brief Traces the encoder over the options, or returns why they do not fit. */
            [[nodiscard]] Result<Graph::ValueID> trace_encode(TextEncoder& encoder,
                                                              const NN::Trace& trace,
                                                              Graph::ValueID tokens,
                                                              Graph::ValueID lengths)
            {
                const Graph::GraphBuilder& builder = trace.builder();
                SA_FATAL_IF(tokens >= builder.graph().value_count()
                                || lengths >= builder.graph().value_count(),
                            "OptionHead was traced over values {} and {}; the graph holds {} "
                            "values",
                            tokens, lengths, builder.graph().value_count());

                // By value: the emissions below grow the value table.
                const Shape texts = builder.graph().value(tokens).shape;
                const DType type = builder.graph().value(tokens).dtype;
                SA_FAIL_IF(texts.rank() != K_OPTION_RANK || type != DType::Int32,
                           StatusCode::InvalidDescription,
                           "OptionHead needs [N, K, T] INT32 option tokens; got {} of {}",
                           texts.to_string(), dtype_name(type));

                const int64_t batch = texts[0];
                const int64_t options = texts[1];
                const Shape given = builder.graph().value(lengths).shape;
                SA_FAIL_IF(given != (Shape{batch, options}), StatusCode::InvalidDescription,
                           "OptionHead needs one length for each option, {}; got {}",
                           (Shape{batch, options}).to_string(), given.to_string());

                SA_TRY_ASSIGN(const Graph::ValueID flat,
                              builder.reshape(tokens, Shape{batch * options, texts[2]}));
                const Graph::ValueID states = encoder.forward(trace, flat, lengths);

                // A failure the encoder recorded sits on the trace; the id is meaningless.
                SA_TRY(trace.status());

                const int64_t dim = builder.graph().value(states).shape[1];
                SA_TRY_ASSIGN(const Graph::ValueID grouped,
                              builder.reshape(states, Shape{batch, options, dim}));

                return grouped;
            }

            /** @brief Traces the score, or returns why an operand does not fit. */
            [[nodiscard]] Result<Graph::ValueID> trace_score(const NN::Trace& trace,
                                                             Graph::ValueID question,
                                                             Graph::ValueID options,
                                                             Graph::ValueID valid)
            {
                const Graph::GraphBuilder& builder = trace.builder();
                const std::size_t known = builder.graph().value_count();
                SA_FATAL_IF(question >= known || options >= known || valid >= known,
                            "OptionHead was traced over values {}, {} and {}; the graph holds {} "
                            "values",
                            question, options, valid, known);

                // By value: the emissions below grow the value table.
                const Shape state = builder.graph().value(question).shape;
                const DType dtype = builder.graph().value(question).dtype;
                const Shape states = builder.graph().value(options).shape;
                const DType states_dtype = builder.graph().value(options).dtype;
                const Shape mask = builder.graph().value(valid).shape;
                const DType mask_dtype = builder.graph().value(valid).dtype;

                SA_FAIL_IF(state.rank() != K_QUESTION_RANK || states.rank() != K_OPTION_RANK
                               || !is_real_float(dtype) || states_dtype != dtype,
                           StatusCode::InvalidDescription,
                           "OptionHead scores an [N, dim] question state against [N, K, dim] "
                           "option states of one half, float32 or float64 type; got {} of {} "
                           "and {} of {}",
                           state.to_string(), dtype_name(dtype), states.to_string(),
                           dtype_name(states_dtype));
                SA_FAIL_IF(states[0] != state[0] || states[2] != state[1],
                           StatusCode::InvalidDescription,
                           "OptionHead was given a question state of {} and option states of {}; "
                           "the batch and the width must agree",
                           state.to_string(), states.to_string());

                const int64_t batch = states[0];
                const int64_t count = states[1];
                SA_FAIL_IF(mask != (Shape{batch, count}) || !is_real_float(mask_dtype),
                           StatusCode::InvalidDescription,
                           "OptionHead needs one real floating valid flag for each option, {}; "
                           "got {} of {}",
                           (Shape{batch, count}).to_string(), mask.to_string(),
                           dtype_name(mask_dtype));

                SA_TRY_ASSIGN(const Graph::ValueID column,
                              builder.reshape(question, Shape{batch, state[1], 1}));
                SA_TRY_ASSIGN(const Graph::ValueID products, builder.matmul(options, column));
                SA_TRY_ASSIGN(const Graph::ValueID scores,
                              builder.reshape(products, Shape{batch, count}));

                Graph::ValueID typed = valid;
                if (mask_dtype != dtype)
                {
                    SA_TRY_ASSIGN(typed, builder.cast(valid, dtype));
                }

                // The held flag is a comparison, so a count above 1 or a rounded 1 still holds.
                const Graph::Traced held = builder.less_equal(
                    builder.constant(Shape{batch, count}, K_HELD_COUNT, dtype), typed);
                const Graph::Traced fill = builder.constant(
                    Shape{batch, count}, Detail::padded_score(dtype), dtype);
                SA_TRY_ASSIGN(const Graph::ValueID logits, builder.select(held, scores, fill));

                return logits;
            }
        } // namespace

        Graph::ValueID OptionHead::encode(const NN::Trace& trace, TextEncoder& encoder,
                                          Graph::ValueID tokens, Graph::ValueID lengths) noexcept
        {
            return trace.value_or_fail(trace_encode(encoder, trace, tokens, lengths), tokens);
        }

        Graph::ValueID OptionHead::score(const NN::Trace& trace, Graph::ValueID question,
                                         Graph::ValueID options, Graph::ValueID valid) noexcept
        {
            return trace.value_or_fail(trace_score(trace, question, options, valid), options);
        }

        Graph::ValueID OptionHead::forward(const NN::Trace& trace, TextEncoder& encoder,
                                           Graph::ValueID question, Graph::ValueID tokens,
                                           Graph::ValueID lengths, Graph::ValueID valid) noexcept
        {
            const Graph::ValueID states = encode(trace, encoder, tokens, lengths);
            if (!trace.status().is_ok())
                return tokens;

            return score(trace, question, states, valid);
        }

        Result<Shape> OptionHead::output_shape(const Shape& input) const noexcept
        {
            SA_FAIL_IF(input.rank() != K_SAMPLE_OPTION_RANK, StatusCode::InvalidDescription,
                       "OptionHead reads a per-sample [K, T] input; got {}", input.to_string());

            return Shape{input[0]};
        }

    } // namespace Language::Blocks
} // namespace SushiAI
