/**************************************************************************/
/* decision_heads.cpp                                                     */
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
 * @file decision_heads.cpp
 * @brief Implements DecisionHeads' trace and its node factory.
 * @author Mustafa Garip
 */

#include <SushiAI/language/blocks/decision_heads.hpp>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/failure/check.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/seam/seam.hpp>
#include "block_attributes.hpp"
#include "text_encoder_attributes.hpp"

namespace SushiAI
{
    namespace Language::Blocks
    {
        namespace
        {
            /** @brief Names the key that holds how many options a list has. */
            constexpr std::string_view K_OPTIONS_KEY = "options";

            /** @brief Lists every key a "decision_heads" row accepts. */
            constexpr std::array<std::string_view, 8> K_DECISION_KEYS =
                Detail::keys_with(Detail::K_ENCODER_KEYS, K_OPTIONS_KEY);

            /** @brief Holds how many values a decision node reads. */
            constexpr std::size_t K_INPUT_COUNT = 5;

            /** @brief Holds how many ports a decision node has. */
            constexpr std::size_t K_PORT_COUNT = 3;

            /** @brief Holds the rank of the [T] question tokens. */
            constexpr int32_t K_QUESTION_RANK = 1;

            /** @brief Holds the rank of the [K, T] option tokens. */
            constexpr int32_t K_OPTION_RANK = 2;

            /** @brief Holds the extent of the per-sample question length: one count. */
            constexpr int64_t K_LENGTH_EXTENT = 1;

            /** @brief Holds the width of the yes-or-no head and of the number head. */
            constexpr int64_t K_SCALAR_WIDTH = 1;

            /** @brief Presents DecisionHeads as a node of five inputs and three ports. */
            class DecisionNode final : public NN::Node
            {
                public:
                    /**
                     * @brief Wraps @p heads and declares its three ports.
                     * @param options The K of the lists, the extent of the choice port.
                     * @return The node, or OutOfMemory.
                     */
                    [[nodiscard]] static Result<std::unique_ptr<NN::Node>> create(
                        DecisionHeads heads, int64_t options)
                    {
                        return SushiAI::Detail::guard_value(
                            SushiAI::Detail::SeamPhase::Allocate, [&]()
                        {
                            return std::unique_ptr<NN::Node>(
                                new DecisionNode(std::move(heads), options));
                        });
                    }

                    /**
                     * @brief Traces the heads and appends the choice, the yes-or-no logit and the
                     *        number.
                     * @return InvalidDescription unless @p inputs holds the five values the heads
                     *         accept; @p outputs is unchanged then.
                     */
                    [[nodiscard]] Status forward(const NN::Trace& trace, NN::ValueSpan inputs,
                                                 std::vector<Graph::ValueID>& outputs) override
                    {
                        SA_TRY(NN::Detail::require_input_count(inputs, K_INPUT_COUNT));
                        const Result<DecisionTrace> traced = heads_.trace_decision(
                            trace, inputs[0], inputs[1], inputs[2], inputs[3], inputs[4]);

                        // A failure a child recorded is the cause of the one returned after it.
                        SA_TRY(trace.status());
                        SA_TRY(traced);

                        const DecisionTrace& values = traced.value();
                        return SushiAI::Detail::guard(SushiAI::Detail::SeamPhase::Allocate,
                                                      [&]() -> Status
                        {
                            outputs.reserve(outputs.size() + K_PORT_COUNT);
                            outputs.push_back(values.choice);
                            outputs.push_back(values.yes_no);
                            outputs.push_back(values.number);

                            return Status();
                        });
                    }

                    /** @brief Walks the heads' state in the order NN::walk_state() yields. */
                    void walk_state(std::string_view prefix,
                                    const NN::StateVisitor& visitor) override
                    {
                        NN::walk_state(heads_, prefix, visitor);
                    }

                    /** @brief Returns five: the question and its length, the options, their
                     *         lengths and their flags. */
                    [[nodiscard]] std::size_t input_count() const noexcept override
                    {
                        return K_INPUT_COUNT;
                    }

                    /** @brief Lists the three ports: the choice, the yes-or-no logit and the
                     *         number. */
                    [[nodiscard]] SushiRuntime::span<const NN::Port> ports() const noexcept override
                    {
                        return SushiRuntime::span<const NN::Port>(ports_.data(), ports_.size());
                    }

                private:
                    /** @brief Binds the heads and the shapes of the three ports. */
                    DecisionNode(DecisionHeads heads, int64_t options)
                        : heads_(std::move(heads)),
                          ports_{NN::Port{std::string(K_CHOICE_PORT), Shape{options}},
                                 NN::Port{std::string(K_YES_NO_PORT), Shape{K_SCALAR_WIDTH}},
                                 NN::Port{std::string(K_NUMBER_PORT), Shape{K_SCALAR_WIDTH}}}
                    {

                    }

                    /** @brief Holds the wrapped heads. */
                    DecisionHeads heads_;

                    /** @brief Holds the ports in the order forward() appends to them. */
                    std::array<NN::Port, K_PORT_COUNT> ports_;
            };

            /** @brief Holds the extents a decision row reads from its five input shapes. */
            struct DecisionInputs
            {
                int64_t question_positions = 0; /**< T of the question tokens. */
                int64_t option_positions = 0;   /**< T' of the option tokens. */
                int64_t options = 0;            /**< K of the option tokens. */
            };

            /**
             * @brief Checks the five per-sample shapes of a decision row against each other.
             * @return The extents, or InvalidDescription unless @p inputs holds a rank-1 [T], a
             *         [1], a rank-2 [K, T'], then a [K] and a [K].
             */
            [[nodiscard]] Result<DecisionInputs> read_decision_inputs(NN::ShapeSpan inputs,
                                                                      std::string_view context)
            {
                SA_FAIL_IF(inputs.size() != K_INPUT_COUNT, StatusCode::InvalidDescription,
                           "{} reads the question, its length, the options, their lengths and "
                           "their flags, but it was given {} inputs",
                           context, inputs.size());
                SA_FAIL_IF(inputs[0].rank() != K_QUESTION_RANK, StatusCode::InvalidDescription,
                           "{} was given question tokens of {}, but it reads a rank-1 [T] input",
                           context, inputs[0].to_string());
                SA_FAIL_IF(inputs[1] != (Shape{K_LENGTH_EXTENT}), StatusCode::InvalidDescription,
                           "{} was given a question length of {}, but it reads one length, [1]",
                           context, inputs[1].to_string());
                SA_FAIL_IF(inputs[2].rank() != K_OPTION_RANK, StatusCode::InvalidDescription,
                           "{} was given option tokens of {}, but it reads a rank-2 [K, T] input",
                           context, inputs[2].to_string());

                const int64_t options = inputs[2][0];
                SA_FAIL_IF(inputs[3] != (Shape{options}), StatusCode::InvalidDescription,
                           "{} was given option lengths of {}, but it reads one length for each "
                           "of its {} options",
                           context, inputs[3].to_string(), options);
                SA_FAIL_IF(inputs[4] != (Shape{options}), StatusCode::InvalidDescription,
                           "{} was given option flags of {}, but it reads one flag for each of "
                           "its {} options",
                           context, inputs[4].to_string(), options);

                return DecisionInputs{inputs[0][0], inputs[2][1], options};
            }
        } // namespace

        DecisionHeads::DecisionHeads(TextEncoder text_encoder) noexcept
            : encoder(std::move(text_encoder))
        {
            const int64_t dim = encoder.embedding.embedding_dim;
            yes_no.in_features = dim;
            yes_no.out_features = K_SCALAR_WIDTH;
            number.in_features = dim;
            number.out_features = K_SCALAR_WIDTH;
        }

        Result<DecisionTrace> DecisionHeads::trace_decision(const NN::Trace& trace,
                                                            Graph::ValueID question_tokens,
                                                            Graph::ValueID question_lengths,
                                                            Graph::ValueID option_tokens,
                                                            Graph::ValueID option_lengths,
                                                            Graph::ValueID option_valid) noexcept
        {
            const Graph::ValueID question = encoder.forward(trace, question_tokens,
                                                            question_lengths);
            SA_TRY(trace.status());

            const Graph::ValueID states = options.encode(trace, encoder, option_tokens,
                                                         option_lengths);
            SA_TRY(trace.status());

            return trace_heads(trace, question, states, option_valid);
        }

        Result<DecisionTrace> DecisionHeads::trace_heads(const NN::Trace& trace,
                                                         Graph::ValueID question_state,
                                                         Graph::ValueID option_states,
                                                         Graph::ValueID option_valid) noexcept
        {
            DecisionTrace traced;
            traced.choice = options.score(trace, question_state, option_states, option_valid);
            SA_TRY(trace.status());

            traced.yes_no = NN::trace_layer(yes_no, trace, question_state);
            SA_TRY(trace.status());

            traced.number = NN::trace_layer(number, trace, question_state);
            SA_TRY(trace.status());

            return traced;
        }

        Result<std::unique_ptr<NN::Node>> make_decision_node(
            NN::ShapeSpan inputs, const Config::Value& attributes,
            std::string_view context) noexcept
        {
            SA_TRY(Detail::reject_unknown_keys(attributes, Detail::keys_of(K_DECISION_KEYS),
                                               context));
            SA_TRY_ASSIGN(const DecisionInputs shapes, read_decision_inputs(inputs, context));
            SA_TRY_ASSIGN(const int64_t options,
                          Detail::read_positive_int(attributes, K_OPTIONS_KEY, context));
            SA_FAIL_IF(options != shapes.options, StatusCode::InvalidDescription,
                       "{} states '{}' as {}, but its option tokens hold {} options", context,
                       K_OPTIONS_KEY, options, shapes.options);

            // One position table serves the question and the options, so it holds the longer.
            SA_TRY_ASSIGN(TextEncoder encoder,
                          Detail::read_text_encoder(
                              attributes, std::max(shapes.question_positions,
                                                   shapes.option_positions),
                              context));

            return DecisionNode::create(DecisionHeads(std::move(encoder)), options);
        }

    } // namespace Language::Blocks
} // namespace SushiAI
