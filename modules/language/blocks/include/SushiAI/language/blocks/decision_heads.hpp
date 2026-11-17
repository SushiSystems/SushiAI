/**************************************************************************/
/* decision_heads.hpp                                                     */
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
 * @file decision_heads.hpp
 * @brief Declares DecisionHeads, one text encoder behind a choice head, a yes-or-no head and a
 *        number head, and the node that publishes the three.
 * @author Mustafa Garip
 */

#pragma once

#include <memory>
#include <string_view>
#include <tuple>
#include <SushiAI/config/value.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/language/blocks/option_head.hpp>
#include <SushiAI/language/blocks/text_encoder.hpp>
#include <SushiAI/nn/layers.hpp>
#include <SushiAI/nn/module.hpp>
#include <SushiAI/nn/node.hpp>
#include <SushiAI/nn/trace.hpp>

namespace SushiAI
{
    namespace Language::Blocks
    {
        /** @brief Names the port that holds the [K] logits of the options. */
        inline constexpr std::string_view K_CHOICE_PORT = "choice";

        /** @brief Names the port that holds the [1] logit of a yes. */
        inline constexpr std::string_view K_YES_NO_PORT = "yes_no";

        /** @brief Names the port that holds the [1] number. */
        inline constexpr std::string_view K_NUMBER_PORT = "number";

        /** @brief Holds the three values a decision model traces. */
        struct DecisionTrace
        {
            Graph::ValueID choice = Graph::K_NO_VALUE; /**< The [N, K] logits of the options. */
            Graph::ValueID yes_no = Graph::K_NO_VALUE; /**< The [N, 1] logit of a yes. */
            Graph::ValueID number = Graph::K_NO_VALUE; /**< The [N, 1] number. */
        };

        /**
         * @brief Reads a question and its options through one TextEncoder and answers with a
         *        choice among the options, a yes-or-no logit and a number.
         *
         * The yes-or-no head and the number head are each a linear layer from the state of the
         * question to one value. README.md, "Decision heads", states the three.
         */
        struct DecisionHeads
        {
            /** @brief Holds the encoder of the question and of every option. */
            TextEncoder encoder{};

            /** @brief Holds the scoring of the options; it holds no state. */
            OptionHead options{};

            /** @brief Holds the linear layer from the question state to the yes-or-no logit. */
            NN::Linear yes_no{};

            /** @brief Holds the linear layer from the question state to the number. */
            NN::Linear number{};

            /** @brief Builds heads of no width, which no input fits. */
            DecisionHeads() = default;

            /** @brief Takes @p encoder and sizes both linear layers to its width. */
            explicit DecisionHeads(TextEncoder encoder) noexcept;

            /** @brief Lists the three children that hold state, under "encoder", "yes_no" and
             *         "number". */
            static constexpr auto fields() noexcept
            {
                return std::tuple{
                    NN::child("encoder", &DecisionHeads::encoder),
                    NN::child("yes_no", &DecisionHeads::yes_no),
                    NN::child("number", &DecisionHeads::number),
                };
            }

            /**
             * @brief Traces the encoder over the question and the options, then the three heads.
             * @param question_tokens An [N, T] INT32 value, with N lengths of at most T.
             * @param option_tokens An [N, K, T'] INT32 value, with [N, K] lengths.
             * @param option_valid An [N, K] real floating value: 1 held, 0 padded.
             * @return The three ids, or the failure of the first argument or child that does not
             *         fit; trace.status() holds a failure a child recorded.
             */
            [[nodiscard]] Result<DecisionTrace> trace_decision(
                const NN::Trace& trace, Graph::ValueID question_tokens,
                Graph::ValueID question_lengths, Graph::ValueID option_tokens,
                Graph::ValueID option_lengths, Graph::ValueID option_valid) noexcept;

            /**
             * @brief Traces the three heads over states an encoder produced.
             * @param question_state An [N, dim] value of a real floating type.
             * @param option_states An [N, K, dim] value of the type of @p question_state.
             * @param option_valid An [N, K] real floating value.
             * @return The three ids, or the failure of the first operand that does not fit.
             */
            [[nodiscard]] Result<DecisionTrace> trace_heads(
                const NN::Trace& trace, Graph::ValueID question_state,
                Graph::ValueID option_states, Graph::ValueID option_valid) noexcept;
        };

        /**
         * @brief Builds a DecisionHeads node from the keys of make_text_encoder_node() and the
         *        required key "options", the K of the lists.
         * @param inputs Five per-sample shapes: [T], [1], [K, T'], [K] and [K].
         * @return A node of five inputs and the ports "choice" [K], "yes_no" [1] and "number" [1],
         *         or InvalidDescription when @p inputs or a key does not fit.
         */
        [[nodiscard]] Result<std::unique_ptr<NN::Node>> make_decision_node(
            NN::ShapeSpan inputs, const Config::Value& attributes,
            std::string_view context) noexcept;

    } // namespace Language::Blocks
} // namespace SushiAI
