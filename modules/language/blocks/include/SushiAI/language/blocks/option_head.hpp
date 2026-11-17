/**************************************************************************/
/* option_head.hpp                                                        */
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
 * @file option_head.hpp
 * @brief Declares OptionHead, which scores each option of a list against the state of a question.
 * @author Mustafa Garip
 */

#pragma once

#include <tuple>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/language/blocks/text_encoder.hpp>
#include <SushiAI/nn/trace.hpp>
#include <SushiAI/shape/shape.hpp>

namespace SushiAI
{
    namespace Language::Blocks
    {
        /**
         * @brief Encodes the K options of each text and scores each against the state of its
         *        question, with the options a text does not fill held out of the choice.
         *
         * The head holds no state: the encoder is the one that encoded the question, which the
         * caller passes in. README.md, "Option scores", states the score and the fill.
         */
        struct OptionHead
        {
            /** @brief Lists no field: the head holds no state. */
            static constexpr auto fields() noexcept
            {
                return std::tuple<>{};
            }

            /**
             * @brief Traces @p encoder over the [N * K, T] texts the options make.
             * @param tokens An [N, K, T] INT32 value, T at most the position rows of @p encoder.
             * @param lengths An [N, K] real floating value: the tokens of each option before its
             *        padding.
             * @return The [N, K, dim] id, or @p tokens after trace.fail() when an argument or the
             *         encoder does not fit.
             */
            Graph::ValueID encode(const NN::Trace& trace, TextEncoder& encoder,
                                  Graph::ValueID tokens, Graph::ValueID lengths) noexcept;

            /**
             * @brief Traces each option state's dot product with the question state, with the
             *        score of an option that is not valid replaced.
             * @param question An [N, dim] value; @p options is [N, K, dim] of its type.
             * @param valid An [N, K] real floating value: at least 1 held, 0 padded.
             * @return The [N, K] logits' id, or @p options after trace.fail() when an argument
             *         misfits. A softmax weighs a padded option exactly 0 while one is held.
             */
            Graph::ValueID score(const NN::Trace& trace, Graph::ValueID question,
                                 Graph::ValueID options, Graph::ValueID valid) noexcept;

            /**
             * @brief Traces encode() and then score().
             * @param question The [N, dim] state of the question, which @p encoder produced.
             * @return The [N, K] logits' id, or @p tokens after trace.fail() when an argument
             *         does not fit.
             */
            Graph::ValueID forward(const NN::Trace& trace, TextEncoder& encoder,
                                   Graph::ValueID question, Graph::ValueID tokens,
                                   Graph::ValueID lengths, Graph::ValueID valid) noexcept;

            /** @brief Returns the per-sample [K] shape forward() produces for a [K, T] input. */
            Result<Shape> output_shape(const Shape& input) const noexcept;
        };

    } // namespace Language::Blocks
} // namespace SushiAI
