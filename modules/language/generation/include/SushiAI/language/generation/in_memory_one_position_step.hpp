/**************************************************************************/
/* in_memory_one_position_step.hpp                                       */
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
 * @file in_memory_one_position_step.hpp
 * @brief Declares InMemoryOnePositionStep, the OnePositionStep that scores a successor and
 *        keeps what it was fed.
 * @author Mustafa Garip
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/language/generation/in_memory_fed_leaves.hpp>
#include <SushiAI/language/generation/one_position_step.hpp>
#include <SushiAI/language/generation/sampler.hpp>

namespace SushiAI::Language::Generation
{
    /**
     * @brief Answers every run with logits of 1 at the token after the fed id, modulo V, and 0
     *        elsewhere, and keeps the ids it was fed and the resets it saw.
     */
    class InMemoryOnePositionStep final : public OnePositionStep
    {
        public:
            /**
             * @brief Builds the step.
             * @param leaves The leaves the step is fed; one of them is the model input.
             * @param output The name the logits are published under.
             * @param vocabulary V, the scores of a position.
             * @param carries_state What carries_state() answers.
             */
            InMemoryOnePositionStep(std::vector<NamedLeafFacts> leaves, std::string output,
                                    std::int64_t vocabulary, bool carries_state);

            /** @brief Lists the leaves it was given. */
            [[nodiscard]] SushiRuntime::span<const std::string> names() const noexcept override;

            /** @brief Returns the facts of the leaf @p name, or InvalidDescription. */
            [[nodiscard]] Result<LeafFacts> leaf(std::string_view name) const noexcept override;

            /** @brief Returns the carries_state it was given. */
            [[nodiscard]] bool carries_state() const noexcept override;

            /** @brief Returns the facts of the [1, 1, V] float32 output, or InvalidDescription. */
            [[nodiscard]] Result<LeafFacts> output(std::string_view name) const noexcept override;

            /** @brief Returns the scores of the last run. */
            [[nodiscard]] LogitSpan logits(std::string_view name) const noexcept override;

            /** @brief Counts one reset. */
            [[nodiscard]] Status reset() override;

            /** @brief Keeps the fed id and scores its successor; fails without a model input. */
            [[nodiscard]] Status run(SushiRuntime::span<const Train::NamedFeed> feeds) override;

            /** @brief Returns the ids it was fed, in order. */
            [[nodiscard]] const std::vector<std::int32_t>& fed_ids() const noexcept;

            /** @brief Returns how often it was reset. */
            [[nodiscard]] std::size_t resets() const noexcept;

        private:
            /// @brief Lists the leaves.
            InMemoryFedLeaves leaves_;

            /// @brief Names the output.
            std::string output_;

            /// @brief Holds V.
            std::int64_t vocabulary_;

            /// @brief Holds the answer of carries_state().
            bool carries_state_;

            /// @brief Holds the scores of the last run.
            std::vector<float> scores_;

            /// @brief Holds the ids fed so far.
            std::vector<std::int32_t> fed_ids_;

            /// @brief Counts the resets.
            std::size_t resets_ = 0;
    };

} // namespace SushiAI::Language::Generation
