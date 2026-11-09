/**************************************************************************/
/* evaluation_one_position_step.hpp                                      */
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
 * @file evaluation_one_position_step.hpp
 * @brief Declares EvaluationOnePositionStep, the OnePositionStep over a compiled
 *        EvaluationStep.
 * @author Mustafa Garip
 */

#pragma once

#include <string>
#include <string_view>
#include <SushiBLAS/tensor/tensor.hpp>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/language/generation/one_position_step.hpp>
#include <SushiAI/train/evaluation_step.hpp>

namespace SushiAI::Language::Generation::Detail
{
    /** @brief Answers the questions of CachedGenerator from a compiled step; it does not own it. */
    class EvaluationOnePositionStep final : public OnePositionStep
    {
        public:
            /** @brief Binds @p step, which outlives this object. */
            explicit EvaluationOnePositionStep(Train::EvaluationStep& step) noexcept;

            /** @brief Lists the input leaves of the step. */
            [[nodiscard]] SushiRuntime::span<const std::string> names() const noexcept override;

            /** @brief Returns the facts of the input leaf @p name, or the failure of the step. */
            [[nodiscard]] Result<LeafFacts> leaf(std::string_view name) const noexcept override;

            /** @brief Returns whether the step keeps state between runs. */
            [[nodiscard]] bool carries_state() const noexcept override;

            /** @brief Returns the facts of the output @p name, or the failure of the step. */
            [[nodiscard]] Result<LeafFacts> output(std::string_view name) const noexcept override;

            /** @brief Returns the scores the last run computed under @p name. */
            [[nodiscard]] LogitSpan logits(std::string_view name) const noexcept override;

            /** @brief Resets the step. */
            [[nodiscard]] Status reset() override;

            /** @brief Runs the step with @p feeds. */
            [[nodiscard]] Status run(SushiRuntime::span<const Train::NamedFeed> feeds) override;

        private:
            /// @brief Runs the position; not owned.
            Train::EvaluationStep* step_;
    };

    /** @brief Reads the shape and the type of @p tensor. */
    [[nodiscard]] LeafFacts facts_of(const SushiBLAS::Tensor& tensor) noexcept;

} // namespace SushiAI::Language::Generation::Detail
