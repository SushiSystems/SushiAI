/**************************************************************************/
/* evaluation_one_position_step.cpp                                      */
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
 * @file evaluation_one_position_step.cpp
 * @brief Implements EvaluationOnePositionStep.
 * @author Mustafa Garip
 */

#include "evaluation_one_position_step.hpp"

#include <cstddef>
#include <SushiAI/failure/check.hpp>

namespace SushiAI::Language::Generation::Detail
{
    LeafFacts facts_of(const SushiBLAS::Tensor& tensor) noexcept
    {
        return LeafFacts{Shape(SushiRuntime::span<const std::int64_t>(
                           tensor.shape.data(), static_cast<std::size_t>(tensor.rank))),
                         tensor.dtype};
    }

    EvaluationOnePositionStep::EvaluationOnePositionStep(Train::EvaluationStep& step) noexcept
        : step_(&step)
    {

    }

    SushiRuntime::span<const std::string> EvaluationOnePositionStep::names() const noexcept
    {
        return step_->input_names();
    }

    Result<LeafFacts> EvaluationOnePositionStep::leaf(std::string_view name) const noexcept
    {
        SA_TRY_ASSIGN(const SushiBLAS::Tensor* const tensor, step_->input(name));
        return facts_of(*tensor);
    }

    bool EvaluationOnePositionStep::carries_state() const noexcept
    {
        return step_->carries_state();
    }

    Result<LeafFacts> EvaluationOnePositionStep::output(std::string_view name) const noexcept
    {
        SA_TRY_ASSIGN(const SushiBLAS::Tensor* const tensor, step_->output(name));
        return facts_of(*tensor);
    }

    LogitSpan EvaluationOnePositionStep::logits(std::string_view name) const noexcept
    {
        const SushiBLAS::Tensor* const tensor = step_->output(name).value();
        return LogitSpan(tensor->data_as<float>(), static_cast<std::size_t>(tensor->shape[2]));
    }

    Status EvaluationOnePositionStep::reset()
    {
        return step_->reset();
    }

    Status EvaluationOnePositionStep::run(SushiRuntime::span<const Train::NamedFeed> feeds)
    {
        return step_->run_by_name(feeds);
    }

} // namespace SushiAI::Language::Generation::Detail
