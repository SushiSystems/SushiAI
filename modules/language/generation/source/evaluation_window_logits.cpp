/**************************************************************************/
/* evaluation_window_logits.cpp                                          */
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
 * @file evaluation_window_logits.cpp
 * @brief Implements EvaluationWindowLogits.
 * @author Mustafa Garip
 */

#include "evaluation_window_logits.hpp"

#include <algorithm>
#include <utility>
#include <SushiAI/dtype/dtype.hpp>
#include <SushiAI/failure/check.hpp>
#include <SushiAI/seam/seam.hpp>
#include <SushiAI/shape/shape.hpp>
#include <SushiAI/data/input.hpp>
#include <SushiAI/data/materialised_dataset.hpp>

namespace SushiAI::Language::Generation::Detail
{
    namespace
    {
        /** @brief Names this unit in a message. */
        constexpr std::string_view K_GENERATOR = "WindowGenerator";

        /** @brief Names the rank of the [1, T, V] logits. */
        constexpr std::int32_t K_LOGITS_RANK = 3;

    } // namespace

    Result<std::unique_ptr<WindowLogits>> EvaluationWindowLogits::make(
        SushiBLAS::Engine& engine, Train::EvaluationStep& step, std::string_view output) noexcept
    {
        SA_TRY_ASSIGN(const SushiBLAS::Tensor* const logits, step.output(output));
        SA_FAIL_IF(logits->rank != K_LOGITS_RANK || logits->shape[0] != 1 ||
                       logits->dtype != DType::Float32,
                   StatusCode::InvalidDescription,
                   "{}: the output '{}' has rank {} and the element type {}; it reads "
                   "float32 logits of [1, T, V]",
                   K_GENERATOR, output, logits->rank, dtype_name(logits->dtype));

        Data::Batch feed;
        SA_TRY(Data::prepare_batch(engine, 1,
                                   Data::InputDescription{Shape{logits->shape[1]}, DType::Int32},
                                   {}, feed));

        return SushiAI::Detail::guard_value(SushiAI::Detail::SeamPhase::Allocate, [&]()
        {
            return std::unique_ptr<WindowLogits>(
                new EvaluationWindowLogits(step, *logits, std::move(feed)));
        });
    }

    EvaluationWindowLogits::EvaluationWindowLogits(Train::EvaluationStep& step,
                                                   const SushiBLAS::Tensor& logits,
                                                   Data::Batch feed) noexcept
        : step_(&step), logits_(&logits), feed_(std::move(feed))
    {

    }

    std::int64_t EvaluationWindowLogits::window() const noexcept
    {
        return logits_->shape[1];
    }

    std::int64_t EvaluationWindowLogits::vocabulary() const noexcept
    {
        return logits_->shape[2];
    }

    Status EvaluationWindowLogits::run(SushiRuntime::span<const Text::TokenID> ids)
    {
        std::copy(ids.begin(), ids.end(), feed_.inputs.data_as<Text::TokenID>());
        return step_->run(feed_.inputs);
    }

    LogitSpan EvaluationWindowLogits::row(std::size_t position) const noexcept
    {
        const std::size_t scores = static_cast<std::size_t>(vocabulary());
        return LogitSpan(logits_->data_as<float>() + position * scores, scores);
    }

} // namespace SushiAI::Language::Generation::Detail
