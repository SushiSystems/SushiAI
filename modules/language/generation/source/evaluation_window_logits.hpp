/**************************************************************************/
/* evaluation_window_logits.hpp                                          */
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
 * @file evaluation_window_logits.hpp
 * @brief Declares EvaluationWindowLogits, the WindowLogits over a compiled EvaluationStep.
 * @author Mustafa Garip
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string_view>
#include <SushiBLAS/engine/engine.hpp>
#include <SushiBLAS/tensor/tensor.hpp>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/data/dataset.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/language/generation/window_logits.hpp>
#include <SushiAI/train/evaluation_step.hpp>

namespace SushiAI::Language::Generation::Detail
{
    /** @brief Runs the [1, T] forward pass of a compiled step; it does not own the step. */
    class EvaluationWindowLogits final : public WindowLogits
    {
        public:
            /**
             * @brief Binds a step traced for [1, T] int32 ids and one [1, T, V] float32 output.
             * @param engine Allocates the window; it, @p step and its pool outlive the result.
             * @param output The name @p step publishes the logits under.
             * @return The source, or InvalidDescription when the output is absent or is
             *         not [1, T, V] float32, or the failure of the allocation.
             */
            [[nodiscard]] static Result<std::unique_ptr<WindowLogits>> make(
                SushiBLAS::Engine& engine, Train::EvaluationStep& step,
                std::string_view output) noexcept;

            /** @brief Returns T. */
            [[nodiscard]] std::int64_t window() const noexcept override;

            /** @brief Returns V. */
            [[nodiscard]] std::int64_t vocabulary() const noexcept override;

            /** @brief Copies @p ids into the feed and runs the step. */
            [[nodiscard]] Status run(SushiRuntime::span<const Text::TokenID> ids) override;

            /** @brief Returns the scores of @p position from the last run. */
            [[nodiscard]] LogitSpan row(std::size_t position) const noexcept override;

        private:
            /** @brief Takes the parts make() built. */
            EvaluationWindowLogits(Train::EvaluationStep& step, const SushiBLAS::Tensor& logits,
                                   Data::Batch feed) noexcept;

            /// @brief Runs the forward pass; not owned.
            Train::EvaluationStep* step_;

            /// @brief Views the [1, T, V] logits the step's pool owns.
            const SushiBLAS::Tensor* logits_;

            /// @brief Holds the [1, T] int32 window each run feeds.
            Data::Batch feed_;
    };

} // namespace SushiAI::Language::Generation::Detail
