/**************************************************************************/
/* window_logits.hpp                                                     */
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
 * @file window_logits.hpp
 * @brief Declares WindowLogits, the seam between WindowGenerator and the forward pass it runs.
 * @author Mustafa Garip
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/language/generation/sampler.hpp>
#include <SushiAI/language/text/token.hpp>

namespace SushiAI::Language::Generation
{
    /** @brief Scores every position of a window of T ids at once. */
    class WindowLogits
    {
        public:
            /** @brief Releases the source through its base. */
            virtual ~WindowLogits() = default;

            /** @brief Returns T, the positions of the window. */
            [[nodiscard]] virtual std::int64_t window() const noexcept = 0;

            /** @brief Returns V, the tokens a position scores. */
            [[nodiscard]] virtual std::int64_t vocabulary() const noexcept = 0;

            /**
             * @brief Scores the window.
             * @param ids Exactly T ids, the context first and zeros behind it.
             * @return Success, or the failure of the pass.
             */
            [[nodiscard]] virtual Status run(SushiRuntime::span<const Text::TokenID> ids) = 0;

            /**
             * @brief Returns the V scores of @p position from the last run.
             * @pre @p position is below T and a run succeeded.
             */
            [[nodiscard]] virtual LogitSpan row(std::size_t position) const noexcept = 0;
    };

} // namespace SushiAI::Language::Generation
