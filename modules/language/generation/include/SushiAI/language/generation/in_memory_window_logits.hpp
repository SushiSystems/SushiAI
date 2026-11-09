/**************************************************************************/
/* in_memory_window_logits.hpp                                           */
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
 * @file in_memory_window_logits.hpp
 * @brief Declares InMemoryWindowLogits, the WindowLogits that scores the successor of each id
 *        and keeps the windows it was run on.
 * @author Mustafa Garip
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/language/generation/sampler.hpp>
#include <SushiAI/language/generation/window_logits.hpp>
#include <SushiAI/language/text/token.hpp>

namespace SushiAI::Language::Generation
{
    /**
     * @brief Scores position p with 1 at the token after the id at p, modulo V, and 0
     *        elsewhere, and keeps every window it was run on.
     */
    class InMemoryWindowLogits final : public WindowLogits
    {
        public:
            /** @brief Builds the source of a window of @p window ids over @p vocabulary tokens. */
            InMemoryWindowLogits(std::int64_t window, std::int64_t vocabulary);

            /** @brief Returns T. */
            [[nodiscard]] std::int64_t window() const noexcept override;

            /** @brief Returns V. */
            [[nodiscard]] std::int64_t vocabulary() const noexcept override;

            /** @brief Keeps @p ids and scores them; fails unless there are exactly T ids. */
            [[nodiscard]] Status run(SushiRuntime::span<const Text::TokenID> ids) override;

            /** @brief Returns the scores of @p position from the last run. */
            [[nodiscard]] LogitSpan row(std::size_t position) const noexcept override;

            /** @brief Returns the windows it was run on, in order. */
            [[nodiscard]] const std::vector<std::vector<Text::TokenID>>& windows() const noexcept;

        private:
            /// @brief Holds T.
            std::int64_t window_;

            /// @brief Holds V.
            std::int64_t vocabulary_;

            /// @brief Holds the T rows of V scores of the last run.
            std::vector<float> scores_;

            /// @brief Holds the windows run so far.
            std::vector<std::vector<Text::TokenID>> windows_;
    };

} // namespace SushiAI::Language::Generation
