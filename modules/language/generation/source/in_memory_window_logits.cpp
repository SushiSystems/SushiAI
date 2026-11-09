/**************************************************************************/
/* in_memory_window_logits.cpp                                           */
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
 * @file in_memory_window_logits.cpp
 * @brief Implements InMemoryWindowLogits.
 * @author Mustafa Garip
 */

#include <SushiAI/language/generation/in_memory_window_logits.hpp>

#include <algorithm>
#include <SushiAI/failure/check.hpp>

namespace SushiAI::Language::Generation
{
    InMemoryWindowLogits::InMemoryWindowLogits(std::int64_t window, std::int64_t vocabulary)
        : window_(window),
          vocabulary_(vocabulary),
          scores_(static_cast<std::size_t>(window * vocabulary), 0.0f)
    {

    }

    std::int64_t InMemoryWindowLogits::window() const noexcept
    {
        return window_;
    }

    std::int64_t InMemoryWindowLogits::vocabulary() const noexcept
    {
        return vocabulary_;
    }

    Status InMemoryWindowLogits::run(SushiRuntime::span<const Text::TokenID> ids)
    {
        SA_FAIL_IF(static_cast<std::int64_t>(ids.size()) != window_,
                   StatusCode::InvalidDescription,
                   "the window is run on {} ids; it holds {}", ids.size(), window_);

        windows_.emplace_back(ids.begin(), ids.end());
        std::fill(scores_.begin(), scores_.end(), 0.0f);
        for (std::size_t position = 0; position < ids.size(); ++position)
        {
            const std::int64_t successor = (ids[position] + 1) % vocabulary_;
            scores_[position * static_cast<std::size_t>(vocabulary_) +
                    static_cast<std::size_t>(successor)] = 1.0f;
        }

        return Status();
    }

    LogitSpan InMemoryWindowLogits::row(std::size_t position) const noexcept
    {
        const std::size_t scores = static_cast<std::size_t>(vocabulary_);
        return LogitSpan(scores_.data() + position * scores, scores);
    }

    const std::vector<std::vector<Text::TokenID>>& InMemoryWindowLogits::windows() const noexcept
    {
        return windows_;
    }

} // namespace SushiAI::Language::Generation
