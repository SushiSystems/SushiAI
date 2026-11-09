/**************************************************************************/
/* window_generator.cpp                                                  */
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
 * @file window_generator.cpp
 * @brief Implements WindowGenerator over a source of window logits.
 * @author Mustafa Garip
 */

#include <SushiAI/language/generation/window_generator.hpp>

#include <algorithm>
#include <cstddef>
#include <utility>
#include <SushiAI/seam/seam.hpp>
#include "evaluation_window_logits.hpp"
#include "request.hpp"

namespace SushiAI::Language::Generation
{
    namespace
    {
        /** @brief Names this unit in a message. */
        constexpr std::string_view K_GENERATOR = "WindowGenerator";

    } // namespace

    Result<std::unique_ptr<WindowGenerator>> WindowGenerator::make(SushiBLAS::Engine& engine,
                                                                  Train::EvaluationStep& step,
                                                                  std::string_view output) noexcept
    {
        SA_TRY_ASSIGN(std::unique_ptr<WindowLogits> source,
                      Detail::EvaluationWindowLogits::make(engine, step, output));

        return make(std::move(source));
    }

    Result<std::unique_ptr<WindowGenerator>> WindowGenerator::make(
        std::unique_ptr<WindowLogits> source) noexcept
    {
        return SushiAI::Detail::guard_value(SushiAI::Detail::SeamPhase::Allocate, [&]()
        {
            return std::unique_ptr<WindowGenerator>(new WindowGenerator(std::move(source)));
        });
    }

    WindowGenerator::WindowGenerator(std::unique_ptr<WindowLogits> source) noexcept
        : source_(std::move(source))
    {

    }

    std::int64_t WindowGenerator::window() const noexcept
    {
        return source_->window();
    }

    std::int64_t WindowGenerator::vocabulary() const noexcept
    {
        return source_->vocabulary();
    }

    Status WindowGenerator::generate(SushiRuntime::span<const Text::TokenID> prompt,
                                     std::int64_t count, const Sampler& sampler,
                                     RandomGenerator& random,
                                     std::vector<Text::TokenID>& generated)
    {
        SA_TRY(Detail::check_request(prompt, count, vocabulary(), K_GENERATOR));

        const std::size_t length = static_cast<std::size_t>(window());
        const std::size_t kept = std::min(prompt.size(), length);
        std::vector<Text::TokenID> context(prompt.end() - static_cast<std::ptrdiff_t>(kept),
                                           prompt.end());

        std::vector<Text::TokenID> ids(length);
        for (std::int64_t made = 0; made < count; ++made)
        {
            // A position behind the context holds 0, which no earlier position reads.
            std::fill(ids.begin(), ids.end(), Text::TokenID{0});
            std::copy(context.begin(), context.end(), ids.begin());
            SA_TRY(source_->run(SushiRuntime::span<const Text::TokenID>(ids.data(), ids.size())));

            const std::size_t last = context.size() - 1;
            const LogitSpan row = source_->row(last);
            SA_TRY(Detail::check_finite(row, last, K_GENERATOR));

            const Text::TokenID token = sampler.sample(row, random);
            generated.push_back(token);
            if (context.size() == length)
                context.erase(context.begin());

            context.push_back(token);
        }

        return Status();
    }

} // namespace SushiAI::Language::Generation
