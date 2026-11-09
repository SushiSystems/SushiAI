/**************************************************************************/
/* in_memory_generator.cpp                                               */
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
 * @file in_memory_generator.cpp
 * @brief Implements InMemoryGenerator.
 * @author Mustafa Garip
 */

#include <SushiAI/language/generation/in_memory_generator.hpp>

#include <cstddef>
#include <string_view>
#include "request.hpp"

namespace SushiAI::Language::Generation
{
    namespace
    {
        /** @brief Names this unit in a message. */
        constexpr std::string_view K_GENERATOR = "InMemoryGenerator";

    } // namespace

    InMemoryGenerator::InMemoryGenerator(std::int64_t window, std::int64_t vocabulary)
        : window_(window), vocabulary_(vocabulary), destroyed_(std::make_shared<bool>(false))
    {

    }

    InMemoryGenerator::~InMemoryGenerator()
    {
        *destroyed_ = true;
    }

    std::int64_t InMemoryGenerator::window() const noexcept
    {
        return window_;
    }

    std::int64_t InMemoryGenerator::vocabulary() const noexcept
    {
        return vocabulary_;
    }

    Status InMemoryGenerator::generate(SushiRuntime::span<const Text::TokenID> prompt,
                                       std::int64_t count, const Sampler& sampler,
                                       RandomGenerator& random,
                                       std::vector<Text::TokenID>& generated)
    {
        SA_TRY(Detail::check_request(prompt, count, vocabulary_, K_GENERATOR));

        Text::TokenID last = prompt[prompt.size() - 1];
        for (std::int64_t made = 0; made < count; ++made)
        {
            std::vector<float> logits(static_cast<std::size_t>(vocabulary_), 0.0f);
            logits[static_cast<std::size_t>((last + 1) % vocabulary_)] = 1.0f;
            last = sampler.sample(LogitSpan(logits.data(), logits.size()), random);
            generated.push_back(last);
        }

        return Status();
    }

    std::shared_ptr<const bool> InMemoryGenerator::destruction_witness() const noexcept
    {
        return destroyed_;
    }

} // namespace SushiAI::Language::Generation
