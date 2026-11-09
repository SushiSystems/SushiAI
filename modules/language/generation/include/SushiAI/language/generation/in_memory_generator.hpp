/**************************************************************************/
/* in_memory_generator.hpp                                               */
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
 * @file in_memory_generator.hpp
 * @brief Declares InMemoryGenerator, the Generator that continues a prompt with successors.
 * @author Mustafa Garip
 */

#pragma once

#include <cstdint>
#include <memory>
#include <vector>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/language/generation/generator.hpp>
#include <SushiAI/language/generation/sampler.hpp>
#include <SushiAI/language/text/token.hpp>

namespace SushiAI::Language::Generation
{
    /**
     * @brief Scores the token after the last one of the context highest, modulo V, and
     *        appends what the sampler picks; it runs no model.
     */
    class InMemoryGenerator final : public Generator
    {
        public:
            /** @brief Builds the generator of a window of @p window ids over @p vocabulary tokens. */
            InMemoryGenerator(std::int64_t window, std::int64_t vocabulary);

            InMemoryGenerator(const InMemoryGenerator&) = delete;
            InMemoryGenerator& operator=(const InMemoryGenerator&) = delete;
            InMemoryGenerator(InMemoryGenerator&&) = delete;
            InMemoryGenerator& operator=(InMemoryGenerator&&) = delete;

            /** @brief Marks the generator as destroyed for the witness. */
            ~InMemoryGenerator() override;

            /** @brief Returns T. */
            [[nodiscard]] std::int64_t window() const noexcept override;

            /** @brief Returns V. */
            [[nodiscard]] std::int64_t vocabulary() const noexcept override;

            /** @brief Appends @p count samples; refuses what the Generator contract refuses. */
            [[nodiscard]] Status generate(SushiRuntime::span<const Text::TokenID> prompt,
                                          std::int64_t count, const Sampler& sampler,
                                          RandomGenerator& random,
                                          std::vector<Text::TokenID>& generated) override;

            /** @brief Returns a flag that reads true once this generator is destroyed. */
            [[nodiscard]] std::shared_ptr<const bool> destruction_witness() const noexcept;

        private:
            /// @brief Holds T.
            std::int64_t window_;

            /// @brief Holds V.
            std::int64_t vocabulary_;

            /// @brief Holds the flag the destructor sets.
            std::shared_ptr<bool> destroyed_;
    };

} // namespace SushiAI::Language::Generation
