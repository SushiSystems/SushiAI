/**************************************************************************/
/* in_memory_fed_leaves.hpp                                              */
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
 * @file in_memory_fed_leaves.hpp
 * @brief Declares InMemoryFedLeaves, the FedLeaves that lists the leaves it was given.
 * @author Mustafa Garip
 */

#pragma once

#include <string>
#include <string_view>
#include <utility>
#include <vector>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/failure/check.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/language/generation/fed_leaves.hpp>

namespace SushiAI::Language::Generation
{
    /** @brief Pairs the name of a leaf with its shape and type. */
    struct NamedLeafFacts
    {
        /** @brief Names the leaf. */
        std::string name;

        /** @brief Holds the shape and the type of the leaf. */
        LeafFacts facts;
    };

    /** @brief Lists the leaves it was given, with no step behind them. */
    class InMemoryFedLeaves : public FedLeaves
    {
        public:
            /** @brief Takes the leaves, in the order names() lists them. */
            explicit InMemoryFedLeaves(std::vector<NamedLeafFacts> leaves)
                : leaves_(std::move(leaves))
            {
                for (const NamedLeafFacts& leaf : leaves_)
                    names_.push_back(leaf.name);
            }

            /** @brief Lists the names of the leaves. */
            [[nodiscard]] SushiRuntime::span<const std::string> names() const noexcept override
            {
                return SushiRuntime::span<const std::string>(names_.data(), names_.size());
            }

            /** @brief Returns the facts of the leaf @p name, or InvalidDescription. */
            [[nodiscard]] Result<LeafFacts> leaf(std::string_view name) const noexcept override
            {
                for (const NamedLeafFacts& held : leaves_)
                {
                    if (held.name == name)
                        return held.facts;
                }

                return Status::from_message(StatusCode::InvalidDescription, [&]()
                {
                    return std::string("the step is fed no leaf '") + std::string(name) + "'";
                });
            }

        private:
            /// @brief Holds the leaves with their facts.
            std::vector<NamedLeafFacts> leaves_;

            /// @brief Holds the names of the leaves, in the same order.
            std::vector<std::string> names_;
    };

} // namespace SushiAI::Language::Generation
