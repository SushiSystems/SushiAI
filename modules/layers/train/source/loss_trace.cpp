/**************************************************************************/
/* loss_trace.cpp                                                       */
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
 * @file loss_trace.cpp
 * @brief Implements LossTrace.
 * @author Mustafa Garip
 */

#include <SushiAI/train/loss_trace.hpp>

#include <SushiAI/failure/check.hpp>

namespace SushiAI
{
    namespace Train
    {
        void LossTrace::add(std::string_view name, Graph::ValueID id) noexcept
        {
            SA_FATAL_IF(has(name),
                        "the objective kept the value '{}' twice; each name holds one value",
                        name);
            values_.emplace_back(std::string(name), id);
        }

        bool LossTrace::has(std::string_view name) const noexcept
        {
            for (const std::pair<std::string, Graph::ValueID>& value : values_)
                if (value.first == name) return true;
            return false;
        }

        Graph::ValueID LossTrace::at(std::string_view name) const noexcept
        {
            for (const std::pair<std::string, Graph::ValueID>& value : values_)
                if (value.first == name) return value.second;

            std::string held;
            for (const std::pair<std::string, Graph::ValueID>& value : values_)
            {
                if (!held.empty()) held += ", ";
                held += "'" + value.first + "'";
            }
            if (held.empty()) held = "(none)";

            SA_FATAL("the loss trace holds no value '{}'; it holds {}", name, held);
        }

        void LossTrace::add_term(LossTrace term) noexcept
        {
            terms_.push_back(std::move(term));
        }

        const LossTrace& LossTrace::term(std::size_t index) const noexcept
        {
            static const LossTrace K_EMPTY;

            return index < terms_.size() ? terms_[index] : K_EMPTY;
        }

    } // namespace Train
} // namespace SushiAI
