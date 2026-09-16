/**************************************************************************/
/* registry.hpp                                                           */
/**************************************************************************/
/*                          This file is part of:                         */
/*                                 SushiAI                                */
/*                https://github.com/SushiSystems/SushiAI                 */
/*                         https://sushisystems.io                        */
/**************************************************************************/
/* Copyright (c) 2026-present Mustafa Garip & Sushi Systems               */
/*                                                                        */
/* Licensed under the Apache License, Version 2.0 (the "License");        */
/* you may not use this file except in compliance with the License.       */
/* You may obtain a copy of the License at                                */
/*                                                                        */
/*     http://www.apache.org/licenses/LICENSE-2.0                         */
/*                                                                        */
/* Unless required by applicable law or agreed to in writing, software    */
/* distributed under the License is distributed on an "AS IS" BASIS,      */
/* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or        */
/* implied. See the License for the specific language governing           */
/* permissions and limitations under the License.                         */
/**************************************************************************/

/**
 * @file registry.hpp
 * @brief Resolves configuration type discriminators against static registries.
 * @author Mustafa Garip
 */

#pragma once

#include <array>
#include <cstddef>
#include <string>
#include <string_view>
#include <SushiAI/core/common.hpp>

namespace SushiAI
{
    namespace Config
    {
        /**
         * @brief Finds the entry named @p type, or throws listing what exists.
         * @param table Registry constant table.
         * @param type Discriminator string to locate.
         * @param what Diagnostic label describing entry category.
         * @return Reference to matching registry entry.
         * @throws Error If @p type is not found in @p table.
         */
        template<typename Entry, std::size_t N>
        const Entry& find_entry(const std::array<Entry, N>& table,
                                std::string_view type, std::string_view what)
        {
            for (const Entry& entry : table)
                if (entry.type == type) return entry;

            std::string available;
            for (const Entry& entry : table)
            {
                if (!available.empty()) available += ", ";
                available += "'";
                available.append(entry.type.data(), entry.type.size());
                available += "'";
            }

            SA_THROW_IF(true, "unknown {} '{}'; this build has: {}", what, type, available);
            return table[0];   // unreachable: SA_THROW_IF(true) always throws
        }

    } // namespace Config
} // namespace SushiAI
