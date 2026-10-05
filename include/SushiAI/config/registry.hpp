/****************************************************************/
/* registry.hpp                                                 */
/* SushiAI - https://github.com/SushiSystems/SushiAI            */
/* Copyright (c) 2026 Sushi Systems                             */
/* Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.    */
/* Commercial use requires a licence from Sushi Systems.        */
/****************************************************************/

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
