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

#pragma once

/**
 * @file registry.hpp
 * @brief The one lookup every "type" discriminator in a config file goes through.
 *
 * @section shape What a registry is here
 *
 * Four tables select what a driver would otherwise have to name: layers,
 * datasets, optimizers and objectives. Each is a file-scope @c constexpr array
 * of @c {name, factory function pointer}, in the shape of @c Ops::ALL and the
 * fusion @c CATALOG, and @c "type" is the discriminator key in all four - one
 * convention, not four. So the lookup and its diagnostic are written once,
 * here, and the four tables differ only in what their factories return.
 *
 * @section registration Nothing self-registers
 *
 * @c sushiai is a **static** library, so a translation unit whose only job is
 * to run a registration constructor resolves no symbol and is dropped by the
 * linker - silently, and only in a downstream consumer's build. That is the
 * same argument docs/ARCHITECTURE.md §4 gives for the backward-rule table and
 * §5.2 for the fusion catalog. Adding an entry is one table row and one
 * factory function, both visible in a diff.
 *
 * Taking the address of a function declared in a header and defined in a .cpp
 * is an address constant expression, so each table stays @c constexpr and this
 * lookup stays a plain walk over it.
 *
 * @section diagnostics An unknown type says what IS available
 *
 * A driver that knows no type names cannot suggest one, so the error has to.
 * @ref find_entry lists the table's whole contents, which is the difference
 * between "unknown layer type 'conv2d'" and a message the reader can act on
 * without opening the source.
 *
 * Host lane: no SushiBLAS engine, no SYCL.
 */

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
         *
         * @tparam Entry A struct with a @c std::string_view member named @c type.
         * @tparam N The table's size, deduced.
         * @param table The registry, a file-scope constant.
         * @param type The discriminator read from the config file.
         * @param what What kind of thing is being looked up, e.g. "layer type".
         * @return The matching entry.
         * @throws Error Naming @p what, @p type and every entry in @p table.
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
