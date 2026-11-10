/**************************************************************************/
/* scratch_names.hpp                                                      */
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
 * @file scratch_names.hpp
 * @brief Declares the function that gives a scratch file or folder a name no other case or
 *        process uses.
 * @author Mustafa Garip
 */

#pragma once

#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <random>
#include <string>

namespace SushiAI::Test
{
    /**
     * @brief Returns @p name with a random 16-digit hex tag put before its extension.
     *
     * Two cases, or two shard processes, that pass the same @p name get different results, so
     * none of them reads a folder another one removes or rewrites.
     */
    inline std::string unique_scratch_name(const std::string& name)
    {
        static thread_local std::mt19937_64 generator{std::random_device{}()};
        char tag[17];
        std::snprintf(tag, sizeof(tag), "%016llx",
                      static_cast<unsigned long long>(generator()));

        const std::filesystem::path given(name);
        return given.stem().string() + "_" + tag + given.extension().string();
    }
} // namespace SushiAI::Test
