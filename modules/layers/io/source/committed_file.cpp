/**************************************************************************/
/* committed_file.cpp                                                     */
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
 * @file committed_file.cpp
 * @brief Implements the rename that makes a written file whole under its name.
 * @author Mustafa Garip
 */

#include "committed_file.hpp"

#include <string>
#include <system_error>

namespace SushiAI::IO::Detail
{
    namespace
    {
        /** @brief Holds what a file is called while it is being written. */
        constexpr const char* K_PARTIAL_EXTENSION = ".partial";
    } // namespace

    std::filesystem::path partial_path_of(const std::filesystem::path& path)
    {
        std::filesystem::path partial = path;
        partial += K_PARTIAL_EXTENSION;
        return partial;
    }

    void discard_partial(const std::filesystem::path& partial) noexcept
    {
        std::error_code ignored;
        std::filesystem::remove(partial, ignored);
    }

    Status commit_partial(const std::filesystem::path& partial,
                          const std::filesystem::path& path) noexcept
    {
        std::error_code refusal;
        std::filesystem::rename(partial, path, refusal);
        if (!refusal)
            return Status();

        discard_partial(partial);
        return Status::from_message(StatusCode::InvalidFile, [&]()
        {
            return SushiLog::format_text("could not put the written file under the name '{}': {}",
                                         path.string(), refusal.message());
        });
    }
} // namespace SushiAI::IO::Detail
