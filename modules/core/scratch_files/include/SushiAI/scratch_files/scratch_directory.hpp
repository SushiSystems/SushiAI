/**************************************************************************/
/* scratch_directory.hpp                                                  */
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
 * @file scratch_directory.hpp
 * @brief Declares ScratchDirectory, a directory under the temporary one removed on both ends.
 * @author Mustafa Garip
 */

#pragma once

#include <filesystem>
#include <system_error>

namespace SushiAI
{
    namespace Test
    {
        /** @brief Names a directory under the temporary one and removes it on both ends. */
        class ScratchDirectory final
        {
            public:
                /** @brief Removes whatever a previous run left under @p name. */
                explicit ScratchDirectory(const char* name) noexcept
                    : path_(std::filesystem::temp_directory_path() / name)
                {
                    std::error_code ignored;
                    std::filesystem::remove_all(path_, ignored);
                }

                /** @brief Removes the directory and what it holds. */
                ~ScratchDirectory()
                {
                    std::error_code ignored;
                    std::filesystem::remove_all(path_, ignored);
                }

                ScratchDirectory(const ScratchDirectory&) = delete;
                ScratchDirectory& operator=(const ScratchDirectory&) = delete;
                ScratchDirectory(ScratchDirectory&&) = delete;
                ScratchDirectory& operator=(ScratchDirectory&&) = delete;

                /** @brief Returns the directory, which need not exist. */
                const std::filesystem::path& path() const noexcept { return path_; }

            private:
                /** @brief Holds the path of the directory. */
                std::filesystem::path path_;
        };

    } // namespace Test
} // namespace SushiAI
