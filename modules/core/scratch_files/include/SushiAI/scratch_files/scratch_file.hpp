/**************************************************************************/
/* scratch_file.hpp                                                       */
/**************************************************************************/
/*                          This file is part of:                         */
/*                                 SushiAI                                */
/*                 https://github.com/SushiSystems/SushiAI                */
/*                         https://sushisystems.io                        */
/**************************************************************************/
/* Copyright (c) 2026-present Mustafa Garip & Sushi Systems               */
/*                                                                        */
/* Licensed under the PolyForm Noncommercial License 1.0.0 (the           */
/* "License"); you may not use this file except in compliance with the    */
/* License. You may obtain a copy of the License at                       */
/*                                                                        */
/*     https://polyformproject.org/licenses/noncommercial/1.0.0           */
/*                                                                        */
/* Noncommercial use is free. Commercial use requires a separate licence  */
/* from Sushi Systems; see COMMERCIAL.md. The software is provided        */
/* "as is", without warranty of any kind.                                 */
/**************************************************************************/

/**
 * @file scratch_file.hpp
 * @brief Declares ScratchFile, a file under the temporary directory removed on both ends.
 * @author Mustafa Garip
 */

#pragma once

#include <filesystem>
#include <system_error>

namespace SushiAI
{
    namespace Test
    {
        /** @brief Names a file under the temporary directory and removes it on both ends. */
        class ScratchFile
        {
            public:
                /** @brief Removes any file a previous run left under @p name. */
                explicit ScratchFile(const char* name)
                    : path_(std::filesystem::temp_directory_path() / name)
                {
                    std::error_code ignored;
                    std::filesystem::remove(path_, ignored);
                }

                /** @brief Removes the file. */
                ~ScratchFile()
                {
                    std::error_code ignored;
                    std::filesystem::remove(path_, ignored);
                }

                ScratchFile(const ScratchFile&) = delete;
                ScratchFile& operator=(const ScratchFile&) = delete;

                /** @brief Returns the path this file removes on destruction. */
                const std::filesystem::path& path() const noexcept { return path_; }

            private:
                /** @brief Holds the path of the scratch file. */
                std::filesystem::path path_;
        };

    } // namespace Test
} // namespace SushiAI
