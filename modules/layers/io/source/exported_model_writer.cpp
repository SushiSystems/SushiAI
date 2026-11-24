/**************************************************************************/
/* exported_model_writer.cpp                                              */
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
 * @file exported_model_writer.cpp
 * @brief Implements the writer that puts the files of an export into a directory.
 * @author Mustafa Garip
 */

#include <SushiAI/io/exported_model_writer.hpp>

#include <filesystem>
#include <fstream>
#include <ios>
#include <string>
#include <string_view>
#include <system_error>
#include <unordered_set>
#include <vector>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/io/exported_model.hpp>
#include <SushiAI/io/imported_model_writer.hpp>
#include <SushiAI/seam/seam.hpp>
#include "committed_file.hpp"

namespace SushiAI
{
    namespace IO
    {
        namespace
        {
            /** @brief Returns InvalidDescription unless every name is a plain, unique file name. */
            [[nodiscard]] Status check_names(const ExportedModel& exported)
            {
                SA_FAIL_IF(exported.files.empty(), StatusCode::InvalidDescription,
                           "the export holds no file");

                return SushiAI::Detail::guard(SushiAI::Detail::SeamPhase::Allocate,
                                              [&]() -> Status
                {
                    std::unordered_set<std::string_view> named;
                    for (const ExportedFile& file : exported.files)
                    {
                        const std::filesystem::path path(file.name);
                        SA_FAIL_IF(file.name.empty() || path.filename() != path ||
                                       file.name == "." || file.name == "..",
                                   StatusCode::InvalidDescription,
                                   "the exported file '{}' is no plain file name", file.name);
                        SA_FAIL_IF(!named.insert(file.name).second,
                                   StatusCode::InvalidDescription,
                                   "the export states the file '{}' twice", file.name);
                    }

                    return Status();
                });
            }

        } // namespace

        Result<std::vector<std::filesystem::path>> write_exported_model(
            const ExportedModel& exported, const std::filesystem::path& directory,
            ExistingFiles existing) noexcept
        {
            SA_TRY(check_names(exported));
            SA_TRY_ASSIGN(std::vector<std::filesystem::path> paths,
                          SushiAI::Detail::guard_value(SushiAI::Detail::SeamPhase::Allocate,
                                                       [&]()
            {
                std::vector<std::filesystem::path> planned;
                for (const ExportedFile& file : exported.files)
                    planned.push_back(directory / std::filesystem::path(file.name));
                return planned;
            }));

            if (existing == ExistingFiles::Refuse)
            {
                for (const std::filesystem::path& path : paths)
                {
                    std::error_code ignored;
                    SA_FAIL_IF(std::filesystem::exists(path, ignored), StatusCode::Incompatible,
                               "'{}' exists already, and the export was not told to replace it",
                               path.string());
                }
            }

            std::error_code failure;
            std::filesystem::create_directories(directory, failure);
            SA_FAIL_IF(failure, StatusCode::InvalidFile, "could not create '{}': {}",
                       directory.string(), failure.message());

            for (std::size_t i = 0; i < paths.size(); ++i)
            {
                const std::vector<std::byte>& bytes = exported.files[i].bytes;
                SA_TRY(Detail::write_committed(paths[i], [&](std::ofstream& out) -> Status
                {
                    out.write(reinterpret_cast<const char*>(bytes.data()),
                              static_cast<std::streamsize>(bytes.size()));
                    return Status();
                }));
            }

            return paths;
        }

    } // namespace IO
} // namespace SushiAI
