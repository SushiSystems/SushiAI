/**************************************************************************/
/* committed_file.hpp                                                     */
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
 * @file committed_file.hpp
 * @brief Declares the write of a file that takes its name only when the whole file was written.
 * @author Mustafa Garip
 */

#pragma once

#include <filesystem>
#include <fstream>
#include <utility>
#include <SushiAI/failure/status.hpp>

namespace SushiAI::IO::Detail
{
    /** @brief Returns the name a file has while it is being written, beside the file. */
    [[nodiscard]] std::filesystem::path partial_path_of(const std::filesystem::path& path);

    /**
     * @brief Gives the finished @p partial file the name @p path, replacing a file of that name.
     * @return Success; InvalidFile when the rename is refused, the partial file is removed then.
     */
    [[nodiscard]] Status commit_partial(const std::filesystem::path& partial,
                                        const std::filesystem::path& path) noexcept;

    /** @brief Removes @p partial; a file that is already gone is no error. */
    void discard_partial(const std::filesystem::path& partial) noexcept;

    /**
     * @brief Fills the partial file of @p path through @p fill and renames it over @p path.
     * @param fill Callable taking the output stream and returning a Status.
     * @return Success; InvalidFile when the file cannot be opened, written or renamed. A failure
     *         leaves a file already named @p path as it was and no partial file behind.
     */
    template<typename Fill>
    [[nodiscard]] Status write_committed(const std::filesystem::path& path, Fill&& fill)
    {
        const std::filesystem::path partial = partial_path_of(path);

        // The stream closes at the end of the lambda, before the rename.
        const Status written = [&]() -> Status
        {
            std::ofstream out(partial, std::ios::binary | std::ios::trunc);
            SA_FAIL_IF(!out, StatusCode::InvalidFile, "could not open '{}' for writing",
                       path.string());
            SA_TRY(fill(out));
            out.flush();
            SA_FAIL_IF(!out, StatusCode::InvalidFile, "failed writing '{}'", path.string());
            return Status();
        }();

        if (!written.is_ok())
        {
            discard_partial(partial);
            return written;
        }
        return commit_partial(partial, path);
    }
} // namespace SushiAI::IO::Detail
