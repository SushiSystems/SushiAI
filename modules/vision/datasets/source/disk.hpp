/**************************************************************************/
/* disk.hpp                                                               */
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
 * @file disk.hpp
 * @brief Declares the file system calls the format readers share, each returning a Status.
 * @author Mustafa Garip
 */

#pragma once

#include <cstdint>
#include <exception>
#include <filesystem>
#include <new>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <SushiAI/failure/result.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/vision/datasets/byte_source.hpp>

/** @brief Holds the units of the datasets module that no caller names. */
namespace SushiAI::Vision::Datasets::Detail
{
    using Datasets::EntryKind;
    using Datasets::ListDepth;

    /** @brief Returns the source that reads the file system, the default of every reader. */
    [[nodiscard]] const ByteSource& file_system() noexcept;

    /** @brief Returns the text of a path with forward slashes, for a message. */
    [[nodiscard]] std::string display_of(const std::filesystem::path& path);

    /**
     * @brief Returns the entries of one kind in or below a directory, sorted by path.
     * @param reader Names the calling reader in a message.
     * @return The paths, each opening with @p directory; FileNotFound when it is not a
     *         directory; InvalidFile when it cannot be listed.
     */
    [[nodiscard]] Result<std::vector<std::filesystem::path>> list_entries(
        const std::filesystem::path& directory, EntryKind kind, ListDepth depth,
        std::string_view reader);

    /**
     * @brief Returns the byte count of a regular file.
     * @param reader Names the calling reader in a message.
     * @return The size; FileNotFound when the path names nothing; InvalidFile when it names
     *         something else or the size cannot be read.
     */
    [[nodiscard]] Result<std::uint64_t> size_of_file(const std::filesystem::path& file,
                                                     std::string_view reader);

    /**
     * @brief Reads every byte of a file as text, changing none.
     * @param reader Names the calling reader in a message.
     * @return The bytes; what size_of_file returns; InvalidFile when the read stops early.
     */
    [[nodiscard]] Result<std::string> read_text_file(const std::filesystem::path& file,
                                                     std::string_view reader);

    /**
     * @class FileSystemSource
     * @brief Serves the files and folders of the file system, through the calls above.
     *
     * It is the source a reader's path overload reads through.
     */
    class FileSystemSource final : public ByteSource
    {
        public:
            [[nodiscard]] PathKind kind_of(const std::filesystem::path& path) const override;

            [[nodiscard]] Result<std::uint64_t> size_of(const std::filesystem::path& file,
                                                        std::string_view reader) const override;

            [[nodiscard]] Result<std::string> read(const std::filesystem::path& file,
                                                   std::string_view reader) const override;

            [[nodiscard]] Result<std::string> read_range(const std::filesystem::path& file,
                                                         std::uint64_t offset,
                                                         std::uint64_t count,
                                                         std::string_view reader) const override;

            [[nodiscard]] Result<std::vector<std::filesystem::path>> list(
                const std::filesystem::path& directory, EntryKind kind, ListDepth depth,
                std::string_view reader) const override;
    };

    /** @brief Holds what a message says of an allocation that failed. */
    inline constexpr std::string_view K_ALLOCATION_FAILED = "an allocation failed";

    /** @brief Holds what a message says of an exception that is no std::exception. */
    inline constexpr std::string_view K_UNKNOWN_EXCEPTION = "an unknown exception";

    /** @brief Returns a failure whose message is "<reader>: <what>"; never a success. */
    [[nodiscard]] Status boundary_failure(StatusCode code, std::string_view reader,
                                          std::string_view what) noexcept;

    /** @brief Returns a failure whose message is "'<path>': <reader>: <what>". */
    [[nodiscard]] Status boundary_failure(StatusCode code, std::string_view reader,
                                          const std::filesystem::path& subject,
                                          std::string_view what) noexcept;

    /**
     * @brief Runs a call and returns its Status or Result, or the Status of what it threw.
     * @param call Called once with no argument; returns a Status or a Result.
     * @param subject Nothing, or the one path a failure message opens with.
     * @return What the call returns; OutOfMemory when it throws std::bad_alloc; InvalidFile
     *         with the exception's text when it throws anything else.
     */
    template <typename Call, typename... Subject>
    [[nodiscard]] auto at_boundary(std::string_view reader, Call&& call,
                                   const Subject&... subject) noexcept
        -> decltype(std::forward<Call>(call)())
    {
        try
        {
            return std::forward<Call>(call)();
        }
        catch (const std::bad_alloc&)
        {
            return boundary_failure(StatusCode::OutOfMemory, reader, subject...,
                                    K_ALLOCATION_FAILED);
        }
        catch (const std::exception& error)
        {
            return boundary_failure(StatusCode::InvalidFile, reader, subject..., error.what());
        }
        catch (...)
        {
            return boundary_failure(StatusCode::InvalidFile, reader, subject...,
                                    K_UNKNOWN_EXCEPTION);
        }
    }
} // namespace SushiAI::Vision::Datasets::Detail
