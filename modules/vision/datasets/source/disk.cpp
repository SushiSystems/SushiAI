/**************************************************************************/
/* disk.cpp                                                               */
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
 * @file disk.cpp
 * @brief Implements the file system calls the format readers share.
 * @author Mustafa Garip
 */

#include "disk.hpp"

#include <algorithm>
#include <fstream>
#include <iterator>
#include <limits>
#include <set>
#include <system_error>

#include <SushiLog/text/text_format.hpp>

namespace SushiAI::Vision::Datasets::Detail
{
    namespace
    {
        /** @brief Reports whether a directory entry is of the kind asked for. */
        [[nodiscard]] bool is_kind(const std::filesystem::directory_entry& entry, EntryKind kind)
        {
            std::error_code ignored;
            return kind == EntryKind::Directory ? entry.is_directory(ignored)
                                                : entry.is_regular_file(ignored);
        }

        /**
         * @brief Appends the entries of one kind that an iterator type visits.
         * @tparam Iterator std::filesystem::directory_iterator or its recursive form.
         */
        template <typename Iterator>
        [[nodiscard]] Status collect(const std::filesystem::path& directory, EntryKind kind,
                                     std::string_view reader,
                                     std::vector<std::filesystem::path>& found)
        {
            std::error_code error;
            Iterator entry(directory, error);

            // increment() makes the iterator the end iterator when it sets error.
            for (const Iterator end; !error && entry != end; entry.increment(error))
            {
                if (is_kind(*entry, kind))
                    found.push_back(entry->path());
            }

            SA_FAIL_IF(error, StatusCode::InvalidFile,
                       "'{}': {}: the directory cannot be listed: {}", display_of(directory),
                       reader, error.message());
            return Status();
        }

        /** @brief Reports whether a path lies strictly below a directory, by its components. */
        [[nodiscard]] bool is_below(const std::filesystem::path& path,
                                    const std::filesystem::path& directory)
        {
            const auto stopped = std::mismatch(directory.begin(), directory.end(), path.begin(),
                                               path.end());
            return stopped.first == directory.end() &&
                   std::distance(path.begin(), path.end()) >
                       std::distance(directory.begin(), directory.end());
        }

        /** @brief Inserts a folder and every folder above it that has a name. */
        void insert_with_parents(const std::filesystem::path& path,
                                 std::set<std::filesystem::path>& folders)
        {
            for (std::filesystem::path folder = path;
                 !folder.empty() && folder != folder.root_path(); folder = folder.parent_path())
            {
                if (!folders.insert(folder).second)
                    break;
            }
        }
    } // namespace

    std::string display_of(const std::filesystem::path& path)
    {
        return path.generic_string();
    }

    Result<std::vector<std::filesystem::path>> list_entries(
        const std::filesystem::path& directory, EntryKind kind, ListDepth depth,
        std::string_view reader)
    {
        std::error_code error;
        SA_FAIL_IF(!std::filesystem::is_directory(directory, error), StatusCode::FileNotFound,
                   "'{}': {}: no such directory", display_of(directory), reader);

        std::vector<std::filesystem::path> found;
        if (depth == ListDepth::Tree)
        {
            SA_TRY(collect<std::filesystem::recursive_directory_iterator>(directory, kind, reader,
                                                                          found));
        }
        else
        {
            SA_TRY(collect<std::filesystem::directory_iterator>(directory, kind, reader, found));
        }

        std::sort(found.begin(), found.end());
        return Result<std::vector<std::filesystem::path>>(std::move(found));
    }

    Result<std::uint64_t> size_of_file(const std::filesystem::path& file, std::string_view reader)
    {
        std::error_code error;
        const std::filesystem::file_status status = std::filesystem::status(file, error);
        SA_FAIL_IF(status.type() == std::filesystem::file_type::not_found,
                   StatusCode::FileNotFound, "'{}': {}: no such file", display_of(file), reader);
        SA_FAIL_IF(!std::filesystem::is_regular_file(status), StatusCode::InvalidFile,
                   "'{}': {}: the path does not name a regular file", display_of(file), reader);

        const std::uintmax_t bytes = std::filesystem::file_size(file, error);
        SA_FAIL_IF(error, StatusCode::InvalidFile, "'{}': {}: the file's size cannot be read: {}",
                   display_of(file), reader, error.message());

        return static_cast<std::uint64_t>(bytes);
    }

    Result<std::string> read_text_file(const std::filesystem::path& file, std::string_view reader)
    {
        SA_TRY_ASSIGN(const std::uint64_t bytes, size_of_file(file, reader));
        SA_FAIL_IF(bytes > static_cast<std::uint64_t>(std::numeric_limits<std::streamsize>::max()),
                   StatusCode::Unsupported, "'{}': {}: {} bytes are more than one read holds",
                   display_of(file), reader, bytes);

        std::ifstream in(file, std::ios::binary);
        SA_FAIL_IF(!in.is_open(), StatusCode::InvalidFile,
                   "'{}': {}: the file cannot be opened for reading", display_of(file), reader);

        std::string text(static_cast<std::size_t>(bytes), '\0');
        in.read(text.data(), static_cast<std::streamsize>(text.size()));
        SA_FAIL_IF(in.gcount() != static_cast<std::streamsize>(text.size()),
                   StatusCode::InvalidFile, "'{}': {}: the read ended after {} of {} bytes",
                   display_of(file), reader, in.gcount(), text.size());

        return Result<std::string>(std::move(text));
    }

    Status boundary_failure(StatusCode code, std::string_view reader,
                            std::string_view what) noexcept
    {
        return Status::from_message(code, [&]()
        {
            return SushiLog::format_text("{}: {}", reader, what);
        });
    }

    Status boundary_failure(StatusCode code, std::string_view reader,
                            const std::filesystem::path& subject, std::string_view what) noexcept
    {
        return Status::from_message(code, [&]()
        {
            return SushiLog::format_text("'{}': {}: {}", display_of(subject), reader, what);
        });
    }

    PathKind FileSystemSource::kind_of(const std::filesystem::path& path) const
    {
        std::error_code ignored;
        const std::filesystem::file_status status = std::filesystem::status(path, ignored);

        if (status.type() == std::filesystem::file_type::not_found)
            return PathKind::Missing;
        if (std::filesystem::is_regular_file(status))
            return PathKind::File;
        if (std::filesystem::is_directory(status))
            return PathKind::Directory;
        return PathKind::Other;
    }

    Result<std::uint64_t> FileSystemSource::size_of(const std::filesystem::path& file,
                                                    std::string_view reader) const
    {
        return size_of_file(file, reader);
    }

    Result<std::int64_t> FileSystemSource::modified_of(const std::filesystem::path& file,
                                                       std::string_view reader) const
    {
        SA_TRY(size_of_file(file, reader));

        std::error_code error;
        const std::filesystem::file_time_type modified =
            std::filesystem::last_write_time(file, error);
        SA_FAIL_IF(static_cast<bool>(error), StatusCode::InvalidFile,
                   "'{}': {}: the modification time cannot be read: {}", display_of(file), reader,
                   error.message());

        return static_cast<std::int64_t>(modified.time_since_epoch().count());
    }

    Result<std::string> FileSystemSource::read(const std::filesystem::path& file,
                                               std::string_view reader) const
    {
        return read_text_file(file, reader);
    }

    Result<std::string> FileSystemSource::read_range(const std::filesystem::path& file,
                                                     std::uint64_t offset, std::uint64_t count,
                                                     std::string_view reader) const
    {
        SA_TRY_ASSIGN(const std::uint64_t bytes, size_of_file(file, reader));

        std::ifstream in(file, std::ios::binary);
        SA_FAIL_IF(!in.is_open(), StatusCode::InvalidFile,
                   "'{}': {}: the file cannot be opened for reading", display_of(file), reader);

        const std::uint64_t wanted = std::min(count, bytes > offset ? bytes - offset : 0);
        SA_FAIL_IF(wanted > static_cast<std::uint64_t>(std::numeric_limits<std::streamsize>::max()),
                   StatusCode::Unsupported, "'{}': {}: {} bytes are more than one read holds",
                   display_of(file), reader, wanted);

        std::string out(static_cast<std::size_t>(wanted), '\0');
        if (wanted != 0)
        {
            in.seekg(static_cast<std::streamoff>(offset), std::ios::beg);
            in.read(out.data(), static_cast<std::streamsize>(out.size()));
            out.resize(static_cast<std::size_t>(in.gcount()));
        }

        return Result<std::string>(std::move(out));
    }

    Result<std::vector<std::filesystem::path>> FileSystemSource::list(
        const std::filesystem::path& directory, EntryKind kind, ListDepth depth,
        std::string_view reader) const
    {
        return list_entries(directory, kind, depth, reader);
    }

    const ByteSource& file_system() noexcept
    {
        static const FileSystemSource K_FILE_SYSTEM;
        return K_FILE_SYSTEM;
    }
} // namespace SushiAI::Vision::Datasets::Detail

namespace SushiAI::Vision::Datasets
{
    void MemorySource::add_file(const std::filesystem::path& path, std::string bytes)
    {
        files_[path] = std::move(bytes);
    }

    void MemorySource::add_directory(const std::filesystem::path& path)
    {
        directories_.insert(path);
    }

    PathKind MemorySource::kind_of(const std::filesystem::path& path) const
    {
        if (files_.count(path) != 0)
            return PathKind::File;

        if (directories_.count(path) != 0)
            return PathKind::Directory;

        for (const auto& entry : files_)
        {
            if (Detail::is_below(entry.first, path))
                return PathKind::Directory;
        }

        return PathKind::Missing;
    }

    Result<std::uint64_t> MemorySource::size_of(const std::filesystem::path& file,
                                                std::string_view reader) const
    {
        const PathKind kind = kind_of(file);
        SA_FAIL_IF(kind == PathKind::Missing, StatusCode::FileNotFound, "'{}': {}: no such file",
                   Detail::display_of(file), reader);
        SA_FAIL_IF(kind != PathKind::File, StatusCode::InvalidFile,
                   "'{}': {}: the path does not name a regular file", Detail::display_of(file),
                   reader);

        return static_cast<std::uint64_t>(files_.at(file).size());
    }

    Result<std::int64_t> MemorySource::modified_of(const std::filesystem::path& file,
                                                   std::string_view reader) const
    {
        SA_TRY(size_of(file, reader));
        return static_cast<std::int64_t>(0);
    }

    Result<std::string> MemorySource::read(const std::filesystem::path& file,
                                           std::string_view reader) const
    {
        SA_TRY(size_of(file, reader));
        return Result<std::string>(files_.at(file));
    }

    Result<std::string> MemorySource::read_range(const std::filesystem::path& file,
                                                 std::uint64_t offset, std::uint64_t count,
                                                 std::string_view reader) const
    {
        SA_TRY_ASSIGN(const std::uint64_t bytes, size_of(file, reader));

        if (offset >= bytes)
            return Result<std::string>(std::string());

        const std::uint64_t wanted = std::min(count, bytes - offset);
        return Result<std::string>(files_.at(file).substr(static_cast<std::size_t>(offset),
                                                          static_cast<std::size_t>(wanted)));
    }

    Result<std::vector<std::filesystem::path>> MemorySource::list(
        const std::filesystem::path& directory, EntryKind kind, ListDepth depth,
        std::string_view reader) const
    {
        SA_FAIL_IF(kind_of(directory) != PathKind::Directory, StatusCode::FileNotFound,
                   "'{}': {}: no such directory", Detail::display_of(directory), reader);

        std::set<std::filesystem::path> candidates;
        if (kind == EntryKind::File)
        {
            for (const auto& entry : files_)
                candidates.insert(entry.first);
        }
        else
        {
            for (const auto& entry : files_)
                Detail::insert_with_parents(entry.first.parent_path(), candidates);
            for (const std::filesystem::path& folder : directories_)
                Detail::insert_with_parents(folder, candidates);
        }

        std::vector<std::filesystem::path> found;
        for (const std::filesystem::path& candidate : candidates)
        {
            if (!Detail::is_below(candidate, directory))
                continue;

            if (depth == ListDepth::Children && candidate.parent_path() != directory)
                continue;

            found.push_back(candidate);
        }

        return Result<std::vector<std::filesystem::path>>(std::move(found));
    }
} // namespace SushiAI::Vision::Datasets
