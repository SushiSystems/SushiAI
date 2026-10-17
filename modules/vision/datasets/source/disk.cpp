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
 * @file disk.cpp
 * @brief Implements the file system calls the format readers share.
 * @author Mustafa Garip
 */

#include "disk.hpp"

#include <algorithm>
#include <fstream>
#include <limits>
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
} // namespace SushiAI::Vision::Datasets::Detail
