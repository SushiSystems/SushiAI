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
 * @brief Implements the file system calls of the text module.
 * @author Mustafa Garip
 */

#include "disk.hpp"

#include <cstddef>
#include <fstream>
#include <ios>
#include <system_error>

#include <SushiBLAS/core/format.hpp>

namespace SushiAI::Language::Text::Detail
{
    std::string display_of(const std::filesystem::path& path)
    {
        return path.generic_string();
    }

    Result<std::string> read_file(const std::filesystem::path& file, std::uint64_t limit,
                                  std::string_view reader)
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
        SA_FAIL_IF(bytes > limit, StatusCode::InvalidFile,
                   "'{}': {}: the file holds {} bytes; the format allows at most {}",
                   display_of(file), reader, bytes, limit);

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

    Status write_file(const std::filesystem::path& file, std::string_view bytes,
                      std::string_view writer)
    {
        std::ofstream out(file, std::ios::binary | std::ios::trunc);
        SA_FAIL_IF(!out.is_open(), StatusCode::InvalidFile,
                   "'{}': {}: the file cannot be opened for writing", display_of(file), writer);

        out.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
        out.flush();
        SA_FAIL_IF(!out, StatusCode::InvalidFile,
                   "'{}': {}: the stream failed while {} bytes were written", display_of(file),
                   writer, bytes.size());

        return Status();
    }

    Status boundary_failure(StatusCode code, std::string_view unit,
                            std::string_view what) noexcept
    {
        return Status::from_message(code, [&]()
        {
            return SushiBLAS::Core::format("{}: {}", unit, what);
        });
    }

    Status boundary_failure(StatusCode code, std::string_view unit,
                            const std::filesystem::path& subject, std::string_view what) noexcept
    {
        return Status::from_message(code, [&]()
        {
            return SushiBLAS::Core::format("'{}': {}: {}", display_of(subject), unit, what);
        });
    }
} // namespace SushiAI::Language::Text::Detail
