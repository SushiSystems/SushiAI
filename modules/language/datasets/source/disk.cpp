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
 * @brief Implements the file system calls of the language datasets module.
 * @author Mustafa Garip
 */

#include "disk.hpp"

#include <system_error>

#include <SushiLog/text/text_format.hpp>

namespace SushiAI::Language::Datasets::Detail
{
    std::string display_of(const std::filesystem::path& path)
    {
        return path.generic_string();
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

    Status boundary_failure(StatusCode code, std::string_view unit,
                            std::string_view what) noexcept
    {
        return Status::from_message(code, [&]()
        {
            return SushiLog::format_text("{}: {}", unit, what);
        });
    }

    Status boundary_failure(StatusCode code, std::string_view unit,
                            const std::filesystem::path& subject, std::string_view what) noexcept
    {
        return Status::from_message(code, [&]()
        {
            return SushiLog::format_text("'{}': {}: {}", display_of(subject), unit, what);
        });
    }
} // namespace SushiAI::Language::Datasets::Detail
