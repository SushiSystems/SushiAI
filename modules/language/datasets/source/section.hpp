/**************************************************************************/
/* section.hpp                                                            */
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
 * @file section.hpp
 * @brief Defines the readers the dataset rows share over the keys of a data block.
 * @author Mustafa Garip
 */

#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <SushiAI/config/section_reader.hpp>
#include <SushiAI/config/value.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/failure/status.hpp>

namespace SushiAI::Language::Datasets::Detail
{
    /** @brief Names the key that holds the positions of one input row. */
    inline constexpr std::string_view K_WINDOW_KEY = "window";

    /**
     * @brief Reads the required string under @p key.
     * @return InvalidDescription when the key is absent or holds anything but a string.
     */
    [[nodiscard]] inline Result<std::string> read_text(const Config::SectionReader& reader,
                                                       std::string_view key)
    {
        SA_TRY_ASSIGN(const Config::Value* const member,
                      reader.find_of(key, Config::Kind::String));

        return member->as_string(key);
    }

    /**
     * @brief Reads the whole number above zero under @p key.
     * @return InvalidDescription when the key is absent, no whole number, or below one.
     */
    [[nodiscard]] inline Result<std::int64_t> read_count(const Config::SectionReader& reader,
                                                         std::string_view key)
    {
        SA_TRY_ASSIGN(const Config::Value* const member, reader.find(key));
        SA_TRY_ASSIGN(const std::int64_t count,
                      Config::read_whole(*member, reader.member(key)));
        SA_FAIL_IF(count <= 0, StatusCode::InvalidDescription,
                   "{} must be positive, but it is {}", reader.member(key), count);

        return count;
    }

} // namespace SushiAI::Language::Datasets::Detail
