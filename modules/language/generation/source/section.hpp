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
 * @brief Defines the readers the sampler section and the task section share over their keys.
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

namespace SushiAI::Language::Generation::Detail
{
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
     * @brief Reads the required whole number under @p key.
     * @return InvalidDescription when the key is absent or holds anything but a whole number.
     */
    [[nodiscard]] inline Result<std::int64_t> read_whole(const Config::SectionReader& reader,
                                                         std::string_view key)
    {
        SA_TRY_ASSIGN(const Config::Value* const member, reader.find(key));

        return Config::read_whole(*member, reader.member(key));
    }

    /**
     * @brief Reads the whole number under @p key, or @p fallback when the key is absent.
     * @return InvalidDescription when the key holds anything but a whole number.
     */
    [[nodiscard]] inline Result<std::int64_t> read_whole_or(const Config::SectionReader& reader,
                                                            std::string_view key,
                                                            std::int64_t fallback)
    {
        if (!reader.has(key))
            return fallback;

        return read_whole(reader, key);
    }

    /**
     * @brief Reads the required number under @p key.
     * @return InvalidDescription when the key is absent or holds anything but a number.
     */
    [[nodiscard]] inline Result<double> read_number(const Config::SectionReader& reader,
                                                    std::string_view key)
    {
        SA_TRY_ASSIGN(const Config::Value* const member, reader.find(key));

        return Config::read_real(*member, reader.member(key));
    }

    /**
     * @brief Reads the number under @p key, or @p fallback when the key is absent.
     * @return InvalidDescription when the key holds anything but a number.
     */
    [[nodiscard]] inline Result<double> read_number_or(const Config::SectionReader& reader,
                                                       std::string_view key, double fallback)
    {
        if (!reader.has(key))
            return fallback;

        return read_number(reader, key);
    }

    /**
     * @brief Reads the boolean under @p key, or @p fallback when the key is absent.
     * @return InvalidDescription when the key holds anything but a boolean.
     */
    [[nodiscard]] inline Result<bool> read_flag_or(const Config::SectionReader& reader,
                                                   std::string_view key, bool fallback)
    {
        if (!reader.has(key))
            return fallback;

        SA_TRY_ASSIGN(const Config::Value* const member,
                      reader.find_of(key, Config::Kind::Bool));

        return member->as_bool(key);
    }

} // namespace SushiAI::Language::Generation::Detail
