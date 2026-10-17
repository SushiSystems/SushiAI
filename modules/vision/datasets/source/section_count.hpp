/**************************************************************************/
/* section_count.hpp                                                      */
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
 * @file section_count.hpp
 * @brief Declares the reader of a whole number of a data section into a 32-bit count.
 * @author Mustafa Garip
 */

#pragma once

#include <cstdint>
#include <string_view>

#include <SushiAI/config/value.hpp>
#include <SushiAI/failure/result.hpp>

namespace SushiAI::Vision::Datasets::Detail
{
    /**
     * @brief Reads @p value as a whole number in [@p lowest, the largest int32].
     * @param key Names the value in a message, after @p context.
     * @return The number; InvalidDescription, naming the bounds, for one outside them.
     * @throws Error If @p value is not a whole number.
     */
    [[nodiscard]] Result<std::int32_t> read_section_count(const Config::Value& value,
                                                          std::string_view key,
                                                          std::string_view context,
                                                          std::int32_t lowest);
} // namespace SushiAI::Vision::Datasets::Detail
