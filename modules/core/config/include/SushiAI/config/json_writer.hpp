/**************************************************************************/
/* json_writer.hpp                                                        */
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
 * @file json_writer.hpp
 * @brief Declares the writer that turns a configuration value into JSON text.
 * @author Mustafa Garip
 */

#pragma once

#include <cstddef>
#include <string>
#include <SushiAI/config/value.hpp>
#include <SushiAI/core/result.hpp>

namespace SushiAI
{
    namespace Config
    {
        /** @brief Holds how many levels of a configuration document take one line per member. */
        inline constexpr std::size_t K_JSON_EXPANDED_DEPTH = 2;

        /**
         * @brief Writes @p value as JSON text that parse_json reads back to an equal tree.
         * @param expanded_depth How many levels from the root take one line per member,
         *        indented four spaces; a deeper container takes one line.
         * @return UTF-8 text with line feed line ends, or InvalidDescription for a number
         *         that is not finite, text that is not UTF-8 or a key stated twice, or
         *         OutOfMemory.
         */
        [[nodiscard]] Result<std::string> write_json(
            const Value& value, std::size_t expanded_depth = K_JSON_EXPANDED_DEPTH) noexcept;

    } // namespace Config
} // namespace SushiAI
