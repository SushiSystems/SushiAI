/**************************************************************************/
/* model_text.hpp                                                         */
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
 * @file model_text.hpp
 * @brief Declares the parsers that turn the JSON text of a test into config values.
 * @author Mustafa Garip
 */

#pragma once

#include <string_view>
#include <SushiAI/config/model_config.hpp>
#include <SushiAI/config/value.hpp>

namespace SushiAI
{
    namespace Test
    {
        /** @brief Parses @p text as the attributes of one row, read from "<test>". */
        [[nodiscard]] inline Config::Value attributes_from(std::string_view text)
        {
            return Config::parse_json(text, "<test>");
        }

        /** @brief Parses @p text as a model file, read from "<test>". */
        [[nodiscard]] inline Config::ModelSpec model_from(std::string_view text)
        {
            return Config::parse_model(Config::parse_json(text, "<test>"), "<test>");
        }

    } // namespace Test
} // namespace SushiAI
