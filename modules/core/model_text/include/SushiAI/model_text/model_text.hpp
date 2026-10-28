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

#include <cstdio>
#include <cstdlib>
#include <string>
#include <string_view>
#include <utility>
#include <SushiAI/config/model_config.hpp>
#include <SushiAI/config/value.hpp>

namespace SushiAI
{
    namespace Test
    {
        namespace Detail
        {
            /**
             * @brief Prints what a test fixture was refused with and stops the process.
             * @param kind Names the fixture: "JSON" or "model".
             */
            [[noreturn]] inline void stop_on_refused_fixture(std::string_view kind,
                                                             std::string_view message)
            {
                std::string text = "sushiai: fatal: the test's ";
                text.append(kind).append(" fixture is refused: ").append(message).append("\n");
                std::fputs(text.c_str(), stderr);
                std::abort();
            }
        } // namespace Detail

        /**
         * @brief Parses @p text as the attributes of one row, read from "<test>".
         * @return The parsed value, or InvalidFile when @p text is not valid JSON.
         */
        [[nodiscard]] inline Result<Config::Value> try_attributes_from(std::string_view text)
        {
            return Config::parse_json(text, "<test>");
        }

        /**
         * @brief Parses @p text as a model file, read from "<test>".
         * @return The spec, or the failure of the parse.
         */
        [[nodiscard]] inline Result<Config::ModelSpec> try_model_from(std::string_view text)
        {
            Result<Config::Value> root = Config::parse_json(text, "<test>");
            if (!root.has_value())
                return root.status();

            // The value is held, so value() succeeds.
            return Config::parse_model(root.value(), "<test>");
        }

        /**
         * @brief Parses @p text as the attributes of one row, read from "<test>".
         * @pre @p text is valid JSON; a fixture a test wrote is, and a test of a refusal calls
         *      try_attributes_from().
         */
        [[nodiscard]] inline Config::Value attributes_from(std::string_view text)
        {
            Result<Config::Value> parsed = try_attributes_from(text);
            if (!parsed.has_value())
                Detail::stop_on_refused_fixture("JSON", parsed.status().message());

            // The value is held, so value() succeeds.
            return std::move(parsed).value();
        }

        /**
         * @brief Parses @p text as a model file, read from "<test>".
         * @pre @p text is a valid model file; a test of a refusal calls try_model_from().
         */
        [[nodiscard]] inline Config::ModelSpec model_from(std::string_view text)
        {
            Result<Config::ModelSpec> parsed = try_model_from(text);
            if (!parsed.has_value())
                Detail::stop_on_refused_fixture("model", parsed.status().message());

            // The value is held, so value() succeeds.
            return std::move(parsed).value();
        }
    } // namespace Test
} // namespace SushiAI
