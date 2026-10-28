/**************************************************************************/
/* darknet_config.hpp                                                     */
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
 * @file darknet_config.hpp
 * @brief Declares the reader of Darknet's .cfg text and the sections and options it yields.
 * @author Mustafa Garip
 */

#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <string_view>
#include <vector>
#include <SushiAI/failure/result.hpp>

namespace SushiAI
{
    namespace Vision::Importers
    {
        /** @brief Holds one key and its value, both with the whitespace removed. */
        struct DarknetOption
        {
            /** @brief Holds the key, the text before the first '='. */
            std::string key;

            /** @brief Holds the value, the text after the first '='. */
            std::string value;

            /** @brief Holds the 1-based line of the file that spells the key. */
            std::int64_t line = 0;
        };

        /** @brief Holds one bracketed section and the options under it, in file order. */
        struct DarknetSection
        {
            /** @brief Holds the name between the brackets, as the file spells it. */
            std::string type;

            /** @brief Holds the 1-based line of the file that opens the section. */
            std::int64_t line = 0;

            /** @brief Holds the options of the section, in file order. */
            std::vector<DarknetOption> options;
        };

        /** @brief Holds a parsed .cfg: its first section and the layers that follow it. */
        struct DarknetConfig
        {
            /** @brief Holds the first section, [net] or [network]. */
            DarknetSection net;

            /** @brief Holds the later sections; index i is Darknet's layer i. */
            std::vector<DarknetSection> layers;
        };

        /**
         * @brief Parses the text of a .cfg file into its sections, without reading what they mean.
         * @param text The whole file. Lines end at a line feed; carriage returns are ignored.
         * @param origin The name the diagnostics quote, usually the file's path.
         * @return The parsed configuration, or InvalidDescription naming origin and the line
         *         at fault, or when the text holds no section.
         */
        [[nodiscard]] Result<DarknetConfig> parse_darknet_config(std::string_view text,
                                                                 std::string_view origin) noexcept;

        /**
         * @brief Reads the .cfg file at @p path and parses it.
         * @return The parsed configuration, FileNotFound when the file cannot be opened,
         *         InvalidFile when a read fails, or the failure of parse_darknet_config.
         */
        [[nodiscard]] Result<DarknetConfig> load_darknet_config(const std::filesystem::path& path) noexcept;
    } // namespace Vision::Importers
} // namespace SushiAI
