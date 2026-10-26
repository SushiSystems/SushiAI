/**************************************************************************/
/* profile_arguments.hpp                                                  */
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
 * @file profile_arguments.hpp
 * @brief Declares read_profile_arguments, which takes the --profile pair out of a command line.
 * @author Mustafa Garip
 * @see applications/programs/README.md
 */

#pragma once

#include <filesystem>
#include <optional>
#include <string>
#include <string_view>
#include <vector>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/failure/result.hpp>

namespace SushiAI
{
    namespace Apps
    {
        /** @brief Holds the flag that names a profiling configuration. */
        inline constexpr std::string_view K_PROFILE_FLAG = "--profile";

        /**
         * @struct ProfileArguments
         * @brief Holds a command line with the `--profile <config>` pair taken out.
         */
        struct ProfileArguments
        {
            /** @brief The arguments left, in order, the program name not among them. */
            std::vector<std::string> positional;

            /** @brief The value of `--profile`; empty when the flag is absent. */
            std::optional<std::filesystem::path> config;
        };

        /**
         * @brief Takes `--profile <config>` out of a command line, wherever it stands.
         * @param arguments The command line without its program name; no entry is null.
         * @return InvalidDescription for a flag with no value, a value that is itself a flag,
         *         or a flag given twice.
         */
        [[nodiscard]] Result<ProfileArguments> read_profile_arguments(
            SushiRuntime::span<const char* const> arguments);

    } // namespace Apps
} // namespace SushiAI
