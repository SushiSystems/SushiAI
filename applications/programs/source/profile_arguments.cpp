/**************************************************************************/
/* profile_arguments.cpp                                                  */
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
 * @file profile_arguments.cpp
 * @brief Implements read_profile_arguments.
 * @author Mustafa Garip
 * @see applications/programs/README.md
 */

#include "profile_arguments.hpp"

#include <cstddef>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/seam/seam.hpp>

namespace SushiAI
{
    namespace Apps
    {
        Result<ProfileArguments> read_profile_arguments(
            SushiRuntime::span<const char* const> arguments)
        {
            ProfileArguments read;
            SA_TRY(::SushiAI::Detail::guard(::SushiAI::Detail::SeamPhase::Allocate,
                                            [&arguments, &read]() -> Status
            {
                for (std::size_t index = 0; index < arguments.size(); ++index)
                {
                    const std::string_view argument = arguments[index];
                    if (argument != K_PROFILE_FLAG)
                    {
                        read.positional.emplace_back(argument);
                        continue;
                    }

                    SA_FAIL_IF(read.config.has_value(), StatusCode::InvalidDescription,
                               "'{}' is given twice; a run has one profile configuration",
                               K_PROFILE_FLAG);
                    SA_FAIL_IF(index + 1 == arguments.size()
                                   || std::string_view(arguments[index + 1]).rfind("-", 0) == 0,
                               StatusCode::InvalidDescription,
                               "'{}' needs the path of a profile configuration", K_PROFILE_FLAG);
                    ++index;
                    read.config = std::filesystem::path(arguments[index]);
                }

                return Status();
            }));

            return read;
        }

    } // namespace Apps
} // namespace SushiAI
