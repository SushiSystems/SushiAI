/**************************************************************************/
/* worker_arguments.cpp                                                   */
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
 * @file worker_arguments.cpp
 * @brief Implements read_worker_arguments.
 * @author Mustafa Garip
 */

#include "worker_arguments.hpp"

#include <cstddef>

namespace SushiAI
{
    namespace Apps
    {
        namespace
        {
            /** @brief Holds the switch SushiRuntime's parser reads for help. */
            constexpr const char* K_HELP_FLAG = "--help";

            /** @brief Holds the switch SushiRuntime's parser reads for the version. */
            constexpr const char* K_VERSION_FLAG = "--version";
        } // namespace

        WorkerArguments read_worker_arguments(SushiRuntime::span<const char* const> arguments)
        {
            WorkerArguments read;
            if (arguments.size() == 0)
                return read;

            const std::string_view first = arguments[0];
            if (first == "-h" || first == K_HELP_FLAG)
            {
                read.call = WorkerCall::Help;
                read.forwarded.push_back(K_HELP_FLAG);
                return read;
            }

            if (first == "-V" || first == K_VERSION_FLAG)
            {
                read.call = WorkerCall::Version;
                read.forwarded.push_back(K_VERSION_FLAG);
                return read;
            }

            if (!first.empty() && first.front() == '-')
                return read;

            read.call = WorkerCall::Serve;
            read.spec_path = std::string(first);
            for (std::size_t index = 1; index < arguments.size(); ++index)
                read.forwarded.push_back(arguments[index]);
            return read;
        }

    } // namespace Apps
} // namespace SushiAI
