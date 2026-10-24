/**************************************************************************/
/* worker_arguments.hpp                                                   */
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
 * @file worker_arguments.hpp
 * @brief Declares read_worker_arguments, which sorts the command line of sushiai_worker.
 * @author Mustafa Garip
 */

#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>
#include <SushiRuntime/core/span.hpp>

namespace SushiAI
{
    namespace Apps
    {
        /** @brief Holds the usage line that names the order of the worker's arguments. */
        inline constexpr std::string_view K_WORKER_USAGE_LINE =
            "usage: sushiai_worker <train.json> --master host:port --name <worker> [flags]";

        /** @brief Names what a worker's command line asks for. */
        enum class WorkerCall : std::uint8_t
        {
            Serve,   /**< A training file comes first; serve its replica. */
            Help,    /**< `-h` or `--help` comes first; print the usage and exit 0. */
            Version, /**< `-V` or `--version` comes first; print the version and exit 0. */
            Refused, /**< No argument, or another flag first; print the usage line. */
        };

        /**
         * @struct WorkerArguments
         * @brief Holds what read_worker_arguments() found on a worker's command line.
         */
        struct WorkerArguments
        {
            /** @brief What the command line asks for. */
            WorkerCall call = WorkerCall::Refused;

            /** @brief The training file; empty unless call is Serve. */
            std::string spec_path;

            /** @brief The arguments SushiRuntime's worker parser reads, in order. */
            std::vector<const char*> forwarded;
        };

        /**
         * @brief Sorts a worker's command line, its program name left out.
         * @param arguments No entry is null; each outlives the result's forwarded view.
         */
        [[nodiscard]] WorkerArguments read_worker_arguments(
            SushiRuntime::span<const char* const> arguments);

    } // namespace Apps
} // namespace SushiAI
