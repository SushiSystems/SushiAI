/**************************************************************************/
/* profile_session.hpp                                                    */
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
 * @file profile_session.hpp
 * @brief Declares ProfileSession, which runs the profiling session of one program run, and end_profiled_run.
 * @author Mustafa Garip
 * @see applications/programs/README.md
 */

#pragma once

#include <filesystem>
#include <memory>
#include <utility>
#include <SushiRuntime/sushiruntime.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/failure/status.hpp>

namespace SushiAI
{
    namespace Apps
    {
        /**
         * @brief Owns the profiling session of one program run: its collector, the runtime's
         *        device source, the registry both came from and the binding of the timelines.
         * @see docs/design/PROFILING/README.md
         */
        class ProfileSession final
        {
            public:
                /**
                 * @brief Reads the SushiProfiler configuration at @p config_path and starts a
                 *        session over @p runtime.
                 * @pre @p runtime and the engine on it outlive the session.
                 * @return FileNotFound for an unreadable file, InvalidDescription for a
                 *         configuration SushiProfiler refuses or a session it cannot start,
                 *         Unsupported when the build has no profiler; each names the file.
                 */
                [[nodiscard]] static Result<std::unique_ptr<ProfileSession>> open(
                    const std::filesystem::path& config_path, const SushiRuntime::Runtime& runtime);

                ProfileSession(const ProfileSession&) = delete;
                ProfileSession& operator=(const ProfileSession&) = delete;
                ProfileSession(ProfileSession&&) = delete;
                ProfileSession& operator=(ProfileSession&&) = delete;

                /**
                 * @brief Unbinds the timelines, stops the session and hands the record to the
                 *        analyses and writers of its configuration, logging each problem.
                 * @return Success, also on a second call, or DeviceFailed when the session
                 *         stopped with problems.
                 */
                [[nodiscard]] Status close();

                /** @brief Calls close() if it was not called, logging its failure. */
                ~ProfileSession();

            private:
                /** @brief Holds what the session is made of; defined in the source. */
                struct Impl;

                /** @brief Takes the parts a successful open() made. */
                explicit ProfileSession(std::unique_ptr<Impl> impl) noexcept;

                /** @brief The parts, or null once close() ran. */
                std::unique_ptr<Impl> impl_;
        };

        /**
         * @brief Ends the profiling of a run that returned @p run, and returns what the program
         *        returns.
         * @param session The session of the run, or null for an unprofiled run.
         * @return @p run when it failed, so a failed close() is logged and not allowed to hide
         *         it; otherwise the result of close().
         */
        [[nodiscard]] Status end_profiled_run(const std::unique_ptr<ProfileSession>& session,
                                              Status run);

    } // namespace Apps
} // namespace SushiAI
