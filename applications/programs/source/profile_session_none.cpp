/**************************************************************************/
/* profile_session_none.cpp                                               */
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
 * @file profile_session_none.cpp
 * @brief Implements ProfileSession for a build without a profiler: it refuses to open.
 * @author Mustafa Garip
 * @see applications/programs/README.md
 */

#include "profile_session.hpp"

#include <SushiLog/text/text_format.hpp>

namespace SushiAI
{
    namespace Apps
    {
        /** @brief Holds nothing: a build without a profiler never makes a session. */
        struct ProfileSession::Impl
        {

        };

        Result<std::unique_ptr<ProfileSession>> ProfileSession::open(
            const std::filesystem::path& config_path, const SushiRuntime::Runtime&)
        {
            return Status::from_message(StatusCode::Unsupported, [&config_path]()
            {
                return SushiLog::format_text(
                    "This build has no profiler, so it cannot run '{}'; configure with "
                    "SA_ENABLE_PROFILER=ON",
                    config_path.string());
            });
        }

        ProfileSession::ProfileSession(std::unique_ptr<Impl> impl) noexcept
            : impl_(std::move(impl))
        {

        }

        Status ProfileSession::close()
        {
            return Status();
        }

        Status end_profiled_run(const std::unique_ptr<ProfileSession>&, Status run)
        {
            return run;
        }

        ProfileSession::~ProfileSession() = default;

    } // namespace Apps
} // namespace SushiAI
