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
/* Licensed under the Apache License, Version 2.0 (the "License"); you    */
/* may not use this file except in compliance with the License. You may   */
/* obtain a copy of the License at                                        */
/*                                                                        */
/*     https://www.apache.org/licenses/LICENSE-2.0                        */
/*                                                                        */
/* Unless required by applicable law or agreed to in writing, software    */
/* distributed under the License is distributed on an "AS IS" BASIS,      */
/* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or        */
/* implied. See the License for the specific language governing           */
/* permissions and limitations under the License.                         */
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
