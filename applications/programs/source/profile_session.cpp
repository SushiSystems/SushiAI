/**************************************************************************/
/* profile_session.cpp                                                    */
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
 * @file profile_session.cpp
 * @brief Implements ProfileSession over SushiProfiler and the runtime's profiling parts.
 * @author Mustafa Garip
 * @see applications/programs/README.md
 */

#include "profile_session.hpp"

#include <fstream>
#include <iostream>
#include <iterator>
#include <optional>
#include <string>
#include <SushiProfiler/config/profiler_config.hpp>
#include <SushiProfiler/registry/registry.hpp>
#include <SushiProfiler/session/session.hpp>
#include <SushiRuntime/profiling/register_profiler_parts.hpp>
#include <SushiLog/text/text_format.hpp>
#include <SushiAI/log/log.hpp>
#include <SushiAI/profile/timeline_binding.hpp>
#include <SushiAI/profile/timelines.hpp>
#include <SushiAI/seam/seam.hpp>

#include "programs_log_category.hpp"

namespace SushiAI
{
    namespace Apps
    {
        /**
         * @brief Holds the parts of a session in the order that destroys them safely.
         *
         * The binding goes first, then the session, which reads the registry and the runtime
         * parts, then the parts the registry points into, then the registry.
         */
        struct ProfileSession::Impl
        {
            /** @brief Holds the factories and contributors the session starts from. */
            SushiProfiler::Registry registry;

            /** @brief Holds the runtime's device source and environment contributor. */
            std::optional<SushiRuntime::RuntimeProfilerParts> parts;

            /** @brief Holds the running session. */
            std::unique_ptr<SushiProfiler::Session> session;

            /** @brief Points the program's timelines at the session; empty after close(). */
            std::optional<Profile::TimelineBinding> binding;
        };

        namespace
        {
            /** @brief Returns the text of the file at @p path, or FileNotFound naming it. */
            [[nodiscard]] Result<std::string> read_text(const std::filesystem::path& path)
            {
                std::ifstream file(path, std::ios::binary);
                SA_FAIL_IF(!file, StatusCode::FileNotFound,
                           "The profile configuration '{}' cannot be opened", path.string());

                return ::SushiAI::Detail::guard_value(::SushiAI::Detail::SeamPhase::Allocate,
                                                      [&file]()
                {
                    return std::string(std::istreambuf_iterator<char>(file),
                                       std::istreambuf_iterator<char>());
                });
            }

            /** @brief Opens the next period of every timeline, which ends the open ones. */
            void end_open_periods() noexcept
            {
                Profile::mark_period(Profile::Timeline::TrainStep);
                Profile::mark_period(Profile::Timeline::InferenceBatch);
            }

            /** @brief Logs each of @p problems as a warning of the programs. */
            void log_problems(const std::vector<std::string>& problems)
            {
                for (const std::string& problem : problems)
                    SA_LOG_WARN(K_PROGRAMS_LOG_CATEGORY, "profiling: {}", problem);
            }
        } // namespace

        Result<std::unique_ptr<ProfileSession>> ProfileSession::open(
            const std::filesystem::path& config_path, const SushiRuntime::Runtime& runtime)
        {
            SA_TRY_ASSIGN(const std::string text, read_text(config_path));
            const SushiProfiler::ConfigReadResult read = SushiProfiler::read_profiler_config(text);
            SA_FAIL_IF(!read.ok(), StatusCode::InvalidDescription,
                       "The profile configuration '{}' is refused: {}", config_path.string(),
                       read.error);

            SA_TRY_ASSIGN(std::unique_ptr<Impl> impl,
                          ::SushiAI::Detail::guard_value(::SushiAI::Detail::SeamPhase::Allocate,
                                                         [&runtime]()
            {
                std::unique_ptr<Impl> made = std::make_unique<Impl>();
                SushiProfiler::register_builtin_writers(made->registry, std::cout);
                made->parts.emplace(made->registry, runtime);
                return made;
            }));
            SA_FAIL_IF(!impl->parts->is_registered(), StatusCode::InvalidDescription,
                       "The runtime's profiling source was refused for '{}'",
                       config_path.string());

            SushiProfiler::SessionStart started =
                SushiProfiler::Session::start(*read.config, impl->registry);
            SA_FAIL_IF(!started.ok(), StatusCode::InvalidDescription,
                       "The profile configuration '{}' did not start a session: {}",
                       config_path.string(), started.error);
            impl->session = std::move(started.session);

            SA_TRY(::SushiAI::Detail::guard(::SushiAI::Detail::SeamPhase::Allocate,
                                            [&impl]() -> Status
            {
                impl->binding.emplace(*impl->session);
                return Status();
            }));

            return ::SushiAI::Detail::guard_value(::SushiAI::Detail::SeamPhase::Allocate,
                                                  [&impl]()
            {
                return std::unique_ptr<ProfileSession>(new ProfileSession(std::move(impl)));
            });
        }

        ProfileSession::ProfileSession(std::unique_ptr<Impl> impl) noexcept
            : impl_(std::move(impl))
        {

        }

        Status ProfileSession::close()
        {
            if (impl_ == nullptr)
                return Status();

            const std::unique_ptr<Impl> impl = std::move(impl_);
            return ::SushiAI::Detail::guard(::SushiAI::Detail::SeamPhase::Execute,
                                            [&impl]() -> Status
            {
                // A period without its end mark is not kept, so the last step of the run gets one.
                end_open_periods();
                impl->binding.reset();

                const SushiProfiler::SessionStop stopped = impl->session->stop();
                log_problems(stopped.problems);
                const SushiProfiler::SessionReport report = impl->session->report(stopped.record);
                log_problems(report.problems);

                SA_FAIL_IF(!stopped.problems.empty(), StatusCode::DeviceFailed,
                           "The profiling session stopped with {} problems; the first: {}",
                           stopped.problems.size(), stopped.problems.front());
                return Status();
            });
        }

        Status end_profiled_run(const std::unique_ptr<ProfileSession>& session, Status run)
        {
            if (session == nullptr)
                return run;

            const Status closed = session->close();
            if (run.is_ok())
                return closed;

            if (!closed.is_ok())
                SA_LOG_WARN(K_PROGRAMS_LOG_CATEGORY, "the profiling record of the failed run: {}",
                            closed.message());
            return run;
        }

        ProfileSession::~ProfileSession()
        {
            const Status closed = close();
            if (!closed.is_ok())
                SA_LOG_ERROR(K_PROGRAMS_LOG_CATEGORY, "profiling: {}", closed.message());
        }

    } // namespace Apps
} // namespace SushiAI
