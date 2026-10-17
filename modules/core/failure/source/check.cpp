/**************************************************************************/
/* check.cpp                                                              */
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
 * @file check.cpp
 * @brief Implements the fatal report handler and the stop that follows a report.
 * @author Mustafa Garip
 */

#include <SushiAI/failure/check.hpp>
#include "failure_log_category.hpp"
#include <SushiLog/log/log_calls.hpp>
#include <SushiLog/log/log_level.hpp>

#include <atomic>
#include <cstdio>
#include <cstdlib>

namespace SushiAI
{
    namespace
    {
        /**
         * @brief Logs @p message at FATAL in K_FAILURE_LOG_CATEGORY, at the call site @p file and
         *        @p line of the failed check, past every level gate.
         */
        void write_fatal_report(const char* file, int line, std::string_view message) noexcept
        {
            SushiLog::log_at(SushiLog::LogLevel::Fatal, K_FAILURE_LOG_CATEGORY, file, line,
                             message);
        }

        /** @brief Returns the slot that holds the installed handler; it is never null. */
        std::atomic<FatalHandler>& installed_handler() noexcept
        {
            static std::atomic<FatalHandler> handler{&write_fatal_report};
            return handler;
        }

    } // namespace

    FatalHandler set_fatal_handler(FatalHandler handler) noexcept
    {
        const FatalHandler next = handler != nullptr ? handler : &write_fatal_report;
        return installed_handler().exchange(next);
    }

    namespace Detail
    {
        void report_fatal(const char* file, int line, std::string_view message) noexcept
        {
            const FatalHandler handler = installed_handler().load();
            handler(file, line, message);
        }

        void fail_fast(const char* file, int line, std::string_view message) noexcept
        {
            report_fatal(file, line, message);

            // A null stream flushes every open output stream, stdout included.
            std::fflush(nullptr);
            std::abort();
        }

    } // namespace Detail
} // namespace SushiAI
