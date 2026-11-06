/**************************************************************************/
/* in_memory_fatal_sink.hpp                                               */
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
 * @file in_memory_fatal_sink.hpp
 * @brief Declares InMemoryFatalSink, a receiver of fatal reports that keeps them in memory.
 * @author Mustafa Garip
 * @see modules/core/failure/README.md
 */

#pragma once

#include <cstddef>
#include <string>
#include <string_view>
#include <vector>
#include <SushiAI/failure/check.hpp>

namespace SushiAI
{
    /**
     * @class InMemoryFatalSink
     * @brief Installs a FatalHandler that appends each report to a list, and puts the previous
     *        handler back when it dies; touches no device, file, thread or clock.
     *
     * The newest live sink receives every report; sinks die in the reverse order they were
     * made. A test reads reports through Detail::report_fatal(), never fail_fast(). No lock.
     */
    class InMemoryFatalSink final
    {
        public:
            /** @brief Holds one report as the handler received it. */
            struct Report
            {
                /** @brief The file of the failed check. */
                std::string file;

                /** @brief The line of the failed check. */
                int line = 0;

                /** @brief The message of the failed check. */
                std::string message;
            };

            /** @brief Installs the recording handler. */
            InMemoryFatalSink() noexcept
            {
                outer_ = active_;
                active_ = this;
                previous_ = set_fatal_handler(&InMemoryFatalSink::record);
            }

            InMemoryFatalSink(const InMemoryFatalSink&) = delete;
            InMemoryFatalSink& operator=(const InMemoryFatalSink&) = delete;
            InMemoryFatalSink(InMemoryFatalSink&&) = delete;
            InMemoryFatalSink& operator=(InMemoryFatalSink&&) = delete;

            /** @brief Puts the handler that was installed before back. */
            ~InMemoryFatalSink()
            {
                static_cast<void>(set_fatal_handler(previous_));
                active_ = outer_;
            }

            /** @brief Returns every report received, first received first. */
            [[nodiscard]] const std::vector<Report>& reports() const noexcept
            {
                return reports_;
            }

            /**
             * @brief Returns how many reports were received but could not be kept because the
             *        host heap refused the copy.
             */
            [[nodiscard]] std::size_t dropped() const noexcept
            {
                return dropped_;
            }

        private:
            /** @brief Appends one report to the active sink. */
            static void record(const char* file, int line, std::string_view message) noexcept
            {
                try
                {
                    active_->reports_.push_back(Report{file, line, std::string(message)});
                }
                catch (...)
                {
                    ++active_->dropped_;
                }
            }

            /// @brief Points at the sink that is installed, or is null.
            static inline InMemoryFatalSink* active_ = nullptr;

            /// @brief Points at the sink that was newest when this one was made, or is null.
            InMemoryFatalSink* outer_ = nullptr;

            /// @brief Holds the handler that was installed before this sink.
            FatalHandler previous_ = nullptr;

            /// @brief Holds the reports received, in order.
            std::vector<Report> reports_;

            /// @brief Counts the reports that could not be kept.
            std::size_t dropped_ = 0;
    };
} // namespace SushiAI
