/**************************************************************************/
/* in_memory_failure_block_counter.hpp                                    */
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
 * @file in_memory_failure_block_counter.hpp
 * @brief Declares InMemoryFailureBlockCounter, a block counter that sees only its own scope.
 * @author Mustafa Garip
 * @see modules/core/failure/README.md
 */

#pragma once

#include <atomic>
#include <cstddef>
#include <SushiAI/failure/check.hpp>
#include <SushiAI/failure/failure_block_counter.hpp>

namespace SushiAI
{
    /**
     * @class InMemoryFailureBlockCounter
     * @brief Counts the failure blocks allocated while it lives, and puts the previous counter
     *        back when it dies; touches no device, file, thread or clock.
     *
     * Every status made under it must be gone before it is; a block left alive stops the
     * process in the destructor, since it would point at a dead counter.
     */
    class InMemoryFailureBlockCounter final : public FailureBlockCounter
    {
        public:
            /** @brief Installs the counter. */
            InMemoryFailureBlockCounter() noexcept : previous_(&set_failure_block_counter(this))
            {

            }

            InMemoryFailureBlockCounter(const InMemoryFailureBlockCounter&) = delete;
            InMemoryFailureBlockCounter& operator=(const InMemoryFailureBlockCounter&) = delete;
            InMemoryFailureBlockCounter(InMemoryFailureBlockCounter&&) = delete;
            InMemoryFailureBlockCounter& operator=(InMemoryFailureBlockCounter&&) = delete;

            /** @brief Puts the counter that was installed before back. */
            ~InMemoryFailureBlockCounter() override
            {
                static_cast<void>(set_failure_block_counter(previous_));
                SA_FATAL_IF(live() != 0, "{} failure blocks outlive the counter that counted them",
                            live());
            }

            /** @brief Counts one block made. */
            void block_made() noexcept override
            {
                made_.fetch_add(1, std::memory_order_relaxed);
            }

            /** @brief Counts one block released. */
            void block_released() noexcept override
            {
                released_.fetch_add(1, std::memory_order_relaxed);
            }

            /** @brief Returns the blocks made under this counter and not yet released. */
            [[nodiscard]] std::size_t live() const noexcept override
            {
                return made_.load(std::memory_order_relaxed)
                     - released_.load(std::memory_order_relaxed);
            }

        private:
            /// @brief Holds the counter that was installed before this one.
            FailureBlockCounter* previous_ = nullptr;

            /// @brief Counts the blocks made.
            std::atomic<std::size_t> made_{0};

            /// @brief Counts the blocks released.
            std::atomic<std::size_t> released_{0};
    };
} // namespace SushiAI
