/**************************************************************************/
/* failure_block_counter.hpp                                              */
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
 * @file failure_block_counter.hpp
 * @brief Declares FailureBlockCounter, the port that counts the failure blocks of Status.
 * @author Mustafa Garip
 * @see modules/core/failure/README.md
 */

#pragma once

#include <cstddef>

namespace SushiAI
{
    /**
     * @class FailureBlockCounter
     * @brief Counts the heap blocks that hold the failures of Status objects.
     *
     * The process counts them in an atomic by default. A test installs an in-memory counter to
     * read how many blocks its own statuses hold.
     */
    class FailureBlockCounter
    {
        public:
            virtual ~FailureBlockCounter() = default;

            /** @brief Records that a heap block was allocated for a failure. */
            virtual void block_made() noexcept = 0;

            /** @brief Records that a heap block was freed. */
            virtual void block_released() noexcept = 0;

            /** @brief Returns how many blocks this counter saw made and not yet released. */
            [[nodiscard]] virtual std::size_t live() const noexcept = 0;
    };

    /**
     * @brief Installs @p counter as the one that counts the blocks allocated from now on.
     * @param counter The counter, which must outlive every block it counted; a null pointer
     *                installs the process counter that live_failure_count() reads.
     * @return The counter that was installed before the call; never null.
     */
    FailureBlockCounter& set_failure_block_counter(FailureBlockCounter* counter) noexcept;
} // namespace SushiAI
