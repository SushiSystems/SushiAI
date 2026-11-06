/**************************************************************************/
/* in_memory_epoch_aware.hpp                                              */
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
 * @file in_memory_epoch_aware.hpp
 * @brief Declares InMemoryEpochAware, the EpochAware that keeps the epoch it was told in memory.
 * @author Mustafa Garip
 * @see modules/core/data/README.md
 */

#pragma once

#include <cstdint>
#include <SushiAI/failure/check.hpp>
#include <SushiAI/data/epoch.hpp>

namespace SushiAI
{
    namespace Data
    {
        /**
         * @class InMemoryEpochAware
         * @brief Holds the epoch last begun and the number of times one began; touches no device,
         *        file, thread or clock.
         *
         * Combine it with a Dataset to give that dataset an epoch to read, or hand it to a test
         * that checks a caller tells every epoch.
         */
        class InMemoryEpochAware : public EpochAware
        {
            public:
                /**
                 * @brief Holds @p epoch as the current one and counts the call.
                 * @param epoch Counted from 0; a negative one stops the process.
                 */
                void begin_epoch(std::int64_t epoch) const noexcept override
                {
                    SA_FATAL_IF(epoch < 0, "InMemoryEpochAware::begin_epoch: epoch {} is negative",
                                epoch);

                    epoch_ = epoch;
                    ++begin_count_;
                }

                /** @brief Returns the epoch last begun, or -1 before the first call. */
                [[nodiscard]] std::int64_t epoch() const noexcept
                {
                    return epoch_;
                }

                /** @brief Returns how many times begin_epoch() returned. */
                [[nodiscard]] std::uint64_t begin_count() const noexcept
                {
                    return begin_count_;
                }

            private:
                /// @brief Holds the epoch last begun; -1 before the first.
                mutable std::int64_t epoch_ = -1;

                /// @brief Counts the calls of begin_epoch() that returned.
                mutable std::uint64_t begin_count_ = 0;
        };

    } // namespace Data
} // namespace SushiAI
