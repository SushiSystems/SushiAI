/**************************************************************************/
/* trailing_ledger.cpp                                                    */
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
 * @file trailing_ledger.cpp
 * @brief Implements TrailingLedger.
 * @author Mustafa Garip
 * @see modules/core/eager/README.md
 */

#include <SushiAI/eager/trailing_ledger.hpp>

#include <cstddef>

#include <SushiAI/failure/status.hpp>

namespace SushiAI
{
    namespace Eager
    {
        Result<TrailingPlan> TrailingLedger::plan(const StepWork& work) const
        {
            TrailingPlan decided;
            decided.retire =
                key_ != K_NO_REGION && (work.trailing != recorder_ || work.trailing_key != key_);
            decided.record =
                work.trailing != nullptr && (decided.retire || work.trailing != recorder_);
            if (!decided.record)
                return decided;

            SA_FAIL_IF(work.trailing_key == K_NO_REGION, StatusCode::InvalidDescription,
                       "The trailing work of a step needs a key above every region; got none");

            for (const RegionWork& region : work.regions)
            {
                SA_FAIL_IF(region.order.key >= work.trailing_key, StatusCode::InvalidDescription,
                           "The trailing work of a step needs a key above every region; key {} "
                           "does not lie above the region under key {}",
                           work.trailing_key, region.order.key);
            }

            return decided;
        }

        void TrailingLedger::begin_recording(RegionKey key) noexcept
        {
            key_ = key;
            ++records_;
        }

        void TrailingLedger::finish_recording(Graph::StepRecorder& recorder,
                                              std::size_t operations) noexcept
        {
            recorder_ = &recorder;
            operations_ = operations;
        }

        void TrailingLedger::clear() noexcept
        {
            recorder_ = nullptr;
            key_ = K_NO_REGION;
            operations_ = 0;
        }

        void TrailingLedger::before_replay()
        {
            if (recorder_ != nullptr)
                recorder_->before_replay();
        }

        void TrailingLedger::after_replay()
        {
            if (recorder_ != nullptr)
                recorder_->after_replay();
        }

    } // namespace Eager
} // namespace SushiAI
