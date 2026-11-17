/**************************************************************************/
/* trailing_chain.cpp                                                     */
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
 * @file trailing_chain.cpp
 * @brief Implements TrailingChain, which orders the copies, the update and the blends.
 * @author Mustafa Garip
 * @see modules/core/eager/README.md
 */

#include <SushiAI/eager/trailing_chain.hpp>

#include <cstddef>
#include <utility>
#include <vector>

#include <SushiBLAS/blas/blas.hpp>

#include <SushiAI/failure/check.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/seam/seam.hpp>

namespace SushiAI
{
    namespace Eager
    {
        TrailingChain::TrailingChain(Graph::StepRecorder* inner, std::vector<CopySlot> copies,
                                     std::vector<Graph::BufferUpdateSlot> blends) noexcept
            : inner_(inner), copies_(std::move(copies)), blends_(std::move(blends))
        {

        }

        Result<std::unique_ptr<TrailingChain>> TrailingChain::make(
            const TrailingSpec& spec, const StorageBank& bank) noexcept
        {
            return ::SushiAI::Detail::guard_value(::SushiAI::Detail::SeamPhase::Allocate,
                                                  [&]() -> Result<std::unique_ptr<TrailingChain>>
            {
                std::vector<CopySlot> copies;
                copies.reserve(spec.copies.size());
                for (const TrailingCopy& copy : spec.copies)
                    copies.push_back(CopySlot{bank.tensor(copy.from), bank.tensor(copy.to)});

                std::vector<Graph::BufferUpdateSlot> blends;
                blends.reserve(spec.blends.size());
                for (const TrailingBlend& blend : spec.blends)
                {
                    blends.push_back(Graph::BufferUpdateSlot{bank.tensor(blend.target),
                                                             bank.tensor(blend.source),
                                                             blend.momentum,
                                                             blend.source_scale});
                }

                return std::unique_ptr<TrailingChain>(
                    new TrailingChain(spec.inner, std::move(copies), std::move(blends)));
            });
        }

        Result<std::size_t> TrailingChain::record_into_step(const Graph::StepRecording& recording)
        {
            SA_TRY(::SushiAI::Detail::guard(::SushiAI::Detail::SeamPhase::Record, [&]() -> Status
            {
                SushiBLAS::BLASOps blas(recording.tasks());
                for (CopySlot& slot : copies_)
                    blas.copy(slot.from, slot.to);
                return Status();
            }));

            std::size_t recorded = copies_.size();
            if (inner_ != nullptr)
            {
                SA_TRY_ASSIGN(const std::size_t updated, inner_->record_into_step(recording));
                recorded += updated;
            }

            SA_TRY_ASSIGN(const std::size_t blended, blends_.record_into_step(recording));
            return recorded + blended;
        }

        void TrailingChain::before_replay()
        {
            if (inner_ != nullptr)
                inner_->before_replay();
        }

        void TrailingChain::after_replay()
        {
            if (inner_ != nullptr)
                inner_->after_replay();
        }

    } // namespace Eager
} // namespace SushiAI
