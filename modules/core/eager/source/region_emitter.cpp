/**************************************************************************/
/* region_emitter.cpp                                                     */
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
 * @file region_emitter.cpp
 * @brief Implements RegionEmitter, which lowers a region's program into a recorder.
 * @author Mustafa Garip
 * @see modules/core/eager/README.md
 */

#include <SushiAI/eager/region_emitter.hpp>

#include <string>

#include <SushiBLAS/engine/task_recorder.hpp>

#include <SushiAI/seam/seam.hpp>
#include <SushiAI/graph/lowering.hpp>

namespace SushiAI
{
    namespace Eager
    {
        RegionEmitter::RegionEmitter(RuntimeBackend& backend, const RegionProgram& program,
                                     Graph::TensorPool& pool,
                                     const Graph::FusionSelector* selector,
                                     RegionID region) noexcept
            : backend_(&backend), program_(&program), pool_(&pool), selector_(selector),
              region_(region), late_counts_(backend.runtime())
        {

        }

        Status RegionEmitter::emit(SushiRuntime::Recorder& recorder) noexcept
        {
            ++emits_;
            late_counts_.release();

            SushiBLAS::TaskRecorder tasks(recorder, backend_->engine().scratch_pool(),
                                          late_counts_, backend_->engine().mode());
            const Graph::DeviceFusionSelector device =
                Graph::DeviceFusionSelector::for_backend(*backend_);
            Graph::Lowering lowering(tasks, backend_->engine().rng(), *pool_,
                                     selector_ != nullptr ? *selector_ : device);
            if (region_ != K_NO_REGION_ID)
                lowering.set_origin_prefix("r" + std::to_string(region_) + "/");

            // record() throws for a graph that mixes layouts; the guard turns that into a status.
            Status recorded = ::SushiAI::Detail::guard(::SushiAI::Detail::SeamPhase::Record, [&]()
            {
                return lowering.record(program_->graph);
            });
            operations_ = lowering.emitted_operations();

            const Status copied = ::SushiAI::Detail::guard(
                ::SushiAI::Detail::SeamPhase::Allocate, [&]()
            {
                const SushiRuntime::span<const Graph::FusionSelection> chosen =
                    lowering.selections();
                selections_.assign(chosen.begin(), chosen.end());
                preludes_ = lowering.take_preludes();
                return Status();
            });
            if (recorded.is_ok())
                recorded = copied;
            SA_TRY(recorded);

            if (const SushiRuntime::Error* refused = recorder.refusal())
                return ::SushiAI::Detail::status_of_error(*refused,
                                                          ::SushiAI::Detail::SeamPhase::Record);

            return Status();
        }

    } // namespace Eager
} // namespace SushiAI
