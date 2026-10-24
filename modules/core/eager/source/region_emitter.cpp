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
 * @file region_emitter.cpp
 * @brief Implements RegionEmitter, which lowers a region's program into a recorder.
 * @author Mustafa Garip
 * @see modules/core/eager/README.md
 */

#include <SushiAI/eager/region_emitter.hpp>

#include <SushiBLAS/engine/task_recorder.hpp>

#include <SushiAI/seam/seam.hpp>
#include <SushiAI/graph/lowering.hpp>

namespace SushiAI
{
    namespace Eager
    {
        RegionEmitter::RegionEmitter(SushiBLAS::Engine& engine, const RegionProgram& program,
                                     Graph::TensorPool& pool,
                                     const Graph::FusionSelector* selector) noexcept
            : engine_(&engine), program_(&program), pool_(&pool), selector_(selector),
              late_counts_(engine.runtime())
        {

        }

        Status RegionEmitter::emit(SushiRuntime::Recorder& recorder)
        {
            ++emits_;
            late_counts_.release();

            SushiBLAS::TaskRecorder tasks(recorder, engine_->scratch_pool(), late_counts_,
                                          engine_->mode());
            const Graph::DeviceFusionSelector device =
                Graph::DeviceFusionSelector::for_engine(*engine_);
            Graph::Lowering lowering(tasks, engine_->rng(), *pool_,
                                     selector_ != nullptr ? *selector_ : device);

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
