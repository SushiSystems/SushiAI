/**************************************************************************/
/* execution.cpp                                                          */
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
 * @file execution.cpp
 * @brief Implements the calls of execution.hpp, each turning what SushiRuntime or SushiBLAS
 *        reports into a Status through the seam.
 * @author Mustafa Garip
 */

#include <SushiAI/execution/execution.hpp>

#include <utility>

namespace SushiAI
{
    namespace Detail
    {
        Result<SushiRuntime::Runtime> create_runtime(
            const SushiRuntime::RuntimeConfig& config) noexcept
        {
            try
            {
                SushiRuntime::Result<SushiRuntime::Runtime> created =
                    SushiRuntime::Runtime::create(config);
                if (!created)
                    return status_of_error(created.error(), SeamPhase::Allocate);

                return std::move(*created);
            }
            catch (...)
            {
                return status_of_current_exception(SeamPhase::Allocate);
            }
        }

        Status run_recorded(SushiBLAS::Engine& engine) noexcept
        {
            return guard(SeamPhase::Execute, [&]()
            {
                const SushiRuntime::Result<void> ran = engine.execute();
                if (!ran)
                    return status_of_error(ran.error(), SeamPhase::Execute);

                return Status();
            });
        }

        Status run_step(SushiRuntime::Graph& graph, SushiRuntime::RunReport& report) noexcept
        {
            return guard(SeamPhase::Execute, [&]()
            {
                const SushiRuntime::Result<SushiRuntime::RunReport*> ran = graph.run(report);
                if (!ran)
                    return status_of_error(ran.error(), SeamPhase::Execute);

                return Status();
            });
        }

        Status discard_recorded(SushiBLAS::Engine& engine) noexcept
        {
            return guard(SeamPhase::Execute, [&]()
            {
                engine.discard_recorded();
                return Status();
            });
        }

        Status abandon_recording(SushiBLAS::Engine& engine, const Status& failure) noexcept
        {
            const Status dropped = discard_recorded(engine);
            if (dropped.is_ok())
                return failure;

            return SA_ANNOTATE(failure, "the tasks recorded before this failure stay on the "
                                        "engine, because discarding them failed ({})",
                               dropped.message());
        }
    } // namespace Detail
} // namespace SushiAI
