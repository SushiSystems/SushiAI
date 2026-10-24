/**************************************************************************/
/* execution.hpp                                                          */
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
 * @file execution.hpp
 * @brief Declares the calls that create a runtime and a graph, record on a graph the caller owns
 *        and run it, each returning a Status.
 * @author Mustafa Garip
 */

#pragma once

#include <utility>
#include <SushiBLAS/engine/engine.hpp>
#include <SushiBLAS/engine/task_recorder.hpp>
#include <SushiRuntime/sushiruntime.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/seam/seam.hpp>

namespace SushiAI
{
    namespace Detail
    {
        /**
         * @brief Creates the runtime an engine is built on.
         * @return The runtime, or the Status of the Error SushiRuntime returned for it.
         */
        [[nodiscard]] Result<SushiRuntime::Runtime> create_runtime(
            const SushiRuntime::RuntimeConfig& config = {}) noexcept;

        /**
         * @brief Creates an empty graph on the runtime of @p engine.
         * @return The graph, or the Status of the failure that kept it from being created.
         */
        [[nodiscard]] Result<SushiRuntime::Graph> create_graph(SushiBLAS::Engine& engine) noexcept;

        /**
         * @brief Runs one step of @p graph and returns when it has finished.
         * @param report Refilled by the run; its previous contents are lost.
         */
        [[nodiscard]] Status run_step(SushiRuntime::Graph& graph,
                                      SushiRuntime::RunReport& report) noexcept;

        /**
         * @brief Runs every task recorded on @p graph once and returns when they have finished.
         * @return Success also when nothing was recorded, or the Status of the failure.
         */
        [[nodiscard]] Status run_once(SushiRuntime::Graph& graph) noexcept;

        /**
         * @brief Records on a graph of its own with @p record, runs that graph and waits.
         * @param record Called once with a SushiBLAS::TaskRecorder over the graph, the scratch
         *               pool and the mode of @p engine; it returns Status.
         * @return Success, the failure of @p record with the graph and every task recorded on it
         *         destroyed, or the failure of the run.
         */
        template<typename Record>
        [[nodiscard]] Status record_and_run(SushiBLAS::Engine& engine, Record&& record) noexcept;

        template<typename Record>
        Status record_and_run(SushiBLAS::Engine& engine, Record&& record) noexcept
        {
            Result<SushiRuntime::Graph> created = create_graph(engine);
            if (!created.has_value())
                return created.status();

            SushiRuntime::Graph& graph = created.value();
            const Status recorded = guard(SeamPhase::Record, [&]() -> Status
            {
                SushiBLAS::TaskRecorder recorder(graph.recorder(), engine.scratch_pool(),
                                                 engine.mode());
                return std::forward<Record>(record)(recorder);
            });
            if (!recorded.is_ok())
                return recorded;

            return run_once(graph);
        }

    } // namespace Detail
} // namespace SushiAI

/** @brief Short alias for the namespace of SushiRuntime, whose Runtime create_runtime() returns. */
namespace sr = SushiRuntime;
