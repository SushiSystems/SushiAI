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
 * @brief Declares the calls that create a runtime, run what an engine recorded and run a graph
 *        step, each returning a Status.
 * @author Mustafa Garip
 */

#pragma once

#include <SushiBLAS/engine/engine.hpp>
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

        /** @brief Runs the tasks recorded on @p engine and returns when they have finished. */
        [[nodiscard]] Status run_recorded(SushiBLAS::Engine& engine) noexcept;

        /**
         * @brief Runs one step of @p graph and returns when it has finished.
         * @param report Refilled by the run; its previous contents are lost.
         */
        [[nodiscard]] Status run_step(SushiRuntime::Graph& graph,
                                      SushiRuntime::RunReport& report) noexcept;

        /**
         * @brief Destroys every task recorded on @p engine that has not run.
         * @pre No thread is recording on @p engine or running its tasks.
         */
        [[nodiscard]] Status discard_recorded(SushiBLAS::Engine& engine) noexcept;

        /**
         * @brief Discards what a failed recording left on @p engine.
         * @param failure The failure of the recording; never a success.
         * @return @p failure, prefixed with the failure of the discard when that fails too.
         */
        [[nodiscard]] Status abandon_recording(SushiBLAS::Engine& engine,
                                               const Status& failure) noexcept;

        /**
         * @brief Runs @p record, then runs the tasks it recorded on @p engine and waits.
         * @param record Called once with no argument; records on @p engine and returns Status.
         * @pre @p engine holds no recorded task that must outlive a failure of @p record.
         * @return Success, the failure of @p record with every task recorded on @p engine
         *         discarded, or the failure of the run.
         */
        template<typename Record>
        [[nodiscard]] Status record_and_run(SushiBLAS::Engine& engine, Record&& record) noexcept;

        template<typename Record>
        Status record_and_run(SushiBLAS::Engine& engine, Record&& record) noexcept
        {
            const Status recorded = guard(SeamPhase::Record, std::forward<Record>(record));
            if (!recorded.is_ok())
                return abandon_recording(engine, recorded);

            return run_recorded(engine);
        }

    } // namespace Detail
} // namespace SushiAI

/** @brief Short alias for the namespace of SushiRuntime, whose Runtime create_runtime() returns. */
namespace sr = SushiRuntime;
