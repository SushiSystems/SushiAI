/**************************************************************************/
/* seam.hpp                                                               */
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
 * @file seam.hpp
 * @brief Declares the guards that turn an exception of SushiBLAS or an Error of SushiRuntime
 *        into a Status.
 * @author Mustafa Garip
 */

#pragma once

#include <cstdint>
#include <exception>
#include <type_traits>
#include <utility>
#include <SushiBLAS/engine/engine.hpp>
#include <SushiRuntime/sushiruntime.hpp>
#include <SushiAI/core/result.hpp>
#include <SushiAI/core/status.hpp>

namespace SushiAI
{
    namespace Detail
    {
        /**
         * @brief Names what a guarded call was doing, which decides the code of an untyped
         *        failure.
         */
        enum class SeamPhase : std::uint8_t
        {
            Record,   /**< Recording an operation. */
            Allocate, /**< Allocating storage. */
            Execute,  /**< Compiling or running a graph. */
        };

        /**
         * @brief Returns the Status of an Error SushiRuntime returned, by its code.
         *
         * Stops the process for a code that reports a defect of the caller.
         * @param phase Decides the code of an Error whose own code is Ok.
         * @return A failure whose message is the Error's; never a success.
         */
        [[nodiscard]] Status status_of_error(const SushiRuntime::Error& error,
                                             SeamPhase phase) noexcept;

        /**
         * @brief Returns the Status of the exception being handled.
         * @param phase What the guarded call was doing when it threw.
         * @pre Called from a handler, while an exception is being handled.
         * @return A failure whose message is the exception's what(); never a success.
         */
        [[nodiscard]] Status status_of_current_exception(SeamPhase phase) noexcept;

        /**
         * @brief Runs @p call and returns its Status, or the Status of what it threw.
         * @param call Called once with no argument; returns Status.
         */
        template<typename Call>
        [[nodiscard]] Status guard(SeamPhase phase, Call&& call) noexcept;

        /**
         * @brief Runs @p call and returns its value, or the Status of what it threw.
         * @param call Called once with no argument; returns a value, not a Status.
         */
        template<typename Call>
        [[nodiscard]] auto guard_value(SeamPhase phase, Call&& call) noexcept
            -> Result<std::decay_t<decltype(std::forward<Call>(call)())>>;

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

        template<typename Call>
        Status guard(SeamPhase phase, Call&& call) noexcept
        {
            try
            {
                return std::forward<Call>(call)();
            }
            catch (...)
            {
                return status_of_current_exception(phase);
            }
        }

        template<typename Call>
        auto guard_value(SeamPhase phase, Call&& call) noexcept
            -> Result<std::decay_t<decltype(std::forward<Call>(call)())>>
        {
            try
            {
                return std::forward<Call>(call)();
            }
            catch (...)
            {
                return status_of_current_exception(phase);
            }
        }

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
