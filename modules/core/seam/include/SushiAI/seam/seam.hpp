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
#include <SushiRuntime/core/error.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/failure/status.hpp>

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
         * Stops the process for DanglingResource, which reports a tensor freed before the
         * step that names it.
         * @param phase Decides the code of CompileFailed, which names no cause.
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

    } // namespace Detail
} // namespace SushiAI
