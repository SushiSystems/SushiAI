/**************************************************************************/
/* seam.cpp                                                               */
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
 * @file seam.cpp
 * @brief Implements the mapping from a sibling's exception to a Status and the guarded runs.
 * @author Mustafa Garip
 */

#include <SushiAI/core/seam.hpp>

#include <new>
#include <stdexcept>
#include <string>
#include <SushiBLAS/core/errors.hpp>
#include <SushiRuntime/core/error.hpp>
#include <SushiRuntime/execution/runtime_context.hpp>
#include <SushiAI/core/common.hpp>

namespace SushiAI
{
    namespace Detail
    {
        namespace
        {
            /** @brief Holds the message of a failure whose exception has no what(). */
            constexpr const char* K_UNKNOWN_EXCEPTION_MESSAGE = "an exception of an unknown type";

            /** @brief Builds a failure of @p code whose message is @p text. */
            [[nodiscard]] Status failure_of(StatusCode code, const char* text) noexcept
            {
                return Status::from_message(code, [&]()
                {
                    return std::string(text);
                });
            }

            /** @brief Returns the code of a failure whose exception type names no cause. */
            [[nodiscard]] constexpr StatusCode untyped_code(SeamPhase phase) noexcept
            {
                if (phase == SeamPhase::Allocate)
                    return StatusCode::OutOfMemory;

                return StatusCode::DeviceFailed;
            }

            /**
             * @brief Returns the Status of a SushiRuntime failure, by its error code.
             *
             * Stops the process for a code that reports a defect of the caller.
             */
            [[nodiscard]] Status status_of_runtime_failure(
                const SushiRuntime::Core::SushiException& failure, SeamPhase phase) noexcept
            {
                using SushiRuntime::Core::Errc;

                const Errc code = failure.error().code;
                switch (code)
                {
                    case Errc::allocation_failed:
                        return failure_of(StatusCode::OutOfMemory, failure.what());

                    case Errc::kernel_failed:
                    case Errc::device_lost:
                        return failure_of(StatusCode::DeviceFailed, failure.what());

                    case Errc::unsupported:
                        return failure_of(StatusCode::Unsupported, failure.what());

                    case Errc::invalid_graph:
                    case Errc::compile_failed:
                    case Errc::invalid_access:
                    case Errc::dangling_resource:
                        SA_FATAL("SushiRuntime reported {}: {}",
                                 SushiRuntime::Core::to_string(code), failure.what());

                    case Errc::ok:
                        break;
                }

                return failure_of(untyped_code(phase), failure.what());
            }

            /**
             * @brief Returns the Status of a std::runtime_error, by its dynamic type.
             *
             * Stops the process for a SushiAI::Error.
             */
            [[nodiscard]] Status status_of_runtime_error(const std::runtime_error& error,
                                                         SeamPhase phase) noexcept
            {
                if (dynamic_cast<const Error*>(&error) != nullptr)
                    SA_FATAL("{}", error.what());

                const auto* failure =
                    dynamic_cast<const SushiRuntime::Core::SushiException*>(&error);
                if (failure != nullptr)
                    return status_of_runtime_failure(*failure, phase);

                return failure_of(untyped_code(phase), error.what());
            }

        } // namespace

        Status status_of_current_exception(SeamPhase phase) noexcept
        {
            try
            {
                throw;
            }
            catch (const SushiBLAS::UnsupportedExecution& error)
            {
                return failure_of(StatusCode::Unsupported, error.what());
            }
            catch (const std::invalid_argument& error)
            {
                return failure_of(StatusCode::InvalidDescription, error.what());
            }
            catch (const std::runtime_error& error)
            {
                return status_of_runtime_error(error, phase);
            }
            catch (const std::bad_alloc& error)
            {
                return failure_of(StatusCode::OutOfMemory, error.what());
            }
            catch (const std::exception& error)
            {
                return failure_of(StatusCode::DeviceFailed, error.what());
            }
            catch (...)
            {
                return failure_of(StatusCode::DeviceFailed, K_UNKNOWN_EXCEPTION_MESSAGE);
            }
        }

        Status run_recorded(SushiBLAS::Engine& engine) noexcept
        {
            return guard(SeamPhase::Execute, [&]()
            {
                engine.execute();
                engine.get_context().wait_all();
                return Status();
            });
        }

        Status compile_graph(SushiRuntime::Graph::TaskGraph& graph,
                             SushiRuntime::Execution::CompiledGraph& plan) noexcept
        {
            return guard(SeamPhase::Execute, [&]()
            {
                plan = graph.compile();
                return Status();
            });
        }

        Status run_plan(SushiRuntime::Execution::RuntimeContext& context,
                        SushiRuntime::Execution::CompiledGraph& plan) noexcept
        {
            return guard(SeamPhase::Execute, [&]()
            {
                context.get_execution_engine().execute(plan);
                context.wait_all();
                return Status();
            });
        }

        Status discard_recorded(SushiRuntime::Graph::TaskGraph& graph) noexcept
        {
            return guard(SeamPhase::Execute, [&]()
            {
                // The returned vector is a temporary; its nodes die at the end of the statement.
                static_cast<void>(graph.take_nodes());
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
