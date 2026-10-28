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
 * @brief Implements the mapping from a sibling's exception or Error to a Status and the
 *        guarded runs.
 * @author Mustafa Garip
 */

#include <SushiAI/seam/seam.hpp>

#include <new>
#include <stdexcept>
#include <string>
#include <utility>
#include <SushiBLAS/core/errors.hpp>

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

        } // namespace

        Status status_of_error(const SushiRuntime::Error& error, SeamPhase phase) noexcept
        {
            using SushiRuntime::ErrorCode;

            const char* text = error.message.c_str();
            switch (error.code)
            {
                case ErrorCode::AllocationFailed:
                    return failure_of(StatusCode::OutOfMemory, text);

                case ErrorCode::KernelFailed:
                case ErrorCode::DeviceLost:
                    return failure_of(StatusCode::DeviceFailed, text);

                case ErrorCode::Unsupported:
                case ErrorCode::InvalidAccess:
                case ErrorCode::InvalidState:
                    return failure_of(StatusCode::Unsupported, text);

                case ErrorCode::InvalidGraph:
                case ErrorCode::InvalidArgument:
                    return failure_of(StatusCode::InvalidDescription, text);

                case ErrorCode::ProtocolMismatch:
                    return failure_of(StatusCode::Incompatible, text);

                case ErrorCode::CompileFailed:
                    return failure_of(untyped_code(phase), text);

                case ErrorCode::DanglingResource:
                    SA_FATAL("SushiRuntime reported {}: {}",
                             SushiRuntime::to_string(error.code), text);
            }

            return failure_of(untyped_code(phase), text);
        }

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
                return failure_of(untyped_code(phase), error.what());
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

    } // namespace Detail
} // namespace SushiAI
