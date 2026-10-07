/**************************************************************************/
/* throwing_bridge.hpp                                                    */
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
 * @file throwing_bridge.hpp
 * @brief Declares bridge_throwing, the one place a thrown Error becomes a Status.
 * @author Mustafa Garip
 */

#pragma once

#include <exception>
#include <new>
#include <string>
#include <utility>
#include <SushiAI/core/check.hpp>
#include <SushiAI/core/common.hpp>
#include <SushiAI/core/result.hpp>
#include <SushiAI/core/status.hpp>

namespace SushiAI
{
    namespace NN
    {
        namespace Detail
        {
            /**
             * @brief Calls @p call, which reports a failure by throwing, and returns a Result.
             * @param call Takes no argument and returns a value, not void.
             * @return The value, InvalidDescription carrying the message of a thrown Error, or
             *         OutOfMemory for std::bad_alloc; any other exception is fatal.
             */
            template<typename Call>
            [[nodiscard]] auto bridge_throwing(Call&& call) noexcept
                -> Result<decltype(std::forward<Call>(call)())>
            {
                try
                {
                    return std::forward<Call>(call)();
                }
                catch (const Error& error)
                {
                    return Status::from_message(StatusCode::InvalidDescription, [&error]()
                    {
                        return std::string(error.what());
                    });
                }
                catch (const std::bad_alloc&)
                {
                    return Status::from_message(StatusCode::OutOfMemory, []()
                    {
                        return std::string("an allocation was refused while a layer was "
                                           "built or traced");
                    });
                }
                catch (const std::exception& unknown)
                {
                    SA_FATAL("a layer threw something other than Error: {}", unknown.what());
                }
            }

        } // namespace Detail
    } // namespace NN
} // namespace SushiAI
