/**************************************************************************/
/* traced.hpp                                                             */
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
 * @file traced.hpp
 * @brief Declares Traced, a traced value or the failure that stopped the trace.
 * @author Mustafa Garip
 */

#pragma once

#include <utility>
#include <SushiAI/failure/check.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/graph/ir.hpp>

namespace SushiAI
{
    namespace Graph
    {
        /**
         * @brief Holds a traced value's id, or the failure that stopped the trace.
         *
         * Both converting constructors are implicit, against the single-argument rule, so an
         * emitter takes a ValueID or another emitter's result as an operand and returns
         * either an id or a Status.
         */
        class [[nodiscard]] Traced final
        {
            public:
                /** @brief Holds the absent-value id, which stands for an operand not given. */
                Traced() noexcept = default;

                /** @brief Holds @p id. */
                Traced(ValueID id) noexcept : id_(id)
                {

                }

                /**
                 * @brief Holds @p failure.
                 * @param failure A failed Status; a success is fatal.
                 */
                Traced(Status failure) noexcept : status_(std::move(failure))
                {
                    SA_FATAL_IF(status_.is_ok(),
                                "a Traced was built from a success Status; it needs an id or a "
                                "failure");
                }

                /** @brief Reports whether no failure is held; an absent operand counts. */
                [[nodiscard]] bool has_value() const noexcept
                {
                    return status_.is_ok();
                }

                /** @brief Returns the failure, or a success when an id is held. */
                [[nodiscard]] const Status& status() const noexcept
                {
                    return status_;
                }

                /**
                 * @brief Returns the held id, which is the absent-value id for an absent operand.
                 * @pre has_value(); a failure is fatal and reports its message.
                 */
                [[nodiscard]] ValueID value() const noexcept
                {
                    SA_FATAL_IF(!status_.is_ok(),
                                "Traced::value() was called on a failure: {}", status_.message());
                    return id_;
                }

            private:
                /** @brief Holds the id; the absent-value id when a failure is held. */
                ValueID id_ = K_NO_VALUE;

                /** @brief Holds the failure; a success when an id is held. */
                Status status_;
        };

    } // namespace Graph

    namespace Detail
    {
        /** @brief Returns the status SA_TRY and SA_TRY_ASSIGN read from a Traced. */
        [[nodiscard]] inline const Status& status_of(const Graph::Traced& traced) noexcept
        {
            return traced.status();
        }

    } // namespace Detail
} // namespace SushiAI
