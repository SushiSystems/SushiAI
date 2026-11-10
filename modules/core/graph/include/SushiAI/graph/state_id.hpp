/**************************************************************************/
/* state_id.hpp                                                           */
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
 * @file state_id.hpp
 * @brief Declares StateID and StateIdentity, which name a piece of model state across graphs.
 * @author Mustafa Garip
 */

#pragma once

#include <atomic>
#include <cstdint>

namespace SushiAI
{
    namespace Graph
    {
        /** @brief Identifies one piece of model state across every graph it is traced into. */
        enum class StateID : std::uint64_t
        {

        };

        /** @brief Marks state that has not been assigned an identity. */
        inline constexpr StateID K_NO_STATE{0};

        /**
         * @class StateIDSource
         * @brief Hands out identities, each above every one it handed out before.
         *
         * The process draws from one source; a test builds its own and reads the order of its
         * draws without touching the process-wide counter.
         */
        class StateIDSource final
        {
            public:
                /** @brief Draws an identity no earlier call on this source returned. */
                [[nodiscard]] StateID draw() noexcept
                {
                    return StateID{next_.fetch_add(1, std::memory_order_relaxed)};
                }

            private:
                /// @brief Holds the next identity to hand out; 0 is K_NO_STATE and is never drawn.
                std::atomic<std::uint64_t> next_{1};
        };

        namespace Detail
        {
            /** @brief Holds the one source of the process. */
            inline StateIDSource process_state_ids;

            /** @brief Draws an identity no earlier call in this process returned. */
            [[nodiscard]] inline StateID draw_state_id() noexcept
            {
                return process_state_ids.draw();
            }

        } // namespace Detail

        /**
         * @brief Holds a state identity that a copy does not share and a move carries away.
         *
         * A copy is new state and starts without an identity. A moved-from holder is left
         * without one, so the identity stays with the object the graphs were traced from.
         */
        class StateIdentity final
        {
            public:
                /** @brief Starts without an identity; the first get() draws one. */
                StateIdentity() noexcept = default;

                /** @brief Starts without an identity, whatever the source holds. */
                StateIdentity(const StateIdentity&) noexcept
                {

                }

                /** @brief Drops this holder's identity, unless the source is this holder. */
                StateIdentity& operator=(const StateIdentity& other) noexcept
                {
                    if (this != &other)
                        id_ = K_NO_STATE;

                    return *this;
                }

                /** @brief Takes the identity of @p other and leaves it without one. */
                StateIdentity(StateIdentity&& other) noexcept : id_(other.id_)
                {
                    other.id_ = K_NO_STATE;
                }

                /** @brief Takes the identity of @p other and leaves it without one. */
                StateIdentity& operator=(StateIdentity&& other) noexcept
                {
                    if (this != &other)
                    {
                        id_ = other.id_;
                        other.id_ = K_NO_STATE;
                    }

                    return *this;
                }

                /** @brief Releases nothing; an identity is a number. */
                ~StateIdentity() = default;

                /**
                 * @brief Returns the identity, drawing one from the process-wide counter if none.
                 * @pre No other thread calls get() on this holder at the same time.
                 */
                [[nodiscard]] StateID get() noexcept
                {
                    if (id_ == K_NO_STATE)
                        id_ = Detail::draw_state_id();

                    return id_;
                }

                /**
                 * @brief Returns the identity, drawing one from @p source if none.
                 * @pre No other thread calls get() on this holder at the same time.
                 */
                [[nodiscard]] StateID get(StateIDSource& source) noexcept
                {
                    if (id_ == K_NO_STATE)
                        id_ = source.draw();

                    return id_;
                }

                /** @brief Returns the identity, or K_NO_STATE when none was drawn. */
                [[nodiscard]] StateID peek() const noexcept
                {
                    return id_;
                }

            private:
                /** @brief Holds the identity; K_NO_STATE until get() draws one. */
                StateID id_ = K_NO_STATE;
        };

    } // namespace Graph
} // namespace SushiAI
