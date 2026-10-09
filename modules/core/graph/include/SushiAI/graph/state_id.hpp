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
 * @file state_id.hpp
 * @brief Declares StateId and StateIdentity, which name a piece of model state across graphs.
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
        enum class StateId : std::uint64_t
        {

        };

        /** @brief Marks state that has not been assigned an identity. */
        inline constexpr StateId K_NO_STATE{0};

        namespace Detail
        {
            /** @brief Holds the next identity to hand out; the one counter of the process. */
            inline std::atomic<std::uint64_t> next_state_id{1};

            /** @brief Draws an identity no earlier call in this process returned. */
            [[nodiscard]] inline StateId draw_state_id() noexcept
            {
                return StateId{next_state_id.fetch_add(1, std::memory_order_relaxed)};
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
                [[nodiscard]] StateId get() noexcept
                {
                    if (id_ == K_NO_STATE)
                        id_ = Detail::draw_state_id();

                    return id_;
                }

                /** @brief Returns the identity, or K_NO_STATE when none was drawn. */
                [[nodiscard]] StateId peek() const noexcept
                {
                    return id_;
                }

            private:
                /** @brief Holds the identity; K_NO_STATE until get() draws one. */
                StateId id_ = K_NO_STATE;
        };

    } // namespace Graph
} // namespace SushiAI
