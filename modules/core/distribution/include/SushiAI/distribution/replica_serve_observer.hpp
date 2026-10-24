/**************************************************************************/
/* replica_serve_observer.hpp                                             */
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
 * @file replica_serve_observer.hpp
 * @brief Declares ReplicaServeObserver, what a replica kernel tells about each task it serves.
 * @author Mustafa Garip
 * @see modules/core/distribution/README.md
 */

#pragma once

#include <cstdint>
#include <SushiAI/distribution/replica_position.hpp>

namespace SushiAI
{
    namespace Distribution
    {
        /**
         * @brief Hears of each replica task a kernel serves, so the master learns that a
         *        replica's node ran on its own machine.
         */
        class ReplicaServeObserver
        {
            public:
                virtual ~ReplicaServeObserver() = default;

                /**
                 * @brief Called on the serving thread before the kernel runs replica
                 *        @p replica at @p position; the kernel serves one task at a time.
                 */
                virtual void on_served(std::uint64_t replica, const ReplicaPosition& position) = 0;

            protected:
                ReplicaServeObserver() = default;
                ReplicaServeObserver(const ReplicaServeObserver&) = default;
                ReplicaServeObserver& operator=(const ReplicaServeObserver&) = default;
                ReplicaServeObserver(ReplicaServeObserver&&) = default;
                ReplicaServeObserver& operator=(ReplicaServeObserver&&) = default;
        };

    } // namespace Distribution
} // namespace SushiAI
