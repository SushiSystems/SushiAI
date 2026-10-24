/**************************************************************************/
/* replica_draws.hpp                                                      */
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
 * @file replica_draws.hpp
 * @brief Declares ReplicaDraws, what every seeded draw of one replica takes.
 * @author Mustafa Garip
 * @see modules/core/distribution/README.md
 */

#pragma once

#include <cstdint>

namespace SushiAI
{
    namespace Distribution
    {
        /** @brief Names the seed, the replica and the run that every draw of a replica takes. */
        struct ReplicaDraws
        {
            /** @brief The run's seed, as the training spec states it. */
            std::uint64_t seed = 0;

            /** @brief The replica's index, k. */
            std::uint64_t replica_index = 0;

            /** @brief The index of the run, so two runs of one spec can draw apart. */
            std::uint64_t run_index = 0;

            /** @brief Returns whether two draws name the same seed, replica and run. */
            [[nodiscard]] bool operator==(const ReplicaDraws& other) const noexcept
            {
                return seed == other.seed && replica_index == other.replica_index
                       && run_index == other.run_index;
            }

            /** @brief Returns whether two draws differ in the seed, the replica or the run. */
            [[nodiscard]] bool operator!=(const ReplicaDraws& other) const noexcept
            {
                return !(*this == other);
            }
        };

        /**
         * @brief Returns the seed a replica's random sites and augmentation draw from.
         * @return splitmix64_at(splitmix64_at(seed, replica_index), run_index), so two
         *         replicas, or two runs, of one seed draw from different streams.
         */
        [[nodiscard]] std::uint64_t replica_seed(const ReplicaDraws& draws) noexcept;

    } // namespace Distribution
} // namespace SushiAI
