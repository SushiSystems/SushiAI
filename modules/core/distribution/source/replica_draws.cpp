/**************************************************************************/
/* replica_draws.cpp                                                      */
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
 * @file replica_draws.cpp
 * @brief Implements replica_seed().
 * @author Mustafa Garip
 */

#include <SushiAI/distribution/replica_draws.hpp>

#include <SushiAI/graph/dropout_mask.hpp>

namespace SushiAI
{
    namespace Distribution
    {
        std::uint64_t replica_seed(const ReplicaDraws& draws) noexcept
        {
            const std::uint64_t replica_stream =
                Graph::splitmix64_at(draws.seed, draws.replica_index);
            return Graph::splitmix64_at(replica_stream, draws.run_index);
        }

    } // namespace Distribution
} // namespace SushiAI
