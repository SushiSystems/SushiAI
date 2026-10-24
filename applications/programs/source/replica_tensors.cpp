/**************************************************************************/
/* replica_tensors.cpp                                                    */
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
 * @file replica_tensors.cpp
 * @brief Implements ReplicaTensors::list(), which names the tensors of a replica's packs.
 * @author Mustafa Garip
 */

#include "replica_tensors.hpp"

#include <SushiAI/failure/status.hpp>
#include <SushiAI/graph/ir.hpp>

namespace SushiAI
{
    namespace Apps
    {
        Result<ReplicaTensors> ReplicaTensors::list(
            SushiRuntime::span<const Optim::ParameterSlot> parameters,
            const Train::TrainingGraph& built, Graph::TensorPool& pool,
            SushiRuntime::span<const Train::BufferUpdateSlot> updates)
        {
            SA_FAIL_IF(built.spec.loss == Graph::K_NO_VALUE, StatusCode::InvalidDescription,
                       "the training graph names no loss, so a replica has none to return");

            ReplicaTensors tensors;
            tensors.parameters.reserve(parameters.size());
            tensors.results.reserve(parameters.size() + 1 + updates.size());
            for (const Optim::ParameterSlot& slot : parameters)
            {
                tensors.parameters.push_back(slot.parameter);
                tensors.results.push_back(slot.gradient);
            }

            tensors.results.push_back(pool.tensor(built.spec.loss));
            for (const Train::BufferUpdateSlot& update : updates)
                tensors.results.push_back(update.source);

            return tensors;
        }

    } // namespace Apps
} // namespace SushiAI
