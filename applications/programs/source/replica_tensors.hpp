/**************************************************************************/
/* replica_tensors.hpp                                                    */
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
 * @file replica_tensors.hpp
 * @brief Declares ReplicaTensors, the tensors a replica's two packs hold, in pack order.
 * @author Mustafa Garip
 */

#pragma once

#include <vector>
#include <SushiBLAS/tensor/tensor.hpp>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/graph/tensor_pool.hpp>
#include <SushiAI/optim/optimizer.hpp>
#include <SushiAI/train/buffer_update_recorder.hpp>
#include <SushiAI/train/training_graph.hpp>

namespace SushiAI
{
    namespace Apps
    {
        /**
         * @struct ReplicaTensors
         * @brief Holds the tensors of a training setup in the order of the parameter pack and
         *        of the result pack.
         *
         * The master and every replica list their own training graph through list(), so entry
         * i of a pack names the same value on every machine.
         */
        struct ReplicaTensors
        {
            /** @brief The parameters, in the order of the parameter slots. */
            std::vector<SushiBLAS::Tensor> parameters;

            /** @brief The gradients in parameter order, the loss, then the batch statistics. */
            std::vector<SushiBLAS::Tensor> results;

            /**
             * @brief Lists the packed tensors of a training graph bound into @p pool.
             * @param parameters The slots bind_parameters() returned for @p built.
             * @param updates The slots bind_buffer_updates() returned for @p built.
             * @return The tensors, or InvalidDescription for a training graph with no loss.
             */
            [[nodiscard]] static Result<ReplicaTensors> list(
                SushiRuntime::span<const Optim::ParameterSlot> parameters,
                const Train::TrainingGraph& built, Graph::TensorPool& pool,
                SushiRuntime::span<const Train::BufferUpdateSlot> updates);
        };

    } // namespace Apps
} // namespace SushiAI
