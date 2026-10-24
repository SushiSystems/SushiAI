/**************************************************************************/
/* replica_model.hpp                                                      */
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
 * @file replica_model.hpp
 * @brief Declares ReplicaModel, the traced model and the rows a replica step runs over.
 * @author Mustafa Garip
 * @see modules/core/distribution/README.md
 */

#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <SushiBLAS/engine/engine.hpp>
#include <SushiBLAS/tensor/tensor.hpp>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/distribution/replica_draws.hpp>
#include <SushiAI/distribution/replica_plan.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/graph/tensor_pool.hpp>

namespace SushiAI
{
    namespace Distribution
    {
        /** @brief Names the rows of one micro-batch of one replica in one step. */
        struct ReplicaRows
        {
            /** @brief The epoch, counted from 0. */
            std::uint64_t epoch = 0;

            /** @brief The batch of the epoch, counted from 0. */
            std::uint64_t batch_index = 0;

            /** @brief The micro-batch of the replica's rows, counted from 0. */
            std::uint64_t micro_batch_index = 0;

            /** @brief The positions of the epoch's sample order the micro-batch holds. */
            RowRange rows{};

            /** @brief The seed, replica and run every draw of the feed takes. */
            ReplicaDraws draws{};
        };

        /**
         * @brief Hands a replica step a traced forward and backward graph and fills its inputs.
         *
         * The layer above implements it from its training graph and its dataset. The graph
         * holds no optimizer; its objective is a mean over the rows of one micro-batch. It
         * owns everything it returns, for as long as it lives.
         */
        class ReplicaModel
        {
            public:
                virtual ~ReplicaModel() = default;

                /** @brief Returns the forward and backward graph, with every leaf in pool(). */
                [[nodiscard]] virtual const Graph::Graph& graph() const noexcept = 0;

                /** @brief Returns the pool the graph's values live in. */
                [[nodiscard]] virtual Graph::TensorPool& pool() noexcept = 0;

                /**
                 * @brief Returns the parameters, in the order of the parameter pack.
                 *
                 * Each is contiguous, and host-addressable memory of the step's engine.
                 */
                [[nodiscard]] virtual SushiRuntime::span<const SushiBLAS::Tensor>
                parameters() const noexcept = 0;

                /**
                 * @brief Returns the result tensors, in the order of the result pack: the
                 *        gradients in the order of parameters(), the loss, then the batch
                 *        statistics.
                 *
                 * Each is contiguous and host-addressable; a replay rewrites every one.
                 */
                [[nodiscard]] virtual SushiRuntime::span<const SushiBLAS::Tensor>
                results() const noexcept = 0;

                /** @brief Returns the 64-bit hash of the model, which its layer above defines. */
                [[nodiscard]] virtual std::uint64_t fingerprint() const noexcept = 0;

                /**
                 * @brief Writes the rows of one micro-batch into the graph's input leaves.
                 * @pre No replay of the graph is running.
                 */
                [[nodiscard]] virtual Status feed(const ReplicaRows& rows) = 0;

            protected:
                ReplicaModel() = default;
                ReplicaModel(const ReplicaModel&) = default;
                ReplicaModel& operator=(const ReplicaModel&) = default;
                ReplicaModel(ReplicaModel&&) = default;
                ReplicaModel& operator=(ReplicaModel&&) = default;
        };

        /**
         * @brief Builds the model of one replica on an engine.
         *
         * Called as factory(engine, draws, micro_batch_rows): the graph's input leaves hold
         * micro_batch_rows rows, and its random sites are seeded with replica_seed(draws). The
         * model's tensors are allocated through @c engine, which outlives the model.
         */
        using ReplicaModelFactory = std::function<Result<std::unique_ptr<ReplicaModel>>(
            SushiBLAS::Engine& engine, const ReplicaDraws& draws, std::int64_t micro_batch_rows)>;

    } // namespace Distribution
} // namespace SushiAI
