/**************************************************************************/
/* training_replica_model.hpp                                             */
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
 * @file training_replica_model.hpp
 * @brief Declares TrainingReplicaModel, the ReplicaModel a training file describes.
 * @author Mustafa Garip
 */

#pragma once

#include <cstdint>
#include <memory>
#include <vector>
#include <SushiBLAS/engine/engine.hpp>
#include <SushiBLAS/tensor/tensor.hpp>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/config/model_config.hpp>
#include <SushiAI/config/train_config.hpp>
#include <SushiAI/data/dataset.hpp>
#include <SushiAI/distribution/replica_model.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/graph/tensor_pool.hpp>

#include "registries.hpp"
#include "replica_tensors.hpp"
#include "training_setup.hpp"

namespace SushiAI
{
    namespace Apps
    {
        /**
         * @struct ReplicaSource
         * @brief Holds the files every replica of one run is built from, and their fingerprint.
         */
        struct ReplicaSource
        {
            /** @brief The rows the training file may name. */
            Registries registries;

            /** @brief The training file. */
            Config::TrainSpec spec;

            /** @brief The architecture file the training file names. */
            Config::ModelSpec model_spec;

            /** @brief The FNV-1a hash of the architecture file's bytes and the objective type. */
            std::uint64_t fingerprint = 0;

            /**
             * @brief Reads the architecture file @p spec names and hashes it.
             * @return The source, or FileNotFound naming the architecture file it cannot open.
             */
            [[nodiscard]] static Result<std::shared_ptr<const ReplicaSource>> make(
                Registries registries, Config::TrainSpec spec, Config::ModelSpec model_spec);
        };

        /**
         * @class TrainingReplicaModel
         * @brief Holds a training setup over one micro-batch and feeds it a replica's rows.
         *
         * Each feed reads the rows of its micro-batch from the replica's own copy of the data,
         * and no other row of the batch.
         */
        class TrainingReplicaModel final : public Distribution::ReplicaModel
        {
            public:
                /**
                 * @brief Returns the factory that builds the model of a replica from @p source.
                 * @param source Not null; every model the factory builds shares it.
                 */
                [[nodiscard]] static Distribution::ReplicaModelFactory factory(
                    std::shared_ptr<const ReplicaSource> source);

                /**
                 * @brief Builds the setup of @p source over @p micro_batch_rows rows.
                 * @return The model, or the first refusal of the setup.
                 */
                [[nodiscard]] static Result<std::unique_ptr<Distribution::ReplicaModel>> make(
                    std::shared_ptr<const ReplicaSource> source, SushiBLAS::Engine& engine,
                    const Distribution::ReplicaDraws& draws, std::int64_t micro_batch_rows);

                TrainingReplicaModel(const TrainingReplicaModel&) = delete;
                TrainingReplicaModel& operator=(const TrainingReplicaModel&) = delete;
                TrainingReplicaModel(TrainingReplicaModel&&) = delete;
                TrainingReplicaModel& operator=(TrainingReplicaModel&&) = delete;
                ~TrainingReplicaModel() override = default;

                /** @brief Returns the training graph over one micro-batch. */
                [[nodiscard]] const Graph::Graph& graph() const noexcept override;

                /** @brief Returns the pool the training graph is bound into. */
                [[nodiscard]] Graph::TensorPool& pool() noexcept override;

                /** @brief Returns the parameters, in the order of the parameter pack. */
                [[nodiscard]] SushiRuntime::span<const SushiBLAS::Tensor>
                parameters() const noexcept override;

                /** @brief Returns the gradients, the loss and the statistics, in pack order. */
                [[nodiscard]] SushiRuntime::span<const SushiBLAS::Tensor>
                results() const noexcept override;

                /** @brief Returns the fingerprint of the files the model was built from. */
                [[nodiscard]] std::uint64_t fingerprint() const noexcept override;

                /**
                 * @brief Writes the rows of @p rows into the leaves of the step.
                 * @return InvalidDescription for rows of another count or outside their batch,
                 *         or the refusal of the read, of the objective's feed or of the copy.
                 */
                [[nodiscard]] Status feed(const Distribution::ReplicaRows& rows) override;

            private:
                /** @brief Holds the parts make() built. */
                TrainingReplicaModel(std::shared_ptr<const ReplicaSource> source,
                                     SushiBLAS::Engine& engine,
                                     std::unique_ptr<TrainingSetup> setup,
                                     ReplicaTensors tensors) noexcept;

                /** @brief Makes the reads that follow return batches of @p epoch. */
                void enter_epoch(std::uint64_t epoch) noexcept;

                /// @brief The files the model was built from; shared by every replica.
                std::shared_ptr<const ReplicaSource> source_;

                /// @brief The replica's engine, which the inputs are copied on; not owned.
                SushiBLAS::Engine* engine_;

                /// @brief The data, the training graph, the state and the pool.
                std::unique_ptr<TrainingSetup> setup_;

                /// @brief The tensors of the two packs, in pack order.
                ReplicaTensors tensors_;

                /// @brief The rows the last feed read; reused by the next one.
                Data::Batch part_;

                /// @brief Whether the dataset has entered an epoch.
                bool entered_ = false;

                /// @brief The epoch the dataset entered last.
                std::uint64_t entered_epoch_ = 0;
        };

    } // namespace Apps
} // namespace SushiAI
