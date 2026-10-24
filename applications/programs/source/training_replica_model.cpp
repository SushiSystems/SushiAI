/**************************************************************************/
/* training_replica_model.cpp                                             */
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
 * @file training_replica_model.cpp
 * @brief Implements TrainingReplicaModel and the files every replica is built from.
 * @author Mustafa Garip
 */

#include "training_replica_model.hpp"

#include <cstddef>
#include <fstream>
#include <iterator>
#include <string>
#include <utility>
#include <SushiBLAS/blas/blas.hpp>
#include <SushiAI/data/epoch.hpp>
#include <SushiAI/distribution/replica_draws.hpp>
#include <SushiAI/execution/execution.hpp>
#include <SushiAI/seam/seam.hpp>

namespace SushiAI
{
    namespace Apps
    {
        namespace
        {
            /** @brief Holds the FNV-1a 64-bit offset basis. */
            constexpr std::uint64_t K_FNV_OFFSET = 0xcbf29ce484222325ull;

            /** @brief Holds the FNV-1a 64-bit prime. */
            constexpr std::uint64_t K_FNV_PRIME = 0x100000001b3ull;

            /** @brief Returns @p hash with the bytes of @p text folded in, one at a time. */
            [[nodiscard]] std::uint64_t fold(std::uint64_t hash, const std::string& text) noexcept
            {
                for (const char byte : text)
                {
                    hash ^= static_cast<unsigned char>(byte);
                    hash *= K_FNV_PRIME;
                }

                return hash;
            }

            /** @brief Returns rows @p first to @p first + @p count of @p tensor, as a view. */
            [[nodiscard]] SushiBLAS::Tensor rows_of(const SushiBLAS::Tensor& tensor,
                                                    std::int64_t first, std::int64_t count)
            {
                return tensor.slice(0, first, first + count);
            }

        } // namespace

        Result<std::shared_ptr<const ReplicaSource>> ReplicaSource::make(
            Registries registries, Config::TrainSpec spec, Config::ModelSpec model_spec)
        {
            std::ifstream file(spec.model_path, std::ios::binary);
            SA_FAIL_IF(!file, StatusCode::FileNotFound,
                       "the architecture file '{}' could not be opened to fingerprint it",
                       spec.model_path.string());

            return SushiAI::Detail::guard_value(SushiAI::Detail::SeamPhase::Allocate, [&]()
            {
                const std::string bytes((std::istreambuf_iterator<char>(file)),
                                        std::istreambuf_iterator<char>());
                auto source = std::make_shared<ReplicaSource>();
                source->fingerprint = fold(fold(K_FNV_OFFSET, bytes), spec.objective_type);
                source->registries = std::move(registries);
                source->spec = std::move(spec);
                source->model_spec = std::move(model_spec);
                return std::shared_ptr<const ReplicaSource>(std::move(source));
            });
        }

        Distribution::ReplicaModelFactory TrainingReplicaModel::factory(
            std::shared_ptr<const ReplicaSource> source)
        {
            return [source = std::move(source)](SushiBLAS::Engine& engine,
                                                const Distribution::ReplicaDraws& draws,
                                                std::int64_t micro_batch_rows)
            {
                return make(source, engine, draws, micro_batch_rows);
            };
        }

        Result<std::unique_ptr<Distribution::ReplicaModel>> TrainingReplicaModel::make(
            std::shared_ptr<const ReplicaSource> source, SushiBLAS::Engine& engine,
            const Distribution::ReplicaDraws& draws, std::int64_t micro_batch_rows)
        {
            SA_FAIL_IF(source == nullptr, StatusCode::InvalidDescription,
                       "a replica model needs the files of its run");

            SA_TRY_ASSIGN(std::unique_ptr<TrainingSetup> setup,
                          TrainingSetup::make(source->registries, source->spec,
                                              source->model_spec, engine,
                                              Distribution::replica_seed(draws),
                                              micro_batch_rows));
            SA_TRY_ASSIGN(ReplicaTensors tensors,
                          ReplicaTensors::list(
                              SushiRuntime::span<const Optim::ParameterSlot>(
                                  setup->parameters.data(), setup->parameters.size()),
                              setup->built, *setup->pool,
                              SushiRuntime::span<const Train::BufferUpdateSlot>(
                                  setup->updates.data(), setup->updates.size())));

            return Result<std::unique_ptr<Distribution::ReplicaModel>>(
                std::unique_ptr<Distribution::ReplicaModel>(new TrainingReplicaModel(
                    std::move(source), engine, std::move(setup), std::move(tensors))));
        }

        TrainingReplicaModel::TrainingReplicaModel(std::shared_ptr<const ReplicaSource> source,
                                                   SushiBLAS::Engine& engine,
                                                   std::unique_ptr<TrainingSetup> setup,
                                                   ReplicaTensors tensors) noexcept
            : source_(std::move(source)), engine_(&engine), setup_(std::move(setup)),
              tensors_(std::move(tensors))
        {

        }

        const Graph::Graph& TrainingReplicaModel::graph() const noexcept
        {
            return setup_->built.graph;
        }

        Graph::TensorPool& TrainingReplicaModel::pool() noexcept
        {
            return *setup_->pool;
        }

        SushiRuntime::span<const SushiBLAS::Tensor> TrainingReplicaModel::parameters()
            const noexcept
        {
            return SushiRuntime::span<const SushiBLAS::Tensor>(tensors_.parameters.data(),
                                                               tensors_.parameters.size());
        }

        SushiRuntime::span<const SushiBLAS::Tensor> TrainingReplicaModel::results() const noexcept
        {
            return SushiRuntime::span<const SushiBLAS::Tensor>(tensors_.results.data(),
                                                               tensors_.results.size());
        }

        std::uint64_t TrainingReplicaModel::fingerprint() const noexcept
        {
            return source_->fingerprint;
        }

        Status TrainingReplicaModel::hold_batch(std::uint64_t epoch, std::uint64_t batch_index)
        {
            if (holds_batch_ && held_epoch_ == epoch && held_index_ == batch_index)
                return Status();

            if (!holds_batch_ || held_epoch_ != epoch)
                Data::begin_epoch(*setup_->dataset, static_cast<std::int64_t>(epoch));

            holds_batch_ = false;
            SA_TRY(setup_->dataset->read(static_cast<std::size_t>(batch_index), held_));
            holds_batch_ = true;
            held_epoch_ = epoch;
            held_index_ = batch_index;
            return Status();
        }

        Status TrainingReplicaModel::feed(const Distribution::ReplicaRows& rows)
        {
            const std::uint64_t batch_rows =
                static_cast<std::uint64_t>(setup_->dataset->batch_size());
            const std::uint64_t first = rows.batch_index * batch_rows;
            SA_FAIL_IF(static_cast<std::int64_t>(rows.rows.count) != setup_->given.batch,
                       StatusCode::InvalidDescription,
                       "{} rows were fed to a replica traced over {}", rows.rows.count,
                       setup_->given.batch);
            SA_FAIL_IF(rows.rows.first < first
                           || rows.rows.first + rows.rows.count > first + batch_rows,
                       StatusCode::InvalidDescription,
                       "rows {} to {} lie outside batch {}, which holds positions {} to {}; the "
                       "cluster's rows must sum to the data's batch size",
                       rows.rows.first, rows.rows.first + rows.rows.count, rows.batch_index,
                       first, first + batch_rows);

            SA_TRY(hold_batch(rows.epoch, rows.batch_index));

            const std::int64_t start = static_cast<std::int64_t>(rows.rows.first - first);
            const std::int64_t count = static_cast<std::int64_t>(rows.rows.count);
            SA_TRY_ASSIGN(const Data::Batch part,
                          SushiAI::Detail::guard_value(SushiAI::Detail::SeamPhase::Record, [&]()
            {
                Data::Batch sliced;
                sliced.inputs = rows_of(held_.inputs, start, count);
                for (const Data::BatchTarget& target : held_.targets)
                    sliced.targets.push_back(
                        Data::BatchTarget{target.role, rows_of(target.values, start, count)});
                return sliced;
            }));

            // No task is recorded here, so the objective may write its leaves on the host.
            SA_TRY(setup_->objective->feed(part, setup_->built.leaves, *setup_->pool));

            SushiBLAS::Tensor& inputs = setup_->pool->tensor(setup_->built.spec.inputs);
            return SushiAI::Detail::record_and_run(*engine_,
                                                   [&](SushiBLAS::TaskRecorder& recorder)
            {
                SushiBLAS::BLASOps blas(recorder);
                blas.copy(part.inputs, inputs);
                return Status();
            });
        }

    } // namespace Apps
} // namespace SushiAI
