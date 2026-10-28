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
#include <exception>
#include <fstream>
#include <iterator>
#include <string>
#include <string_view>
#include <utility>
#include <SushiBLAS/blas/blas.hpp>
#include <SushiBLAS/engine/task_recorder.hpp>
#include <SushiAI/config/json_writer.hpp>
#include <SushiAI/config/model_config.hpp>
#include <SushiAI/data/epoch.hpp>
#include <SushiAI/distribution/cluster_section.hpp>
#include <SushiAI/distribution/replica_plan.hpp>
#include <SushiAI/distribution/replica_draws.hpp>
#include <SushiAI/execution/execution.hpp>
#include <SushiAI/hash/fnv1a.hpp>
#include <SushiAI/profile/profile.hpp>
#include <SushiAI/seam/seam.hpp>

#include "training_cluster_section.hpp"

namespace SushiAI
{
    namespace Apps
    {
        namespace
        {
            /** @brief Folds @p text and then its length into @p hash, so two parts never merge. */
            [[nodiscard]] std::uint64_t fold_text(std::uint64_t hash, std::string_view text)
            {
                return Hash::fnv1a_word(Hash::fnv1a_text(hash, text), text.size());
            }

            /** @brief Folds the JSON text of @p value into @p hash. */
            [[nodiscard]] Result<std::uint64_t> fold_value(std::uint64_t hash,
                                                           const Config::Value& value)
            {
                SA_TRY_ASSIGN(const std::string text, Config::write_json(value));
                return fold_text(hash, text);
            }

            /**
             * @brief Returns the fingerprint of a replica built from @p bytes and @p spec.
             * @return The hash, or the refusal of the cluster section, of the seed or of a
             *         key's JSON text.
             */
            [[nodiscard]] Result<std::uint64_t> fingerprint_of(const std::string& bytes,
                                                               const Config::TrainSpec& spec,
                                                               std::uint64_t run_index)
            {
                SA_TRY_ASSIGN(const Distribution::ClusterSection section,
                              read_training_cluster(spec));
                SA_TRY_ASSIGN(const std::uint64_t seed,
                              Config::read_run_seed(spec, "the training file"));

                std::uint64_t hash = fold_text(Hash::K_FNV1A_OFFSET_BASIS, bytes);
                hash = fold_text(hash, spec.objective_type);
                SA_TRY_ASSIGN(hash, fold_value(hash, spec.objective_attributes));
                hash = fold_text(hash, spec.data.type);
                SA_TRY_ASSIGN(hash, fold_value(hash, spec.data.attributes));
                hash = Hash::fnv1a_word(hash, seed);
                hash = Hash::fnv1a_word(hash, run_index);

                const Distribution::ReplicaPlan& plan = section.plan();
                hash = Hash::fnv1a_word(hash, plan.replica_count());
                for (std::size_t replica = 0; replica < plan.replica_count(); ++replica)
                {
                    hash = Hash::fnv1a_word(hash,
                                            static_cast<std::uint64_t>(plan.rows_of(replica)));
                    hash = Hash::fnv1a_word(
                        hash, static_cast<std::uint64_t>(plan.micro_batch_of(replica)));
                }
                return hash;
            }
        } // namespace

        Result<std::shared_ptr<const ReplicaSource>> ReplicaSource::load(
            const Config::TrainSpec& spec)
        {
            SA_TRY_ASSIGN(Config::ModelSpec model_spec, Config::load_model(spec.model_path));
            SA_TRY_ASSIGN(Registries registries, compose_registries());
            return make(std::move(registries), spec, std::move(model_spec));
        }

        Result<std::shared_ptr<const ReplicaSource>> ReplicaSource::make(
            Registries registries, Config::TrainSpec spec, Config::ModelSpec model_spec)
        {
            std::ifstream file(spec.model_path, std::ios::binary);
            SA_FAIL_IF(!file, StatusCode::FileNotFound,
                       "the architecture file '{}' could not be opened to fingerprint it",
                       spec.model_path.string());

            SA_TRY_ASSIGN(const std::string bytes,
                          SushiAI::Detail::guard_value(SushiAI::Detail::SeamPhase::Allocate,
                                                       [&file]()
            {
                return std::string((std::istreambuf_iterator<char>(file)),
                                   std::istreambuf_iterator<char>());
            }));
            SA_TRY_ASSIGN(const std::uint64_t fingerprint,
                          fingerprint_of(bytes, spec, K_RUN_INDEX));

            return SushiAI::Detail::guard_value(SushiAI::Detail::SeamPhase::Allocate, [&]()
            {
                auto source = std::make_shared<ReplicaSource>();
                source->fingerprint = fingerprint;
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
                                  setup->parameters().data(), setup->parameters().size()),
                              setup->built(), setup->pool(),
                              SushiRuntime::span<const Train::BufferUpdateSlot>(
                                  setup->updates().data(), setup->updates().size())));

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
            return setup_->built().graph;
        }

        Graph::TensorPool& TrainingReplicaModel::pool() noexcept
        {
            return setup_->pool();
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

        void TrainingReplicaModel::enter_epoch(std::uint64_t epoch) noexcept
        {
            if (entered_ && entered_epoch_ == epoch)
                return;

            Data::begin_epoch(setup_->dataset(), static_cast<std::int64_t>(epoch));
            entered_ = true;
            entered_epoch_ = epoch;
        }

        Status TrainingReplicaModel::feed(const Distribution::ReplicaRows& rows)
        {
            const Data::Dataset& dataset = setup_->dataset();
            const std::uint64_t batch_rows = static_cast<std::uint64_t>(dataset.batch_size());
            const std::uint64_t first = rows.batch_index * batch_rows;
            SA_FAIL_IF(static_cast<std::int64_t>(rows.rows.count) != setup_->given().batch,
                       StatusCode::InvalidDescription,
                       "{} rows were fed to a replica traced over {}", rows.rows.count,
                       setup_->given().batch);
            SA_FAIL_IF(rows.batch_index >= dataset.batch_count() || rows.rows.first < first
                           || rows.rows.first + rows.rows.count > first + batch_rows,
                       StatusCode::InvalidDescription,
                       "rows {} to {} lie outside batch {}, which holds positions {} to {}; the "
                       "cluster's rows must sum to the data's batch size",
                       rows.rows.first, rows.rows.first + rows.rows.count, rows.batch_index,
                       first, first + batch_rows);

            enter_epoch(rows.epoch);
            const Data::RowRange range{static_cast<std::int64_t>(rows.rows.first - first),
                                       static_cast<std::int64_t>(rows.rows.count)};
            {
                SA_ZONE("data", "read");
                SA_TRY(dataset.read_rows(static_cast<std::size_t>(rows.batch_index), range,
                                         part_));
            }

            // No task is recorded here, so the objective may write its leaves on the host.
            const Train::TrainingGraph& built = setup_->built();
            {
                SA_ZONE("host", "feed");
                SA_TRY(setup_->objective().feed(part_, built.leaves, setup_->pool()));
            }

            SushiBLAS::Tensor& inputs = setup_->pool().tensor(built.spec.inputs);
            SA_ZONE("copy", "input");
            return SushiAI::Detail::record_and_run(*engine_,
                                                   [&](SushiBLAS::TaskRecorder& recorder)
            {
                SushiBLAS::BLASOps blas(recorder);
                blas.copy(part_.inputs, inputs);
                return Status();
            });
        }

    } // namespace Apps
} // namespace SushiAI
