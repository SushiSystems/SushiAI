/**************************************************************************/
/* cluster_run.cpp                                                        */
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
 * @file cluster_run.cpp
 * @brief Implements the training run of `sushiai train` over the replicas of a cluster.
 * @author Mustafa Garip
 */

#include "cluster_run.hpp"

#include <cstddef>
#include <iostream>
#include <memory>
#include <utility>
#include <vector>
#include <SushiBLAS/tensor/tensor.hpp>
#include <SushiLog/text/text_format.hpp>
#include <SushiLog/text/value_text.hpp>
#include <SushiAI/distribution/gradient_exchange.hpp>
#include <SushiAI/distribution/host_access.hpp>
#include <SushiAI/distribution/parameter_publish.hpp>
#include <SushiAI/dtype/dtype.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/profile/profile.hpp>
#include <SushiAI/train/training_step.hpp>

#include "epoch_loop.hpp"
#include "replica_tensors.hpp"

namespace SushiAI
{
    namespace Apps
    {
        namespace
        {
            /** @brief Returns the one element of the summed loss, read on the host. */
            [[nodiscard]] double read_loss(const SushiBLAS::Tensor& loss) noexcept
            {
                if (loss.dtype == DType::Float64)
                    return *loss.data_as<double>();

                return static_cast<double>(*loss.data_as<float>());
            }

            /** @brief Holds the recorders of the master's step and the step that runs them. */
            struct ClusterStep
            {
                /** @brief Packs the parameters after the optimizer. */
                std::unique_ptr<Distribution::ParameterPublish> publish;

                /** @brief Records the replica tasks and the sums before the optimizer. */
                std::unique_ptr<Distribution::GradientExchange> exchange;

                /** @brief The graph the step compiles, which holds no node. */
                std::unique_ptr<Graph::Graph> graph;

                /** @brief The compiled exchange, update and publish; declared last, dies first. */
                std::unique_ptr<Train::TrainingStep> step;
            };

            /**
             * @brief Builds the publish, the exchange and the step of @p run over @p tensors.
             * @return The parts, or the first refusal of a bind or of the compile.
             */
            [[nodiscard]] Result<ClusterStep> build_step(ClusterRun& run,
                                                         const ReplicaTensors& tensors)
            {
                ClusterStep built;
                SA_TRY_ASSIGN(built.publish,
                              Distribution::ParameterPublish::make(
                                  run.prepared.engine, run.binding, run.local.parameter_layout(),
                                  tensors.parameters));

                Distribution::ExchangeRun exchange_run;
                exchange_run.operation = run.local.operation();
                exchange_run.seed = run.prepared.start.seed;
                exchange_run.run_index = run.run_index;
                exchange_run.batches_per_epoch =
                    static_cast<std::uint64_t>(run.prepared.dataset.batch_count());
                SA_TRY_ASSIGN(built.exchange,
                              Distribution::GradientExchange::make(
                                  run.prepared.engine, run.binding, run.section.plan(),
                                  exchange_run,
                                  built.publish->pack(), run.local.result_layout(),
                                  tensors.results));

                Train::TrainingStepOptions options;
                options.optimizer_placement = Train::OptimizerPlacement::InPlan;
                options.before_update = built.exchange.get();
                options.after_update = built.publish.get();

                built.graph = std::make_unique<Graph::Graph>();
                SA_TRY_ASSIGN(built.step,
                              Train::TrainingStep::make(run.prepared.engine, *built.graph,
                                                        run.prepared.pool,
                                                        run.prepared.optimizer,
                                                        std::move(run.prepared.updates),
                                                        options));
                return built;
            }

        } // namespace

        Result<RunSummary> run_on_cluster(ClusterRun& run)
        {
            PreparedRun& prepared = run.prepared;
            const Data::Dataset& dataset = prepared.dataset;
            SA_FAIL_IF(run.section.plan().batch_rows() != dataset.batch_size(),
                       StatusCode::InvalidDescription,
                       "the cluster's replicas take {} rows a batch, but the '{}' data reads "
                       "batches of {}; the rows of 'replicas' sum to the data's batch size",
                       run.section.plan().batch_rows(), dataset.name(), dataset.batch_size());

            SA_TRY(check_start_batch(prepared));
            const std::size_t batches = dataset.batch_count();

            SA_TRY_ASSIGN(const ReplicaTensors tensors,
                          ReplicaTensors::list(
                              run.parameters, prepared.built, prepared.pool,
                              SushiRuntime::span<const Train::BufferUpdateSlot>(
                                  prepared.updates.data(), prepared.updates.size())));
            SA_TRY_ASSIGN(ClusterStep built, build_step(run, tensors));

            // A resumed run replays from the step count and the position the checkpoint saved.
            SA_TRY(built.step->restore_run_count(
                static_cast<std::size_t>(prepared.start.step_runs)));
            built.exchange->set_next_step(prepared.start.epoch * batches + prepared.start.batch);

            // The master's pool allocates Residency::Shared, so the host reads the loss.
            const SushiBLAS::Tensor& loss = tensors.results[tensors.parameters.size()];
            Distribution::require_host_visible(loss, "Cluster loss", 0);

            // The replicas read their own rows, so the master's data enters no epoch.
            EpochSteps steps;
            steps.run_step = [&built](std::size_t)
            {
                SA_ZONE("device", "run");
                return built.step->run();
            };
            steps.report = [&loss](std::int64_t epoch, std::int64_t epochs)
            {
                std::cout << SushiLog::format_text("  epoch {}/{} loss {}\n",
                                                   SushiLog::right(epoch, 3),
                                                   SushiLog::left(epochs, 3),
                                                   SushiLog::fixed(read_loss(loss), 6));
            };
            SA_TRY(run_epochs(prepared, steps));

            RunSummary summary;
            summary.compile_count = built.step->step().compile_count();
            summary.progress = finished_progress(prepared, built.step->step().run_count());
            return summary;
        }

    } // namespace Apps
} // namespace SushiAI
