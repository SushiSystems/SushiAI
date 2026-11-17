/**************************************************************************/
/* decision_run.cpp                                                       */
/**************************************************************************/
/*                          This file is part of:                         */
/*                                 SushiAI                                */
/*                 https://github.com/SushiSystems/SushiAI                */
/*                         https://sushisystems.io                        */
/**************************************************************************/
/* Copyright (c) 2026-present Mustafa Garip & Sushi Systems               */
/*                                                                        */
/* Licensed under the Apache License, Version 2.0 (the "License"); you    */
/* may not use this file except in compliance with the License. You may   */
/* obtain a copy of the License at                                        */
/*                                                                        */
/*     https://www.apache.org/licenses/LICENSE-2.0                        */
/*                                                                        */
/* Unless required by applicable law or agreed to in writing, software    */
/* distributed under the License is distributed on an "AS IS" BASIS,      */
/* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or        */
/* implied. See the License for the specific language governing           */
/* permissions and limitations under the License.                         */
/**************************************************************************/

/**
 * @file decision_run.cpp
 * @brief Implements the compilation of one step of a decision model.
 * @author Mustafa Garip
 */

#include "decision_run.hpp"

#include <utility>
#include <vector>

#include <SushiAI/config/train_config.hpp>
#include <SushiAI/dtype/dtype.hpp>
#include <SushiAI/optim/optimizer_registry.hpp>
#include <SushiAI/optim/sgd.hpp>
#include <SushiAI/seam/seam.hpp>
#include <SushiAI/shape/shape.hpp>
#include <SushiAI/train/buffer_update_recorder.hpp>
#include <SushiAI/train/objective_context.hpp>
#include <SushiAI/train/role_match.hpp>
#include <SushiAI/train/steps.hpp>

namespace SushiAI::Language::DecisionModels
{
    namespace
    {
        /** @brief Holds the rate of the SGD a Probe run applies: the weights stay as they are. */
        constexpr double K_PROBE_RATE = 0.0;

        /** @brief Names the objective block in a message. */
        constexpr std::string_view K_OBJECTIVE_CONTEXT = "the objective block";

        /** @brief Returns the optimizer of a run over @p slots. */
        [[nodiscard]] Result<std::unique_ptr<Optim::Optimizer>> make_optimizer(
            RunKind kind, ExecutionBackend& backend, std::vector<Optim::ParameterSlot> slots,
            const Config::TrainSpec& train) noexcept
        {
            if (kind == RunKind::Training)
            {
                return Optim::make_optimizer(backend, train.optimizer_type, std::move(slots),
                                             train.optimizer_attributes, "the optimizer block");
            }

            Optim::SGDOptions reader;
            reader.learning_rate = K_PROBE_RATE;
            SA_TRY_ASSIGN(std::unique_ptr<Optim::SGD> frozen,
                          Optim::SGD::make(backend, std::move(slots), reader));

            return std::unique_ptr<Optim::Optimizer>(std::move(frozen));
        }

        /** @brief Returns what the objective factory reads of the batch and the published outputs. */
        [[nodiscard]] Result<Train::ObjectiveContext> make_context(
            const DecisionSessionOptions& options, const Train::TracedStep& probe,
            const Data::Dataset& dataset) noexcept
        {
            Train::ObjectiveContext given;
            given.batch = dataset.batch_size();
            given.input_sample = dataset.sample_shape();
            given.task = options.model.task;
            SA_TRY_ASSIGN(given.outputs, Train::list_output_shapes(probe.graph, probe.outputs));

            return given;
        }
    } // namespace

    DecisionRun::DecisionRun(SushiBLAS::Engine& engine) : pool_(engine)
    {

    }

    Result<std::unique_ptr<DecisionRun>> DecisionRun::make(
        RunKind kind, ExecutionBackend& backend, NN::ModelGraph& model,
        std::unique_ptr<NN::ModelState>& state, const DecisionSessionOptions& options,
        const Data::Dataset& dataset) noexcept
    {
        SA_TRY_ASSIGN(std::unique_ptr<DecisionRun> run,
                      SushiAI::Detail::guard_value(SushiAI::Detail::SeamPhase::Allocate, [&]()
        {
            return std::unique_ptr<DecisionRun>(new DecisionRun(backend.engine()));
        }));

        const Shape sample = dataset.sample_shape();
        SA_FAIL_IF(sample != options.model.input_shape, StatusCode::InvalidDescription,
                   "the model file declares the input shape {}, but the '{}' data supplies {}",
                   options.model.input_shape.to_string(), dataset.name(), sample.to_string());
        SA_TRY_ASSIGN(const Shape batch_shape, Train::batch_shape(dataset.batch_size(), sample));

        Train::GraphOptions graph_options;
        graph_options.dtype = options.model.input_dtype;
        SA_TRY_ASSIGN(graph_options.seed, Config::read_run_seed(options.train, "the training file"));

        // The probe trace states the shapes the objective factory may read.
        SA_TRY_ASSIGN(const Train::TracedStep probe,
                      Train::build_probe_graph(model, batch_shape, graph_options));
        SA_TRY_ASSIGN(const Train::ObjectiveContext given, make_context(options, probe, dataset));
        SA_TRY_ASSIGN(run->objective_,
                      Train::make_objective(*options.objectives, options.train.objective_type,
                                            given, options.train.objective_attributes,
                                            K_OBJECTIVE_CONTEXT));

        SA_TRY_ASSIGN(run->built_, Train::build_training_graph(model, *run->objective_,
                                                               batch_shape, graph_options));
        SA_TRY(Train::match_roles(run->built_.leaves.required(), dataset));

        if (state == nullptr)
        {
            SA_TRY_ASSIGN(NN::ModelState created,
                          NN::ModelState::create(backend.engine(), model));
            SA_TRY(SushiAI::Detail::guard(SushiAI::Detail::SeamPhase::Allocate, [&]() -> Status
            {
                state = std::make_unique<NN::ModelState>(std::move(created));
                return Status();
            }));
        }

        SA_TRY_ASSIGN(std::vector<Optim::ParameterSlot> slots,
                      Train::bind_parameters(backend, run->built_, *state, run->pool_));
        SA_TRY_ASSIGN(std::vector<Train::BufferUpdateSlot> updates,
                      Train::bind_buffer_updates(run->built_.updates, *state, run->pool_));
        SA_TRY_ASSIGN(run->optimizer_, make_optimizer(kind, backend, std::move(slots),
                                                      options.train));

        Train::TrainerOptions trainer_options;
        trainer_options.track_metrics = kind == RunKind::Training;
        SA_TRY_ASSIGN(run->trainer_,
                      Train::Trainer::make(backend, run->built_, run->pool_, *run->objective_,
                                           *run->optimizer_, trainer_options,
                                           std::move(updates)));

        return run;
    }

    Train::Trainer& DecisionRun::trainer() noexcept
    {
        return *trainer_;
    }

    Result<const SushiBLAS::Tensor*> DecisionRun::output(std::string_view name) const noexcept
    {
        SA_TRY_ASSIGN(const Graph::ValueID id, built_.outputs.at(name));

        return &pool_.tensor(id);
    }
} // namespace SushiAI::Language::DecisionModels
