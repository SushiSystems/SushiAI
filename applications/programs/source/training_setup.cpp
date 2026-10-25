/**************************************************************************/
/* training_setup.cpp                                                     */
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
 * @file training_setup.cpp
 * @brief Implements TrainingSetup, which builds what a training file describes on one engine.
 * @author Mustafa Garip
 */

#include "training_setup.hpp"

#include <cstdint>
#include <memory>
#include <string_view>
#include <utility>
#include <SushiAI/config/data_fit.hpp>
#include <SushiAI/data/dataset_registry.hpp>
#include <SushiAI/data/roles.hpp>
#include <SushiAI/data/target.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/nn/node_registry.hpp>
#include <SushiAI/train/objective_registry.hpp>
#include <SushiAI/train/role_match.hpp>
#include <SushiAI/train/steps.hpp>
#include <SushiAI/train/trainer.hpp>

namespace SushiAI
{
    namespace Apps
    {
        namespace
        {
            /** @brief Lists the roles whose description states the class count of a dataset. */
            constexpr std::string_view K_CLASS_ROLES[] = {Data::K_ROLE_LABELS,
                                                          Data::K_ROLE_CLASSES};

            /** @brief Returns the class count the first class role of @p dataset states, else 0. */
            [[nodiscard]] std::int64_t class_count_of(const Data::Dataset& dataset) noexcept
            {
                for (const std::string_view role : K_CLASS_ROLES)
                {
                    const Data::TargetDescription* const described =
                        Data::find_target(dataset.targets(), role);
                    if (described != nullptr)
                        return described->classes;
                }

                return 0;
            }

        } // namespace

        TrainingSetup::TrainingSetup(std::unique_ptr<Data::Dataset> read_dataset,
                                     NN::ModelGraph built_model, Train::ObjectiveContext context,
                                     SushiBLAS::Engine& engine) noexcept
            : dataset_(std::move(read_dataset)), model_(std::move(built_model)),
              given_(std::move(context)), pool_(std::make_unique<Graph::TensorPool>(engine))
        {

        }

        Result<std::unique_ptr<TrainingSetup>> TrainingSetup::make(
            const Registries& registries, const Config::TrainSpec& spec,
            const Config::ModelSpec& model_spec, SushiBLAS::Engine& engine,
            std::uint64_t seed, std::optional<std::int64_t> rows, SetupScope scope)
        {
            SA_TRY_ASSIGN(const Config::Value data_block,
                          Config::fit_data_to_model(spec.data.attributes, model_spec,
                                                    spec.model_path.string()));
            SA_TRY_ASSIGN(std::unique_ptr<Data::Dataset> dataset,
                          Data::make_dataset(registries.datasets, engine, spec.data.type,
                                             data_block, "the data block"));

            const Shape sample_shape = dataset->sample_shape();
            SA_FAIL_IF(sample_shape != model_spec.input_shape, StatusCode::InvalidDescription,
                       "'{}' declares the input shape {}, but the '{}' data supplies {}",
                       spec.model_path.string(), model_spec.input_shape.to_string(),
                       dataset->name(), sample_shape.to_string());

            SA_TRY_ASSIGN(NN::ModelGraph model, NN::build_model(registries.nodes, model_spec));

            const std::int64_t batch_rows = rows ? *rows : dataset->batch_size();
            SA_TRY_ASSIGN(const Shape batch_shape, Train::batch_shape(batch_rows, sample_shape));

            // The probe trace states the published shapes an objective factory may read.
            Train::GraphOptions graph_options;
            graph_options.dtype = model_spec.input_dtype;
            graph_options.seed = seed;
            SA_TRY_ASSIGN(const Train::TracedStep probe,
                          Train::build_probe_graph(model, batch_shape, graph_options));

            Train::ObjectiveContext given;
            given.batch = batch_rows;
            given.classes = class_count_of(*dataset);
            given.input_sample = sample_shape;
            given.task = model_spec.task;
            SA_TRY_ASSIGN(given.outputs, Train::list_output_shapes(probe.graph, probe.outputs));

            std::unique_ptr<TrainingSetup> setup(new TrainingSetup(
                std::move(dataset), std::move(model), std::move(given), engine));

            SA_TRY_ASSIGN(setup->objective_,
                          Train::make_objective(registries.objectives, spec.objective_type,
                                                setup->given_, spec.objective_attributes,
                                                "the objective block"));

            SA_TRY_ASSIGN(setup->built_,
                          Train::build_training_graph(setup->model_, *setup->objective_,
                                                      batch_shape, graph_options));
            SA_TRY(Train::match_roles(setup->built_.leaves.required(), *setup->dataset_));

            // The trace above declared every parameter and buffer the state lists.
            SA_TRY_ASSIGN(NN::ModelState state, NN::ModelState::create(engine, setup->model_));
            setup->state_ = std::make_unique<NN::ModelState>(std::move(state));

            if (scope == SetupScope::Step)
            {
                SA_TRY_ASSIGN(setup->parameters_,
                              Train::bind_parameters(setup->built_, *setup->state_,
                                                     *setup->pool_));
            }
            else
            {
                SA_TRY_ASSIGN(setup->parameters_,
                              Train::bind_step_results(setup->built_, *setup->state_,
                                                       *setup->pool_));
            }

            SA_TRY_ASSIGN(setup->updates_,
                          Train::bind_buffer_updates(setup->built_.updates, *setup->state_,
                                                     *setup->pool_));

            return Result<std::unique_ptr<TrainingSetup>>(std::move(setup));
        }

        const Data::Dataset& TrainingSetup::dataset() const noexcept
        {
            return *dataset_;
        }

        const NN::ModelGraph& TrainingSetup::model() const noexcept
        {
            return model_;
        }

        const Train::ObjectiveContext& TrainingSetup::given() const noexcept
        {
            return given_;
        }

        const Train::Objective& TrainingSetup::objective() const noexcept
        {
            return *objective_;
        }

        const Train::TrainingGraph& TrainingSetup::built() const noexcept
        {
            return built_;
        }

        NN::ModelState& TrainingSetup::state() noexcept
        {
            return *state_;
        }

        Graph::TensorPool& TrainingSetup::pool() noexcept
        {
            return *pool_;
        }

        const std::vector<Optim::ParameterSlot>& TrainingSetup::parameters() const noexcept
        {
            return parameters_;
        }

        const std::vector<Train::BufferUpdateSlot>& TrainingSetup::updates() const noexcept
        {
            return updates_;
        }

        std::vector<Train::BufferUpdateSlot> TrainingSetup::take_updates() noexcept
        {
            return std::move(updates_);
        }

    } // namespace Apps
} // namespace SushiAI
