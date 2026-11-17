/**************************************************************************/
/* inference_preparation.cpp                                              */
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
 * @file inference_preparation.cpp
 * @brief Defines the build `sushiai inference` runs before a report starts.
 * @author Mustafa Garip
 * @see applications/programs/README.md, "inference_preparation.hpp".
 */

#include "inference_preparation.hpp"

#include <cstdint>
#include <utility>
#include <vector>
#include <SushiAI/config/data_adapt.hpp>
#include <SushiAI/config/value.hpp>
#include <SushiAI/io/checkpoint.hpp>
#include <SushiAI/profile/profile.hpp>
#include <SushiAI/shape/shape.hpp>

namespace SushiAI
{
    namespace Apps
    {
        InferenceBuild::InferenceBuild(ExecutionBackend& backend) : pool_(backend.engine())
        {

        }

        PreparedInference& InferenceBuild::prepared() noexcept
        {
            return *prepared_;
        }

        Result<std::unique_ptr<InferenceBuild>> prepare_inference(const Registries& registries,
                                                                  ExecutionBackend& backend,
                                                                  const InferenceRequest& request)
        {
            std::unique_ptr<InferenceBuild> made(new InferenceBuild(backend));
            const Config::ModelSpec& model_spec = request.model;

            SA_TRY_ASSIGN(const Config::Value data_block,
                          Config::adapt_data_to_model(request.data.attributes, model_spec,
                                                      request.model_path.string()));
            SA_TRY_ASSIGN(std::unique_ptr<Data::Dataset> read_dataset,
                          Data::make_dataset(registries.datasets, backend.engine(),
                                             request.data.type, data_block, "the data block"));
            made->dataset_ = request.hooks.wrap_dataset
                ? request.hooks.wrap_dataset(std::move(read_dataset))
                : std::move(read_dataset);

            const Shape sample_shape = made->dataset_->sample_shape();
            SA_FAIL_IF(sample_shape != model_spec.input_shape,
                       StatusCode::InvalidDescription,
                       "'{}' declares the input shape {}, but the '{}' data supplies {}",
                       request.model_path.string(), model_spec.input_shape.to_string(),
                       made->dataset_->name(), sample_shape.to_string());

            // Forward pass only: no loss, backward or optimizer; see the programs README.
            SA_TRY_ASSIGN(NN::ModelGraph model, NN::build_model(registries.nodes, model_spec));
            made->graph_.emplace(std::move(model));
            const std::int64_t rows = made->dataset_->batch_size();

            SA_TRY_ASSIGN(const Shape batch_shape, Train::batch_shape(rows, sample_shape));
            // The model file states the element type of the input leaf; int32 reads indices.
            Train::GraphOptions graph_options;
            graph_options.dtype = model_spec.input_dtype;
            SA_TRY_ASSIGN(Train::TracedStep traced,
                          Train::build_evaluation_graph(*made->graph_, batch_shape, graph_options));
            made->traced_.emplace(std::move(traced));

            // A model without a task is a classifier with one output; a task names its own.
            const bool tasked = !model_spec.task.is_null();
            const std::vector<NN::ModelOutputs::Entry>& published = made->traced_->outputs.all();
            SA_FAIL_IF(!tasked && published.size() != 1, StatusCode::InvalidDescription,
                       "'{}' publishes {} outputs, but the report counts classes and reads one",
                       request.model_path.string(), published.size());

            SA_TRY_ASSIGN(NN::ModelState state,
                          NN::ModelState::create(backend.engine(), *made->graph_));
            made->state_.emplace(std::move(state));
            if (request.hooks.fill_state)
            {
                SA_TRY(request.hooks.fill_state(backend, *made->state_));
            }
            else
            {
                SA_ZONE("io", "checkpoint");
                SA_TRY(IO::load_checkpoint(request.weights_path, *made->state_));
            }

            SA_TRY_ASSIGN(made->step_, Train::EvaluationStep::make(backend, *made->traced_,
                                                                   *made->state_, made->pool_));

            // The report is the row of the task type; see applications/programs/README.md.
            made->prepared_.emplace(PreparedInference{
                request.model_path, request.weights_path, model_spec, made->graph_->size(),
                *made->dataset_, *made->step_, published, backend, &*made->graph_,
                &*made->state_, request.detection, request.comparison,
            });
            return std::move(made);
        }
    } // namespace Apps
} // namespace SushiAI
