/**************************************************************************/
/* model_state_entries.cpp                                                */
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
 * @file model_state_entries.cpp
 * @brief Defines the listing of the state records of a traced model, without an engine.
 * @author Mustafa Garip
 */

#include "model_state_entries.hpp"

#include <cstdint>
#include <vector>
#include <SushiAI/config/model_config.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/nn/model_graph.hpp>
#include <SushiAI/nn/model_state.hpp>
#include <SushiAI/nn/module.hpp>
#include <SushiAI/nn/node_registry.hpp>
#include <SushiAI/shape/shape.hpp>
#include <SushiAI/train/steps.hpp>

namespace SushiAI
{
    namespace Apps
    {
        namespace
        {
            /** @brief Holds the rows of the sample the model is traced over. */
            constexpr std::int64_t K_TRACE_ROWS = 1;
        } // namespace

        Result<std::vector<NN::StateEntry>> model_state_entries(
            const NN::NodeRegistry& registry, const Config::ModelSpec& model_spec)
        {
            SA_TRY_ASSIGN(NN::ModelGraph model, NN::build_model(registry, model_spec));
            SA_TRY_ASSIGN(const Shape batch_shape,
                          Train::batch_shape(K_TRACE_ROWS, model_spec.input_shape));
            Train::GraphOptions options;
            options.dtype = model_spec.input_dtype;
            SA_TRY_ASSIGN(const Train::TracedStep traced,
                          Train::build_evaluation_graph(model, batch_shape, options));
            static_cast<void>(traced);

            return NN::list_state_entries(model);
        }

    } // namespace Apps
} // namespace SushiAI
