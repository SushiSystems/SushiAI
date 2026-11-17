/**************************************************************************/
/* model_state_entries.hpp                                                */
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
 * @file model_state_entries.hpp
 * @brief Declares the listing of the state records a model file states, without an engine.
 * @author Mustafa Garip
 * @see docs/design/MODEL_IMPORT/README.md
 */

#pragma once

#include <vector>
#include <SushiAI/config/model_config.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/nn/model_state.hpp>
#include <SushiAI/nn/node_registry.hpp>

namespace SushiAI
{
    namespace Apps
    {
        /**
         * @brief Builds the model @p model_spec describes, traces it once over one sample and
         *        lists every parameter and buffer it declares.
         * @param registry The rows a type of @p model_spec may name.
         * @param model_spec The parsed model file.
         * @return The entries, in the order `ModelState::create` would list them, with no tensor
         *         allocated; or the failure of the build or the trace.
         */
        [[nodiscard]] Result<std::vector<NN::StateEntry>> model_state_entries(
            const NN::NodeRegistry& registry, const Config::ModelSpec& model_spec);

    } // namespace Apps
} // namespace SushiAI
