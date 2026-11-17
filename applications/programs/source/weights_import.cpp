/**************************************************************************/
/* weights_import.cpp                                                     */
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
 * @file weights_import.cpp
 * @brief Defines `sushiai import weights` over the model records, the name map and the weight source.
 * @author Mustafa Garip
 */

#include "weights_import.hpp"

#include <memory>
#include <vector>
#include <SushiAI/config/model_config.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/io/imported_model.hpp>
#include <SushiAI/io/imported_model_writer.hpp>
#include <SushiAI/io/name_map.hpp>
#include <SushiAI/io/npz_weight_source.hpp>
#include <SushiAI/io/safetensors_weight_source.hpp>
#include <SushiAI/io/torch_weight_source.hpp>
#include <SushiAI/io/weight_import.hpp>
#include <SushiAI/io/weight_source.hpp>
#include <SushiAI/nn/model_state.hpp>
#include "model_state_entries.hpp"

namespace SushiAI
{
    namespace Apps
    {
        Result<IO::WeightSourceRegistry> compose_weight_sources()
        {
            IO::WeightSourceRegistry sources;
            // The npz row goes before the torch row: both are zips, and only npz reads names.
            SA_TRY(sources.add(IO::K_NPZ_WEIGHT_SOURCE_ROW));
            SA_TRY(sources.add(IO::K_TORCH_WEIGHT_SOURCE_ROW));
            SA_TRY(sources.add(IO::K_SAFETENSORS_WEIGHT_SOURCE_ROW));
            return sources;
        }

        Result<WeightsImportReport> import_weights_file(const WeightsImportRequest& request,
                                                        const Registries& registries)
        {
            SA_TRY_ASSIGN(const Config::ModelSpec model_spec, Config::load_model(request.model));
            SA_TRY_ASSIGN(const std::vector<NN::StateEntry> records,
                          model_state_entries(registries.nodes, model_spec));
            SA_TRY_ASSIGN(const IO::NameMap map, IO::NameMap::read(request.map));

            SA_TRY_ASSIGN(const IO::WeightSourceRegistry sources, compose_weight_sources());
            IO::WeightSourceOptions options;
            options.state_key = map.state_key();
            SA_TRY_ASSIGN(const std::unique_ptr<IO::WeightSource> source,
                          IO::open_weight_file(sources, request.weights, options));

            SA_TRY_ASSIGN(const std::vector<IO::ImportedTensor> tensors,
                          IO::import_state_weights(
                              SushiRuntime::span<const NN::StateEntry>(records.data(),
                                                                       records.size()),
                              *source, map));
            SA_TRY(IO::write_imported_weights(
                SushiRuntime::span<const IO::ImportedTensor>(tensors.data(), tensors.size()),
                request.output, request.existing));

            WeightsImportReport report;
            report.written = tensors.size();
            report.dropped = source->tensors().size() - tensors.size();
            return report;
        }

    } // namespace Apps
} // namespace SushiAI
