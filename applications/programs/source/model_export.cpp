/**************************************************************************/
/* model_export.cpp                                                       */
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
 * @file model_export.cpp
 * @brief Defines `sushiai export <format>` over the model files, the exporter rows and the writer.
 * @author Mustafa Garip
 */

#include "model_export.hpp"

#include <utility>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/io/exported_model_writer.hpp>
#include <SushiAI/io/exporter_registry.hpp>
#include <SushiAI/io/imported_model.hpp>
#include <SushiAI/io/model_files.hpp>

namespace SushiAI
{
    namespace Apps
    {
        Result<ModelExportReport> export_model_files(const ModelExportRequest& request,
                                                     const Registries& registries)
        {
            IO::ModelFilePaths paths;
            paths.model = request.model;
            paths.weights = request.weights;
            paths.training = request.training;
            SA_TRY_ASSIGN(const IO::ImportedModel model, IO::read_model_files(paths));

            // The exporter is picked by string, not by type; see applications/programs/README.md.
            SA_TRY_ASSIGN(IO::ExportedModel exported,
                          IO::export_model(registries.exporters, request.format, model,
                                           request.name));

            ModelExportReport report;
            SA_TRY_ASSIGN(report.written,
                          IO::write_exported_model(exported, request.output, request.existing));
            report.notes = std::move(exported.notes);
            return report;
        }

    } // namespace Apps
} // namespace SushiAI
