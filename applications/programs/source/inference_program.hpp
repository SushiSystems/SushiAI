/**************************************************************************/
/* inference_program.hpp                                                  */
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
 * @file inference_program.hpp
 * @brief Declares the assembly `sushiai inference` runs once it holds the parsed files.
 * @author Mustafa Garip
 * @see applications/programs/README.md, "inference_program.hpp".
 */

#pragma once

#include <filesystem>
#include <SushiAI/config/model_config.hpp>
#include <SushiAI/config/train_config.hpp>
#include <SushiAI/execution/execution_backend.hpp>
#include <SushiAI/failure/status.hpp>

namespace SushiAI
{
    namespace Apps
    {
        /**
         * @brief Evaluates a checkpoint over a dataset and prints its report.
         * @param model_spec The parsed model file.
         * @param data_spec The parsed evaluation file.
         * @param backend Runs the forward pass; its engine allocates the tensors.
         * @param detection The thresholds the file states; an unset one keeps the report's default.
         * @return Success, or the first failure; InvalidDescription for data the model refuses.
         */
        [[nodiscard]] Status run_inference_file(const std::filesystem::path& model_path,
                                                const std::filesystem::path& checkpoint_path,
                                                const Config::ModelSpec& model_spec,
                                                const Config::DataSpec& data_spec,
                                                ExecutionBackend& backend,
                                                const Config::DetectionSettings& detection = {});
    } // namespace Apps
} // namespace SushiAI
