/**************************************************************************/
/* calibrate_decision_model.hpp                                           */
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
 * @file calibrate_decision_model.hpp
 * @brief Declares the step that fits the temperatures of a trained decision model on its
 *        calibration split and stores them in the model file.
 * @author Mustafa Garip
 */

#pragma once

#include <filesystem>

#include <SushiAI/config/model_config.hpp>
#include <SushiAI/data/dataset.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/language/decision_models/decision_calibration.hpp>
#include <SushiAI/language/decision_models/decision_session.hpp>

namespace SushiAI::Language::DecisionModels
{
    /**
     * @brief Reads the answers of @p session to @p calibration_split, fits one temperature for
     *        each classifying head and stores both in @p spec.
     * @param calibration_split A split the model was not trained on.
     * @param spec The model file; its task is absent or of the type "decision".
     * @return The temperatures, or the failure of the read, the fit or store_calibration().
     */
    [[nodiscard]] Result<DecisionCalibration> calibrate_decision_model(
        DecisionSession& session, const Data::Dataset& calibration_split,
        Config::ModelSpec& spec) noexcept;

    /**
     * @brief Calibrates as the overload above does, over the model file at @p model_file, and
     *        writes the file back with the temperatures in its task.
     * @return The temperatures; FileNotFound or InvalidFile when the file cannot be read or is
     *         no model file, or when it cannot be written; or what the overload above returns.
     *         The file is left as it was when the fit fails.
     */
    [[nodiscard]] Result<DecisionCalibration> calibrate_decision_model(
        DecisionSession& session, const Data::Dataset& calibration_split,
        const std::filesystem::path& model_file) noexcept;
} // namespace SushiAI::Language::DecisionModels
