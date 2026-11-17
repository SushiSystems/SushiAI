/**************************************************************************/
/* calibrate_decision_model.cpp                                           */
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
 * @file calibrate_decision_model.cpp
 * @brief Implements the calibration step of a decision model.
 * @author Mustafa Garip
 */

#include <SushiAI/language/decision_models/calibrate_decision_model.hpp>

#include <fstream>
#include <string>

#include <SushiAI/config/model_writer.hpp>
#include <SushiAI/seam/seam.hpp>

namespace SushiAI::Language::DecisionModels
{
    namespace
    {
        /** @brief Replaces the file at @p file with @p text. */
        [[nodiscard]] Status write_text(const std::filesystem::path& file,
                                        const std::string& text) noexcept
        {
            return SushiAI::Detail::guard(SushiAI::Detail::SeamPhase::Allocate, [&]() -> Status
            {
                std::ofstream out(file, std::ios::binary | std::ios::trunc);
                out.write(text.data(), static_cast<std::streamsize>(text.size()));
                out.flush();
                SA_FAIL_IF(!out, StatusCode::InvalidFile, "the model file '{}' cannot be written",
                           file.string());

                return Status();
            });
        }
    } // namespace

    Result<DecisionCalibration> calibrate_decision_model(DecisionSession& session,
                                                         const Data::Dataset& calibration_split,
                                                         Config::ModelSpec& spec) noexcept
    {
        SA_TRY_ASSIGN(const DecisionPredictions predictions, session.predict(calibration_split));
        SA_TRY_ASSIGN(const DecisionCalibration fitted, fit_decision_calibration(predictions));
        SA_TRY(store_calibration(spec, fitted));

        return fitted;
    }

    Result<DecisionCalibration> calibrate_decision_model(
        DecisionSession& session, const Data::Dataset& calibration_split,
        const std::filesystem::path& model_file) noexcept
    {
        SA_TRY_ASSIGN(Config::ModelSpec spec, Config::load_model(model_file));
        SA_TRY_ASSIGN(const DecisionCalibration fitted,
                      calibrate_decision_model(session, calibration_split, spec));
        SA_TRY_ASSIGN(const std::string text, Config::write_model(spec));
        SA_TRY(write_text(model_file, text));

        return fitted;
    }
} // namespace SushiAI::Language::DecisionModels
