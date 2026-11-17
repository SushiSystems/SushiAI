/**************************************************************************/
/* decision_calibration.cpp                                               */
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
 * @file decision_calibration.cpp
 * @brief Implements the fit of the two temperatures and their reading and writing in a model file.
 * @author Mustafa Garip
 */

#include <SushiAI/language/decision_models/decision_calibration.hpp>

#include <array>
#include <cmath>
#include <string>
#include <utility>
#include <vector>

#include <SushiRuntime/core/span.hpp>

#include <SushiAI/config/value.hpp>
#include <SushiAI/language/training/temperature_scaling.hpp>
#include <SushiAI/seam/seam.hpp>

namespace SushiAI::Language::DecisionModels
{
    namespace
    {
        /** @brief Names the task in a message. */
        constexpr std::string_view K_TASK_CONTEXT = "the model file's 'task'";

        /** @brief Names the calibration object in a message. */
        constexpr std::string_view K_CALIBRATION_CONTEXT = "the 'task' object's 'calibration'";

        /** @brief Names the key a task states its type under. */
        constexpr std::string_view K_TYPE_KEY = "type";

        /** @brief Lists every key the calibration object accepts. */
        constexpr std::array<std::string_view, 2> K_TEMPERATURE_KEYS = {K_CHOICE_TEMPERATURE_KEY,
                                                                        K_YES_NO_TEMPERATURE_KEY};

        /** @brief Fails unless @p temperature is a number fit_temperature() could have returned. */
        [[nodiscard]] Status check_temperature(double temperature, std::string_view key) noexcept
        {
            SA_FAIL_IF(!std::isfinite(temperature)
                           || temperature < Training::K_MIN_TEMPERATURE
                           || temperature > Training::K_MAX_TEMPERATURE,
                       StatusCode::InvalidDescription,
                       "the temperature '{}' is {}, which is outside [{}, {}]", key, temperature,
                       Training::K_MIN_TEMPERATURE, Training::K_MAX_TEMPERATURE);

            return Status();
        }

        /** @brief Fits the temperature of one head over @p rows. */
        [[nodiscard]] Result<double> fit_head(const LogitRows& rows) noexcept
        {
            return Training::fit_temperature(
                SushiRuntime::span<const float>(rows.logits.data(), rows.logits.size()),
                SushiRuntime::span<const std::int32_t>(rows.labels.data(), rows.labels.size()),
                rows.classes);
        }

        /** @brief Reads the temperature stored under @p key of @p calibration. */
        [[nodiscard]] Result<double> read_temperature(const Config::Value& calibration,
                                                      std::string_view key) noexcept
        {
            SA_TRY_ASSIGN(const Config::Value& entry, calibration.at(key, K_CALIBRATION_CONTEXT));
            SA_TRY_ASSIGN(const double temperature, entry.as_real(key));
            SA_TRY(check_temperature(temperature, key));

            return temperature;
        }

        /** @brief Fails unless the task of @p spec is of the type "decision". */
        [[nodiscard]] Status check_task_type(const Config::Value& task) noexcept
        {
            SA_FAIL_IF(!task.has(K_TYPE_KEY), StatusCode::InvalidDescription,
                       "{} states no type, so it is no decision task", K_TASK_CONTEXT);
            SA_TRY_ASSIGN(const Config::Value& type, task.at(K_TYPE_KEY, K_TASK_CONTEXT));
            SA_TRY_ASSIGN(const std::string& name, type.as_string(K_TASK_CONTEXT));
            SA_FAIL_IF(name != K_DECISION_TASK_TYPE, StatusCode::InvalidDescription,
                       "{} is of the type '{}', not '{}'", K_TASK_CONTEXT, name,
                       K_DECISION_TASK_TYPE);

            return Status();
        }
    } // namespace

    Result<DecisionCalibration> fit_decision_calibration(
        const DecisionPredictions& predictions) noexcept
    {
        SA_TRY_ASSIGN(const LogitRows choice, choice_rows(predictions));
        SA_TRY_ASSIGN(const LogitRows yes_no, yes_no_rows(predictions));

        DecisionCalibration fitted;
        SA_TRY_ASSIGN(fitted.choice, fit_head(choice));
        SA_TRY_ASSIGN(fitted.yes_no, fit_head(yes_no));

        return fitted;
    }

    Status store_calibration(Config::ModelSpec& spec, const DecisionCalibration& calibration) noexcept
    {
        SA_TRY(check_temperature(calibration.choice, K_CHOICE_TEMPERATURE_KEY));
        SA_TRY(check_temperature(calibration.yes_no, K_YES_NO_TEMPERATURE_KEY));
        if (spec.task.has(K_TYPE_KEY))
            SA_TRY(check_task_type(spec.task));

        return SushiAI::Detail::guard(SushiAI::Detail::SeamPhase::Allocate, [&]() -> Status
        {
            std::vector<std::string> keys;
            std::vector<Config::Value> values;
            if (spec.task.has(K_TYPE_KEY))
            {
                for (std::size_t index = 0; index < spec.task.keys().size(); ++index)
                {
                    if (spec.task.keys()[index] == K_CALIBRATION_KEY)
                        continue;

                    keys.push_back(spec.task.keys()[index]);
                    values.push_back(spec.task.values()[index]);
                }
            }
            else
            {
                keys.push_back(std::string(K_TYPE_KEY));
                values.push_back(Config::Value::string(std::string(K_DECISION_TASK_TYPE)));
            }

            keys.push_back(std::string(K_CALIBRATION_KEY));
            values.push_back(Config::Value::object(
                {std::string(K_CHOICE_TEMPERATURE_KEY), std::string(K_YES_NO_TEMPERATURE_KEY)},
                {Config::Value::real(calibration.choice), Config::Value::real(calibration.yes_no)}));
            spec.task = Config::Value::object(std::move(keys), std::move(values));

            return Status();
        });
    }

    Result<DecisionCalibration> read_calibration(const Config::ModelSpec& spec) noexcept
    {
        SA_TRY(check_task_type(spec.task));
        if (!spec.task.has(K_CALIBRATION_KEY))
            return DecisionCalibration{};

        SA_TRY_ASSIGN(const Config::Value& stored, spec.task.at(K_CALIBRATION_KEY, K_TASK_CONTEXT));
        SA_TRY(stored.reject_unknown_keys(
            K_CALIBRATION_CONTEXT,
            SushiRuntime::span<const std::string_view>(K_TEMPERATURE_KEYS.data(),
                                                       K_TEMPERATURE_KEYS.size())));

        DecisionCalibration read;
        SA_TRY_ASSIGN(read.choice, read_temperature(stored, K_CHOICE_TEMPERATURE_KEY));
        SA_TRY_ASSIGN(read.yes_no, read_temperature(stored, K_YES_NO_TEMPERATURE_KEY));

        return read;
    }
} // namespace SushiAI::Language::DecisionModels
