/**************************************************************************/
/* data_normalise.cpp                                                     */
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
 * @file data_normalise.cpp
 * @brief Implements how a data block takes the input normalisation its model file states.
 * @author Mustafa Garip
 */

#include <SushiAI/config/data_normalise.hpp>

#include <string>
#include <utility>
#include <vector>

#include <SushiAI/config/input_normalise.hpp>
#include <SushiAI/failure/status.hpp>

namespace SushiAI
{
    namespace Config
    {
        namespace
        {
            /** @brief Names the key of the normalisation in a model's input and in a data block. */
            constexpr std::string_view K_NORMALISE_KEY = "normalise";

            /** @brief Names the data block's normalisation in a diagnostic. */
            constexpr std::string_view K_DATA_NORMALISE = "the data block's 'normalise'";
        } // namespace

        Result<Value> normalise_data_to_model(const Value& data, const ModelSpec& model,
                                              std::string_view model_name) noexcept
        {
            if (!model.input_normalise.has_value())
                return data;

            if (data.has(K_NORMALISE_KEY))
            {
                SA_TRY_ASSIGN(const Value& stated, data.at(K_NORMALISE_KEY, K_DATA_NORMALISE));
                SA_TRY_ASSIGN(const InputNormalise own,
                              parse_input_normalise(stated, K_DATA_NORMALISE));

                SA_FAIL_IF(own != *model.input_normalise, StatusCode::InvalidDescription,
                           "'{}' states a 'normalise' for its images, and {} states another",
                           model_name, K_DATA_NORMALISE);
                return data;
            }

            std::vector<std::string> keys = data.keys();
            std::vector<Value> values = data.values();
            values.reserve(keys.size() + 1);

            keys.emplace_back(K_NORMALISE_KEY);
            values.push_back(input_normalise_value(*model.input_normalise));
            return Value::object(std::move(keys), std::move(values));
        }

    } // namespace Config
} // namespace SushiAI
