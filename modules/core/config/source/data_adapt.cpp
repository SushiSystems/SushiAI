/**************************************************************************/
/* data_adapt.cpp                                                         */
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
 * @file data_adapt.cpp
 * @brief Implements how a data block takes the image fit and the normalisation its model states.
 * @author Mustafa Garip
 */

#include <SushiAI/config/data_adapt.hpp>

#include <SushiAI/config/data_fit.hpp>
#include <SushiAI/config/data_normalise.hpp>
#include <SushiAI/failure/status.hpp>

namespace SushiAI
{
    namespace Config
    {
        Result<Value> adapt_data_to_model(const Value& data, const ModelSpec& model,
                                          std::string_view model_name) noexcept
        {
            SA_TRY_ASSIGN(const Value fitted, fit_data_to_model(data, model, model_name));
            return normalise_data_to_model(fitted, model, model_name);
        }

    } // namespace Config
} // namespace SushiAI
