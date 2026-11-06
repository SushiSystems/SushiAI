/**************************************************************************/
/* data_fit.hpp                                                           */
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
 * @file data_fit.hpp
 * @brief Declares how a data block takes the image fit its model file states.
 * @author Mustafa Garip
 */

#pragma once

#include <string_view>

#include <SushiAI/config/model_config.hpp>
#include <SushiAI/config/value.hpp>
#include <SushiAI/failure/result.hpp>

namespace SushiAI
{
    namespace Config
    {
        /**
         * @brief Returns @p data with the fit of @p model under "fit".
         * @param model_name Names the model file in a diagnostic.
         * @return @p data unchanged when the model states no fit or the block states the same
         *         one; the block with "fit" added when it states none; InvalidDescription naming
         *         both fits when the block states another.
         */
        [[nodiscard]] Result<Value> fit_data_to_model(const Value& data, const ModelSpec& model,
                                                      std::string_view model_name) noexcept;

    } // namespace Config
} // namespace SushiAI
