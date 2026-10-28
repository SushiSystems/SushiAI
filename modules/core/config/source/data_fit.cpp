/**************************************************************************/
/* data_fit.cpp                                                           */
/**************************************************************************/
/*                          This file is part of:                         */
/*                                 SushiAI                                */
/*                 https://github.com/SushiSystems/SushiAI                */
/*                         https://sushisystems.io                        */
/**************************************************************************/
/* Copyright (c) 2026-present Mustafa Garip & Sushi Systems               */
/*                                                                        */
/* Licensed under the PolyForm Noncommercial License 1.0.0 (the           */
/* "License"); you may not use this file except in compliance with the    */
/* License. You may obtain a copy of the License at                       */
/*                                                                        */
/*     https://polyformproject.org/licenses/noncommercial/1.0.0           */
/*                                                                        */
/* Noncommercial use is free. Commercial use requires a separate licence  */
/* from Sushi Systems; see COMMERCIAL.md. The software is provided        */
/* "as is", without warranty of any kind.                                 */
/**************************************************************************/

/**
 * @file data_fit.cpp
 * @brief Implements how a data block takes the image fit its model file states.
 * @author Mustafa Garip
 */

#include <SushiAI/config/data_fit.hpp>

#include <string>
#include <utility>
#include <vector>

#include <SushiAI/failure/status.hpp>

namespace SushiAI
{
    namespace Config
    {
        namespace
        {
            /** @brief Names the key of the fit in a model's input and in a data block. */
            constexpr std::string_view K_FIT_KEY = "fit";

            /** @brief Names the data block in a diagnostic. */
            constexpr std::string_view K_DATA_BLOCK = "the data block";
        } // namespace

        Result<Value> fit_data_to_model(const Value& data, const ModelSpec& model,
                                        std::string_view model_name)
        {
            if (model.input_fit.empty())
                return data;

            if (data.has(K_FIT_KEY))
            {
                SA_TRY_ASSIGN(const Value& stated, data.at(K_FIT_KEY, K_DATA_BLOCK));
                bool same = false;
                if (stated.kind() == Kind::String)
                {
                    SA_TRY_ASSIGN(const std::string& fit, stated.as_string(K_DATA_BLOCK));
                    same = fit == model.input_fit;
                }

                SA_FAIL_IF(!same, StatusCode::InvalidDescription,
                           "'{}' fits its images by '{}', and {} states another 'fit'", model_name,
                           model.input_fit, K_DATA_BLOCK);
                return data;
            }

            std::vector<std::string> keys = data.keys();
            std::vector<Value> values = data.values();
            values.reserve(keys.size() + 1);

            keys.emplace_back(K_FIT_KEY);
            values.push_back(Value::string(model.input_fit));
            return Value::object(std::move(keys), std::move(values));
        }

    } // namespace Config
} // namespace SushiAI
