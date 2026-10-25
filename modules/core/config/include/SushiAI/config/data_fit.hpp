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
                                                      std::string_view model_name);

    } // namespace Config
} // namespace SushiAI
