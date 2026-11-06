/**************************************************************************/
/* anchor_objective_factory.hpp                                           */
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
 * @file anchor_objective_factory.hpp
 * @brief Declares the factory that builds an anchor objective from the files of a run.
 * @author Mustafa Garip
 */

#pragma once

#include <memory>
#include <string_view>

#include <SushiAI/config/value.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/train/objective.hpp>
#include <SushiAI/train/objective_context.hpp>
#include <SushiAI/vision/training/anchor_objective.hpp>

namespace SushiAI::Vision::Training
{
    /**
     * @brief Builds the anchor objective from the model's task and the "anchor" section.
     * @param given Its task is an anchor_detection section, its input sample [H, W, C], and
     *        its outputs hold each head of the task as [batch, H, W, A * (5 + C)].
     * @param attributes The objective section, with or without its "type".
     * @return The objective; InvalidDescription for no task or a head the model does not
     *         publish; Incompatible for other extents or classes; or a reader's refusal.
     */
    [[nodiscard]] Result<AnchorObjective> build_anchor_objective(
        const Train::ObjectiveContext& given, const Config::Value& attributes) noexcept;

    /**
     * @brief Builds the anchor objective as a row of a Train::ObjectiveRegistry builds it.
     * @param context What to call this objective in a diagnostic; it prefixes a refusal.
     * @return The objective, never null, or the failure of build_anchor_objective.
     */
    [[nodiscard]] Result<std::unique_ptr<Train::Objective>> make_anchor_objective(
        const Train::ObjectiveContext& given, const Config::Value& attributes,
        std::string_view context) noexcept;

} // namespace SushiAI::Vision::Training
