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
        const Train::ObjectiveContext& given, const Config::Value& attributes);

    /**
     * @brief Builds the anchor objective as a row of a Train::ObjectiveRegistry builds it.
     * @param context What to call this objective in a diagnostic; it prefixes a refusal.
     * @return The objective, never null, or the failure of build_anchor_objective.
     */
    [[nodiscard]] Result<std::unique_ptr<Train::Objective>> make_anchor_objective(
        const Train::ObjectiveContext& given, const Config::Value& attributes,
        std::string_view context);

} // namespace SushiAI::Vision::Training
