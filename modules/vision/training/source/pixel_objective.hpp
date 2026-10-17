/**************************************************************************/
/* pixel_objective.hpp                                                    */
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
 * @file pixel_objective.hpp
 * @brief Declares what the objectives over [N, H, W, C] logits and a mask share.
 * @author Mustafa Garip
 */

#pragma once

#include <cstdint>
#include <string_view>
#include <SushiAI/config/value.hpp>
#include <SushiAI/dtype/dtype.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/shape/shape.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/data/target.hpp>
#include <SushiAI/graph/builder.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/nn/model_outputs.hpp>
#include <SushiAI/train/objective_context.hpp>
#include <SushiAI/vision/training/pixel_extents.hpp>

namespace SushiAI::Vision::Training::Detail
{
    /** @brief Names the key that holds the index of a pixel of no class. */
    inline constexpr std::string_view K_IGNORE_INDEX_KEY = "ignore_index";

    /**
     * @brief Returns InvalidDescription unless the leaves of a per-pixel objective can be made.
     * @param largest The most elements N * H * W * C may hold.
     * @param objective Names the objective in the message.
     * @return InvalidDescription for an extent below 1, a product above @p largest, an ignore
     *         index that names a class, or an element type that is neither float32 nor float64.
     */
    [[nodiscard]] Status validate_pixel_objective(const PixelExtents& extents,
                                                  std::int32_t ignore_index, DType dtype,
                                                  std::int64_t largest,
                                                  std::string_view objective);

    /** @brief Returns the description of the mask role: int32 [H, W] of any class count. */
    [[nodiscard]] Data::TargetDescription describe_mask_role(const PixelExtents& extents);

    /**
     * @brief Emits the only output of @p outputs viewed as [N * H * W, C] rows.
     * @pre @p outputs holds one value of the extents given; anything else stops the process,
     *      naming @p objective.
     */
    [[nodiscard]] Graph::ValueID emit_pixel_rows(const Graph::GraphBuilder& builder,
                                                 const Graph::Graph& graph,
                                                 const NN::ModelOutputs& outputs,
                                                 const PixelExtents& extents,
                                                 std::string_view objective);

    /**
     * @brief Returns the extents of the one [N, H, W, C] output of @p given.
     * @param context What to call the objective in a diagnostic.
     * @return InvalidDescription for a model that publishes another count of outputs, one of
     *         another rank or batch, one whose H and W are not those of a [H, W, channels]
     *         input, or for data of more classes than C.
     */
    [[nodiscard]] Result<PixelExtents> read_pixel_extents(const Train::ObjectiveContext& given,
                                                          std::string_view context);

    /**
     * @brief Reads K_IGNORE_INDEX_KEY, the only key of @p attributes, or the default.
     * @return Segment::K_DEFAULT_IGNORE_INDEX when the key is absent; InvalidDescription for
     *         another key, or for a value that is not a whole number an int32 holds.
     */
    [[nodiscard]] Result<std::int32_t> read_ignore_index(const Config::Value& attributes,
                                                         std::string_view context);

} // namespace SushiAI::Vision::Training::Detail
