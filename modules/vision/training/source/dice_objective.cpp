/**************************************************************************/
/* dice_objective.cpp                                                     */
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
 * @file dice_objective.cpp
 * @brief Implements the soft Dice objective and the factory a training file reaches it by.
 * @author Mustafa Garip
 */

#include <SushiAI/vision/training/dice_objective.hpp>

#include <limits>
#include <utility>
#include <SushiAI/core/seam.hpp>
#include <SushiAI/core/shape.hpp>
#include <SushiAI/data/roles.hpp>
#include <SushiAI/data/target.hpp>
#include <SushiAI/train/batch_role.hpp>
#include <SushiAI/vision/training/dice_targets.hpp>
#include <SushiAI/vision/training/pixel_losses.hpp>
#include <SushiAI/vision/training/segmentation_metrics.hpp>

#include "pixel_objective.hpp"

namespace SushiAI::Vision::Training
{
    namespace
    {
        /** @brief Holds the largest index K_GATHER_ROWS addresses a table row by. */
        constexpr std::int64_t K_LARGEST_INDEX = std::numeric_limits<std::int32_t>::max();
    } // namespace

    Result<std::unique_ptr<DiceObjective>> DiceObjective::make(const PixelExtents& extents,
                                                               std::int32_t ignore_index,
                                                               DType dtype)
    {
        SA_TRY(Detail::validate_pixel_objective(extents, ignore_index, dtype, K_LARGEST_INDEX,
                                                K_DICE_TYPE));

        return SushiAI::Detail::guard_value(SushiAI::Detail::SeamPhase::Allocate, [&]()
        {
            return std::unique_ptr<DiceObjective>(
                new DiceObjective(extents, ignore_index, dtype));
        });
    }

    DiceObjective::DiceObjective(const PixelExtents& extents, std::int32_t ignore_index,
                                 DType dtype) noexcept
        : extents_(extents), ignore_index_(ignore_index), dtype_(dtype)
    {

    }

    void DiceObjective::declare(const Graph::GraphBuilder& builder,
                                Train::ObjectiveLeaves& leaves) const
    {
        const std::int64_t pixels = extents_.pixels();
        const std::int64_t classes = extents_.classes;
        leaves.add(K_DICE_SCORE_INDICES_LEAF,
                   builder.input(Shape{pixels}, K_DICE_SCORE_INDICES_LEAF, DType::Int32));
        leaves.add(K_DICE_CLASS_INDICES_LEAF,
                   builder.input(Shape{pixels}, K_DICE_CLASS_INDICES_LEAF, DType::Int32));
        leaves.add(K_DICE_KEPT_LEAF, builder.input(Shape{pixels, 1}, K_DICE_KEPT_LEAF, dtype_));
        leaves.add(K_DICE_CLASS_PIXELS_LEAF,
                   builder.input(Shape{classes}, K_DICE_CLASS_PIXELS_LEAF, dtype_));
        leaves.add(K_DICE_CLASS_WEIGHTS_LEAF,
                   builder.input(Shape{classes}, K_DICE_CLASS_WEIGHTS_LEAF, dtype_));
        leaves.require(Detail::describe_mask_role(extents_));
    }

    Graph::ValueID DiceObjective::loss(const Graph::GraphBuilder& builder,
                                       const Graph::Graph& graph,
                                       const NN::ModelOutputs& outputs,
                                       const Train::ObjectiveLeaves& leaves) const
    {
        const Graph::ValueID rows =
            Detail::emit_pixel_rows(builder, graph, outputs, extents_, K_DICE_TYPE);
        const DiceOperands operands{leaves.at(K_DICE_SCORE_INDICES_LEAF),
                                    leaves.at(K_DICE_CLASS_INDICES_LEAF),
                                    leaves.at(K_DICE_KEPT_LEAF),
                                    leaves.at(K_DICE_CLASS_PIXELS_LEAF),
                                    leaves.at(K_DICE_CLASS_WEIGHTS_LEAF)};

        return emit_soft_dice(builder, graph, rows, operands);
    }

    Status DiceObjective::feed(const Data::Batch& batch, const Train::ObjectiveLeaves& leaves,
                               Graph::TensorPool& pool) const
    {
        SA_TRY_ASSIGN(const Data::BatchTarget* const mask,
                      Train::require_batch_role(batch, Data::K_ROLE_MASK));

        DiceLeaves tensors;
        SA_TRY_ASSIGN(tensors.score_indices,
                      Train::find_leaf_tensor(leaves, K_DICE_SCORE_INDICES_LEAF, pool));
        SA_TRY_ASSIGN(tensors.class_indices,
                      Train::find_leaf_tensor(leaves, K_DICE_CLASS_INDICES_LEAF, pool));
        SA_TRY_ASSIGN(tensors.kept, Train::find_leaf_tensor(leaves, K_DICE_KEPT_LEAF, pool));
        SA_TRY_ASSIGN(tensors.class_pixels,
                      Train::find_leaf_tensor(leaves, K_DICE_CLASS_PIXELS_LEAF, pool));
        SA_TRY_ASSIGN(tensors.class_weights,
                      Train::find_leaf_tensor(leaves, K_DICE_CLASS_WEIGHTS_LEAF, pool));

        return fill_dice_targets(mask->values, extents_.classes, ignore_index_, tensors);
    }

    Status DiceObjective::measure(const Train::StepView& step, Train::Metrics& metrics) const
    {
        return measure_segmentation(step, extents_, ignore_index_, metrics);
    }

    Result<std::unique_ptr<Train::Objective>> make_dice_objective(
        const Train::ObjectiveContext& given, const Config::Value& attributes,
        std::string_view context)
    {
        SA_TRY_ASSIGN(const std::int32_t ignore_index,
                      Detail::read_ignore_index(attributes, context));
        SA_TRY_ASSIGN(const PixelExtents extents, Detail::read_pixel_extents(given, context));
        SA_TRY_ASSIGN(std::unique_ptr<DiceObjective> made,
                      DiceObjective::make(extents, ignore_index));
        std::unique_ptr<Train::Objective> base = std::move(made);

        return std::move(base);
    }

} // namespace SushiAI::Vision::Training
