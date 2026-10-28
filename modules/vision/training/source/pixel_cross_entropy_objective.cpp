/**************************************************************************/
/* pixel_cross_entropy_objective.cpp                                      */
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
 * @file pixel_cross_entropy_objective.cpp
 * @brief Implements the per-pixel cross entropy and the factory a training file reaches it by.
 * @author Mustafa Garip
 */

#include <SushiAI/vision/training/pixel_cross_entropy_objective.hpp>

#include <limits>
#include <utility>
#include <SushiAI/seam/seam.hpp>
#include <SushiAI/shape/shape.hpp>
#include <SushiAI/data/roles.hpp>
#include <SushiAI/data/target.hpp>
#include <SushiAI/train/batch_role.hpp>
#include <SushiAI/vision/training/pixel_losses.hpp>
#include <SushiAI/vision/training/pixel_targets.hpp>
#include <SushiAI/vision/training/segmentation_metrics.hpp>

#include "pixel_objective.hpp"

namespace SushiAI::Vision::Training
{
    namespace
    {
        /** @brief Holds the largest index K_GATHER_ROWS addresses a table row by. */
        constexpr std::int64_t K_LARGEST_INDEX = std::numeric_limits<std::int32_t>::max();
    } // namespace

    Result<std::unique_ptr<PixelCrossEntropyObjective>> PixelCrossEntropyObjective::make(
        const PixelExtents& extents, std::int32_t ignore_index, DType dtype) noexcept
    {
        SA_TRY(Detail::validate_pixel_objective(extents, ignore_index, dtype, K_LARGEST_INDEX,
                                                K_PIXEL_CROSS_ENTROPY_TYPE));

        return SushiAI::Detail::guard_value(SushiAI::Detail::SeamPhase::Allocate, [&]()
        {
            return std::unique_ptr<PixelCrossEntropyObjective>(
                new PixelCrossEntropyObjective(extents, ignore_index, dtype));
        });
    }

    PixelCrossEntropyObjective::PixelCrossEntropyObjective(const PixelExtents& extents,
                                                           std::int32_t ignore_index,
                                                           DType dtype) noexcept
        : extents_(extents), ignore_index_(ignore_index), dtype_(dtype)
    {

    }

    void PixelCrossEntropyObjective::declare(const Graph::GraphBuilder& builder,
                                             Train::ObjectiveLeaves& leaves) const
    {
        const std::int64_t pixels = extents_.pixels();
        leaves.add(K_PIXEL_TARGETS_LEAF,
                   builder.input(Shape{pixels}, K_PIXEL_TARGETS_LEAF, DType::Int32));
        leaves.add(K_PIXEL_WEIGHTS_LEAF,
                   builder.input(Shape{pixels, 1}, K_PIXEL_WEIGHTS_LEAF, dtype_));
        leaves.require(Detail::describe_mask_role(extents_));
    }

    Result<Graph::ValueID> PixelCrossEntropyObjective::loss(
        const Graph::GraphBuilder& builder, const Graph::Graph& graph,
        const NN::ModelOutputs& outputs, const Train::ObjectiveLeaves& leaves) const
    {
        SA_TRY_ASSIGN(const Graph::ValueID rows,
                      Detail::emit_pixel_rows(builder, graph, outputs, extents_,
                                              K_PIXEL_CROSS_ENTROPY_TYPE));
        SA_TRY_ASSIGN(const Graph::ValueID loss,
                      emit_pixel_cross_entropy(builder, graph, rows,
                                               leaves.at(K_PIXEL_TARGETS_LEAF),
                                               leaves.at(K_PIXEL_WEIGHTS_LEAF)));

        return loss;
    }

    Status PixelCrossEntropyObjective::feed(const Data::Batch& batch,
                                            const Train::ObjectiveLeaves& leaves,
                                            Graph::TensorPool& pool) const
    {
        SA_TRY_ASSIGN(const Data::BatchTarget* const mask,
                      Train::require_batch_role(batch, Data::K_ROLE_MASK));
        SA_TRY_ASSIGN(SushiBLAS::Tensor* const indices,
                      Train::find_leaf_tensor(leaves, K_PIXEL_TARGETS_LEAF, pool));
        SA_TRY_ASSIGN(SushiBLAS::Tensor* const weights,
                      Train::find_leaf_tensor(leaves, K_PIXEL_WEIGHTS_LEAF, pool));

        return fill_pixel_targets(mask->values, extents_.classes, ignore_index_, *indices,
                                  *weights);
    }

    Status PixelCrossEntropyObjective::measure(const Train::StepView& step,
                                               Train::Metrics& metrics) const
    {
        return measure_segmentation(step, extents_, ignore_index_, metrics);
    }

    Result<std::unique_ptr<Train::Objective>> make_pixel_cross_entropy_objective(
        const Train::ObjectiveContext& given, const Config::Value& attributes,
        std::string_view context) noexcept
    {
        SA_TRY_ASSIGN(const std::int32_t ignore_index,
                      Detail::read_ignore_index(attributes, context));
        SA_TRY_ASSIGN(const PixelExtents extents, Detail::read_pixel_extents(given, context));
        SA_TRY_ASSIGN(std::unique_ptr<PixelCrossEntropyObjective> made,
                      PixelCrossEntropyObjective::make(extents, ignore_index));
        std::unique_ptr<Train::Objective> base = std::move(made);

        return std::move(base);
    }

} // namespace SushiAI::Vision::Training
