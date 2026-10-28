/**************************************************************************/
/* free_objective.cpp                                                     */
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
 * @file free_objective.cpp
 * @brief Implements FreeObjective over the assignment stage's leaves and the loss graph.
 * @author Mustafa Garip
 */

#include <SushiAI/vision/training/free_objective.hpp>

#include <cstddef>
#include <initializer_list>
#include <string>
#include <utility>

#include <SushiAI/dtype/dtype.hpp>
#include <SushiAI/failure/check.hpp>
#include <SushiAI/shape/shape.hpp>
#include <SushiAI/vision/training/free_detection_loss.hpp>

namespace SushiAI::Vision::Training
{
    namespace
    {
        /**
         * @brief Returns the id of output @p name.
         * @pre @p outputs are the outputs the objective's facts were derived from.
         */
        [[nodiscard]] Graph::ValueID port_of(const NN::ModelOutputs& outputs,
                                             const std::string& name)
        {
            const Result<Graph::ValueID> found = outputs.at(name);
            SA_FATAL_IF(!found.has_value(), "{}", found.status().message());
            return found.value();
        }

        /** @brief Returns the field of @p level a leaf of @p kind fills, or null for none. */
        [[nodiscard]] Graph::ValueID* level_field(FreeLevelLossValues& level,
                                                  FreeLeafKind kind) noexcept
        {
            switch (kind)
            {
                case FreeLeafKind::TargetScores:
                    return &level.target_scores;
                case FreeLeafKind::BoxMask:
                    return &level.box_mask;
                case FreeLeafKind::NormalizedWeight:
                    return &level.normalized_weight;
                case FreeLeafKind::TargetBoxes:
                    return &level.target_boxes;
                case FreeLeafKind::TargetDistances:
                    return &level.target_distances;
                case FreeLeafKind::L1Scale:
                    return &level.l1_scale;
                case FreeLeafKind::SafeBoxes:
                    return &level.safe_boxes;
                case FreeLeafKind::DistributionWeights:
                    return &level.distribution_weights;
                case FreeLeafKind::AnchorPoints:
                    return &level.anchor_points;
                case FreeLeafKind::AnchorStrides:
                case FreeLeafKind::TargetScoresSum:
                    break;
            }
            return nullptr;
        }
    } // namespace

    Result<FreeObjective> FreeObjective::make(FreeModelFacts facts, Detect::Free::FreeTask task,
                                              std::int64_t batch, FreeLossGains gains,
                                              FreeBranchWeights weights)
    {
        SA_TRY(validate_free_loss_gains(gains));
        SA_TRY(validate_free_branch_weights(weights));
        SA_TRY_ASSIGN(const FreeAssignmentStage stage,
                      FreeAssignmentStage::make(facts, task, batch));

        std::vector<FreeLeaf> leaves = stage.leaves();
        if (facts.has_one_to_one)
        {
            SA_TRY_ASSIGN(const FreeAssignmentStage one_to_one,
                          FreeAssignmentStage::make(facts, task, batch, FreeBranch::OneToOne));
            leaves.insert(leaves.end(), one_to_one.leaves().begin(), one_to_one.leaves().end());
        }
        return Result<FreeObjective>(FreeObjective(std::move(facts), std::move(task), batch,
                                                   gains, weights, std::move(leaves)));
    }

    FreeObjective::FreeObjective(FreeModelFacts facts, Detect::Free::FreeTask task,
                                 std::int64_t batch, FreeLossGains gains,
                                 FreeBranchWeights weights,
                                 std::vector<FreeLeaf> stage_leaves) noexcept
        : facts_(std::move(facts)),
          task_(std::move(task)),
          batch_(batch),
          gains_(gains),
          weights_(weights),
          stage_leaves_(std::move(stage_leaves))
    {

    }

    void FreeObjective::declare(const Graph::GraphBuilder& builder,
                                Train::ObjectiveLeaves& leaves) const
    {
        for (const FreeLeaf& leaf : stage_leaves_)
        {
            if (reads_leaf(leaf.kind))
                leaves.add(leaf.role, builder.input(leaf.shape, leaf.role, DType::Float32));
        }

        if (facts_.reg_max > 1)
        {
            leaves.add(K_FREE_PROJECTION_ROLE,
                       builder.input(Shape{facts_.reg_max, 1}, std::string(K_FREE_PROJECTION_ROLE),
                                     DType::Float32));
        }

        if (facts_.has_one_to_one)
        {
            for (const std::string_view role :
                 {K_FREE_ONE_TO_MANY_WEIGHT_ROLE, K_FREE_ONE_TO_ONE_WEIGHT_ROLE})
                leaves.add(role, builder.input(Shape{1}, std::string(role), DType::Float32));
        }
    }

    Result<Graph::ValueID> FreeObjective::loss(
        const Graph::GraphBuilder& builder, const Graph::Graph& graph,
        const NN::ModelOutputs& outputs, const Train::ObjectiveLeaves& leaves) const
    {
        SA_TRY_ASSIGN(const Graph::ValueID one_to_many,
                      branch_loss(builder, graph, outputs, leaves, FreeBranch::OneToMany));
        if (!facts_.has_one_to_one)
            return one_to_many;

        SA_TRY_ASSIGN(const Graph::ValueID one_to_one,
                      branch_loss(builder, graph, outputs, leaves, FreeBranch::OneToOne));
        SA_TRY_ASSIGN(const Graph::ValueID total,
                      builder.add(
                          builder.mul(one_to_many, leaves.at(K_FREE_ONE_TO_MANY_WEIGHT_ROLE)),
                          builder.mul(one_to_one, leaves.at(K_FREE_ONE_TO_ONE_WEIGHT_ROLE))));

        return total;
    }

    Result<Graph::ValueID> FreeObjective::branch_loss(const Graph::GraphBuilder& builder,
                                                      const Graph::Graph& graph,
                                                      const NN::ModelOutputs& outputs,
                                                      const Train::ObjectiveLeaves& leaves,
                                                      FreeBranch branch) const
    {
        FreeLossValues values;
        values.levels.resize(facts_.levels.size());
        for (std::size_t level = 0; level < facts_.levels.size(); ++level)
        {
            const FreeLevelPorts& ports = branch == FreeBranch::OneToOne
                                        ? facts_.levels[level].one_to_one
                                        : facts_.levels[level].one_to_many;
            values.levels[level].boxes = port_of(outputs, ports.boxes);
            values.levels[level].scores = port_of(outputs, ports.scores);
        }

        for (const FreeLeaf& leaf : stage_leaves_)
        {
            if (leaf.branch != branch || !reads_leaf(leaf.kind))
                continue;

            if (leaf.kind == FreeLeafKind::TargetScoresSum)
            {
                values.target_scores_sum = leaves.at(leaf.role);
                continue;
            }
            Graph::ValueID* const field =
                level_field(values.levels[static_cast<std::size_t>(leaf.level)], leaf.kind);
            SA_FATAL_IF(field == nullptr, "the leaf '{}' has no place in the loss", leaf.role);
            *field = leaves.at(leaf.role);
        }

        if (facts_.reg_max > 1)
            values.projection = leaves.at(K_FREE_PROJECTION_ROLE);

        SA_TRY_ASSIGN(const FreeLossTermIDs terms,
                      emit_free_detection_loss(builder, graph, facts_, values, gains_));

        return terms.total;
    }

    std::string_view FreeObjective::name() const noexcept
    {
        return K_FREE_OBJECTIVE_TYPE;
    }

    bool FreeObjective::reads_leaf(FreeLeafKind kind) const noexcept
    {
        switch (kind)
        {
            case FreeLeafKind::AnchorStrides:
                return false;
            case FreeLeafKind::TargetDistances:
            case FreeLeafKind::L1Scale:
                return gains_.l1_gain > 0.0f;
            default:
                return true;
        }
    }

    const std::vector<FreeLeaf>& FreeObjective::stage_leaves() const noexcept
    {
        return stage_leaves_;
    }

    const FreeModelFacts& FreeObjective::facts() const noexcept
    {
        return facts_;
    }

    const Detect::Free::FreeTask& FreeObjective::task() const noexcept
    {
        return task_;
    }

    std::int64_t FreeObjective::batch() const noexcept
    {
        return batch_;
    }

    const FreeLossGains& FreeObjective::gains() const noexcept
    {
        return gains_;
    }

    const FreeBranchWeights& FreeObjective::branch_weights() const noexcept
    {
        return weights_;
    }

    Status add_free_loss_metrics(const FreeLossTerms& terms, Train::Metrics& metrics)
    {
        SA_TRY(metrics.add(Train::Metric{"box", terms.box, false}));
        SA_TRY(metrics.add(Train::Metric{"class", terms.classes, false}));
        SA_TRY(metrics.add(Train::Metric{"distribution", terms.distribution, false}));
        return metrics.add(Train::Metric{"l1", terms.l1, false});
    }

} // namespace SushiAI::Vision::Training
