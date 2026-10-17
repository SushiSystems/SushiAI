/**************************************************************************/
/* anchor_objective.cpp                                                   */
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
 * @file anchor_objective.cpp
 * @brief Implements AnchorObjective over emit_head_delta_loss and measure_head_cost.
 * @author Mustafa Garip
 */

#include <SushiAI/vision/training/anchor_objective.hpp>

#include <cstdint>
#include <utility>

#include <SushiAI/failure/check.hpp>
#include <SushiAI/dtype/dtype.hpp>
#include <SushiAI/failure/error.hpp>
#include <SushiAI/shape/shape.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/vision/training/head_delta_loss.hpp>

namespace SushiAI::Vision::Training
{
    namespace
    {
        /** @brief Holds the prefix of a delta leaf's role, ahead of the head's output name. */
        constexpr std::string_view K_LEAF_PREFIX = "delta.";

        /**
         * @brief Checks one head against the layout, the heads before it and the first batch.
         * @param index Position of the head in @p heads and in the layout.
         * @return Success, or InvalidDescription naming the head and what it breaks.
         */
        [[nodiscard]] Status validate_anchor_head(const std::vector<AnchorHead>& heads,
                                                  std::size_t index,
                                                  const Detect::Anchor::AnchorLayout& layout)
        {
            const AnchorHead& head = heads[index];

            SA_FAIL_IF(head.output.empty(), StatusCode::InvalidDescription,
                       "anchor head {} names no model output", index);
            SA_FAIL_IF(head.shape.batch < 1 || head.shape.height < 1 || head.shape.width < 1,
                       StatusCode::InvalidDescription,
                       "anchor head '{}' has the extents [{}, {}, {}]; each must be positive",
                       head.output, head.shape.batch, head.shape.height, head.shape.width);
            SA_FAIL_IF(head.shape.batch != heads.front().shape.batch,
                       StatusCode::InvalidDescription,
                       "anchor head '{}' has a batch of {}, but head '{}' has {}", head.output,
                       head.shape.batch, heads.front().output, heads.front().shape.batch);

            for (std::size_t earlier = 0; earlier < index; ++earlier)
            {
                SA_FAIL_IF(heads[earlier].output == head.output, StatusCode::InvalidDescription,
                           "anchor heads {} and {} both read the output '{}'", earlier, index,
                           head.output);
            }

            SA_TRY(Detect::Anchor::validate_head(head.options, layout,
                                                 static_cast<std::int64_t>(index)));

            return Detect::Anchor::validate_head_training(head.training);
        }
    } // namespace

    Result<AnchorObjective> AnchorObjective::make(Detect::Anchor::NetworkSize network,
                                                  Detect::Anchor::AnchorLayout layout,
                                                  std::vector<AnchorHead> heads)
    {
        SA_FAIL_IF(network.height < 1 || network.width < 1, StatusCode::InvalidDescription,
                   "the network input is {} by {} pixels; both sides must be positive",
                   network.height, network.width);
        SA_TRY(Detect::Anchor::validate_anchor_layout(layout));
        SA_FAIL_IF(heads.size() != layout.head_anchors.size(), StatusCode::InvalidDescription,
                   "the objective was given {} heads, but the anchor layout holds {}",
                   heads.size(), layout.head_anchors.size());

        for (std::size_t index = 0; index < heads.size(); ++index)
            SA_TRY(validate_anchor_head(heads, index, layout));

        return Result<AnchorObjective>(
            AnchorObjective(network, std::move(layout), std::move(heads)));
    }

    AnchorObjective::AnchorObjective(Detect::Anchor::NetworkSize network,
                                     Detect::Anchor::AnchorLayout layout,
                                     std::vector<AnchorHead> heads) noexcept
        : network_(network),
          layout_(std::move(layout)),
          heads_(std::move(heads))
    {

    }

    void AnchorObjective::declare(const Graph::GraphBuilder& builder,
                                  Train::ObjectiveLeaves& leaves) const
    {
        for (std::size_t index = 0; index < heads_.size(); ++index)
        {
            const AnchorHead& head = heads_[index];
            const std::int64_t channels = Detect::Anchor::count_head_channels(
                head.options, layout_, static_cast<std::int64_t>(index));
            const Shape shape{head.shape.batch, head.shape.height, head.shape.width, channels};
            const std::string role = leaf_role(index);

            leaves.add(role, builder.input(shape, role, DType::Float32));
        }
    }

    Graph::ValueID AnchorObjective::loss(const Graph::GraphBuilder& builder, const Graph::Graph&,
                                         const NN::ModelOutputs& outputs,
                                         const Train::ObjectiveLeaves& leaves) const
    {
        const double scale = 1.0 / static_cast<double>(heads_.front().shape.batch);
        Graph::ValueID total = Graph::K_NO_VALUE;

        for (std::size_t index = 0; index < heads_.size(); ++index)
        {
            const Result<Graph::ValueID> head = outputs.at(heads_[index].output);
            SA_THROW_IF(!head.has_value(), "{}", head.status().message());

            const Graph::ValueID term = emit_head_delta_loss(
                builder, head.value(), leaves.at(leaf_role(index)), scale);
            total = index == 0 ? term : builder.add(total, term);
        }

        return total;
    }

    std::string_view AnchorObjective::name() const noexcept
    {
        return "anchor_delta";
    }

    Detect::Anchor::HeadCost AnchorObjective::measure_cost(
        SushiRuntime::span<const Detect::Anchor::HeadDelta> deltas) const noexcept
    {
        SA_FATAL_IF(deltas.size() != heads_.size(),
                    "measure_cost was given {} deltas for {} heads", deltas.size(),
                    heads_.size());

        Detect::Anchor::HeadCost sum;
        for (std::size_t index = 0; index < heads_.size(); ++index)
        {
            const AnchorHead& head = heads_[index];
            const Detect::Anchor::HeadCost cost = Detect::Anchor::measure_head_cost(
                deltas[index], head.shape, head.options, head.training);

            sum.total += cost.total;
            sum.box += cost.box;
            sum.non_box += cost.non_box;
            sum.stored += cost.stored;
        }

        return sum;
    }

    std::string AnchorObjective::leaf_role(std::size_t head) const
    {
        SA_FATAL_IF(head >= heads_.size(), "head {} is outside an objective of {} heads", head,
                    heads_.size());

        return std::string(K_LEAF_PREFIX) + heads_[head].output;
    }

    Detect::Anchor::NetworkSize AnchorObjective::network() const noexcept
    {
        return network_;
    }

    const Detect::Anchor::AnchorLayout& AnchorObjective::layout() const noexcept
    {
        return layout_;
    }

    const std::vector<AnchorHead>& AnchorObjective::heads() const noexcept
    {
        return heads_;
    }

} // namespace SushiAI::Vision::Training
