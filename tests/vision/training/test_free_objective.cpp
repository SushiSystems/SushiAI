/**************************************************************************/
/* test_free_objective.cpp                                                */
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
 * @file test_free_objective.cpp
 * @brief Tests the anchor-free objective's leaves, its loss root and its metrics.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstdint>
#include <utility>

#include <gtest/gtest.h>

#include <SushiAI/failure/result.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/graph/builder.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/nn/model_outputs.hpp>
#include <SushiAI/shape/shape.hpp>
#include <SushiAI/train/metrics.hpp>
#include <SushiAI/train/objective.hpp>
#include <SushiAI/train/objective_context.hpp>
#include <SushiAI/vision/detect/free/free_task.hpp>
#include <SushiAI/vision/training/free_detection_loss_host.hpp>
#include <SushiAI/vision/training/free_loss_gains.hpp>
#include <SushiAI/vision/training/free_model_facts.hpp>
#include <SushiAI/vision/training/free_objective.hpp>

#include "free_objective_ports.hpp"

using namespace SushiAI;
using namespace SushiAI::FreeObjectiveTest;
using Graph::GraphBuilder;
using Vision::Training::FreeLossGains;
using Vision::Training::FreeModelFacts;
using Vision::Training::FreeObjective;

namespace
{
    /** @brief Holds the leaves the stage fills at reg_max 2: ten kinds on two levels, a sum. */
    constexpr std::size_t K_STAGE_LEAVES = 21;

    /** @brief Holds the stage leaves left out at l1_gain 0: strides, distances, L1 scale. */
    constexpr std::size_t K_SKIPPED_LEAVES = 6;

    /** @brief Returns the objective of make_context() under @p gains. */
    FreeObjective make_objective(const FreeLossGains& gains)
    {
        const Train::ObjectiveContext given = make_context();
        FreeModelFacts facts =
            Vision::Training::derive_free_model_facts(given.input_sample, given.outputs).value();
        return FreeObjective::make(std::move(facts), Vision::Detect::Free::FreeTask{}, K_BATCH,
                                   gains)
            .value();
    }

    /** @brief Adds every port of list_ports() to @p outputs as a parameter of its shape. */
    void add_ports(const GraphBuilder& builder, NN::ModelOutputs& outputs)
    {
        for (const Train::OutputShape& port : list_ports())
            SA_EXPECT_OK(outputs.add(port.name, builder.parameter(port.shape, port.name)));
    }
} // namespace

/** @brief Tests that declare adds the leaves the loss reads and the projection, no more. */
TEST(Small_FreeObjective, DeclaresTheLeavesTheLossReadsAndTheProjection)
{
    const FreeObjective objective = make_objective(FreeLossGains{});
    ASSERT_EQ(objective.stage_leaves().size(), K_STAGE_LEAVES);

    Graph::Graph graph;
    const GraphBuilder builder(graph);
    Train::ObjectiveLeaves leaves;
    objective.declare(builder, leaves);

    EXPECT_EQ(leaves.all().size(), K_STAGE_LEAVES - K_SKIPPED_LEAVES + 1);
    ASSERT_TRUE(leaves.has(Vision::Training::K_FREE_PROJECTION_ROLE));
    EXPECT_EQ(graph.value(leaves.at(Vision::Training::K_FREE_PROJECTION_ROLE)).shape,
              (Shape{K_REG_MAX, 1}));
    EXPECT_FALSE(leaves.has("free.anchor_strides.level0"));
    EXPECT_FALSE(leaves.has("free.l1_scale.level0"));
    EXPECT_EQ(graph.value(leaves.at("free.target_scores.level1")).shape,
              (Shape{K_BATCH, 2, 2, K_CLASSES}));
}

/** @brief Tests that an L1 gain above 0 declares the distance and scale leaves as well. */
TEST(Small_FreeObjective, AnL1GainDeclaresTheL1Leaves)
{
    FreeLossGains gains;
    gains.l1_gain = 1.0f;
    const FreeObjective objective = make_objective(gains);

    Graph::Graph graph;
    const GraphBuilder builder(graph);
    Train::ObjectiveLeaves leaves;
    objective.declare(builder, leaves);

    EXPECT_EQ(leaves.all().size(), K_STAGE_LEAVES - 2 + 1);
    EXPECT_TRUE(leaves.has("free.target_distances.level1"));
    EXPECT_TRUE(leaves.has("free.l1_scale.level0"));
}

/** @brief Tests that the loss over both levels' ports is a one-element root. */
TEST(Small_FreeObjective, TheLossIsAOneElementRoot)
{
    const FreeObjective objective = make_objective(FreeLossGains{});

    Graph::Graph graph;
    const GraphBuilder builder(graph);
    Train::ObjectiveLeaves leaves;
    objective.declare(builder, leaves);
    NN::ModelOutputs outputs;
    add_ports(builder, outputs);

    const Graph::ValueID root = objective.loss(builder, graph, outputs, leaves);

    ASSERT_NE(root, Graph::K_NO_VALUE);
    EXPECT_EQ(graph.value(root).shape, (Shape{1}));
    EXPECT_EQ(objective.name(), Vision::Training::K_FREE_OBJECTIVE_TYPE);
}

/** @brief Tests that outputs without a port the facts name stop the process naming the port. */
TEST(Small_FreeObjective, AMissingPortIsFatalNamingThePort)
{
    const FreeObjective objective = make_objective(FreeLossGains{});

    Graph::Graph graph;
    const GraphBuilder builder(graph);
    Train::ObjectiveLeaves leaves;
    objective.declare(builder, leaves);
    NN::ModelOutputs outputs;

    SA_EXPECT_FATAL((void)objective.loss(builder, graph, outputs, leaves),
                    "one2many.level0.boxes");
}

/** @brief Tests that the metrics name each loss term with its value. */
TEST(Small_FreeObjective, TheMetricsNameEachTerm)
{
    Vision::Training::FreeLossTerms terms;
    terms.box = 1.0;
    terms.classes = 2.0;
    terms.distribution = 3.0;
    terms.l1 = 4.0;

    Train::Metrics metrics;
    SA_EXPECT_OK(Vision::Training::add_free_loss_metrics(terms, metrics));

    EXPECT_EQ(metrics.value_or("box", 0.0), 1.0);
    EXPECT_EQ(metrics.value_or("class", 0.0), 2.0);
    EXPECT_EQ(metrics.value_or("distribution", 0.0), 3.0);
    EXPECT_EQ(metrics.value_or("l1", 0.0), 4.0);
}
