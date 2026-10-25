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
 * @brief Tests the anchor-free objective's leaves, its loss root, its factory and its metrics.
 * @author Mustafa Garip
 */

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include <SushiAI/config/value.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/failure_expectations/throw_expectations.hpp>
#include <SushiAI/graph/builder.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/nn/model_outputs.hpp>
#include <SushiAI/shape/shape.hpp>
#include <SushiAI/train/metrics.hpp>
#include <SushiAI/train/objective.hpp>
#include <SushiAI/train/objective_context.hpp>
#include <SushiAI/vision/detect/free/free_task.hpp>
#include <SushiAI/vision/detect/free/head_contract.hpp>
#include <SushiAI/vision/training/free_detection_loss_host.hpp>
#include <SushiAI/vision/training/free_loss_gains.hpp>
#include <SushiAI/vision/training/free_model_facts.hpp>
#include <SushiAI/vision/training/free_objective.hpp>
#include <SushiAI/vision/training/free_objective_factory.hpp>

using namespace SushiAI;
using Graph::GraphBuilder;
using Train::ObjectiveContext;
using Train::OutputShape;
using Vision::Detect::Free::detect_port_name;
using Vision::Detect::Free::K_DETECT_BOXES;
using Vision::Detect::Free::K_DETECT_ONE_TO_MANY;
using Vision::Detect::Free::K_DETECT_SCORES;
using Vision::Training::FreeLossGains;
using Vision::Training::FreeModelFacts;
using Vision::Training::FreeObjective;

namespace
{
    /** @brief Holds the images per step. */
    constexpr std::int64_t K_BATCH = 2;

    /** @brief Holds the classes of the ports. */
    constexpr std::int64_t K_CLASSES = 2;

    /** @brief Holds the distribution bins per box side. */
    constexpr std::int64_t K_REG_MAX = 2;

    /** @brief Holds the side of the square input in pixels. */
    constexpr std::int64_t K_INPUT_SIDE = 32;

    /** @brief Holds the side of each level: 4 at stride 8, 2 at stride 16. */
    constexpr std::array<std::int64_t, 2> K_SIDES = {4, 2};

    /** @brief Holds the leaves the stage fills at reg_max 2: ten kinds on two levels, a sum. */
    constexpr std::size_t K_STAGE_LEAVES = 21;

    /** @brief Holds the stage leaves left out at l1_gain 0: strides, distances, L1 scale. */
    constexpr std::size_t K_SKIPPED_LEAVES = 6;

    /** @brief Returns the shape of the box (@p boxes) or the score port of @p level. */
    Shape port_shape(std::size_t level, bool boxes)
    {
        const std::int64_t side = K_SIDES[level];
        return Shape{K_BATCH, side, side, boxes ? 4 * K_REG_MAX : K_CLASSES};
    }

    /** @brief Returns the one-to-many ports of both levels, as a trace lists them. */
    std::vector<OutputShape> list_ports()
    {
        std::vector<OutputShape> outputs;
        for (std::size_t level = 0; level < K_SIDES.size(); ++level)
        {
            outputs.push_back(OutputShape{detect_port_name(K_DETECT_ONE_TO_MANY, level,
                                                           K_DETECT_BOXES),
                                          port_shape(level, true)});
            outputs.push_back(OutputShape{detect_port_name(K_DETECT_ONE_TO_MANY, level,
                                                           K_DETECT_SCORES),
                                          port_shape(level, false)});
        }
        return outputs;
    }

    /** @brief Returns the context of a [32, 32, 3] model that publishes both levels. */
    ObjectiveContext make_context()
    {
        ObjectiveContext given;
        given.batch = K_BATCH;
        given.classes = K_CLASSES;
        given.input_sample = Shape{K_INPUT_SIDE, K_INPUT_SIDE, 3};
        given.task = Vision::Detect::Free::write_free_task(Vision::Detect::Free::FreeTask{})
                         .value();
        given.outputs = list_ports();
        return given;
    }

    /** @brief Returns the objective of make_context() under @p gains. */
    FreeObjective make_objective(const FreeLossGains& gains)
    {
        const ObjectiveContext given = make_context();
        FreeModelFacts facts =
            Vision::Training::derive_free_model_facts(given.input_sample, given.outputs).value();
        return FreeObjective::make(std::move(facts), Vision::Detect::Free::FreeTask{}, K_BATCH,
                                   gains)
            .value();
    }

    /** @brief Adds every port of list_ports() to @p outputs as a parameter of its shape. */
    void add_ports(const GraphBuilder& builder, NN::ModelOutputs& outputs)
    {
        for (const OutputShape& port : list_ports())
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

/** @brief Tests that a model without a port the facts name throws naming the port. */
TEST(Small_FreeObjective, AMissingPortThrowsNamingThePort)
{
    const FreeObjective objective = make_objective(FreeLossGains{});

    Graph::Graph graph;
    const GraphBuilder builder(graph);
    Train::ObjectiveLeaves leaves;
    objective.declare(builder, leaves);
    NN::ModelOutputs outputs;

    SA_EXPECT_ERROR((void)objective.loss(builder, graph, outputs, leaves),
                    "one2many.level0.boxes");
}

/** @brief Tests that the factory derives the facts and reads the gains of the section. */
TEST(Small_FreeObjective, TheFactoryDerivesTheFactsAndReadsTheGains)
{
    const Config::Value section =
        Config::parse_json(R"({"type": "anchor_free", "box_gain": 5.0})", "<test>");

    SA_ASSERT_OK_AND_ASSIGN(const FreeObjective objective,
                            Vision::Training::build_free_objective(make_context(), section));

    EXPECT_EQ(objective.facts().levels.size(), K_SIDES.size());
    EXPECT_EQ(objective.facts().reg_max, K_REG_MAX);
    EXPECT_EQ(objective.facts().levels[1].stride, 16);
    EXPECT_EQ(objective.gains().box_gain, 5.0f);
    EXPECT_EQ(objective.gains().class_gain, FreeLossGains{}.class_gain);
}

/** @brief Tests that a key that is no gain is refused by name. */
TEST(Small_FreeObjective, TheFactoryRefusesAKeyThatIsNoGain)
{
    const Config::Value section =
        Config::parse_json(R"({"type": "anchor_free", "reg_max": 16})", "<test>");

    SA_EXPECT_FAILURE(Vision::Training::build_free_objective(make_context(), section),
                      StatusCode::InvalidDescription, "reg_max");
}

/** @brief Tests that the registry entry point builds an objective of the anchor-free type. */
TEST(Small_FreeObjective, TheRegistryEntryBuildsTheObjective)
{
    const Config::Value section = Config::parse_json(R"({"type": "anchor_free"})", "<test>");

    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<Train::Objective> objective,
                            Vision::Training::make_free_objective(make_context(), section,
                                                                  "the objective block"));

    ASSERT_NE(objective, nullptr);
    EXPECT_EQ(objective->name(), Vision::Training::K_FREE_OBJECTIVE_TYPE);
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
