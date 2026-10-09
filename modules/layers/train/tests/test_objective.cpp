/**************************************************************************/
/* test_objective.cpp                                                     */
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
 * @file test_objective.cpp
 * @brief Tests the objective seam: the leaves it declares, the loss it emits and the resulting IR.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <memory>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include <SushiAI/config/value.hpp>
#include <SushiAI/core/ops.hpp>
#include <SushiAI/core/result.hpp>
#include <SushiAI/core/shape.hpp>
#include <SushiAI/core/status.hpp>
#include <SushiAI/graph/builder.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/nn/layers.hpp>
#include <SushiAI/nn/model_graph.hpp>
#include <SushiAI/nn/model_outputs.hpp>
#include <SushiAI/nn/node.hpp>
#include <SushiAI/train/objective.hpp>
#include <SushiAI/train/steps.hpp>
#include <SushiAI/train/training_graph.hpp>
#include <common/test_status.hpp>
#include <common/test_throws.hpp>
#include <common/test_utils.hpp>

using namespace SushiAI;
using Train::CrossEntropyObjective;
using Train::ObjectiveLeaves;

namespace
{
    using Mlp = NN::Sequential<NN::Linear, NN::ReLU, NN::Linear>;

    Mlp make_model(int64_t features, int64_t hidden, int64_t classes)
    {
        return Mlp(NN::Linear{features, hidden}, NN::ReLU{}, NN::Linear{hidden, classes});
    }

    /** @brief Renders a graph as one line per node: op, inputs, outputs. */
    std::vector<std::string> describe(const Graph::Graph& graph)
    {
        std::vector<std::string> rows;
        rows.reserve(graph.node_count());

        for (std::size_t i = 0; i < graph.node_count(); ++i)
        {
            const Graph::Node& node = graph.node(static_cast<Graph::NodeId>(i));

            std::string row = std::to_string(node.op.value);
            row += " <-";
            for (Graph::ValueId in : node.inputs) row += " " + std::to_string(in);
            row += " ->";
            for (Graph::ValueId out : node.outputs) row += " " + std::to_string(out);
            rows.push_back(row);
        }
        return rows;
    }

    using Test::build_two_output_model;
}

TEST(Unit_Objective, BuildGraphAndBuildClassifierProduceTheSameIr)
{
    const int64_t batch = 4;
    const int64_t features = 6;
    const int64_t classes = 3;

    Mlp through_wrapper = make_model(features, 8, classes);
    SA_ASSERT_OK_AND_ASSIGN(
        const Train::TrainingGraph a,
        Train::build_classifier(through_wrapper, batch, features, classes));

    Mlp through_seam = make_model(features, 8, classes);
    const CrossEntropyObjective objective(batch, classes);
    SA_ASSERT_OK_AND_ASSIGN(
        const Train::TrainingGraph b,
        Train::build_training_graph(through_seam, objective, Shape{batch, features}));

    EXPECT_EQ(a.graph.value_count(), b.graph.value_count());
    EXPECT_EQ(a.graph.node_count(), b.graph.node_count());
    EXPECT_EQ(describe(a.graph), describe(b.graph));

    // The value ids the loop reads are positions, so they must agree too.
    EXPECT_EQ(a.spec.inputs, b.spec.inputs);
    EXPECT_EQ(a.spec.targets, b.spec.targets);
    EXPECT_EQ(a.spec.logits, b.spec.logits);
    EXPECT_EQ(a.spec.loss, b.spec.loss);
}

TEST(Unit_Objective, TheTargetLeafIsDeclaredWhereItAlwaysWas)
{
    // The target leaf sits between the input and the model; see tests/functional/unit/README.md.
    Mlp model = make_model(6, 8, 3);
    const CrossEntropyObjective objective(4, 3);
    SA_ASSERT_OK_AND_ASSIGN(const Train::TrainingGraph built,
                            Train::build_training_graph(model, objective, Shape{4, 6}));

    EXPECT_EQ(built.spec.inputs, 0);
    EXPECT_EQ(built.spec.targets, 1);
    EXPECT_TRUE(built.leaves.has("targets"));
    EXPECT_EQ(built.leaves.at("targets"), built.spec.targets);
}

TEST(Unit_Objective, TheObjectiveOwnsTheOutputShapeCheck)
{
    // The objective owns the output shape message; see tests/functional/unit/README.md.
    Mlp model = make_model(6, 8, 3);
    const CrossEntropyObjective objective(4, 5);   // asks for 5 classes, model gives 3

    SA_EXPECT_FAILURE(Train::build_training_graph(model, objective, Shape{4, 6}),
                      StatusCode::InvalidDescription, "[4, 3]", "[4, 5]");
}

TEST(Unit_Objective, LeavesRefuseADuplicateRoleAndNameAnAbsentOne)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);

    ObjectiveLeaves leaves;
    leaves.add("targets", builder.input(Shape{2, 3}, "targets"));

    SA_EXPECT_ERROR(leaves.add("targets", builder.input(Shape{2, 3}, "again")),
                    "'targets'", "twice");
    SA_EXPECT_ERROR((void)leaves.at("collocation"), "'collocation'", "'targets'");

    EXPECT_TRUE(leaves.has("targets"));
    EXPECT_FALSE(leaves.has("collocation"));
    EXPECT_EQ(leaves.all().size(), 1u);
}

/** @brief Tests that the loss is computed from the value the model published as "output". */
TEST(Unit_Objective, CrossEntropyReadsTheOnlyOutput)
{
    Mlp model = make_model(6, 8, 3);
    const CrossEntropyObjective objective(4, 3);
    SA_ASSERT_OK_AND_ASSIGN(const Train::TrainingGraph built,
                            Train::build_training_graph(model, objective, Shape{4, 6}));

    ASSERT_EQ(built.outputs.all().size(), 1u);
    SA_ASSERT_OK_AND_ASSIGN(const Graph::ValueId published,
                            built.outputs.at(NN::K_SINGLE_PORT_NAME));

    const Graph::NodeId loss_node = built.graph.value(built.spec.loss).producer;
    ASSERT_NE(loss_node, Graph::K_NO_NODE);
    ASSERT_EQ(built.graph.node(loss_node).op, Ops::K_NLL_LOSS.id);

    // NLL_LOSS reads the log-softmax, and the log-softmax reads the model's output.
    const Graph::ValueId log_probabilities = built.graph.node(loss_node).inputs.at(0);
    const Graph::NodeId softmax_node = built.graph.value(log_probabilities).producer;
    ASSERT_NE(softmax_node, Graph::K_NO_NODE);
    ASSERT_EQ(built.graph.node(softmax_node).op, Ops::K_LOG_SOFTMAX.id);
    EXPECT_EQ(built.graph.node(softmax_node).inputs.at(0), published);
}

/** @brief Tests that cross-entropy refuses a model with two outputs and names both. */
TEST(Unit_Objective, CrossEntropyRefusesTwoOutputs)
{
    SA_ASSERT_OK_AND_ASSIGN(NN::ModelGraph model, build_two_output_model());
    const CrossEntropyObjective objective(4, 3);

    SA_EXPECT_FAILURE(Train::build_training_graph(model, objective, Shape{4, 6}),
                      StatusCode::InvalidDescription, "features", "logits");
}

TEST(Unit_Objective, AnEmptyLeafSetSaysSoRatherThanListingNothing)
{
    const ObjectiveLeaves leaves;
    SA_EXPECT_ERROR((void)leaves.at("targets"), "'targets'", "(none)");
}

TEST(Unit_Objective, TheRegistryBuildsCrossEntropyAndListsWhatItHas)
{
    const Config::Value empty = Config::parse_json("{}", "<test>");

    const std::unique_ptr<Train::Objective> objective =
        Train::make_objective("cross_entropy", 4, 3, empty, "objective");
    EXPECT_EQ(objective->name(), "cross_entropy");

    SA_EXPECT_ERROR((void)Train::make_objective("yolo", 4, 3, empty, "objective"),
                    "yolo", "objective type", "'cross_entropy'");
}

TEST(Unit_Objective, CrossEntropyTakesNoKeysAtAll)
{
    // Cross-entropy accepts no key; see tests/functional/unit/README.md.
    const Config::Value stray = Config::parse_json(R"({"label_smoothing": 0.1})", "<test>");
    SA_EXPECT_ERROR((void)Train::make_objective("cross_entropy", 4, 3, stray, "objective"),
                    "objective", "'label_smoothing'");
}
