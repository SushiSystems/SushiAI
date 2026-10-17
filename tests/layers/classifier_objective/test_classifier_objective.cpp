/**************************************************************************/
/* test_classifier_objective.cpp                                          */
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
 * @file test_classifier_objective.cpp
 * @brief Tests that classifier_objective() rebuilds the objective a classifier was traced with.
 * @author Mustafa Garip
 */

#include <cstdint>
#include <gtest/gtest.h>
#include <SushiAI/classifier_objective/classifier_objective.hpp>
#include <SushiAI/shape/shape.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/graph/builder.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/nn/layers.hpp>
#include <SushiAI/train/objective.hpp>
#include <SushiAI/train/steps.hpp>
#include <SushiAI/train/training_graph.hpp>

using namespace SushiAI;

/** @brief Shortens the namespace of the test helpers, which Test alone hides in a case. */
namespace SaTest = SushiAI::Test;

namespace
{
    /** @brief Holds the rows of a step. */
    constexpr int64_t K_BATCH = 4;

    /** @brief Holds the features of a sample. */
    constexpr int64_t K_FEATURES = 6;

    /** @brief Holds the classes the target leaf spans. */
    constexpr int64_t K_CLASSES = 3;

    /** @brief Declares the leaves of @p objective on a fresh graph and returns the targets'. */
    Graph::Value declared_targets(const Train::Objective& objective)
    {
        Graph::Graph graph;
        const Graph::GraphBuilder builder(graph);
        Train::ObjectiveLeaves leaves;
        objective.declare(builder, leaves);
        return graph.value(leaves.at("targets"));
    }
} // namespace

/** @brief Tests that the rebuilt objective declares the target leaf the traced graph holds. */
TEST(Small_ClassifierObjective, TheObjectiveDeclaresTheTracedTargetLeaf)
{
    NN::Linear model{K_FEATURES, K_CLASSES};
    SA_ASSERT_OK_AND_ASSIGN(const Train::TrainingGraph built,
                            Train::build_classifier(model, K_BATCH, K_FEATURES, K_CLASSES));

    const Train::CrossEntropyObjective objective = SaTest::classifier_objective(built);
    const Graph::Value targets = declared_targets(objective);

    const Graph::Value& traced = built.graph.value(built.spec.targets);
    EXPECT_EQ(targets.shape, traced.shape);
    EXPECT_EQ(targets.shape, (Shape{K_BATCH, K_CLASSES}));
    EXPECT_EQ(targets.dtype, traced.dtype);
    EXPECT_EQ(objective.name(), "cross_entropy");
}

/** @brief Tests that a classifier traced in another element type keeps that type. */
TEST(Small_ClassifierObjective, TheObjectiveKeepsTheTracedElementType)
{
    NN::Linear model{K_FEATURES, K_CLASSES};
    SA_ASSERT_OK_AND_ASSIGN(
        const Train::TrainingGraph built,
        Train::build_classifier(model, K_BATCH, K_FEATURES, K_CLASSES, DType::Float64));

    const Graph::Value targets = declared_targets(SaTest::classifier_objective(built));

    EXPECT_EQ(targets.dtype, DType::Float64);
}
