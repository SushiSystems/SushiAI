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
/* Licensed under the Apache License, Version 2.0 (the "License"); you    */
/* may not use this file except in compliance with the License. You may   */
/* obtain a copy of the License at                                        */
/*                                                                        */
/*     https://www.apache.org/licenses/LICENSE-2.0                        */
/*                                                                        */
/* Unless required by applicable law or agreed to in writing, software    */
/* distributed under the License is distributed on an "AS IS" BASIS,      */
/* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or        */
/* implied. See the License for the specific language governing           */
/* permissions and limitations under the License.                         */
/**************************************************************************/

/**
 * @file test_classifier_objective.cpp
 * @brief Tests that classifier_objective() rebuilds the objective a classifier was traced with.
 * @author Mustafa Garip
 */

#include <cstdint>
#include <string>
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

/** @brief Tests that the rebuilt objective declares the target leaf shape the traced graph holds. */
TEST(Small_ClassifierObjective, TheObjectiveDeclaresTheTracedTargetShape)
{
    NN::Linear model{K_FEATURES, K_CLASSES};
    SA_ASSERT_OK_AND_ASSIGN(const Train::TrainingGraph built,
                            Train::build_classifier(model, K_BATCH, K_FEATURES, K_CLASSES));

    const Graph::Value targets = declared_targets(SaTest::classifier_objective(built));

    EXPECT_EQ(targets.shape, built.graph.value(built.spec.targets).shape);
    EXPECT_EQ(targets.shape, (Shape{K_BATCH, K_CLASSES}));
}

namespace
{
/** @brief Holds the element type a classifier is traced in. */
struct DTypeRow
{
    std::string name; /**< The row's name. */
    DType dtype;      /**< The element type the classifier is traced in. */
};
} // namespace

namespace
{
/** @brief Runs the element-type promise over each traced element type. */
class Small_ClassifierObjectiveDType : public ::testing::TestWithParam<DTypeRow>
{
};
} // namespace

/** @brief Tests that a classifier traced in an element type gets a target leaf of that type. */
TEST_P(Small_ClassifierObjectiveDType, TheObjectiveKeepsTheTracedElementType)
{
    const DTypeRow& row = GetParam();
    NN::Linear model{K_FEATURES, K_CLASSES};
    SA_ASSERT_OK_AND_ASSIGN(
        const Train::TrainingGraph built,
        Train::build_classifier(model, K_BATCH, K_FEATURES, K_CLASSES, row.dtype));

    const Graph::Value targets = declared_targets(SaTest::classifier_objective(built));

    EXPECT_EQ(targets.dtype, row.dtype);
    EXPECT_EQ(targets.dtype, built.graph.value(built.spec.targets).dtype);
}

INSTANTIATE_TEST_SUITE_P(
    Rows, Small_ClassifierObjectiveDType,
    ::testing::Values(DTypeRow{"Float32", DType::Float32}, DTypeRow{"Float64", DType::Float64}),
    [](const ::testing::TestParamInfo<DTypeRow>& info) { return info.param.name; });
