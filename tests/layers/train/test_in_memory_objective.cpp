/**************************************************************************/
/* test_in_memory_objective.cpp                                           */
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
 * @file test_in_memory_objective.cpp
 * @brief Tests that InMemoryObjective keeps the promise of the Objective interface.
 * @author Mustafa Garip
 */

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>
#include <gtest/gtest.h>
#include <SushiBLAS/tensor/tensor.hpp>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/allocation/allocation.hpp>
#include <SushiAI/data/dataset.hpp>
#include <SushiAI/data/target.hpp>
#include <SushiAI/dtype/dtype.hpp>
#include <SushiAI/execution/host_execution_backend.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/graph/builder.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/graph/tensor_pool.hpp>
#include <SushiAI/nn/model_outputs.hpp>
#include <SushiAI/shape/shape.hpp>
#include <SushiAI/tensor_values/tensor_values.hpp>
#include <SushiAI/train/in_memory_objective.hpp>
#include <SushiAI/train/metrics.hpp>
#include <SushiAI/train/objective.hpp>
#include <SushiAI/train/step_view.hpp>

using namespace SushiAI;
using Train::InMemoryObjective;
namespace SaTest = SushiAI::Test;

namespace
{
    /** @brief Holds the elements of each leaf the feed cases declare. */
    constexpr std::int64_t K_ELEMENTS = 3;

    /** @brief Holds the value a leaf has before a feed, which no scripted fill writes. */
    constexpr double K_STALE = 7.0;

    /** @brief Provides the host engine and the pool the feed cases bind leaves in. */
    class Small_InMemoryObjectiveFeed : public ::testing::Test
    {
        protected:
            /** @brief Declares the scripted leaves of @p objective into the fixture's graph. */
            void declare_leaves(const InMemoryObjective& objective)
            {
                const Graph::GraphBuilder builder(graph);
                objective.declare(builder, leaves);
            }

            /** @brief Binds a float32 tensor of K_STALE to the declared leaf @p role. */
            void bind_stale(const std::string& role)
            {
                Result<SushiBLAS::Tensor> made =
                    allocate_tensor(backend.engine(), Shape{K_ELEMENTS}, DType::Float32);
                ASSERT_TRUE(made.has_value());
                SaTest::write_tensor(made.value(), std::vector<double>(K_ELEMENTS, K_STALE));
                SA_EXPECT_OK(pool.bind(leaves.at(role), made.value()));
            }

            /** @brief The host backend the tensors are allocated from. */
            HostExecutionBackend backend;

            /** @brief The graph the leaves are inputs of. */
            Graph::Graph graph;

            /** @brief The leaves the objective declared. */
            Train::ObjectiveLeaves leaves;

            /** @brief Binds the tensor of every declared leaf. */
            Graph::TensorPool pool{backend.engine()};
    };
} // namespace

/** @brief Tests that declare adds the scripted leaves and roles and counts the call. */
TEST(Small_InMemoryObjective, DeclareAddsTheScriptedLeavesAndRoles)
{
    InMemoryObjective objective;
    objective.script_leaf("targets", 4);
    objective.script_required(Data::TargetDescription{"labels", Shape{1}});
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    Train::ObjectiveLeaves leaves;
    const Train::Objective& port = objective;

    port.declare(builder, leaves);

    EXPECT_TRUE(leaves.has("targets"));
    EXPECT_EQ(leaves.at("targets"), 4u);
    ASSERT_EQ(leaves.required().size(), 1u);
    EXPECT_EQ(leaves.required()[0].role, "labels");
    EXPECT_EQ(objective.declare_count(), 1u);
}

/** @brief Tests that loss returns the scripted value and counts the call. */
TEST(Small_InMemoryObjective, LossReturnsTheScriptedValue)
{
    InMemoryObjective objective;
    objective.script_loss(9);
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const NN::ModelOutputs outputs;
    const Train::ObjectiveLeaves leaves;

    SA_ASSERT_OK_AND_ASSIGN(const Train::LossTrace loss_trace,
                            objective.loss(builder, graph, outputs, leaves));
    const Graph::ValueID loss = loss_trace.loss();

    EXPECT_EQ(loss, 9u);
    EXPECT_EQ(objective.loss_count(), 1u);
}

/** @brief Tests that a loss nobody scripted is refused as InvalidDescription. */
TEST(Small_InMemoryObjective, AnUnscriptedLossIsRefused)
{
    const InMemoryObjective objective;
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const NN::ModelOutputs outputs;
    const Train::ObjectiveLeaves leaves;

    SA_EXPECT_FAILURE(objective.loss(builder, graph, outputs, leaves),
                      StatusCode::InvalidDescription, "no loss value");
}

/** @brief Tests that a scripted failure is returned in place of the loss value. */
TEST(Small_InMemoryObjective, AScriptedFailureReplacesTheLoss)
{
    InMemoryObjective objective;
    objective.script_loss(9);
    objective.script_loss_failure(Status::from_message(StatusCode::InvalidDescription, []()
    {
        return std::string("the output 'logits' is unread");
    }));
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const NN::ModelOutputs outputs;
    const Train::ObjectiveLeaves leaves;

    SA_EXPECT_FAILURE(objective.loss(builder, graph, outputs, leaves),
                      StatusCode::InvalidDescription, "logits");
}

/** @brief Tests that the objective names itself and takes no batch to feed it. */
TEST(Small_InMemoryObjective, NamesItselfAndIsFedByNoBatch)
{
    const InMemoryObjective objective;

    EXPECT_EQ(objective.name(), "in-memory");
    EXPECT_FALSE(objective.is_fed_by_batch());
}

/** @brief Tests that a scripted input leaf is declared as an input of the builder's graph. */
TEST(Small_InMemoryObjective, ADeclaredInputLeafIsAnInputOfTheGraph)
{
    InMemoryObjective objective;
    objective.script_input_leaf("targets", Shape{2, 3}, DType::Float64);
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    Train::ObjectiveLeaves leaves;

    objective.declare(builder, leaves);

    ASSERT_TRUE(leaves.has("targets"));
    EXPECT_EQ(graph.value(leaves.at("targets")).shape, (Shape{2, 3}));
    EXPECT_EQ(graph.value(leaves.at("targets")).dtype, DType::Float64);
}

/** @brief Tests that the scripted loss emitter is called and its value returned. */
TEST(Small_InMemoryObjective, ALossEmitterBuildsTheLoss)
{
    InMemoryObjective objective;
    objective.script_loss_emitter([](const Graph::GraphBuilder&, const Graph::Graph&,
                                     const NN::ModelOutputs&, const Train::ObjectiveLeaves&)
    {
        return Result<Graph::ValueID>(Graph::ValueID(5));
    });
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const NN::ModelOutputs outputs;
    const Train::ObjectiveLeaves leaves;

    SA_ASSERT_OK_AND_ASSIGN(const Train::LossTrace loss_trace,
                            objective.loss(builder, graph, outputs, leaves));
    const Graph::ValueID loss = loss_trace.loss();

    EXPECT_EQ(loss, 5u);
    EXPECT_EQ(objective.loss_count(), 1u);
}

/** @brief Tests that a scripted fed-by-batch answer is returned. */
TEST(Small_InMemoryObjective, ScriptedFedByBatchIsReturned)
{
    InMemoryObjective objective;
    objective.script_fed_by_batch(true);

    EXPECT_TRUE(objective.is_fed_by_batch());
}

/** @brief Tests that feed writes the scripted leaf and leaves the leaf it was not told to. */
TEST_F(Small_InMemoryObjectiveFeed, FeedFillsTheScriptedLeafAndSkipsTheOther)
{
    InMemoryObjective objective;
    objective.script_input_leaf("targets", Shape{K_ELEMENTS});
    objective.script_input_leaf("weights", Shape{K_ELEMENTS});
    objective.script_feed_fill("targets", 1.0);
    declare_leaves(objective);
    bind_stale("targets");
    bind_stale("weights");
    const Data::Batch batch;

    SA_EXPECT_OK(objective.feed(batch, leaves, pool));

    EXPECT_EQ(SaTest::read_tensor(pool.tensor(leaves.at("targets"))),
              (std::vector<double>{1.0, 1.0, 1.0}));
    EXPECT_EQ(SaTest::read_tensor(pool.tensor(leaves.at("weights"))),
              (std::vector<double>{K_STALE, K_STALE, K_STALE}));
    EXPECT_EQ(objective.feed_count(), 1u);
}

/** @brief Tests that a fill of a leaf that is not declared is refused naming the role. */
TEST_F(Small_InMemoryObjectiveFeed, FeedRefusesAFillOfAnUndeclaredLeaf)
{
    InMemoryObjective objective;
    objective.script_feed_fill("targets", 1.0);
    declare_leaves(objective);
    const Data::Batch batch;

    SA_EXPECT_FAILURE(objective.feed(batch, leaves, pool), StatusCode::InvalidDescription,
                      "targets");
}

/** @brief Tests that a feed nobody scripted keeps the Unsupported of the base and is counted. */
TEST_F(Small_InMemoryObjectiveFeed, AnUnscriptedFeedIsUnsupportedAndCounted)
{
    const InMemoryObjective objective;
    const Data::Batch batch;

    SA_EXPECT_FAILURE(objective.feed(batch, leaves, pool), StatusCode::Unsupported, "in-memory");
    EXPECT_EQ(objective.feed_count(), 1u);
}

/** @brief Tests that a scripted feed failure is returned after the fills. */
TEST_F(Small_InMemoryObjectiveFeed, AScriptedFeedFailureIsReturned)
{
    InMemoryObjective objective;
    objective.script_feed_failure(Status::from_message(StatusCode::Incompatible, []()
    {
        return std::string("the batch has no labels");
    }));
    const Data::Batch batch;

    SA_EXPECT_FAILURE(objective.feed(batch, leaves, pool), StatusCode::Incompatible, "labels");
}

/** @brief Tests that measure appends the scripted metrics in order and counts the call. */
TEST(Small_InMemoryObjective, MeasureAppendsTheScriptedMetrics)
{
    InMemoryObjective objective;
    objective.script_metric("accuracy", 0.25);
    objective.script_metric("recall", 0.75);
    const Data::Batch batch;
    const Train::StepView step{batch, 0.0, SushiRuntime::span<const Train::StepOutput>()};
    Train::Metrics metrics;

    SA_EXPECT_OK(objective.measure(step, metrics));

    ASSERT_EQ(metrics.all().size(), 2u);
    EXPECT_EQ(metrics.all()[0].name, "accuracy");
    EXPECT_EQ(metrics.all()[0].value, 0.25);
    EXPECT_EQ(metrics.all()[1].name, "recall");
    EXPECT_EQ(metrics.all()[1].value, 0.75);
    EXPECT_EQ(objective.measure_count(), 1u);
}

/** @brief Tests that an objective with no scripted metric appends none. */
TEST(Small_InMemoryObjective, MeasureAppendsNothingWhenNothingIsScripted)
{
    const InMemoryObjective objective;
    const Data::Batch batch;
    const Train::StepView step{batch, 0.0, SushiRuntime::span<const Train::StepOutput>()};
    Train::Metrics metrics;

    SA_EXPECT_OK(objective.measure(step, metrics));

    EXPECT_TRUE(metrics.all().empty());
}

/** @brief Tests that the stated names are the scripted ones, whatever measure appends. */
TEST(Small_InMemoryObjective, StatedNamesNeedNotBeTheAppendedOnes)
{
    InMemoryObjective objective;
    objective.script_stated_names({"accuracy", "recall"});
    objective.script_metric("f1", 0.5);

    const SushiRuntime::span<const std::string_view> names = objective.metric_names();

    EXPECT_EQ((std::vector<std::string_view>(names.begin(), names.end())),
              (std::vector<std::string_view>{"accuracy", "recall"}));
}

/** @brief Tests that an objective with no scripted names states none. */
TEST(Small_InMemoryObjective, NoNameIsStatedWhenNoneIsScripted)
{
    const InMemoryObjective objective;

    EXPECT_TRUE(objective.metric_names().empty());
}
