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
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>
#include <gtest/gtest.h>
#include <SushiBLAS/tensor/tensor.hpp>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/config/value.hpp>
#include <SushiAI/core/allocation.hpp>
#include <SushiAI/core/common.hpp>
#include <SushiAI/core/ops.hpp>
#include <SushiAI/core/result.hpp>
#include <SushiAI/core/shape.hpp>
#include <SushiAI/core/status.hpp>
#include <SushiAI/data/dataset.hpp>
#include <SushiAI/data/roles.hpp>
#include <SushiAI/data/target.hpp>
#include <SushiAI/graph/builder.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/graph/tensor_pool.hpp>
#include <SushiAI/nn/layers.hpp>
#include <SushiAI/nn/model_graph.hpp>
#include <SushiAI/nn/model_outputs.hpp>
#include <SushiAI/nn/node.hpp>
#include <SushiAI/train/metrics.hpp>
#include <SushiAI/train/objective.hpp>
#include <SushiAI/train/step_view.hpp>
#include <SushiAI/train/steps.hpp>
#include <SushiAI/train/training_graph.hpp>
#include <common/test_status.hpp>
#include <common/test_throws.hpp>
#include <common/test_utils.hpp>

using namespace SushiAI;
using Train::CrossEntropyObjective;
using Train::ObjectiveLeaves;

namespace SaTest = SushiAI::Test;

namespace
{
    /** @brief Provides the engine the scored tensors are allocated on. */
    class Unit_ObjectiveScore : public SaTest::EngineTest
    {
        protected:
            /** @brief Returns [3, 2] logits whose rows have their largest value at 1, 0, 1. */
            [[nodiscard]] Result<SushiBLAS::Tensor> logits()
            {
                SA_TRY_ASSIGN(SushiBLAS::Tensor output, allocate_tensor(*engine, Shape{3, 2}));
                SaTest::write_tensor(output,
                                     std::vector<double>{0.1, 0.9, 0.8, 0.2, -1.0, 0.5});
                return output;
            }

            /** @brief Returns a batch whose labels role holds @p values as int32. */
            [[nodiscard]] Result<Data::Batch> batch_of(const std::vector<int32_t>& values)
            {
                Data::BatchTarget labels;
                labels.role = std::string(Data::K_ROLE_LABELS);
                SA_TRY_ASSIGN(labels.values,
                              allocate_tensor(*engine,
                                              Shape{static_cast<int64_t>(values.size())},
                                              DType::Int32));
                for (std::size_t i = 0; i < values.size(); ++i)
                    labels.values.data_as<int32_t>()[i] = values[i];

                Data::Batch batch;
                batch.targets.push_back(std::move(labels));
                return batch;
            }

            /** @brief Returns the view of a step over @p batch whose loss is 0. */
            [[nodiscard]] static Train::StepView view_of(
                const Data::Batch& batch, const std::vector<Train::StepOutput>& outputs)
            {
                return Train::StepView{
                    batch, 0.0,
                    SushiRuntime::span<const Train::StepOutput>(outputs.data(), outputs.size())};
            }
    };

    /** @brief Bounds the difference between a counted ratio and its exact value. */
    constexpr double K_RATIO_TOLERANCE = 1e-12;

    /** @brief An objective that keeps the feed and the metrics of the base. */
    class StepFedObjective final : public Train::Objective
    {
        public:
            /** @brief Declares no leaf. */
            void declare(const Graph::GraphBuilder&, ObjectiveLeaves&) const override
            {

            }

            /** @brief Returns the absent value; the tests never build a graph with it. */
            Graph::ValueID loss(const Graph::GraphBuilder&, const Graph::Graph&,
                                const NN::ModelOutputs&, const ObjectiveLeaves&) const override
            {
                return Graph::K_NO_VALUE;
            }

            /** @brief Returns "step_fed". */
            std::string_view name() const noexcept override { return "step_fed"; }
    };

    using MLP = NN::Sequential<NN::Linear, NN::ReLU, NN::Linear>;

    MLP make_model(int64_t features, int64_t hidden, int64_t classes)
    {
        return MLP(NN::Linear{features, hidden}, NN::ReLU{}, NN::Linear{hidden, classes});
    }

    /** @brief Renders a graph as one line per node: op, inputs, outputs. */
    std::vector<std::string> describe(const Graph::Graph& graph)
    {
        std::vector<std::string> rows;
        rows.reserve(graph.node_count());

        for (std::size_t i = 0; i < graph.node_count(); ++i)
        {
            const Graph::Node& node = graph.node(static_cast<Graph::NodeID>(i));

            std::string row = std::to_string(node.op.value);
            row += " <-";
            for (Graph::ValueID in : node.inputs) row += " " + std::to_string(in);
            row += " ->";
            for (Graph::ValueID out : node.outputs) row += " " + std::to_string(out);
            rows.push_back(row);
        }
        return rows;
    }

    using Test::build_two_output_model;
}

TEST(Unit_Objective, BuildGraphAndBuildClassifierProduceTheSameIR)
{
    const int64_t batch = 4;
    const int64_t features = 6;
    const int64_t classes = 3;

    MLP through_wrapper = make_model(features, 8, classes);
    SA_ASSERT_OK_AND_ASSIGN(
        const Train::TrainingGraph a,
        Train::build_classifier(through_wrapper, batch, features, classes));

    MLP through_seam = make_model(features, 8, classes);
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
    // The target leaf sits between the input and the model; see tests/unit/README.md.
    MLP model = make_model(6, 8, 3);
    const CrossEntropyObjective objective(4, 3);
    SA_ASSERT_OK_AND_ASSIGN(const Train::TrainingGraph built,
                            Train::build_training_graph(model, objective, Shape{4, 6}));

    EXPECT_EQ(built.spec.inputs, 0);
    EXPECT_EQ(built.spec.targets, 1);
    EXPECT_TRUE(built.leaves.has("targets"));
    EXPECT_EQ(built.leaves.at("targets"), built.spec.targets);
}

/** @brief Tests that cross-entropy requires the labels role, one int32 per sample. */
TEST(Unit_Objective, CrossEntropyRequiresTheLabelsRole)
{
    MLP model = make_model(6, 8, 3);
    const CrossEntropyObjective objective(4, 3);
    SA_ASSERT_OK_AND_ASSIGN(const Train::TrainingGraph built,
                            Train::build_training_graph(model, objective, Shape{4, 6}));

    const std::vector<Data::TargetDescription>& required = built.leaves.required();
    ASSERT_EQ(required.size(), 1u);
    EXPECT_EQ(required.front().role, Data::K_ROLE_LABELS);
    EXPECT_EQ(required.front().sample_shape, Shape{});
    EXPECT_EQ(required.front().dtype, DType::Int32);
    EXPECT_EQ(required.front().classes, 3);
}

/** @brief Tests that the leaves keep every required role, in the order they were required. */
TEST(Unit_Objective, LeavesKeepEveryRequiredRoleInOrder)
{
    ObjectiveLeaves leaves;
    EXPECT_TRUE(leaves.required().empty());

    leaves.require(Data::TargetDescription{"labels", Shape{}, DType::Int32, 3});
    leaves.require(Data::TargetDescription{"boxes", Shape{8, 4}, DType::Float32, 0});
    leaves.require(Data::TargetDescription{"labels", Shape{}, DType::Int32, 3});

    ASSERT_EQ(leaves.required().size(), 3u);
    EXPECT_EQ(leaves.required()[0].role, "labels");
    EXPECT_EQ(leaves.required()[1].role, "boxes");
    EXPECT_EQ(leaves.required()[1].sample_shape, (Shape{8, 4}));
    EXPECT_EQ(leaves.required()[2].role, "labels");
    EXPECT_TRUE(leaves.all().empty());
}

/** @brief Tests that accuracy is the fraction of rows whose label is the largest logit. */
TEST_F(Unit_ObjectiveScore, AccuracyIsReadFromTheLabelsRole)
{
    const CrossEntropyObjective objective(3, 2);
    SA_ASSERT_OK_AND_ASSIGN(const SushiBLAS::Tensor output, logits());
    SA_ASSERT_OK_AND_ASSIGN(const Data::Batch batch, batch_of({1, 0, 0}));
    const std::vector<Train::StepOutput> outputs{Train::StepOutput{"output", &output}};

    Train::Metrics metrics;
    SA_EXPECT_OK(objective.measure(view_of(batch, outputs), metrics));

    ASSERT_EQ(metrics.all().size(), 1u);
    const Train::Metric* const accuracy = metrics.find("accuracy");
    ASSERT_NE(accuracy, nullptr);
    EXPECT_TRUE(accuracy->is_fraction);
    EXPECT_NEAR(accuracy->value, 2.0 / 3.0, K_RATIO_TOLERANCE);
}

/** @brief Tests that a batch without the labels role is refused and adds no metric. */
TEST_F(Unit_ObjectiveScore, ABatchWithoutTheLabelsRoleIsRefused)
{
    const CrossEntropyObjective objective(3, 2);
    SA_ASSERT_OK_AND_ASSIGN(const SushiBLAS::Tensor output, logits());
    const Data::Batch empty{};
    const std::vector<Train::StepOutput> outputs{Train::StepOutput{"output", &output}};

    Train::Metrics metrics;
    SA_EXPECT_FAILURE(objective.measure(view_of(empty, outputs), metrics),
                      StatusCode::InvalidDescription, "'labels'");
    EXPECT_TRUE(metrics.all().empty());
}

/** @brief Tests that accuracy refuses a step that published two outputs. */
TEST_F(Unit_ObjectiveScore, AccuracyReadsOneOutput)
{
    const CrossEntropyObjective objective(3, 2);
    SA_ASSERT_OK_AND_ASSIGN(const SushiBLAS::Tensor output, logits());
    SA_ASSERT_OK_AND_ASSIGN(const Data::Batch batch, batch_of({1, 0, 0}));
    const std::vector<Train::StepOutput> outputs{Train::StepOutput{"features", &output},
                                                 Train::StepOutput{"logits", &output}};

    Train::Metrics metrics;
    SA_EXPECT_FAILURE(objective.measure(view_of(batch, outputs), metrics),
                      StatusCode::InvalidDescription, "one model output", "published 2");
}

/** @brief Tests that feed writes the one-hot row of every label into the target leaf. */
TEST_F(Unit_ObjectiveScore, FeedWritesTheOneHotRowsOfTheLabels)
{
    const CrossEntropyObjective objective(3, 2);
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    ObjectiveLeaves leaves;
    objective.declare(builder, leaves);

    Graph::TensorPool pool(*engine);
    SA_ASSERT_OK_AND_ASSIGN(SushiBLAS::Tensor targets, allocate_tensor(*engine, Shape{3, 2}));
    SA_EXPECT_OK(pool.bind(leaves.at("targets"), targets));
    SA_ASSERT_OK_AND_ASSIGN(const Data::Batch batch, batch_of({1, 0, 0}));

    SA_EXPECT_OK(objective.feed(batch, leaves, pool));

    EXPECT_EQ(SaTest::read_tensor(pool.tensor(leaves.at("targets"))),
              (std::vector<double>{0.0, 1.0, 1.0, 0.0, 1.0, 0.0}));
}

/** @brief Tests that the base refuses to feed a batch and measures nothing. */
TEST_F(Unit_ObjectiveScore, AnObjectiveFedByItsOwnStepRefusesABatch)
{
    const StepFedObjective objective;
    const ObjectiveLeaves leaves;
    Graph::TensorPool pool(*engine);
    const Data::Batch batch{};

    SA_EXPECT_FAILURE(objective.feed(batch, leaves, pool), StatusCode::Unsupported,
                      "'step_fed'", "fed by its own step");

    Train::Metrics metrics;
    SA_EXPECT_OK(objective.measure(view_of(batch, {}), metrics));
    EXPECT_TRUE(metrics.all().empty());
}

/** @brief Tests that a leaf the objective did not declare, or the pool does not hold, is named. */
TEST_F(Unit_ObjectiveScore, ALeafWithoutATensorIsNamed)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    ObjectiveLeaves leaves;
    leaves.add("targets", builder.input(Shape{2, 3}, "targets"));
    Graph::TensorPool pool(*engine);

    SA_EXPECT_FAILURE(Train::find_leaf_tensor(leaves, "collocation", pool),
                      StatusCode::InvalidDescription, "no leaf 'collocation'");
    SA_EXPECT_FAILURE(Train::find_leaf_tensor(leaves, "targets", pool),
                      StatusCode::InvalidDescription, "no tensor", "'targets'");
}

TEST(Unit_Objective, TheObjectiveOwnsTheOutputShapeCheck)
{
    // The objective owns the output shape message; see tests/unit/README.md.
    MLP model = make_model(6, 8, 3);
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
    MLP model = make_model(6, 8, 3);
    const CrossEntropyObjective objective(4, 3);
    SA_ASSERT_OK_AND_ASSIGN(const Train::TrainingGraph built,
                            Train::build_training_graph(model, objective, Shape{4, 6}));

    ASSERT_EQ(built.outputs.all().size(), 1u);
    SA_ASSERT_OK_AND_ASSIGN(const Graph::ValueID published,
                            built.outputs.at(NN::K_SINGLE_PORT_NAME));

    const Graph::NodeID loss_node = built.graph.value(built.spec.loss).producer;
    ASSERT_NE(loss_node, Graph::K_NO_NODE);
    ASSERT_EQ(built.graph.node(loss_node).op, Ops::K_NLL_LOSS.id);

    // NLL_LOSS reads the log-softmax, and the log-softmax reads the model's output.
    const Graph::ValueID log_probabilities = built.graph.node(loss_node).inputs.at(0);
    const Graph::NodeID softmax_node = built.graph.value(log_probabilities).producer;
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

/** @brief Tests that the factory builds cross-entropy over the batch and classes it is given. */
TEST(Unit_Objective, TheFactoryBuildsCrossEntropyOverTheContext)
{
    const Config::Value empty = Config::parse_json("{}", "<test>");
    Train::ObjectiveContext given;
    given.batch = 4;
    given.classes = 3;

    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<Train::Objective> objective,
                            Train::make_cross_entropy(given, empty, "objective"));
    EXPECT_EQ(objective->name(), "cross_entropy");

    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    ObjectiveLeaves leaves;
    objective->declare(builder, leaves);
    EXPECT_EQ(graph.value(leaves.at("targets")).shape, (Shape{4, 3}));
}

/** @brief Tests that cross-entropy refuses every key, naming the one it met. */
TEST(Unit_Objective, CrossEntropyTakesNoKeysAtAll)
{
    // Cross-entropy accepts no key; see tests/unit/README.md.
    const Config::Value stray = Config::parse_json(R"({"label_smoothing": 0.1})", "<test>");
    Train::ObjectiveContext given;
    given.batch = 4;
    given.classes = 3;

    SA_EXPECT_FAILURE(Train::make_cross_entropy(given, stray, "objective"),
                      StatusCode::InvalidDescription, "objective", "'label_smoothing'");
}

/** @brief Tests that data which states no class count cannot build cross-entropy. */
TEST(Unit_Objective, CrossEntropyNeedsTheClassCountOfTheData)
{
    const Config::Value empty = Config::parse_json("{}", "<test>");
    Train::ObjectiveContext given;
    given.batch = 4;

    SA_EXPECT_FAILURE(Train::make_cross_entropy(given, empty, "objective"),
                      StatusCode::InvalidDescription, "objective", "class count", "0 classes");
}
