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
 * @file test_objective.cpp
 * @brief Tests the Objective interface and the factories of the dense objectives.
 * @author Mustafa Garip
 */

#include <cmath>
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
#include <SushiAI/allocation/allocation.hpp>
#include <SushiAI/dtype/dtype.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/shape/shape.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/data/dataset.hpp>
#include <SushiAI/data/roles.hpp>
#include <SushiAI/data/target.hpp>
#include <SushiAI/execution/host_execution_backend.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/graph/builder.hpp>
#include <SushiAI/graph/host_evaluator.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/graph/tensor_pool.hpp>
#include <SushiAI/nn/layers.hpp>
#include <SushiAI/nn/model_outputs.hpp>
#include <SushiAI/nn/node.hpp>
#include <SushiAI/tensor_values/tensor_values.hpp>
#include <SushiAI/train/binary_cross_entropy_objective.hpp>
#include <SushiAI/train/in_memory_objective.hpp>
#include <SushiAI/train/mean_squared_error_objective.hpp>
#include <SushiAI/train/metrics.hpp>
#include <SushiAI/train/multi_label_objective.hpp>
#include <SushiAI/train/objective.hpp>
#include <SushiAI/train/objective_context.hpp>
#include <SushiAI/train/step_view.hpp>
#include <SushiAI/train/steps.hpp>
#include <SushiAI/train/training_graph.hpp>
#include <support/tolerance.hpp>

using namespace SushiAI;
using Train::CrossEntropyObjective;
using Train::ObjectiveLeaves;
namespace SaTest = SushiAI::Test;
using Train::BinaryCrossEntropyObjective;
using Train::MeanSquaredErrorObjective;
using Train::MultiLabelObjective;

namespace
{
    /**
     * @brief The rounded operations from the counts to the accuracy: the counts are exact
     *        integers, so the one quotient rounds.
     */
    constexpr std::size_t K_RATIO_OPERATIONS = 1;

    /**
     * @brief Counts the roundings on the longest path to the host loss of three rows of four
     *        classes: 8 in the log-softmax, 1 in the product, 11 in the sum of 12 and 1 in the
     *        mean are 21, and the reference takes as many, so the two differ by at most 42.
     */
    constexpr std::size_t K_LOSS_ROUNDED_OPERATIONS = 42;

    /** @brief Provides the host engine the scored tensors, the roles and the leaves live on. */
    class Small_ObjectiveScore : public ::testing::Test
    {
        protected:
            /** @brief Returns [3, 2] logits whose rows have their largest value at 1, 0, 1. */
            [[nodiscard]] Result<SushiBLAS::Tensor> logits()
            {
                SA_TRY_ASSIGN(SushiBLAS::Tensor output,
                              allocate_tensor(backend.engine(), Shape{3, 2}));
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
                              allocate_tensor(backend.engine(),
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

            /** @brief The host backend the tensors and the pool are allocated from. */
            HostExecutionBackend backend;
    };

    /** @brief Holds the leaf a lookup asks for and the words of the refusal it gets. */
    struct LeafLookupRow
    {
        std::string name;       /**< The row's name. */
        std::string role;       /**< The role looked up. */
        std::string first;      /**< A text the message holds. */
        std::string second;     /**< Another text the message holds. */
    };

    /** @brief Looks up a leaf tensor the objective did not declare or the pool does not hold. */
    class Small_ObjectiveLeafLookup : public Small_ObjectiveScore,
                                      public ::testing::WithParamInterface<LeafLookupRow>
    {

    };

    /** @brief Runs the cross-entropy loss graph of each set of logits on the host evaluator. */
    class Small_ObjectiveLoss : public ::testing::TestWithParam<std::vector<double>>
    {

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
}

TEST(Small_Objective, BuildGraphAndBuildClassifierProduceTheSameIR)
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

/** @brief Tests that the spec's targets value is the leaf the objective declared as "targets". */
TEST(Small_Objective, TheSpecTargetsAreTheLeafTheObjectiveDeclared)
{
    MLP model = make_model(6, 8, 3);
    const CrossEntropyObjective objective(4, 3);
    SA_ASSERT_OK_AND_ASSIGN(const Train::TrainingGraph built,
                            Train::build_training_graph(model, objective, Shape{4, 6}));

    EXPECT_TRUE(built.leaves.has("targets"));
    EXPECT_EQ(built.leaves.at("targets"), built.spec.targets);
}

/** @brief Tests that cross-entropy requires the labels role, one int32 per sample. */
TEST(Small_Objective, CrossEntropyRequiresTheLabelsRole)
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

/** @brief Tests that cross-entropy refuses an output of another shape with a Status. */
TEST(Small_Objective, CrossEntropyRefusesAnOutputOfAnotherShape)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const CrossEntropyObjective objective(4, 3);
    ObjectiveLeaves leaves;
    objective.declare(builder, leaves);
    NN::ModelOutputs outputs;
    SA_EXPECT_OK(outputs.add("logits", builder.input(Shape{4, 5}, "logits")));

    SA_EXPECT_FAILURE(objective.loss(builder, graph, outputs, leaves),
                      StatusCode::InvalidDescription, "[4, 5]", "[4, 3]");
}

/** @brief Tests that cross-entropy refuses a model with two outputs with a Status. */
TEST(Small_Objective, CrossEntropyRefusesAModelWithTwoOutputs)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const CrossEntropyObjective objective(4, 3);
    ObjectiveLeaves leaves;
    objective.declare(builder, leaves);
    NN::ModelOutputs outputs;
    SA_EXPECT_OK(outputs.add("first", builder.input(Shape{4, 3}, "first")));
    SA_EXPECT_OK(outputs.add("second", builder.input(Shape{4, 3}, "second")));

    SA_EXPECT_FAILURE(objective.loss(builder, graph, outputs, leaves),
                      StatusCode::InvalidDescription, "exactly one model output", "second");
}

/** @brief Tests that the leaves keep every required role, in the order they were required. */
TEST(Small_Objective, LeavesKeepEveryRequiredRoleInOrder)
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
TEST_F(Small_ObjectiveScore, AccuracyIsReadFromTheLabelsRole)
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
    EXPECT_TRUE(TestSupport::is_within(TestSupport::measure_relative_error<double>(
        accuracy->value, 2.0 / 3.0, K_RATIO_OPERATIONS)));
}

/** @brief Tests that a batch without the labels role is refused and adds no metric. */
TEST_F(Small_ObjectiveScore, ABatchWithoutTheLabelsRoleIsRefused)
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
TEST_F(Small_ObjectiveScore, AccuracyReadsOneOutput)
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
TEST_F(Small_ObjectiveScore, FeedWritesTheOneHotRowsOfTheLabels)
{
    const CrossEntropyObjective objective(3, 2);
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    ObjectiveLeaves leaves;
    objective.declare(builder, leaves);

    Graph::TensorPool pool(backend.engine());
    SA_ASSERT_OK_AND_ASSIGN(SushiBLAS::Tensor targets,
                            allocate_tensor(backend.engine(), Shape{3, 2}));
    SA_EXPECT_OK(pool.bind(leaves.at("targets"), targets));
    SA_ASSERT_OK_AND_ASSIGN(const Data::Batch batch, batch_of({1, 0, 0}));

    SA_EXPECT_OK(objective.feed(batch, leaves, pool));

    EXPECT_EQ(SaTest::read_tensor(pool.tensor(leaves.at("targets"))),
              (std::vector<double>{0.0, 1.0, 1.0, 0.0, 1.0, 0.0}));
}

/** @brief Tests that the base refuses to feed a batch: the real objective that keeps it. */
TEST_F(Small_ObjectiveScore, AnObjectiveFedByItsOwnStepRefusesABatch)
{
    const Train::InMemoryObjective objective;
    const ObjectiveLeaves leaves;
    Graph::TensorPool pool(backend.engine());
    const Data::Batch batch{};

    SA_EXPECT_FAILURE(objective.feed(batch, leaves, pool), StatusCode::Unsupported,
                      "'in-memory'", "fed by its own step");
    EXPECT_FALSE(objective.is_fed_by_batch());
}

/** @brief Tests that the base measures nothing and adds no metric. */
TEST_F(Small_ObjectiveScore, AnObjectiveWithoutMetricsMeasuresNothing)
{
    const Train::InMemoryObjective objective;
    const Data::Batch batch{};

    Train::Metrics metrics;
    SA_EXPECT_OK(objective.measure(view_of(batch, {}), metrics));
    EXPECT_TRUE(metrics.all().empty());
}

/** @brief Tests that a leaf the objective did not declare, or the pool does not hold, is named. */
TEST_P(Small_ObjectiveLeafLookup, ALeafWithoutATensorIsNamed)
{
    const LeafLookupRow& row = GetParam();
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    ObjectiveLeaves leaves;
    leaves.add("targets", builder.input(Shape{2, 3}, "targets"));
    Graph::TensorPool pool(backend.engine());

    SA_EXPECT_FAILURE(Train::find_leaf_tensor(leaves, row.role, pool),
                      StatusCode::InvalidDescription, row.first, row.second);
}

INSTANTIATE_TEST_SUITE_P(
    Rows, Small_ObjectiveLeafLookup,
    ::testing::Values(LeafLookupRow{"ALeafTheObjectiveDidNotDeclare", "collocation",
                                    "no leaf 'collocation'", "no leaf 'collocation'"},
                      LeafLookupRow{"ALeafThePoolHoldsNoTensorFor", "targets", "no tensor",
                                    "'targets'"}),
    [](const ::testing::TestParamInfo<LeafLookupRow>& info) { return info.param.name; });

/** @brief Tests that the leaves report what they hold and what they do not. */
TEST(Small_Objective, LeavesReportWhatTheyHold)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    ObjectiveLeaves leaves;
    leaves.add("targets", builder.input(Shape{2, 3}, "targets"));

    EXPECT_TRUE(leaves.has("targets"));
    EXPECT_FALSE(leaves.has("collocation"));
    EXPECT_EQ(leaves.all().size(), 1u);
}

/** @brief Tests that adding a role twice stops the process and names the role. */
TEST(Medium_ObjectiveLeavesDeathTest, ARoleAddedTwiceIsFatal)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    ObjectiveLeaves leaves;
    leaves.add("targets", builder.input(Shape{2, 3}, "targets"));
    const Graph::ValueID again = builder.input(Shape{2, 3}, "again");

    SA_EXPECT_FATAL(leaves.add("targets", again), "'targets'", "twice");
}

/** @brief Tests that asking for an absent role stops the process and lists the roles held. */
TEST(Medium_ObjectiveLeavesDeathTest, AnAbsentRoleIsFatalAndTheRolesHeldAreListed)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    ObjectiveLeaves leaves;
    leaves.add("targets", builder.input(Shape{2, 3}, "targets"));

    SA_EXPECT_FATAL((void)leaves.at("collocation"), "'collocation'", "'targets'");
}

/** @brief Tests that an empty leaf set says so rather than listing nothing. */
TEST(Medium_ObjectiveLeavesDeathTest, AnEmptyLeafSetSaysSoRatherThanListingNothing)
{
    const ObjectiveLeaves leaves;

    SA_EXPECT_FATAL((void)leaves.at("targets"), "'targets'", "(none)");
}

/**
 * @brief Tests that the loss is computed from the value the model published: for each set of
 *        logits it equals the mean over the batch of the cross-entropy against the one-hot
 *        targets, so the loss moves when the output moves.
 */
TEST_P(Small_ObjectiveLoss, CrossEntropyReadsTheOnlyOutput)
{
    constexpr std::size_t K_ROWS = 3;
    constexpr std::size_t K_CLASSES = 4;
    const std::vector<double>& logits = GetParam();
    const std::vector<double> targets{0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0};
    const CrossEntropyObjective objective(K_ROWS, K_CLASSES);

    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    ObjectiveLeaves leaves;
    objective.declare(builder, leaves);
    NN::ModelOutputs outputs;
    const Graph::ValueID logits_leaf = builder.input(Shape{3, 4}, "logits");
    SA_EXPECT_OK(outputs.add("output", logits_leaf));
    SA_ASSERT_OK_AND_ASSIGN(const Graph::ValueID loss,
                            objective.loss(builder, graph, outputs, leaves));
    Graph::HostEvaluator evaluator(graph);
    SA_EXPECT_OK(evaluator.write(logits_leaf, logits));
    SA_EXPECT_OK(evaluator.write(leaves.at("targets"), targets));
    SA_EXPECT_OK(evaluator.run());

    double expected = 0.0;
    for (std::size_t row = 0; row < K_ROWS; ++row)
    {
        double normaliser = 0.0;
        for (std::size_t c = 0; c < K_CLASSES; ++c)
            normaliser += std::exp(logits[row * K_CLASSES + c]);
        for (std::size_t c = 0; c < K_CLASSES; ++c)
            expected -= targets[row * K_CLASSES + c] *
                        (logits[row * K_CLASSES + c] - std::log(normaliser));
    }
    expected /= static_cast<double>(K_ROWS);

    SA_ASSERT_OK_AND_ASSIGN(const std::vector<double> read, evaluator.read(loss));
    ASSERT_EQ(read.size(), 1u);
    EXPECT_TRUE(TestSupport::is_within(
        TestSupport::measure_relative_error<double>(read[0], expected, K_LOSS_ROUNDED_OPERATIONS)));
}

INSTANTIATE_TEST_SUITE_P(
    Logits, Small_ObjectiveLoss,
    ::testing::Values(
        std::vector<double>{1.0, 2.0, 0.5, -1.0, 0.0, 0.25, 3.0, -2.0, 1.5, 1.5, 0.0, 0.5},
        std::vector<double>{-0.5, 0.75, 2.5, 1.0, 1.25, -1.5, 0.0, 0.5, -2.0, 0.25, 1.75, 1.0}));

/** @brief Tests that the factory builds cross-entropy over the batch and classes it is given. */
TEST(Small_Objective, TheFactoryBuildsCrossEntropyOverTheContext)
{
    const Config::Value empty = SaTest::require_value(Config::parse_json("{}", "<test>"));
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
TEST(Small_Objective, CrossEntropyTakesNoKeysAtAll)
{
    // Cross-entropy accepts no key; see tests/layers/train/README.md.
    const Config::Value stray =
        SaTest::require_value(Config::parse_json(R"({"label_smoothing": 0.1})", "<test>"));
    Train::ObjectiveContext given;
    given.batch = 4;
    given.classes = 3;

    SA_EXPECT_FAILURE(Train::make_cross_entropy(given, stray, "objective"),
                      StatusCode::InvalidDescription, "objective", "'label_smoothing'");
}

/** @brief Tests that data which states no class count cannot build cross-entropy. */
TEST(Small_Objective, CrossEntropyNeedsTheClassCountOfTheData)
{
    const Config::Value empty = SaTest::require_value(Config::parse_json("{}", "<test>"));
    Train::ObjectiveContext given;
    given.batch = 4;

    SA_EXPECT_FAILURE(Train::make_cross_entropy(given, empty, "objective"),
                      StatusCode::InvalidDescription, "objective", "class count", "0 classes");
}

namespace
{
    /** @brief Holds the samples of the batch the graph cases build. */
    constexpr int64_t K_BATCH = 4;

    /** @brief Holds the features of a sample of the graph cases. */
    constexpr int64_t K_FEATURES = 6;

    /** @brief Returns what a factory is given for a model that publishes @p output. */
    Train::ObjectiveContext context_of(const Shape& output)
    {
        Train::ObjectiveContext given;
        given.batch = K_BATCH;
        given.input_sample = Shape{K_FEATURES};
        given.outputs.push_back(Train::OutputShape{"output", output});
        return given;
    }

    /** @brief Returns the attributes @p text spells. */
    Config::Value keys(const char* text)
    {
        return SaTest::require_value(Config::parse_json(text, "<test>"));
    }

} // namespace

/** @brief Tests that the binary make refuses a batch below 1. */
TEST(Small_ObjectiveGraph, BinaryMakeRefusesABatchBelowOne)
{
    SA_EXPECT_FAILURE(BinaryCrossEntropyObjective::make(0), StatusCode::InvalidDescription,
                      "binary_cross_entropy", "at least 1");
}

/** @brief Tests that the binary make refuses a threshold on the bound. */
TEST(Small_ObjectiveGraph, BinaryMakeRefusesAThresholdOnTheBound)
{
    SA_EXPECT_FAILURE(BinaryCrossEntropyObjective::make(K_BATCH, 1.0),
                      StatusCode::InvalidDescription, "above 0 and below 1");
}

/** @brief Tests that the binary make refuses an int32 label type. */
TEST(Small_ObjectiveGraph, BinaryMakeRefusesAnInt32LabelType)
{
    SA_EXPECT_FAILURE(BinaryCrossEntropyObjective::make(K_BATCH, 0.5, DType::Int32),
                      StatusCode::InvalidDescription, "float32", "int32");
}

/** @brief Tests that the multi-label make refuses a width below 1. */
TEST(Small_ObjectiveGraph, MultiLabelMakeRefusesAWidthBelowOne)
{
    SA_EXPECT_FAILURE(MultiLabelObjective::make(K_BATCH, 0), StatusCode::InvalidDescription,
                      "multi_label", "at least 1");
}

/** @brief Tests that the multi-label make refuses a threshold on the bound. */
TEST(Small_ObjectiveGraph, MultiLabelMakeRefusesAThresholdOnTheBound)
{
    SA_EXPECT_FAILURE(MultiLabelObjective::make(K_BATCH, 3, 0.0),
                      StatusCode::InvalidDescription, "above 0 and below 1");
}

/** @brief Tests that the regression make refuses a width below 1. */
TEST(Small_ObjectiveGraph, MeanSquaredErrorMakeRefusesAWidthBelowOne)
{
    SA_EXPECT_FAILURE(MeanSquaredErrorObjective::make(K_BATCH, 0),
                      StatusCode::InvalidDescription, "mean_squared_error", "at least 1");
}

/** @brief Tests that the binary factory takes its threshold and batch from its context. */
TEST(Small_ObjectiveGraph, TheBinaryFactoryReadsTheOneOutput)
{
    SA_ASSERT_OK_AND_ASSIGN(
        const std::unique_ptr<Train::Objective> binary,
        Train::make_binary_cross_entropy(context_of(Shape{K_BATCH, 1}),
                                         keys(R"({"threshold": 0.9})"), "objective"));

    EXPECT_EQ(static_cast<const BinaryCrossEntropyObjective&>(*binary).threshold(), 0.9);
    EXPECT_EQ(static_cast<const BinaryCrossEntropyObjective&>(*binary).batch(), K_BATCH);
}

/** @brief Tests that the multi-label factory takes its class count from the output. */
TEST(Small_ObjectiveGraph, TheMultiLabelFactoryReadsTheOneOutput)
{
    SA_ASSERT_OK_AND_ASSIGN(
        const std::unique_ptr<Train::Objective> multi_label,
        Train::make_multi_label(context_of(Shape{K_BATCH, 5}), keys("{}"), "objective"));

    EXPECT_EQ(static_cast<const MultiLabelObjective&>(*multi_label).classes(), 5);
    EXPECT_EQ(static_cast<const MultiLabelObjective&>(*multi_label).threshold(), 0.5);
}

/** @brief Tests that the regression factory takes its width from the output. */
TEST(Small_ObjectiveGraph, TheMeanSquaredErrorFactoryReadsTheOneOutput)
{
    SA_ASSERT_OK_AND_ASSIGN(
        const std::unique_ptr<Train::Objective> regression,
        Train::make_mean_squared_error(context_of(Shape{K_BATCH, 2}), keys("{}"),
                                       "objective"));

    EXPECT_EQ(static_cast<const MeanSquaredErrorObjective&>(*regression).width(), 2);
}

/** @brief Tests that the binary factory refuses an output of more than one logit per sample. */
TEST(Small_ObjectiveGraph, TheBinaryFactoryRefusesAnOutputOfTwoLogits)
{
    SA_EXPECT_FAILURE(
        Train::make_binary_cross_entropy(context_of(Shape{K_BATCH, 2}), keys("{}"), "objective"),
        StatusCode::InvalidDescription, "objective", "one logit per sample", "[4, 2]");
}

/** @brief Tests that the multi-label factory refuses an output of rank 3. */
TEST(Small_ObjectiveGraph, TheMultiLabelFactoryRefusesAnOutputOfRankThree)
{
    SA_EXPECT_FAILURE(
        Train::make_multi_label(context_of(Shape{K_BATCH, 3, 5}), keys("{}"), "objective"),
        StatusCode::InvalidDescription, "objective", "[4, columns]", "[4, 3, 5]");
}

/** @brief Tests that the regression factory refuses an output of another batch. */
TEST(Small_ObjectiveGraph, TheMeanSquaredErrorFactoryRefusesAnOutputOfAnotherBatch)
{
    SA_EXPECT_FAILURE(
        Train::make_mean_squared_error(context_of(Shape{8, 2}), keys("{}"), "objective"),
        StatusCode::InvalidDescription, "objective", "[4, columns]", "[8, 2]");
}

/** @brief Tests that a factory refuses a model that publishes a second output. */
TEST(Small_ObjectiveGraph, AFactoryRefusesAModelWithTwoOutputs)
{
    Train::ObjectiveContext two = context_of(Shape{K_BATCH, 1});
    two.outputs.push_back(Train::OutputShape{"second", Shape{K_BATCH, 1}});

    SA_EXPECT_FAILURE(Train::make_binary_cross_entropy(two, keys("{}"), "objective"),
                      StatusCode::InvalidDescription, "objective", "publishes 2");
}

/** @brief Tests that the regression factory refuses a stray key and names it. */
TEST(Small_ObjectiveGraph, TheMeanSquaredErrorFactoryRefusesAStrayKey)
{
    SA_EXPECT_FAILURE(
        Train::make_mean_squared_error(context_of(Shape{K_BATCH, 2}), keys(R"({"huber": 1})"),
                                       "objective"),
        StatusCode::InvalidDescription, "objective", "'huber'");
}

/** @brief Tests that the multi-label factory refuses a stray key and names the one it knows. */
TEST(Small_ObjectiveGraph, TheMultiLabelFactoryRefusesAStrayKey)
{
    SA_EXPECT_FAILURE(
        Train::make_multi_label(context_of(Shape{K_BATCH, 3}), keys(R"({"weights": 1})"),
                                "objective"),
        StatusCode::InvalidDescription, "objective", "'weights'", "'threshold'");
}
