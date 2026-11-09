/**************************************************************************/
/* test_binary_cross_entropy_objective.cpp                                */
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
 * @file test_binary_cross_entropy_objective.cpp
 * @brief Tests the binary cross-entropy objective: its graph, its feed and its metrics.
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
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/graph/tensor_pool.hpp>
#include <SushiAI/nn/layers.hpp>
#include <SushiAI/tensor_values/tensor_values.hpp>
#include <SushiAI/train/binary_cross_entropy_objective.hpp>
#include <SushiAI/train/decision_threshold.hpp>
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
using Train::BinaryCrossEntropyObjective;
using Train::MeanSquaredErrorObjective;
using Train::MultiLabelObjective;
using Train::ObjectiveLeaves;
namespace SaTest = SushiAI::Test;

namespace
{
    /** @brief The rounded operations on the longest path from a count to a ratio. */
    constexpr std::size_t K_RATIO_OPERATIONS = 4;

    /** @brief Holds the samples of the batch the graph cases build. */
    constexpr int64_t K_BATCH = 4;

    /** @brief Holds the features of a sample of the graph cases. */
    constexpr int64_t K_FEATURES = 6;

    using MLP = NN::Sequential<NN::Linear, NN::ReLU, NN::Linear>;

    /** @brief Returns a model of K_FEATURES inputs and @p outputs outputs. */
    MLP make_model(int64_t outputs)
    {
        return MLP(NN::Linear{K_FEATURES, 8}, NN::ReLU{}, NN::Linear{8, outputs});
    }

    /** @brief Provides the host engine the outputs, the roles and the leaves are allocated on. */
    class Small_BinaryCrossEntropyObjective : public ::testing::Test
    {
        protected:
            /** @brief Returns a float32 tensor of @p shape that holds @p values. */
            [[nodiscard]] Result<SushiBLAS::Tensor> tensor_of(const Shape& shape,
                                                              const std::vector<double>& values)
            {
                SA_TRY_ASSIGN(SushiBLAS::Tensor tensor, allocate_tensor(backend.engine(), shape));
                SaTest::write_tensor(tensor, values);
                return tensor;
            }

            /** @brief Returns a batch that holds @p values of @p shape under @p role. */
            [[nodiscard]] Result<Data::Batch> batch_of(std::string_view role, const Shape& shape,
                                                       const std::vector<double>& values)
            {
                Data::BatchTarget target;
                target.role = std::string(role);
                SA_TRY_ASSIGN(target.values, tensor_of(shape, values));

                Data::Batch batch;
                batch.targets.push_back(std::move(target));
                return batch;
            }

            /**
             * @brief Declares the leaves of @p objective and binds a tensor of 7s to "targets".
             * @return The id of the leaf.
             */
            [[nodiscard]] Result<Graph::ValueID> bind_targets(const Train::Objective& objective,
                                                              Graph::Graph& graph,
                                                              ObjectiveLeaves& leaves,
                                                              Graph::TensorPool& pool)
            {
                const Graph::GraphBuilder builder(graph);
                objective.declare(builder, leaves);
                const Graph::ValueID id = leaves.at("targets");
                const Shape shape = graph.value(id).shape;

                SA_TRY_ASSIGN(SushiBLAS::Tensor stale,
                              tensor_of(shape, std::vector<double>(
                                                   static_cast<std::size_t>(shape.num_elements()),
                                                   7.0)));
                SA_TRY(pool.bind(id, stale));
                return id;
            }

            /** @brief Returns the view of a step over @p batch whose loss is 0. */
            [[nodiscard]] static Train::StepView view_of(
                const Data::Batch& batch, const std::vector<Train::StepOutput>& outputs)
            {
                return Train::StepView{
                    batch, 0.0,
                    SushiRuntime::span<const Train::StepOutput>(outputs.data(), outputs.size())};
            }

            /**
             * @brief Measures the five rows of the metric cases at @p threshold into @p metrics.
             *
             * Rows: hit, false alarm, miss, correct rejection, hit, on the logits 2, 1, -1, -3, 4.
             */
            void measure_five_rows(double threshold, Train::Metrics& metrics)
            {
                SA_ASSERT_OK_AND_ASSIGN(const SushiBLAS::Tensor logits,
                                        tensor_of(Shape{5, 1}, {2.0, 1.0, -1.0, -3.0, 4.0}));
                SA_ASSERT_OK_AND_ASSIGN(
                    const Data::Batch batch,
                    batch_of(Data::K_ROLE_LABELS, Shape{5}, {1.0, 0.0, 1.0, 0.0, 1.0}));
                const std::vector<Train::StepOutput> outputs{Train::StepOutput{"output", &logits}};
                SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<BinaryCrossEntropyObjective> objective,
                                        BinaryCrossEntropyObjective::make(5, threshold));

                SA_EXPECT_OK(objective->measure(view_of(batch, outputs), metrics));
            }

            /** @brief The host backend the tensors and the pool are allocated from. */
            HostExecutionBackend backend;
    };

} // namespace

/** @brief Tests that the binary objective declares an [N, 1] leaf and float32 [N] labels. */
TEST(Small_BinaryCrossEntropyObjectiveGraph, BinaryDeclaresAColumnLeafAndRealLabels)
{
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<BinaryCrossEntropyObjective> objective,
                            BinaryCrossEntropyObjective::make(K_BATCH));
    MLP model = make_model(1);
    SA_ASSERT_OK_AND_ASSIGN(
        const Train::TrainingGraph built,
        Train::build_training_graph(model, *objective, Shape{K_BATCH, K_FEATURES}));

    EXPECT_EQ(built.graph.value(built.leaves.at("targets")).shape, (Shape{K_BATCH, 1}));
    EXPECT_EQ(built.spec.targets, built.leaves.at("targets"));
    EXPECT_EQ(built.graph.value(built.spec.loss).shape.num_elements(), 1);

    const std::vector<Data::TargetDescription>& required = built.leaves.required();
    ASSERT_EQ(required.size(), 1u);
    EXPECT_EQ(required.front().role, Data::K_ROLE_LABELS);
    EXPECT_EQ(required.front().sample_shape, Shape{});
    EXPECT_EQ(required.front().dtype, DType::Float32);
    EXPECT_EQ(required.front().classes, 0);
}

/** @brief Tests that the binary objective cuts at one half unless it is given a threshold. */
TEST(Small_BinaryCrossEntropyObjectiveGraph, TheThresholdDefaultsToOneHalf)
{
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<BinaryCrossEntropyObjective> objective,
                            BinaryCrossEntropyObjective::make(K_BATCH));

    EXPECT_EQ(objective->threshold(), Train::K_DEFAULT_THRESHOLD);
}

/** @brief Tests that the binary feed refuses a label that is not 0 or 1 and keeps the leaf. */
TEST_F(Small_BinaryCrossEntropyObjective, BinaryFeedRefusesALabelThatIsNotZeroOrOne)
{
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<BinaryCrossEntropyObjective> objective,
                            BinaryCrossEntropyObjective::make(K_BATCH));
    Graph::Graph graph;
    ObjectiveLeaves leaves;
    Graph::TensorPool pool(backend.engine());
    SA_ASSERT_OK_AND_ASSIGN(const Graph::ValueID targets,
                            bind_targets(*objective, graph, leaves, pool));

    SA_ASSERT_OK_AND_ASSIGN(
        const Data::Batch soft,
        batch_of(Data::K_ROLE_LABELS, Shape{K_BATCH}, {1.0, 0.25, 0.0, 1.0}));
    SA_EXPECT_FAILURE(objective->feed(soft, leaves, pool), StatusCode::Incompatible,
                      "element 1", "0 or 1");
    EXPECT_EQ(SaTest::read_tensor(pool.tensor(targets)), std::vector<double>(4, 7.0));
}

/** @brief Tests that the binary feed copies labels of 0 and 1 into the leaf. */
TEST_F(Small_BinaryCrossEntropyObjective, BinaryFeedCopiesLabelsOfZeroAndOne)
{
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<BinaryCrossEntropyObjective> objective,
                            BinaryCrossEntropyObjective::make(K_BATCH));
    Graph::Graph graph;
    ObjectiveLeaves leaves;
    Graph::TensorPool pool(backend.engine());
    SA_ASSERT_OK_AND_ASSIGN(const Graph::ValueID targets,
                            bind_targets(*objective, graph, leaves, pool));

    SA_ASSERT_OK_AND_ASSIGN(
        const Data::Batch batch,
        batch_of(Data::K_ROLE_LABELS, Shape{K_BATCH}, {1.0, 0.0, 0.0, 1.0}));
    SA_EXPECT_OK(objective->feed(batch, leaves, pool));
    EXPECT_EQ(SaTest::read_tensor(pool.tensor(targets)),
              (std::vector<double>{1.0, 0.0, 0.0, 1.0}));
}

/** @brief Tests that the binary metrics are accuracy, precision and recall, in that order. */
TEST_F(Small_BinaryCrossEntropyObjective, BinaryMeasuresTheThreeMetricsInOrder)
{
    Train::Metrics metrics;
    measure_five_rows(Train::K_DEFAULT_THRESHOLD, metrics);

    ASSERT_EQ(metrics.all().size(), 3u);
    EXPECT_EQ(metrics.all()[0].name, "accuracy");
    EXPECT_EQ(metrics.all()[1].name, "precision");
    EXPECT_EQ(metrics.all()[2].name, "recall");
}

/** @brief Tests the three binary metrics on five rows at the threshold one half. */
TEST_F(Small_BinaryCrossEntropyObjective, BinaryMeasuresAccuracyPrecisionAndRecall)
{
    Train::Metrics metrics;
    measure_five_rows(Train::K_DEFAULT_THRESHOLD, metrics);

    EXPECT_TRUE(TestSupport::is_within(TestSupport::measure_relative_error<double>(
        metrics.find("accuracy")->value, 3.0 / 5.0, K_RATIO_OPERATIONS)));
    EXPECT_TRUE(TestSupport::is_within(TestSupport::measure_relative_error<double>(
        metrics.find("precision")->value, 2.0 / 3.0, K_RATIO_OPERATIONS)));
    EXPECT_TRUE(TestSupport::is_within(TestSupport::measure_relative_error<double>(
        metrics.find("recall")->value, 2.0 / 3.0, K_RATIO_OPERATIONS)));
}

/** @brief Tests the recall a higher threshold costs: sigmoid(4) is 0.98, sigmoid(2) is 0.88. */
TEST_F(Small_BinaryCrossEntropyObjective, ARaisedThresholdKeepsTheLastRowAlone)
{
    Train::Metrics metrics;
    measure_five_rows(0.95, metrics);

    EXPECT_TRUE(TestSupport::is_within(TestSupport::measure_relative_error<double>(
        metrics.find("precision")->value, 1.0, K_RATIO_OPERATIONS)));
    EXPECT_TRUE(TestSupport::is_within(TestSupport::measure_relative_error<double>(
        metrics.find("recall")->value, 1.0 / 3.0, K_RATIO_OPERATIONS)));
}
