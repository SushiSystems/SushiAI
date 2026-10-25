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
#include <SushiAI/engine_fixture/engine_test.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/graph/builder.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/graph/tensor_pool.hpp>
#include <SushiAI/nn/layers.hpp>
#include <SushiAI/tensor_values/tensor_values.hpp>
#include <SushiAI/train/binary_cross_entropy_objective.hpp>
#include <SushiAI/train/mean_squared_error_objective.hpp>
#include <SushiAI/train/metrics.hpp>
#include <SushiAI/train/multi_label_objective.hpp>
#include <SushiAI/train/objective.hpp>
#include <SushiAI/train/objective_context.hpp>
#include <SushiAI/train/step_view.hpp>
#include <SushiAI/train/steps.hpp>
#include <SushiAI/train/training_graph.hpp>

using namespace SushiAI;
using Train::BinaryCrossEntropyObjective;
using Train::MeanSquaredErrorObjective;
using Train::MultiLabelObjective;
using Train::ObjectiveLeaves;
namespace SaTest = SushiAI::Test;

namespace
{
    /** @brief Bounds the difference between a ratio of small counts and its exact value. */
    constexpr double K_RATIO_TOLERANCE = 1e-12;

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

    /** @brief Provides the engine the outputs, the roles and the leaves are allocated on. */
    class Medium_BinaryCrossEntropyObjective : public SaTest::EngineTest
    {
        protected:
            /** @brief Returns a float32 tensor of @p shape that holds @p values. */
            [[nodiscard]] Result<SushiBLAS::Tensor> tensor_of(const Shape& shape,
                                                              const std::vector<double>& values)
            {
                SA_TRY_ASSIGN(SushiBLAS::Tensor tensor, allocate_tensor(*engine, shape));
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

    EXPECT_EQ(objective->name(), "binary_cross_entropy");
    EXPECT_EQ(objective->threshold(), 0.5);
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

/** @brief Tests that the binary feed copies the labels and refuses one that is not 0 or 1. */
TEST_F(Medium_BinaryCrossEntropyObjective, BinaryFeedCopiesLabelsOfZeroAndOne)
{
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<BinaryCrossEntropyObjective> objective,
                            BinaryCrossEntropyObjective::make(K_BATCH));
    Graph::Graph graph;
    ObjectiveLeaves leaves;
    Graph::TensorPool pool(*engine);
    SA_ASSERT_OK_AND_ASSIGN(const Graph::ValueID targets,
                            bind_targets(*objective, graph, leaves, pool));

    SA_ASSERT_OK_AND_ASSIGN(
        const Data::Batch soft,
        batch_of(Data::K_ROLE_LABELS, Shape{K_BATCH}, {1.0, 0.25, 0.0, 1.0}));
    SA_EXPECT_FAILURE(objective->feed(soft, leaves, pool), StatusCode::Incompatible,
                      "element 1", "0 or 1");
    EXPECT_EQ(SaTest::read_tensor(pool.tensor(targets)), std::vector<double>(4, 7.0));

    SA_ASSERT_OK_AND_ASSIGN(
        const Data::Batch batch,
        batch_of(Data::K_ROLE_LABELS, Shape{K_BATCH}, {1.0, 0.0, 0.0, 1.0}));
    SA_EXPECT_OK(objective->feed(batch, leaves, pool));
    EXPECT_EQ(SaTest::read_tensor(pool.tensor(targets)),
              (std::vector<double>{1.0, 0.0, 0.0, 1.0}));
}

/** @brief Tests the three binary metrics on five rows, and the recall a higher threshold costs. */
TEST_F(Medium_BinaryCrossEntropyObjective, BinaryMeasuresAccuracyPrecisionAndRecall)
{
    // Rows: hit, false alarm, miss, correct rejection, hit.
    SA_ASSERT_OK_AND_ASSIGN(const SushiBLAS::Tensor logits,
                            tensor_of(Shape{5, 1}, {2.0, 1.0, -1.0, -3.0, 4.0}));
    SA_ASSERT_OK_AND_ASSIGN(
        const Data::Batch batch,
        batch_of(Data::K_ROLE_LABELS, Shape{5}, {1.0, 0.0, 1.0, 0.0, 1.0}));
    const std::vector<Train::StepOutput> outputs{Train::StepOutput{"output", &logits}};

    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<BinaryCrossEntropyObjective> objective,
                            BinaryCrossEntropyObjective::make(5));
    Train::Metrics metrics;
    SA_EXPECT_OK(objective->measure(view_of(batch, outputs), metrics));

    ASSERT_EQ(metrics.all().size(), 3u);
    EXPECT_EQ(metrics.all()[0].name, "accuracy");
    EXPECT_EQ(metrics.all()[1].name, "precision");
    EXPECT_EQ(metrics.all()[2].name, "recall");
    EXPECT_NEAR(metrics.all()[0].value, 3.0 / 5.0, K_RATIO_TOLERANCE);
    EXPECT_NEAR(metrics.all()[1].value, 2.0 / 3.0, K_RATIO_TOLERANCE);
    EXPECT_NEAR(metrics.all()[2].value, 2.0 / 3.0, K_RATIO_TOLERANCE);

    // sigmoid(2) is 0.88 and sigmoid(4) is 0.98, so 0.95 keeps the last row alone.
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<BinaryCrossEntropyObjective> strict,
                            BinaryCrossEntropyObjective::make(5, 0.95));
    Train::Metrics at_strict;
    SA_EXPECT_OK(strict->measure(view_of(batch, outputs), at_strict));
    EXPECT_NEAR(at_strict.find("precision")->value, 1.0, K_RATIO_TOLERANCE);
    EXPECT_NEAR(at_strict.find("recall")->value, 1.0 / 3.0, K_RATIO_TOLERANCE);
}
