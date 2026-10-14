/**************************************************************************/
/* test_mean_squared_error_objective.cpp                                  */
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
 * @file test_mean_squared_error_objective.cpp
 * @brief Tests the mean squared error objective: its graph, its feed and its metric.
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
#include <SushiAI/SushiAI.h>
#include <SushiAI/config/value.hpp>
#include <SushiAI/core/allocation.hpp>
#include <SushiAI/core/common.hpp>
#include <SushiAI/core/result.hpp>
#include <SushiAI/core/shape.hpp>
#include <SushiAI/core/status.hpp>
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
    class Unit_MeanSquaredErrorObjective : public SaTest::EngineTest
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

/** @brief Tests that the regression objective declares an [N, K] leaf and the values role. */
TEST(Unit_MeanSquaredErrorObjectiveGraph, MeanSquaredErrorDeclaresTheValuesRole)
{
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<MeanSquaredErrorObjective> objective,
                            MeanSquaredErrorObjective::make(K_BATCH, 2));
    MLP model = make_model(2);
    SA_ASSERT_OK_AND_ASSIGN(
        const Train::TrainingGraph built,
        Train::build_training_graph(model, *objective, Shape{K_BATCH, K_FEATURES}));

    EXPECT_EQ(objective->name(), "mean_squared_error");
    EXPECT_EQ(objective->width(), 2);
    EXPECT_EQ(built.graph.value(built.leaves.at("targets")).shape, (Shape{K_BATCH, 2}));
    EXPECT_EQ(built.graph.value(built.spec.loss).shape.num_elements(), 1);

    ASSERT_EQ(built.leaves.required().size(), 1u);
    EXPECT_EQ(built.leaves.required().front().role, Data::K_ROLE_VALUES);
    EXPECT_EQ(built.leaves.required().front().sample_shape, (Shape{2}));
    EXPECT_EQ(built.leaves.required().front().dtype, DType::Float32);
}

/** @brief Tests that the regression feed reads the values role and names it when absent. */
TEST_F(Unit_MeanSquaredErrorObjective, MeanSquaredErrorFeedReadsTheValuesRole)
{
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<MeanSquaredErrorObjective> objective,
                            MeanSquaredErrorObjective::make(2, 2));
    Graph::Graph graph;
    ObjectiveLeaves leaves;
    Graph::TensorPool pool(*engine);
    SA_ASSERT_OK_AND_ASSIGN(const Graph::ValueID targets,
                            bind_targets(*objective, graph, leaves, pool));

    SA_ASSERT_OK_AND_ASSIGN(
        const Data::Batch labelled,
        batch_of(Data::K_ROLE_LABELS, Shape{2, 2}, {1.0, 0.0, 0.0, 1.0}));
    SA_EXPECT_FAILURE(objective->feed(labelled, leaves, pool), StatusCode::InvalidDescription,
                      "'values'");

    SA_ASSERT_OK_AND_ASSIGN(
        const Data::Batch batch,
        batch_of(Data::K_ROLE_VALUES, Shape{2, 2}, {1.5, -2.0, 0.25, 8.0}));
    SA_EXPECT_OK(objective->feed(batch, leaves, pool));
    EXPECT_EQ(SaTest::read_tensor(pool.tensor(targets)),
              (std::vector<double>{1.5, -2.0, 0.25, 8.0}));
}

/** @brief Tests that the regression metric is the mean absolute error of every element. */
TEST_F(Unit_MeanSquaredErrorObjective, MeanSquaredErrorMeasuresTheMeanAbsoluteError)
{
    SA_ASSERT_OK_AND_ASSIGN(const SushiBLAS::Tensor predicted,
                            tensor_of(Shape{2, 2}, {1.0, 2.0, 3.0, 4.0}));
    SA_ASSERT_OK_AND_ASSIGN(
        const Data::Batch batch,
        batch_of(Data::K_ROLE_VALUES, Shape{2, 2}, {1.5, 2.0, 2.0, 6.0}));
    const std::vector<Train::StepOutput> outputs{Train::StepOutput{"output", &predicted}};

    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<MeanSquaredErrorObjective> objective,
                            MeanSquaredErrorObjective::make(2, 2));
    Train::Metrics metrics;
    SA_EXPECT_OK(objective->measure(view_of(batch, outputs), metrics));

    ASSERT_EQ(metrics.all().size(), 1u);
    EXPECT_EQ(metrics.all()[0].name, "mean_absolute_error");
    EXPECT_FALSE(metrics.all()[0].is_fraction);
    EXPECT_NEAR(metrics.all()[0].value, 3.5 / 4.0, K_RATIO_TOLERANCE);
}
