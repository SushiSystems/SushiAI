/**************************************************************************/
/* test_dense_objectives.cpp                                              */
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
 * @file test_dense_objectives.cpp
 * @brief Tests the binary, multi-label and regression objectives through Train::Trainer.
 * @author Mustafa Garip
 */

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>
#include <gtest/gtest.h>
#include <SushiBLAS/engine/engine.hpp>
#include <SushiBLAS/tensor/tensor.hpp>
#include <SushiAI/core/common.hpp>
#include <SushiAI/core/result.hpp>
#include <SushiAI/core/shape.hpp>
#include <SushiAI/core/status.hpp>
#include <SushiAI/data/dataset.hpp>
#include <SushiAI/data/input.hpp>
#include <SushiAI/data/materialised_dataset.hpp>
#include <SushiAI/data/roles.hpp>
#include <SushiAI/data/target.hpp>
#include <SushiAI/graph/tensor_pool.hpp>
#include <SushiAI/nn/layers.hpp>
#include <SushiAI/nn/model_state.hpp>
#include <SushiAI/optim/optimizer.hpp>
#include <SushiAI/optim/sgd.hpp>
#include <SushiAI/train/binary_cross_entropy_objective.hpp>
#include <SushiAI/train/mean_squared_error_objective.hpp>
#include <SushiAI/train/metrics.hpp>
#include <SushiAI/train/multi_label_objective.hpp>
#include <SushiAI/train/objective.hpp>
#include <SushiAI/train/steps.hpp>
#include <SushiAI/train/trainer.hpp>
#include <SushiAI/train/training_graph.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/SushiAI.h>
#include <SushiAI/engine_fixture/engine_test.hpp>
#include <SushiAI/tensor_values/tensor_values.hpp>

using namespace SushiAI;

// SaTest stays reachable inside a TEST_F body; see tests/integration/README.md.
namespace SaTest = SushiAI::Test;

namespace
{
    /** @brief Holds the samples of a batch, N. */
    constexpr int64_t K_BATCH = 4;

    /** @brief Holds the features of a sample. */
    constexpr int64_t K_FEATURES = 3;

    /** @brief Bounds the absolute error of a float32 loss of order one. */
    constexpr double K_TOLERANCE = 1e-5;

    /** @brief Bounds the absolute error of a float32 loss of order one hundred. */
    constexpr double K_LARGE_TOLERANCE = 1e-3;

    /** @brief Bounds the difference between a ratio of small counts and its exact value. */
    constexpr double K_RATIO_TOLERANCE = 1e-12;

    /** @brief Stands for a metric the step did not report; no ratio and no error equals it. */
    constexpr double K_ABSENT = -1.0;

    /** @brief Returns the [N, F] inputs: element i holds ((5 i) mod 7 - 3) / 2 times @p scale. */
    std::vector<double> make_inputs(double scale)
    {
        std::vector<double> inputs;
        for (int64_t index = 0; index < K_BATCH * K_FEATURES; ++index)
            inputs.push_back(scale * static_cast<double>((5 * index) % 7 - 3) / 2.0);

        return inputs;
    }

    /** @brief Returns the input of a batch: K_FEATURES float32 values per sample. */
    Data::InputDescription input_description()
    {
        return Data::InputDescription{Shape{K_FEATURES}, DType::Float32};
    }

    /** @brief Returns one float32 role of @p sample values per sample. */
    std::vector<Data::TargetDescription> real_role(std::string_view role, const Shape& sample)
    {
        return {Data::TargetDescription{std::string(role), sample, DType::Float32, 0}};
    }

    /** @brief Returns a batch of make_inputs(@p scale) with @p values under the one role. */
    Result<Data::Batch> make_batch(SushiBLAS::Engine& engine,
                                   const std::vector<Data::TargetDescription>& roles,
                                   const std::vector<double>& values, double scale = 1.0)
    {
        SA_TRY_ASSIGN(Data::Batch batch,
                      Data::allocate_batch(engine, K_BATCH, input_description(), roles));
        SaTest::write_tensor(batch.inputs, make_inputs(scale));
        SaTest::write_tensor(batch.targets.front().values, values);
        return batch;
    }

    /** @brief Writes ((3 i) mod 5 - 2) / 4 into element i of every tensor of @p state. */
    Status write_state(NN::ModelState& state)
    {
        for (const NN::StateEntry& entry : state.entries())
        {
            SA_TRY_ASSIGN(SushiBLAS::Tensor* const tensor, state.tensor(entry.state));
            std::vector<double> values;
            for (int64_t index = 0; index < tensor->num_elements; ++index)
                values.push_back(static_cast<double>((3 * index) % 5 - 2) / 4.0);
            SaTest::write_tensor(*tensor, values);
        }

        return Status();
    }

    /** @brief Holds what a case reads of a trainer and the pool its step writes. */
    using Visit = std::function<Status(Train::Trainer&, const SushiBLAS::Tensor& output)>;

    /**
     * @brief Trains a Linear of @p width outputs under @p objective and hands it to @p visit.
     * @param rate The rate of the SGD; 0 leaves the weights write_state() wrote.
     */
    Status with_trainer(SushiBLAS::Engine& engine, const Train::Objective& objective,
                        int64_t width, double rate, const Visit& visit)
    {
        NN::Linear model{K_FEATURES, width};
        SA_TRY_ASSIGN(Train::TrainingGraph built,
                      Train::build_training_graph(model, objective, Shape{K_BATCH, K_FEATURES}));
        SA_TRY_ASSIGN(NN::ModelState state, NN::ModelState::create(engine, model));

        Graph::TensorPool pool(engine);
        SA_TRY_ASSIGN(std::vector<Optim::ParameterSlot> slots,
                      Train::bind_parameters(built, state, pool));
        SA_TRY(write_state(state));

        Optim::SGDOptions sgd;
        sgd.learning_rate = rate;
        SA_TRY_ASSIGN(const std::unique_ptr<Optim::SGD> optimizer,
                      Optim::SGD::make(engine, slots, sgd));
        SA_TRY_ASSIGN(const std::unique_ptr<Train::Trainer> trainer,
                      Train::Trainer::make(engine, built, pool, objective, *optimizer));

        SA_TRY_ASSIGN(const Graph::ValueID output, built.outputs.only());
        return visit(*trainer, pool.tensor(output));
    }

    /** @brief Returns max(x, 0) - x y + log(1 + exp(-|x|)), averaged over every element. */
    double logistic_loss_on_host(const std::vector<double>& logits,
                                 const std::vector<double>& targets)
    {
        double total = 0.0;
        for (std::size_t index = 0; index < logits.size(); ++index)
        {
            const double x = logits[index];
            total += std::max(x, 0.0) - x * targets[index] + std::log1p(std::exp(-std::abs(x)));
        }

        return total / static_cast<double>(logits.size());
    }

    /** @brief Provides the engine the states, the batches and the steps are built on. */
    class Integration_DenseObjectives : public SaTest::EngineTest
    {

    };
} // namespace

/** @brief Tests the binary loss and its three metrics against the host over the step's logits. */
TEST_F(Integration_DenseObjectives, TheBinaryLossAndMetricsEqualTheHostComputation)
{
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<Train::BinaryCrossEntropyObjective> objective,
                            Train::BinaryCrossEntropyObjective::make(K_BATCH));
    const std::vector<double> labels = {1.0, 0.0, 0.0, 1.0};
    SA_ASSERT_OK_AND_ASSIGN(const Data::Batch batch,
                            make_batch(*engine, real_role(Data::K_ROLE_LABELS, Shape{}), labels));

    SA_EXPECT_OK(with_trainer(*engine, *objective, 1, 0.0,
                              [&](Train::Trainer& trainer,
                                  const SushiBLAS::Tensor& output) -> Status
    {
        SA_TRY_ASSIGN(const Train::StepMetrics step, trainer.run_step(batch));
        const std::vector<double> logits = SaTest::read_tensor(output);

        EXPECT_NEAR(step.loss, logistic_loss_on_host(logits, labels), K_TOLERANCE);

        double right = 0.0;
        double predicted = 0.0;
        double hits = 0.0;
        for (std::size_t row = 0; row < logits.size(); ++row)
        {
            const bool positive = logits[row] >= 0.0;
            right += positive == (labels[row] == 1.0) ? 1.0 : 0.0;
            predicted += positive ? 1.0 : 0.0;
            hits += positive && labels[row] == 1.0 ? 1.0 : 0.0;
        }

        // Both signs occur among the logits, so neither ratio divides by zero.
        EXPECT_GT(predicted, 0.0);
        EXPECT_LT(predicted, static_cast<double>(K_BATCH));
        EXPECT_EQ(step.metrics.all().size(), 3u);
        EXPECT_NEAR(step.metrics.value_or("accuracy", K_ABSENT), right / 4.0, K_RATIO_TOLERANCE);
        EXPECT_NEAR(step.metrics.value_or("precision", K_ABSENT), hits / predicted,
                    K_RATIO_TOLERANCE);
        EXPECT_NEAR(step.metrics.value_or("recall", K_ABSENT), hits / 2.0, K_RATIO_TOLERANCE);
        return Status();
    }));
}

/** @brief Tests that logits above one hundred in magnitude give the finite loss of the host. */
TEST_F(Integration_DenseObjectives, ALargeLogitGivesAFiniteBinaryLoss)
{
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<Train::BinaryCrossEntropyObjective> objective,
                            Train::BinaryCrossEntropyObjective::make(K_BATCH));
    const std::vector<double> labels = {1.0, 0.0, 0.0, 1.0};
    SA_ASSERT_OK_AND_ASSIGN(
        const Data::Batch batch,
        make_batch(*engine, real_role(Data::K_ROLE_LABELS, Shape{}), labels, 400.0));

    SA_EXPECT_OK(with_trainer(*engine, *objective, 1, 0.0,
                              [&](Train::Trainer& trainer,
                                  const SushiBLAS::Tensor& output) -> Status
    {
        SA_TRY_ASSIGN(const Train::StepMetrics step, trainer.run_step(batch));
        const std::vector<double> logits = SaTest::read_tensor(output);

        // exp of such a logit overflows float32; see tests/integration/README.md.
        double largest = 0.0;
        for (const double logit : logits)
            largest = std::max(largest, std::abs(logit));
        EXPECT_GT(largest, 100.0);

        EXPECT_TRUE(std::isfinite(step.loss));
        EXPECT_NEAR(step.loss, logistic_loss_on_host(logits, labels), K_LARGE_TOLERANCE);
        return Status();
    }));
}

/** @brief Tests the multi-label loss against the host over the N * C logits of the step. */
TEST_F(Integration_DenseObjectives, TheMultiLabelLossEqualsTheHostComputation)
{
    constexpr int64_t K_CLASSES = 3;
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<Train::MultiLabelObjective> objective,
                            Train::MultiLabelObjective::make(K_BATCH, K_CLASSES));
    const std::vector<double> labels = {1.0, 0.0, 1.0, 0.0, 0.0, 1.0,
                                        1.0, 1.0, 0.0, 0.0, 1.0, 0.0};
    SA_ASSERT_OK_AND_ASSIGN(
        const Data::Batch batch,
        make_batch(*engine, real_role(Data::K_ROLE_LABELS, Shape{K_CLASSES}), labels));

    SA_EXPECT_OK(with_trainer(*engine, *objective, K_CLASSES, 0.0,
                              [&](Train::Trainer& trainer,
                                  const SushiBLAS::Tensor& output) -> Status
    {
        SA_TRY_ASSIGN(const Train::StepMetrics step, trainer.run_step(batch));
        const std::vector<double> logits = SaTest::read_tensor(output);

        EXPECT_EQ(logits.size(), labels.size());
        EXPECT_NEAR(step.loss, logistic_loss_on_host(logits, labels), K_TOLERANCE);

        double total = 0.0;
        for (std::size_t column = 0; column < 3; ++column)
        {
            double hits = 0.0;
            double wrong = 0.0;
            for (std::size_t row = 0; row < 4; ++row)
            {
                const bool positive = logits[row * 3 + column] >= 0.0;
                const bool actual = labels[row * 3 + column] == 1.0;
                hits += positive && actual ? 1.0 : 0.0;
                wrong += positive != actual ? 1.0 : 0.0;
            }
            total += hits == 0.0 ? 0.0 : 2.0 * hits / (2.0 * hits + wrong);
        }

        EXPECT_EQ(step.metrics.all().size(), 1u);
        EXPECT_NEAR(step.metrics.value_or("f1", K_ABSENT), total / 3.0, K_RATIO_TOLERANCE);
        return Status();
    }));
}

/** @brief Tests the squared error and the absolute error against the host over the outputs. */
TEST_F(Integration_DenseObjectives, TheSquaredErrorEqualsTheHostComputation)
{
    constexpr int64_t K_WIDTH = 2;
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<Train::MeanSquaredErrorObjective> objective,
                            Train::MeanSquaredErrorObjective::make(K_BATCH, K_WIDTH));
    const std::vector<double> values = {0.5, -1.0, 2.0, 0.25, -0.75, 1.5, 0.0, 3.0};
    SA_ASSERT_OK_AND_ASSIGN(
        const Data::Batch batch,
        make_batch(*engine, real_role(Data::K_ROLE_VALUES, Shape{K_WIDTH}), values));

    SA_EXPECT_OK(with_trainer(*engine, *objective, K_WIDTH, 0.0,
                              [&](Train::Trainer& trainer,
                                  const SushiBLAS::Tensor& output) -> Status
    {
        SA_TRY_ASSIGN(const Train::StepMetrics step, trainer.run_step(batch));
        const std::vector<double> predicted = SaTest::read_tensor(output);

        double squared = 0.0;
        double absolute = 0.0;
        for (std::size_t index = 0; index < values.size(); ++index)
        {
            const double difference = predicted[index] - values[index];
            squared += difference * difference;
            absolute += std::abs(difference);
        }

        EXPECT_EQ(predicted.size(), values.size());
        EXPECT_NEAR(step.loss, squared / 8.0, K_TOLERANCE);
        EXPECT_EQ(step.metrics.all().size(), 1u);
        EXPECT_NEAR(step.metrics.value_or("mean_absolute_error", K_ABSENT), absolute / 8.0,
                    K_TOLERANCE);
        return Status();
    }));
}

/** @brief Tests that fit lowers the regression loss and reports the metric of the last step. */
TEST_F(Integration_DenseObjectives, FitLowersTheSquaredError)
{
    constexpr int64_t K_WIDTH = 2;
    constexpr int64_t K_EPOCHS = 40;
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<Train::MeanSquaredErrorObjective> objective,
                            Train::MeanSquaredErrorObjective::make(K_BATCH, K_WIDTH));
    const std::vector<Data::TargetDescription> roles =
        real_role(Data::K_ROLE_VALUES, Shape{K_WIDTH});
    SA_ASSERT_OK_AND_ASSIGN(
        Data::Batch batch,
        make_batch(*engine, roles, {0.5, -1.0, 2.0, 0.25, -0.75, 1.5, 0.0, 3.0}));
    std::vector<Data::Batch> batches;
    batches.push_back(std::move(batch));
    const Data::MaterialisedDataset dataset(std::move(batches), K_BATCH, input_description(),
                                            roles, "regression");

    SA_EXPECT_OK(with_trainer(*engine, *objective, K_WIDTH, 0.05,
                              [&](Train::Trainer& trainer, const SushiBLAS::Tensor&) -> Status
    {
        trainer.options().epochs = K_EPOCHS;
        SA_TRY_ASSIGN(const Train::RunReport report, trainer.fit(dataset));

        EXPECT_EQ(report.steps, static_cast<std::size_t>(K_EPOCHS));
        EXPECT_EQ(report.compile_count, 1u);
        EXPECT_LT(report.last_loss, report.first_loss) << report.to_string();
        EXPECT_NE(report.last_metrics.find("mean_absolute_error"), nullptr);
        EXPECT_NE(report.to_string().find("mean_absolute_error"), std::string::npos);
        return Status();
    }));
}

/** @brief Tests that fit refuses a dataset whose labels are class indices before any step. */
TEST_F(Integration_DenseObjectives, FitRefusesLabelsOfAnotherType)
{
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<Train::BinaryCrossEntropyObjective> objective,
                            Train::BinaryCrossEntropyObjective::make(K_BATCH));
    const std::vector<Data::TargetDescription> roles = {Data::TargetDescription{
        std::string(Data::K_ROLE_LABELS), Shape{}, DType::Int32, 2}};
    SA_ASSERT_OK_AND_ASSIGN(
        Data::Batch batch,
        Data::allocate_batch(*engine, K_BATCH, input_description(), roles));
    SaTest::write_tensor(batch.inputs, make_inputs(1.0));
    std::vector<Data::Batch> batches;
    batches.push_back(std::move(batch));
    const Data::MaterialisedDataset dataset(std::move(batches), K_BATCH, input_description(),
                                            roles, "classes");

    SA_EXPECT_OK(with_trainer(*engine, *objective, 1, 0.05,
                              [&](Train::Trainer& trainer, const SushiBLAS::Tensor&) -> Status
    {
        SA_EXPECT_FAILURE(trainer.fit(dataset), StatusCode::Incompatible, "'classes'",
                          "'labels'", "int32", "float32");
        EXPECT_EQ(trainer.step_count(), 0u);
        return Status();
    }));
}
