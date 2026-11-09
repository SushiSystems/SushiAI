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
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>
#include <gtest/gtest.h>
#include <SushiBLAS/engine/engine.hpp>
#include <SushiBLAS/tensor/tensor.hpp>
#include <SushiAI/dtype/dtype.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/shape/shape.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/data/dataset.hpp>
#include <SushiAI/data/input.hpp>
#include <SushiAI/data/materialised_dataset.hpp>
#include <SushiAI/data/roles.hpp>
#include <SushiAI/data/target.hpp>
#include <SushiAI/engine_fixture/engine_test.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/graph/tensor_pool.hpp>
#include <SushiAI/nn/layers.hpp>
#include <SushiAI/nn/model_state.hpp>
#include <SushiAI/optim/optimizer.hpp>
#include <SushiAI/optim/sgd.hpp>
#include <SushiAI/tensor_values/tensor_values.hpp>
#include <SushiAI/train/binary_cross_entropy_objective.hpp>
#include <SushiAI/train/mean_squared_error_objective.hpp>
#include <SushiAI/train/metrics.hpp>
#include <SushiAI/train/multi_label_objective.hpp>
#include <SushiAI/train/objective.hpp>
#include <SushiAI/train/steps.hpp>
#include <SushiAI/train/trainer.hpp>
#include <SushiAI/train/training_graph.hpp>
#include <support/tolerance.hpp>

using namespace SushiAI;
// SaTest stays reachable inside a TEST_F body; see tests/layers/train/README.md.
namespace SaTest = SushiAI::Test;

namespace
{
    /** @brief Holds the samples of a batch, N. */
    constexpr int64_t K_BATCH = 4;

    /** @brief Holds the features of a sample. */
    constexpr int64_t K_FEATURES = 3;

    /** @brief Holds the classes of the multi-label objective. */
    constexpr int64_t K_CLASSES = 3;

    /** @brief Holds the outputs of the regression objective. */
    constexpr int64_t K_WIDTH = 2;

    /** @brief Holds the epochs of the regression fit. */
    constexpr int64_t K_EPOCHS = 40;

    /**
     * @brief Counts the roundings of a logistic term beyond the sum: the maximum, the product with
     *        the target, the exponential, the logarithm and the two additions.
     */
    constexpr std::size_t K_LOGISTIC_OPERATIONS = 6;

    /** @brief Counts the roundings of a squared term beyond the sum: the difference and the square. */
    constexpr std::size_t K_SQUARE_OPERATIONS = 2;

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
    Status with_trainer(SushiAI::ExecutionBackend& backend, const Train::Objective& objective,
                        int64_t width, double rate, const Visit& visit)
    {
        NN::Linear model{K_FEATURES, width};
        SA_TRY_ASSIGN(Train::TrainingGraph built,
                      Train::build_training_graph(model, objective, Shape{K_BATCH, K_FEATURES}));
        SA_TRY_ASSIGN(NN::ModelState state, NN::ModelState::create(backend.engine(), model));

        Graph::TensorPool pool(backend.engine());
        SA_TRY_ASSIGN(std::vector<Optim::ParameterSlot> slots,
                      Train::bind_parameters(backend, built, state, pool));
        SA_TRY(write_state(state));

        Optim::SGDOptions sgd;
        sgd.learning_rate = rate;
        SA_TRY_ASSIGN(const std::unique_ptr<Optim::SGD> optimizer,
                      Optim::SGD::make(backend, slots, sgd));
        SA_TRY_ASSIGN(const std::unique_ptr<Train::Trainer> trainer,
                      Train::Trainer::make(backend, built, pool, objective, *optimizer));

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

    /** @brief Returns the mean of the squared differences of @p predicted and @p values. */
    double squared_error_on_host(const std::vector<double>& predicted,
                                 const std::vector<double>& values)
    {
        double total = 0.0;
        for (std::size_t index = 0; index < values.size(); ++index)
            total += (predicted[index] - values[index]) * (predicted[index] - values[index]);

        return total / static_cast<double>(values.size());
    }

    /** @brief Returns the values the regression cases fit, two per sample. */
    std::vector<double> regression_values()
    {
        return {0.5, -1.0, 2.0, 0.25, -0.75, 1.5, 0.0, 3.0};
    }

    /** @brief Fits the regression objective for K_EPOCHS epochs over one batch. */
    Result<Train::RunReport> fit_regression(SushiAI::ExecutionBackend& backend)
    {
        SA_TRY_ASSIGN(const std::unique_ptr<Train::MeanSquaredErrorObjective> objective,
                      Train::MeanSquaredErrorObjective::make(K_BATCH, K_WIDTH));
        const std::vector<Data::TargetDescription> roles =
            real_role(Data::K_ROLE_VALUES, Shape{K_WIDTH});
        SA_TRY_ASSIGN(Data::Batch batch,
                      make_batch(backend.engine(), roles, regression_values()));
        std::vector<Data::Batch> batches;
        batches.push_back(std::move(batch));
        const Data::MaterialisedDataset dataset(std::move(batches), K_BATCH, input_description(),
                                                roles, "regression");

        std::optional<Train::RunReport> report;
        SA_TRY(with_trainer(backend, *objective, K_WIDTH, 0.05,
                            [&](Train::Trainer& trainer, const SushiBLAS::Tensor&) -> Status
        {
            trainer.options().epochs = K_EPOCHS;
            SA_TRY_ASSIGN(Train::RunReport fitted, trainer.fit(dataset));
            report = std::move(fitted);
            return Status();
        }));

        return std::move(*report);
    }

    /** @brief Provides the engine the states, the batches and the steps are built on. */
    class Medium_DenseObjectives : public SaTest::EngineTest
    {
        protected:
            /**
             * @brief Expects @p loss to equal the logistic loss of @p logits and @p labels within
             *        the roundings of one term and one sum over the elements.
             */
            static void expect_logistic_loss(double loss, const std::vector<double>& logits,
                                             const std::vector<double>& labels)
            {
                EXPECT_TRUE(TestSupport::is_within(TestSupport::measure_relative_error<float>(
                    loss, logistic_loss_on_host(logits, labels),
                    logits.size() + K_LOGISTIC_OPERATIONS)));
            }
    };
} // namespace

/** @brief Tests the binary loss against the host over the step's logits. */
TEST_F(Medium_DenseObjectives, TheBinaryLossEqualsTheHostComputation)
{
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<Train::BinaryCrossEntropyObjective> objective,
                            Train::BinaryCrossEntropyObjective::make(K_BATCH));
    const std::vector<double> labels = {1.0, 0.0, 0.0, 1.0};
    SA_ASSERT_OK_AND_ASSIGN(const Data::Batch batch,
                            make_batch(*engine, real_role(Data::K_ROLE_LABELS, Shape{}), labels));

    SA_EXPECT_OK(with_trainer(backend(), *objective, 1, 0.0,
                              [&](Train::Trainer& trainer,
                                  const SushiBLAS::Tensor& output) -> Status
    {
        SA_TRY_ASSIGN(const Train::StepMetrics step, trainer.run_step(batch));
        expect_logistic_loss(step.loss, SaTest::read_tensor(output), labels);
        return Status();
    }));
}

/** @brief Tests that logits above one hundred in magnitude give the finite loss of the host. */
TEST_F(Medium_DenseObjectives, ALargeLogitGivesAFiniteBinaryLoss)
{
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<Train::BinaryCrossEntropyObjective> objective,
                            Train::BinaryCrossEntropyObjective::make(K_BATCH));
    const std::vector<double> labels = {1.0, 0.0, 0.0, 1.0};

    // Inputs scaled by 400 overflow exp in float32; see tests/layers/train/README.md.
    SA_ASSERT_OK_AND_ASSIGN(
        const Data::Batch batch,
        make_batch(*engine, real_role(Data::K_ROLE_LABELS, Shape{}), labels, 400.0));

    SA_EXPECT_OK(with_trainer(backend(), *objective, 1, 0.0,
                              [&](Train::Trainer& trainer,
                                  const SushiBLAS::Tensor& output) -> Status
    {
        SA_TRY_ASSIGN(const Train::StepMetrics step, trainer.run_step(batch));

        EXPECT_TRUE(std::isfinite(step.loss));
        expect_logistic_loss(step.loss, SaTest::read_tensor(output), labels);
        return Status();
    }));
}

/** @brief Tests the multi-label loss against the host over the N * C logits of the step. */
TEST_F(Medium_DenseObjectives, TheMultiLabelLossEqualsTheHostComputation)
{
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<Train::MultiLabelObjective> objective,
                            Train::MultiLabelObjective::make(K_BATCH, K_CLASSES));
    const std::vector<double> labels = {1.0, 0.0, 1.0, 0.0, 0.0, 1.0,
                                        1.0, 1.0, 0.0, 0.0, 1.0, 0.0};
    SA_ASSERT_OK_AND_ASSIGN(
        const Data::Batch batch,
        make_batch(*engine, real_role(Data::K_ROLE_LABELS, Shape{K_CLASSES}), labels));

    SA_EXPECT_OK(with_trainer(backend(), *objective, K_CLASSES, 0.0,
                              [&](Train::Trainer& trainer,
                                  const SushiBLAS::Tensor& output) -> Status
    {
        SA_TRY_ASSIGN(const Train::StepMetrics step, trainer.run_step(batch));
        const std::vector<double> logits = SaTest::read_tensor(output);

        EXPECT_EQ(logits.size(), labels.size());
        expect_logistic_loss(step.loss, logits, labels);
        return Status();
    }));
}

/** @brief Tests the squared error against the host over the outputs of the step. */
TEST_F(Medium_DenseObjectives, TheSquaredErrorEqualsTheHostComputation)
{
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<Train::MeanSquaredErrorObjective> objective,
                            Train::MeanSquaredErrorObjective::make(K_BATCH, K_WIDTH));
    const std::vector<double> values = regression_values();
    SA_ASSERT_OK_AND_ASSIGN(
        const Data::Batch batch,
        make_batch(*engine, real_role(Data::K_ROLE_VALUES, Shape{K_WIDTH}), values));

    SA_EXPECT_OK(with_trainer(backend(), *objective, K_WIDTH, 0.0,
                              [&](Train::Trainer& trainer,
                                  const SushiBLAS::Tensor& output) -> Status
    {
        SA_TRY_ASSIGN(const Train::StepMetrics step, trainer.run_step(batch));
        const std::vector<double> predicted = SaTest::read_tensor(output);

        EXPECT_EQ(predicted.size(), values.size());
        EXPECT_TRUE(TestSupport::is_within(TestSupport::measure_relative_error<float>(
            step.loss, squared_error_on_host(predicted, values),
            values.size() + K_SQUARE_OPERATIONS)));
        return Status();
    }));
}

/** @brief Tests that fit takes one step per epoch and lowers the regression loss. */
TEST_F(Medium_DenseObjectives, FitLowersTheSquaredError)
{
    SA_ASSERT_OK_AND_ASSIGN(const Train::RunReport report, fit_regression(backend()));

    EXPECT_EQ(report.steps, static_cast<std::size_t>(K_EPOCHS));
    EXPECT_LT(report.last_loss, report.first_loss) << report.to_string();
}

/** @brief Tests that the report carries the metric of the last step and prints its name. */
TEST_F(Medium_DenseObjectives, TheFitReportCarriesTheRegressionMetric)
{
    SA_ASSERT_OK_AND_ASSIGN(const Train::RunReport report, fit_regression(backend()));

    EXPECT_NE(report.last_metrics.find("mean_absolute_error"), nullptr);
    EXPECT_NE(report.to_string().find("mean_absolute_error"), std::string::npos);
}
