/**************************************************************************/
/* test_convergence.cpp                                                   */
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
 * @file test_convergence.cpp
 * @brief Tests that an MLP converges on the synthetic dataset and that its updates follow their rules.
 * @author Mustafa Garip
 */

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>
#include <gtest/gtest.h>
#include <SushiBLAS/engine/engine.hpp>
#include <SushiBLAS/tensor/tensor.hpp>
#include <SushiAI/classifier_objective/classifier_objective.hpp>
#include <SushiAI/data/dataset.hpp>
#include <SushiAI/data/roles.hpp>
#include <SushiAI/data/target.hpp>
#include <SushiAI/dtype/dtype.hpp>
#include <SushiAI/engine_fixture/engine_test.hpp>
#include <SushiAI/engine_fixture/test_environment.hpp>
#include <SushiAI/execution/host_execution_backend.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/graph/tensor_pool.hpp>
#include <SushiAI/nn/batch_norm2d.hpp>
#include <SushiAI/nn/buffer.hpp>
#include <SushiAI/nn/flatten.hpp>
#include <SushiAI/nn/init.hpp>
#include <SushiAI/nn/layers.hpp>
#include <SushiAI/nn/model_state.hpp>
#include <SushiAI/nn/unflatten.hpp>
#include <SushiAI/optim/adamw.hpp>
#include <SushiAI/optim/optimizer.hpp>
#include <SushiAI/optim/sgd.hpp>
#include <SushiAI/tensor_values/tensor_values.hpp>
#include <SushiAI/train/buffer_update_recorder.hpp>
#include <SushiAI/train/trainer.hpp>
#include <SushiAI/train/training_graph.hpp>
#include <SushiAI/train/training_step.hpp>
#include <support/tolerance.hpp>

using namespace SushiAI;
// SaTest stays reachable inside a TEST_F body; see tests/layers/train/README.md.
namespace SaTest = SushiAI::Test;
using Graph::TensorPool;

namespace
{
    /** @brief The model most cases train, so a failure names one variable. */
    using MLP = NN::Sequential<NN::Linear, NN::ReLU, NN::Linear>;

    /** @brief A model with a batch normalisation over 12 features read as 2 x 2 positions. */
    using NormMLP = NN::Sequential<NN::Unflatten, NN::BatchNorm2d, NN::Flatten, NN::Linear>;

    /** @brief Samples per step of the normalised model. */
    constexpr int64_t K_NORM_BATCH = 8;

    /** @brief Channels the batch normalisation spans. */
    constexpr int64_t K_NORM_CHANNELS = 3;

    /** @brief Positions per sample, 2 by 2. */
    constexpr int64_t K_NORM_POSITIONS = 4;

    /** @brief Features per sample of the normalised model. */
    constexpr int64_t K_NORM_FEATURES = K_NORM_POSITIONS * K_NORM_CHANNELS;

    /** @brief Classes the normalised model ends in. */
    constexpr int64_t K_NORM_CLASSES = 4;

    /** @brief The seed of the data the cases over the normalised model and the feed draw. */
    constexpr std::uint64_t K_DATA_SEED = 20261009ull;

    /** @brief Samples per step of the zeroed classifier. */
    constexpr int64_t K_ZERO_BATCH = 8;

    /** @brief Classes of the zeroed classifier. */
    constexpr int64_t K_ZERO_CLASSES = 5;

    /**
     * @brief Counts the roundings on the longest path to the first loss in float: the log-softmax
     *        of five classes rounds 10, the product with the target once, the sum of the 40
     *        products 39 times at worst and the mean once.
     */
    constexpr std::size_t K_FIRST_LOSS_OPERATIONS = 51;

    /** @brief The fraction of its first loss the last loss of a run must stay under. */
    constexpr double K_FALL_FRACTION = 0.25;

    /** @brief The accuracy a separable problem must pass. */
    constexpr double K_MIN_ACCURACY = 0.9;

    /**
     * @brief Counts the roundings on the path from a parameter to its value after one update:
     *        the moments, the two corrections, the square root, the quotient, the product with
     *        the rate and the subtraction.
     */
    constexpr std::size_t K_STEP_OPERATIONS = 8;

    /** @brief Counts the roundings of the blend of a running statistic with a batch mean. */
    constexpr std::size_t K_BLEND_OPERATIONS = 3;

    /** @brief Counts the values one channel mean sums: the samples times the positions. */
    constexpr std::size_t K_MEAN_TERMS = static_cast<std::size_t>(K_NORM_BATCH * K_NORM_POSITIONS);

    /** @brief Returns the options of a synthetic dataset of @p batches batches. */
    Data::SyntheticOptions data_for(int64_t batch, int64_t batches, int64_t features,
                                    int64_t classes)
    {
        Data::SyntheticOptions options;
        options.batch_size = batch;
        options.batches = batches;
        options.features = features;
        options.classes = classes;
        return options;
    }

    /** @brief Returns the mean of each channel over the samples and positions of @p values. */
    std::vector<double> channel_means(const std::vector<double>& values)
    {
        const std::size_t channels = static_cast<std::size_t>(K_NORM_CHANNELS);
        std::vector<double> means(channels, 0.0);
        const int64_t rows = static_cast<int64_t>(values.size()) / K_NORM_FEATURES;

        for (int64_t row = 0; row < rows; ++row)
        {
            for (int64_t position = 0; position < K_NORM_POSITIONS; ++position)
            {
                for (int64_t channel = 0; channel < K_NORM_CHANNELS; ++channel)
                {
                    const std::size_t index = static_cast<std::size_t>(
                        row * K_NORM_FEATURES + position * K_NORM_CHANNELS + channel);
                    means[static_cast<std::size_t>(channel)] += values[index];
                }
            }
        }

        for (double& mean : means)
            mean /= static_cast<double>(rows * K_NORM_POSITIONS);

        return means;
    }

    /** @brief Returns the absolute values of @p values. */
    std::vector<double> magnitudes(std::vector<double> values)
    {
        std::transform(values.begin(), values.end(), values.begin(),
                       [](double value) { return std::abs(value); });
        return values;
    }

    /** @brief Returns the dotted name of the running mean buffer, or an empty name. */
    std::string running_mean_name(const NN::ModelState& state)
    {
        const std::string suffix = "running_mean";

        for (const NN::StateEntry& entry : state.entries())
        {
            const bool is_buffer = entry.kind == NN::StateKind::Buffer;
            const bool ends_with_suffix = entry.name.size() >= suffix.size() &&
                entry.name.compare(entry.name.size() - suffix.size(), suffix.size(), suffix) == 0;

            if (is_buffer && ends_with_suffix)
                return entry.name;
        }

        return {};
    }

    /** @brief Provides the engine, the classifier and the trainer every training case builds. */
    class Medium_Training : public SaTest::EngineTest
    {
        protected:
            /** @brief Traces, binds and initialises @p model; sets @c ready on success. */
            template <typename Model>
            void prepare(Model& model, int64_t batch, int64_t features, int64_t classes,
                         bool zero_final_layer = true)
            {
                SA_ASSERT_OK_AND_ASSIGN(
                    built, Train::build_classifier(model, batch, features, classes));
                SA_ASSERT_OK_AND_ASSIGN(state, NN::ModelState::create(*engine, model));
                pool = std::make_unique<TensorPool>(*engine);
                SA_ASSERT_OK_AND_ASSIGN(slots,
                                        Train::bind_parameters(backend(), built, *state, *pool));

                // A nonzero classifier gives every coordinate a gradient to move by.
                NN::InitPolicy policy;
                policy.zero_final_layer = zero_final_layer;
                SA_EXPECT_OK(NN::initialize(backend(), *state, policy));
                ready = true;
            }

            /** @brief Makes the SGD optimizer over the bound slots. */
            void use_sgd(const Optim::SGDOptions& options)
            {
                SA_ASSERT_OK_AND_ASSIGN(optimizer, Optim::SGD::make(backend(), slots, options));
            }

            /** @brief Makes the AdamW optimizer over the bound slots. */
            void use_adamw(const Optim::AdamWOptions& options)
            {
                SA_ASSERT_OK_AND_ASSIGN(optimizer, Optim::AdamW::make(backend(), slots, options));
            }

            /** @brief Makes the trainer over the prepared classifier and the chosen optimizer. */
            void make_trainer(const Train::TrainerOptions& options,
                              std::vector<Train::BufferUpdateSlot> updates = {})
            {
                objective = std::make_unique<Train::CrossEntropyObjective>(
                    SaTest::classifier_objective(built));
                SA_ASSERT_OK_AND_ASSIGN(
                    trainer, Train::Trainer::make(backend(), built, *pool, *objective, *optimizer,
                                                  options, std::move(updates)));
            }

            /** @brief Makes the dataset the cases read from. */
            void make_dataset(const Data::SyntheticOptions& options)
            {
                SA_ASSERT_OK_AND_ASSIGN(dataset,
                                        Data::SyntheticClassification::make(backend(), options));
            }

            /**
             * @brief Runs one step over batch @p index of the dataset.
             * @return The metrics of the step, empty when the read or the step failed.
             */
            Train::StepMetrics step_at(std::size_t index)
            {
                Data::Batch batch;
                EXPECT_TRUE(SaTest::status_is_ok(dataset->read(index, batch)));
                const Result<Train::StepMetrics> metrics = trainer->run_step(batch);
                EXPECT_TRUE(metrics.has_value());
                return metrics.has_value() ? metrics.value() : Train::StepMetrics{};
            }

            /** @brief Reads the first parameter and its gradient around one step over batch 0. */
            void capture_first_step()
            {
                before = SaTest::read_tensor(slots[0].parameter);
                step_at(0);
                after = SaTest::read_tensor(slots[0].parameter);
                gradient = SaTest::read_tensor(slots[0].gradient);
            }

            /** @brief Builds the separable problem and fits it with SGD, momentum included. */
            void fit_separable_with_sgd()
            {
                MLP model{NN::Linear{8, 24}, NN::ReLU{}, NN::Linear{24, 4}};
                prepare(model, 16, 8, 4);
                ASSERT_TRUE(ready);

                Optim::SGDOptions sgd;
                sgd.learning_rate = 0.2;
                sgd.momentum = 0.9;
                use_sgd(sgd);
                fit_separable();
            }

            /** @brief Builds the separable problem and fits it with AdamW. */
            void fit_separable_with_adamw()
            {
                MLP model{NN::Linear{8, 24}, NN::ReLU{}, NN::Linear{24, 4}};
                prepare(model, 16, 8, 4);
                ASSERT_TRUE(ready);

                Optim::AdamWOptions adam;
                adam.learning_rate = 0.02;
                use_adamw(adam);
                fit_separable();
            }

            /** @brief Expects the first update to have moved coordinate @p i by the learning rate. */
            void expect_unit_step(std::size_t i, double learning_rate, double eps)
            {
                // The move falls short of the rate by eps / (|g| + eps) of the gradient's own size.
                const double moved = (after[i] - before[i]) / -learning_rate;
                const double expected = gradient[i] > 0.0 ? 1.0 : -1.0;
                const double shortfall = eps / (std::abs(gradient[i]) + eps);
                const double rounding = static_cast<double>(K_STEP_OPERATIONS) *
                    TestSupport::unit_roundoff<float>() *
                    (std::abs(before[i]) + learning_rate) / learning_rate;

                EXPECT_TRUE(TestSupport::is_within(
                    TestSupport::measure_error(moved, expected, shortfall + rounding)))
                    << "at " << i << " (gradient " << gradient[i] << ")";
            }

            /** @brief Expects coordinate @p i to hold its old value minus rate times gradient. */
            void expect_sgd_update(std::size_t i, double learning_rate)
            {
                const double expected = before[i] - learning_rate * gradient[i];

                // The product rounds once and the subtraction once.
                const double bound = 2.0 * TestSupport::unit_roundoff<float>() *
                    (std::abs(learning_rate * gradient[i]) + std::abs(expected));

                EXPECT_TRUE(
                    TestSupport::is_within(TestSupport::measure_error(after[i], expected, bound)))
                    << "at " << i;
            }

            /** @brief Expects the running mean of @p channel to follow the host update. */
            void expect_running_mean(std::size_t channel)
            {
                const double momentum = NN::NormOptions{}.momentum;
                const double expected_first =
                    NN::updated_statistic(0.0, means_first[channel], momentum, 1.0);
                const double expected_second =
                    NN::updated_statistic(expected_first, means_second[channel], momentum, 1.0);
                const double unit = TestSupport::unit_roundoff<float>();

                // The momentum scales the sum's terms; the previous value keeps what is left.
                const double length = static_cast<double>(K_MEAN_TERMS + K_BLEND_OPERATIONS);
                const double first_bound = length * unit * momentum * magnitudes_first[channel];
                const double second_bound =
                    length * unit * ((1.0 - momentum) * std::abs(expected_first) +
                                     momentum * magnitudes_second[channel]);

                EXPECT_TRUE(TestSupport::is_within(TestSupport::measure_error(
                    after_first[channel], expected_first, first_bound)))
                    << "channel " << channel << " after one step";
                EXPECT_TRUE(TestSupport::is_within(TestSupport::measure_error(
                    after_second[channel], expected_second, second_bound)))
                    << "channel " << channel << " after two steps";
                EXPECT_NE(after_second[channel], after_first[channel])
                    << "channel " << channel << " did not move on the second step";
            }

            std::optional<NN::ModelState> state;       /**< @brief Holds the parameters. */
            Train::TrainingGraph built;                /**< @brief Holds the training graph. */
            std::unique_ptr<TensorPool> pool;          /**< @brief Holds the step's values. */
            std::vector<Optim::ParameterSlot> slots;   /**< @brief Pairs parameters and gradients. */
            std::unique_ptr<Optim::Optimizer> optimizer;                    /**< @brief Updates them. */
            std::unique_ptr<Train::CrossEntropyObjective> objective;        /**< @brief Scores a step. */
            std::unique_ptr<Train::Trainer> trainer;                        /**< @brief Runs the steps. */
            std::unique_ptr<Data::SyntheticClassification> dataset;         /**< @brief Feeds them. */
            std::optional<Train::RunReport> fitted;    /**< @brief Holds the report of a fit. */
            std::vector<double> before;                /**< @brief Holds a parameter before a step. */
            std::vector<double> after;                 /**< @brief Holds it after the step. */
            std::vector<double> gradient;              /**< @brief Holds its gradient. */
            std::vector<double> means_first;           /**< @brief Holds the channel means of batch 0. */
            std::vector<double> means_second;          /**< @brief Holds the channel means of batch 1. */
            std::vector<double> magnitudes_first;      /**< @brief Holds the mean magnitudes of batch 0. */
            std::vector<double> magnitudes_second;     /**< @brief Holds the mean magnitudes of batch 1. */
            std::vector<double> after_first;           /**< @brief Holds the running mean after step 1. */
            std::vector<double> after_second;          /**< @brief Holds the running mean after step 2. */
            bool ready = false;                        /**< @brief Holds whether prepare() ended. */

        private:
            /** @brief Fits 30 epochs of six batches over the prepared separable problem. */
            void fit_separable()
            {
                Train::TrainerOptions options;
                options.epochs = 30;
                make_trainer(options);
                make_dataset(data_for(16, 6, 8, 4));
                SA_ASSERT_OK_AND_ASSIGN(fitted, trainer->fit(*dataset));
            }
    };

    /** @brief Provides the host backend the refusals of binding are decided on. */
    class Small_TrainingBinding : public ::testing::Test
    {
        protected:
            /** @brief The host backend; declared first so that it outlives every part below. */
            HostExecutionBackend backend;
    };

    /** @brief Builds two datasets of one option set, each on an Engine of its own. */
    class Medium_TrainingDataset : public ::testing::TestWithParam<std::size_t>
    {
        protected:
            /** @brief Makes the two engines, their backends and the two datasets. */
            void SetUp() override
            {
                Data::SyntheticOptions options = data_for(8, 3, 5, 4);
                options.seed = 4242;

                // Each dataset gets a fresh Engine; see tests/layers/train/README.md.
                first_engine =
                    std::make_unique<SushiBLAS::Engine>(SaTest::TestEnvironment::runtime());
                second_engine =
                    std::make_unique<SushiBLAS::Engine>(SaTest::TestEnvironment::runtime());
                first_backend = std::make_unique<RuntimeExecutionBackend>(
                    SaTest::TestEnvironment::runtime(), *first_engine);
                second_backend = std::make_unique<RuntimeExecutionBackend>(
                    SaTest::TestEnvironment::runtime(), *second_engine);

                SA_ASSERT_OK_AND_ASSIGN(
                    first, Data::SyntheticClassification::make(*first_backend, options));
                SA_ASSERT_OK_AND_ASSIGN(
                    second, Data::SyntheticClassification::make(*second_backend, options));
            }

            std::unique_ptr<SushiBLAS::Engine> first_engine;               /**< @brief The first engine. */
            std::unique_ptr<SushiBLAS::Engine> second_engine;              /**< @brief The second engine. */
            std::unique_ptr<RuntimeExecutionBackend> first_backend;        /**< @brief Runs on the first. */
            std::unique_ptr<RuntimeExecutionBackend> second_backend;       /**< @brief Runs on the second. */
            std::unique_ptr<Data::SyntheticClassification> first;          /**< @brief Draws on the first. */
            std::unique_ptr<Data::SyntheticClassification> second;         /**< @brief Draws on the second. */
    };
}

/** @brief Tests that a zeroed classifier gives a first loss of ln(C), every class equally likely. */
TEST_F(Medium_Training, InitialLossIsExactlyLogCWithAZeroedClassifier)
{
    // Zero logits make the first loss exactly ln(C); see tests/layers/train/README.md.
    MLP model{NN::Linear{6, 12}, NN::ReLU{}, NN::Linear{12, K_ZERO_CLASSES}};
    prepare(model, K_ZERO_BATCH, 6, K_ZERO_CLASSES);
    ASSERT_TRUE(ready);
    use_sgd(Optim::SGDOptions{});

    Train::TrainerOptions options;
    options.track_metrics = false;
    make_trainer(options);
    make_dataset(data_for(K_ZERO_BATCH, 1, 6, K_ZERO_CLASSES));

    const Train::StepMetrics metrics = step_at(0);

    EXPECT_TRUE(TestSupport::is_within(TestSupport::measure_relative_error<float>(
        metrics.loss, std::log(static_cast<double>(K_ZERO_CLASSES)), K_FIRST_LOSS_OPERATIONS)));
}

/** @brief Tests that twelve replays leave the pool with the tensors and bytes it held after one. */
TEST_F(Medium_Training, TheStepAllocatesNothingInSteadyState)
{
    // Replays must reuse the same buffers; see tests/layers/train/README.md.
    MLP model{NN::Linear{6, 8}, NN::ReLU{}, NN::Linear{8, 3}};
    prepare(model, 8, 6, 3);
    ASSERT_TRUE(ready);
    use_sgd(Optim::SGDOptions{});
    make_trainer(Train::TrainerOptions{});
    make_dataset(data_for(8, 4, 6, 3));

    step_at(0);
    const std::size_t tensors_after_first = pool->tensor_count();
    const std::size_t bytes_after_first = pool->allocated_bytes();

    step_at(0);
    step_at(1);
    step_at(2);
    step_at(3);
    step_at(0);
    step_at(1);
    step_at(2);
    step_at(3);
    step_at(0);
    step_at(1);
    step_at(2);
    step_at(3);

    EXPECT_EQ(pool->tensor_count(), tensors_after_first);
    EXPECT_EQ(pool->allocated_bytes(), bytes_after_first);
}

/** @brief Tests that SGD with momentum ends 30 epochs under a quarter of its first loss. */
TEST_F(Medium_Training, SGDDrivesTheLossDownOnASeparableProblem)
{
    fit_separable_with_sgd();
    ASSERT_TRUE(fitted.has_value());

    EXPECT_LT(fitted->last_loss, fitted->first_loss * K_FALL_FRACTION) << fitted->to_string();
}

/** @brief Tests that SGD with momentum ends 30 epochs at an accuracy above 0.9. */
TEST_F(Medium_Training, SGDReachesAHighAccuracyOnASeparableProblem)
{
    fit_separable_with_sgd();
    ASSERT_TRUE(fitted.has_value());

    ASSERT_NE(fitted->last_metrics.find("accuracy"), nullptr);
    EXPECT_GT(fitted->last_metrics.find("accuracy")->value, K_MIN_ACCURACY)
        << fitted->to_string();
}

/** @brief Tests that AdamW ends 30 epochs under a quarter of its first loss at an accuracy above 0.9. */
TEST_F(Medium_Training, AdamWDrivesTheLossDownOnASeparableProblem)
{
    fit_separable_with_adamw();
    ASSERT_TRUE(fitted.has_value());

    EXPECT_LT(fitted->last_loss, fitted->first_loss * K_FALL_FRACTION) << fitted->to_string();
    ASSERT_NE(fitted->last_metrics.find("accuracy"), nullptr);
    EXPECT_GT(fitted->last_metrics.find("accuracy")->value, K_MIN_ACCURACY)
        << fitted->to_string();
}

/** @brief Tests that the first AdamW step moves each of the 12 weights by about the rate. */
TEST_F(Medium_Training, AdamWsFirstStepMovesEveryParameterByAboutTheLearningRate)
{
    // Step one moves each coordinate by about lr; see tests/layers/train/README.md.
    NN::Linear layer{4, 3};
    prepare(layer, 8, 4, 3, false);
    ASSERT_TRUE(ready);

    Optim::AdamWOptions adam;
    adam.learning_rate = 0.05;
    adam.weight_decay = 0.0;
    use_adamw(adam);

    Train::TrainerOptions options;
    options.track_metrics = false;
    make_trainer(options);
    make_dataset(data_for(8, 1, 4, 3));
    capture_first_step();

    ASSERT_EQ(before.size(), 12u);
    ASSERT_EQ(after.size(), 12u);
    ASSERT_EQ(gradient.size(), 12u);
    expect_unit_step(0, adam.learning_rate, adam.eps);
    expect_unit_step(1, adam.learning_rate, adam.eps);
    expect_unit_step(2, adam.learning_rate, adam.eps);
    expect_unit_step(3, adam.learning_rate, adam.eps);
    expect_unit_step(4, adam.learning_rate, adam.eps);
    expect_unit_step(5, adam.learning_rate, adam.eps);
    expect_unit_step(6, adam.learning_rate, adam.eps);
    expect_unit_step(7, adam.learning_rate, adam.eps);
    expect_unit_step(8, adam.learning_rate, adam.eps);
    expect_unit_step(9, adam.learning_rate, adam.eps);
    expect_unit_step(10, adam.learning_rate, adam.eps);
    expect_unit_step(11, adam.learning_rate, adam.eps);
}

/** @brief Tests that one SGD step at rate 0.1 and no momentum moves each of 12 weights by -rate * gradient. */
TEST_F(Medium_Training, SGDWithoutMomentumIsExactlyMinusLearningRateTimesTheGradient)
{
    NN::Linear layer{4, 3};
    prepare(layer, 8, 4, 3, false);
    ASSERT_TRUE(ready);

    Optim::SGDOptions sgd;
    sgd.learning_rate = 0.1;
    use_sgd(sgd);

    Train::TrainerOptions options;
    options.track_metrics = false;
    make_trainer(options);
    make_dataset(data_for(8, 1, 4, 3));
    capture_first_step();

    ASSERT_EQ(before.size(), 12u);
    ASSERT_EQ(after.size(), 12u);
    ASSERT_EQ(gradient.size(), 12u);
    expect_sgd_update(0, sgd.learning_rate);
    expect_sgd_update(1, sgd.learning_rate);
    expect_sgd_update(2, sgd.learning_rate);
    expect_sgd_update(3, sgd.learning_rate);
    expect_sgd_update(4, sgd.learning_rate);
    expect_sgd_update(5, sgd.learning_rate);
    expect_sgd_update(6, sgd.learning_rate);
    expect_sgd_update(7, sgd.learning_rate);
    expect_sgd_update(8, sgd.learning_rate);
    expect_sgd_update(9, sgd.learning_rate);
    expect_sgd_update(10, sgd.learning_rate);
    expect_sgd_update(11, sgd.learning_rate);
}

/** @brief Tests that binding a parameter no gradient reaches stops on its name. */
TEST_F(Small_TrainingBinding, ADisconnectedParameterIsRejectedRatherThanSilentlySkipped)
{
    NN::Linear layer{4, 3};
    SA_ASSERT_OK_AND_ASSIGN(Train::TrainingGraph built,
                            Train::build_classifier(layer, 4, 4, 3));

    // "ghost" is a parameter no gradient reaches; see tests/layers/train/README.md.
    built.parameters.push_back(Train::ParameterBinding{"ghost", layer.weight.id_in(built.graph),
                                                       Graph::K_NO_VALUE});

    SA_ASSERT_OK_AND_ASSIGN(NN::ModelState state,
                            NN::ModelState::create(backend.engine(), layer));
    TensorPool pool(backend.engine());
    SA_EXPECT_FAILURE(Train::bind_parameters(backend, built, state, pool),
                      StatusCode::InvalidDescription,
                      "'ghost'");
}

/** @brief Tests that bind_parameters marks the weights as decaying and the biases as not. */
TEST_F(Small_TrainingBinding, BoundSlotsDecayByTheRoleOfTheirParameter)
{
    MLP model{NN::Linear{6, 8}, NN::ReLU{}, NN::Linear{8, 3}};
    SA_ASSERT_OK_AND_ASSIGN(Train::TrainingGraph built,
                            Train::build_classifier(model, 4, 6, 3));
    SA_ASSERT_OK_AND_ASSIGN(NN::ModelState state,
                            NN::ModelState::create(backend.engine(), model));
    TensorPool pool(backend.engine());
    SA_ASSERT_OK_AND_ASSIGN(std::vector<Optim::ParameterSlot> slots,
                            Train::bind_parameters(backend, built, state, pool));

    ASSERT_EQ(slots.size(), 4u);
    EXPECT_EQ(slots[0].name, "0.weight");
    EXPECT_TRUE(slots[0].decays);
    EXPECT_EQ(slots[1].name, "0.bias");
    EXPECT_FALSE(slots[1].decays);
    EXPECT_EQ(slots[2].name, "2.weight");
    EXPECT_TRUE(slots[2].decays);
    EXPECT_EQ(slots[3].name, "2.bias");
    EXPECT_FALSE(slots[3].decays);
}

/** @brief Tests that two fresh engines draw the same inputs for one batch index. */
TEST_P(Medium_TrainingDataset, TwoDatasetsOfOneSeedHoldTheSameInputs)
{
    ASSERT_EQ(first->batch_count(), 3u);
    EXPECT_EQ(first->name(), "synthetic-classification");

    Data::Batch first_batch;
    Data::Batch second_batch;
    ASSERT_TRUE(SaTest::status_is_ok(first->read(GetParam(), first_batch)));
    ASSERT_TRUE(SaTest::status_is_ok(second->read(GetParam(), second_batch)));

    EXPECT_EQ(SaTest::read_tensor(first_batch.inputs), SaTest::read_tensor(second_batch.inputs));
}

/** @brief Tests that round-robin labels hold each of the four classes twice in every batch. */
TEST_P(Medium_TrainingDataset, EveryBatchHoldsEachClassTwice)
{
    Data::Batch batch;
    ASSERT_TRUE(SaTest::status_is_ok(first->read(GetParam(), batch)));

    // Round-robin labels balance every batch; see tests/layers/train/README.md.
    const Data::BatchTarget* labels = Data::find_target(batch.targets, Data::K_ROLE_LABELS);
    ASSERT_NE(labels, nullptr);
    ASSERT_EQ(labels->values.dtype, DType::Int32);
    ASSERT_EQ(labels->values.num_elements, 8);

    const int32_t* data = labels->values.data_as<int32_t>();
    const std::vector<int32_t> values(data, data + labels->values.num_elements);

    EXPECT_EQ(std::count(values.begin(), values.end(), 0), 2);
    EXPECT_EQ(std::count(values.begin(), values.end(), 1), 2);
    EXPECT_EQ(std::count(values.begin(), values.end(), 2), 2);
    EXPECT_EQ(std::count(values.begin(), values.end(), 3), 2);
}

INSTANTIATE_TEST_SUITE_P(Batches, Medium_TrainingDataset, ::testing::Values(0u, 1u, 2u));

/** @brief Tests that the running mean follows the host update of each step's batch mean. */
TEST_F(Medium_Training, RunningMeanMovesAcrossTwoStepsAsUpdatedStatisticSays)
{
    // Each step applies the host form of the update to the batch mean of that step.
    NormMLP model{NN::Unflatten{2, 2, K_NORM_CHANNELS},
                  NN::BatchNorm2d(K_NORM_CHANNELS, NN::NormOptions{}), NN::Flatten{},
                  NN::Linear{K_NORM_FEATURES, K_NORM_CLASSES}};
    prepare(model, K_NORM_BATCH, K_NORM_FEATURES, K_NORM_CLASSES);
    ASSERT_TRUE(ready);

    SA_ASSERT_OK_AND_ASSIGN(std::vector<Train::BufferUpdateSlot> updates,
                            Train::bind_buffer_updates(built.updates, *state, *pool));
    use_sgd(Optim::SGDOptions{});
    make_trainer(Train::TrainerOptions{}, std::move(updates));

    Data::SyntheticOptions data_options =
        data_for(K_NORM_BATCH, 2, K_NORM_FEATURES, K_NORM_CLASSES);
    data_options.seed = K_DATA_SEED;
    make_dataset(data_options);

    Data::Batch dataset_batch;
    ASSERT_TRUE(SaTest::status_is_ok(dataset->read(0, dataset_batch)));
    const std::vector<double> first_inputs = SaTest::read_tensor(dataset_batch.inputs);
    means_first = channel_means(first_inputs);
    magnitudes_first = channel_means(magnitudes(first_inputs));
    ASSERT_TRUE(SaTest::status_is_ok(dataset->read(1, dataset_batch)));
    const std::vector<double> second_inputs = SaTest::read_tensor(dataset_batch.inputs);
    means_second = channel_means(second_inputs);
    magnitudes_second = channel_means(magnitudes(second_inputs));

    SA_ASSERT_OK_AND_ASSIGN(SushiBLAS::Tensor* running, state->tensor(running_mean_name(*state)));

    const Train::StepMetrics first = step_at(0);
    after_first = SaTest::read_tensor(*running);
    const Train::StepMetrics second = step_at(1);
    after_second = SaTest::read_tensor(*running);

    EXPECT_GT(first.loss, 0.0);
    EXPECT_GT(second.loss, 0.0);
    ASSERT_EQ(after_first.size(), 3u);
    expect_running_mean(0);
    expect_running_mean(1);
    expect_running_mean(2);
}

/** @brief Tests that the per-step placement still updates a weight when the model holds a batch norm. */
TEST_F(Medium_Training, APerStepOptimizerStillStepsWhenTheModelHoldsABatchNorm)
{
    NormMLP model{NN::Unflatten{2, 2, K_NORM_CHANNELS},
                  NN::BatchNorm2d(K_NORM_CHANNELS, NN::NormOptions{}), NN::Flatten{},
                  NN::Linear{K_NORM_FEATURES, K_NORM_CLASSES}};
    prepare(model, K_NORM_BATCH, K_NORM_FEATURES, K_NORM_CLASSES);
    ASSERT_TRUE(ready);

    SA_ASSERT_OK_AND_ASSIGN(std::vector<Train::BufferUpdateSlot> updates,
                            Train::bind_buffer_updates(built.updates, *state, *pool));
    use_sgd(Optim::SGDOptions{});

    Train::TrainerOptions options;
    options.optimizer_placement = Train::OptimizerPlacement::PerStep;
    make_trainer(options, std::move(updates));

    Data::SyntheticOptions data_options =
        data_for(K_NORM_BATCH, 1, K_NORM_FEATURES, K_NORM_CLASSES);
    data_options.seed = K_DATA_SEED;
    make_dataset(data_options);

    const auto weight = std::find_if(slots.begin(), slots.end(),
                                     [](const Optim::ParameterSlot& slot)
                                     { return slot.name == "3.weight"; });
    ASSERT_NE(weight, slots.end()) << "the model has no parameter named '3.weight'";
    const std::vector<double> weight_before = SaTest::read_tensor(weight->parameter);

    const Train::StepMetrics metrics = step_at(0);

    // The classifier starts at zero, so the first step's gradient reaches its weight alone.
    EXPECT_GT(metrics.loss, 0.0);
    EXPECT_EQ(trainer->step_count(), 1u);
    EXPECT_NE(SaTest::read_tensor(weight->parameter), weight_before);
}

/** @brief Tests that a batch of 4 samples is refused by a trainer compiled for 8, naming both shapes. */
TEST_F(Medium_Training, ABatchOfAnotherShapeIsRefusedWithBothShapes)
{
    MLP model{NN::Linear{6, 12}, NN::ReLU{}, NN::Linear{12, 5}};
    prepare(model, 8, 6, 5);
    ASSERT_TRUE(ready);
    use_sgd(Optim::SGDOptions{});
    make_trainer(Train::TrainerOptions{});

    Data::SyntheticOptions data_options = data_for(4, 1, 6, 5);
    data_options.seed = K_DATA_SEED;
    make_dataset(data_options);

    Data::Batch wrong_batch;
    ASSERT_TRUE(SaTest::status_is_ok(dataset->read(0, wrong_batch)));
    SA_EXPECT_FAILURE(trainer->run_step(wrong_batch), StatusCode::Incompatible, "[4, 6]",
                      "[8, 6]");
}
