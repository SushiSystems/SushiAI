/**************************************************************************/
/* test_training.cpp                                                      */
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
 * @file test_training.cpp
 * @brief Tests that an MLP converges on the synthetic dataset and compiles its step once.
 * @author Mustafa Garip
 */

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <gtest/gtest.h>
#include <SushiAI/nn/model_state.hpp>
#include <SushiAI/train/buffer_update_recorder.hpp>
#include <common/test_status.hpp>
#include <common/test_utils.hpp>

using namespace SushiAI;

// SaTest stays reachable inside a TEST_F body; see tests/integration/README.md.
namespace SaTest = SushiAI::Test;
using Graph::TensorPool;
using Graph::ValueId;

namespace
{
    class Integration_Training : public SaTest::EngineTest {};

    /** @brief The model every test here trains, so a failure names one variable. */
    using Mlp = NN::Sequential<NN::Linear, NN::ReLU, NN::Linear>;

    Mlp make_model(int64_t features, int64_t hidden, int64_t classes)
    {
        return Mlp(NN::Linear{features, hidden}, NN::ReLU{}, NN::Linear{hidden, classes});
    }

    /** @brief A model with a batch normalisation over 12 features read as 2 x 2 positions. */
    using NormMlp = NN::Sequential<NN::Unflatten, NN::BatchNorm2d, NN::Flatten, NN::Linear>;

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

    /** @brief Absolute tolerance for values of order one that a float32 tensor holds. */
    constexpr double K_TOLERANCE = 1e-5;

    /** @brief The least change of a value that counts as having moved. */
    constexpr double K_MOVED = 1e-7;

    /** @brief Builds the normalised model over K_NORM_FEATURES and K_NORM_CLASSES. */
    NormMlp make_norm_model()
    {
        return NormMlp(NN::Unflatten{2, 2, K_NORM_CHANNELS},
                       NN::BatchNorm2d(K_NORM_CHANNELS, NN::NormOptions{}), NN::Flatten{},
                       NN::Linear{K_NORM_FEATURES, K_NORM_CLASSES});
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
}

TEST_F(Integration_Training, InitialLossIsExactlyLogCWithAZeroedClassifier)
{
    // Zero logits make the first loss exactly ln(C); see tests/integration/README.md.
    const int64_t batch = 8;
    const int64_t features = 6;
    const int64_t classes = 5;

    Mlp model = make_model(features, 12, classes);
    SA_ASSERT_OK_AND_ASSIGN(Train::TrainingGraph built,
                            Train::build_classifier(model, batch, features, classes));

    SA_ASSERT_OK_AND_ASSIGN(NN::ModelState state, NN::ModelState::create(*engine, model));
    TensorPool pool(*engine);
    SA_ASSERT_OK_AND_ASSIGN(std::vector<Optim::ParameterSlot> slots,
                            Train::bind_parameters(built, state, pool));
    SA_EXPECT_OK(NN::initialize(*engine, state));

    SA_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Optim::SGD> optimizer,
                            Optim::SGD::make(*engine, slots, Optim::SGDOptions{}));
    Train::TrainerOptions options;
    options.track_accuracy = false;
    SA_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Train::Trainer> trainer,
                            Train::Trainer::make(*engine, built.graph, pool, built.spec,
                                                 *optimizer, options));

    Data::SyntheticOptions data_options;
    data_options.batch_size = batch;
    data_options.batches = 1;
    data_options.features = features;
    data_options.classes = classes;
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<Data::SyntheticClassification> dataset,
                            Data::SyntheticClassification::make(*engine, data_options));

    SA_ASSERT_OK_AND_ASSIGN(const Train::StepMetrics metrics,
                            trainer->run_step(dataset->batch(0)));
    EXPECT_NEAR(metrics.loss, std::log(static_cast<double>(classes)), 1e-5);
}

TEST_F(Integration_Training, TheStepIsCompiledExactlyOnceAcrossManySteps)
{
    // The step must compile once over many steps; see tests/integration/README.md.
    const int64_t batch = 16;
    const int64_t features = 8;
    const int64_t classes = 4;

    Mlp model = make_model(features, 16, classes);
    SA_ASSERT_OK_AND_ASSIGN(Train::TrainingGraph built,
                            Train::build_classifier(model, batch, features, classes));

    SA_ASSERT_OK_AND_ASSIGN(NN::ModelState state, NN::ModelState::create(*engine, model));
    TensorPool pool(*engine);
    SA_ASSERT_OK_AND_ASSIGN(std::vector<Optim::ParameterSlot> slots,
                            Train::bind_parameters(built, state, pool));
    SA_EXPECT_OK(NN::initialize(*engine, state));

    SA_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Optim::AdamW> optimizer,
                            Optim::AdamW::make(*engine, slots, Optim::AdamWOptions{}));
    SA_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Train::Trainer> trainer,
                            Train::Trainer::make(*engine, built.graph, pool, built.spec,
                                                 *optimizer));

    Data::SyntheticOptions data_options;
    data_options.batch_size = batch;
    data_options.batches = 8;
    data_options.features = features;
    data_options.classes = classes;
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<Data::SyntheticClassification> dataset,
                            Data::SyntheticClassification::make(*engine, data_options));

    EXPECT_EQ(trainer->compile_count(), 1u);

    Train::TrainerOptions options;
    options.epochs = 5;
    trainer->options() = options;

    SA_ASSERT_OK_AND_ASSIGN(const Train::RunReport report, trainer->fit(*dataset));

    EXPECT_EQ(report.steps, 40u);
    EXPECT_EQ(report.compile_count, 1u);
    EXPECT_EQ(trainer->compile_count(), 1u);
    EXPECT_EQ(trainer->step_count(), 40u);
    EXPECT_GT(report.operations_per_step, 0u);
}

TEST_F(Integration_Training, TheStepAllocatesNothingInSteadyState)
{
    // Replays must reuse the same buffers; see tests/integration/README.md.
    const int64_t batch = 8;
    const int64_t features = 6;
    const int64_t classes = 3;

    Mlp model = make_model(features, 8, classes);
    SA_ASSERT_OK_AND_ASSIGN(Train::TrainingGraph built,
                            Train::build_classifier(model, batch, features, classes));

    SA_ASSERT_OK_AND_ASSIGN(NN::ModelState state, NN::ModelState::create(*engine, model));
    TensorPool pool(*engine);
    SA_ASSERT_OK_AND_ASSIGN(std::vector<Optim::ParameterSlot> slots,
                            Train::bind_parameters(built, state, pool));
    SA_EXPECT_OK(NN::initialize(*engine, state));

    SA_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Optim::SGD> optimizer,
                            Optim::SGD::make(*engine, slots, Optim::SGDOptions{}));
    SA_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Train::Trainer> trainer,
                            Train::Trainer::make(*engine, built.graph, pool, built.spec,
                                                 *optimizer));

    Data::SyntheticOptions data_options;
    data_options.batch_size = batch;
    data_options.batches = 4;
    data_options.features = features;
    data_options.classes = classes;
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<Data::SyntheticClassification> dataset,
                            Data::SyntheticClassification::make(*engine, data_options));

    SA_EXPECT_OK(trainer->run_step(dataset->batch(0)));
    const std::size_t tensors_after_first = pool.tensor_count();
    const std::size_t bytes_after_first = pool.allocated_bytes();

    for (int i = 0; i < 12; ++i)
        SA_EXPECT_OK(trainer->run_step(dataset->batch(static_cast<std::size_t>(i % 4))));

    EXPECT_EQ(pool.tensor_count(), tensors_after_first);
    EXPECT_EQ(pool.allocated_bytes(), bytes_after_first);
}

TEST_F(Integration_Training, SgdDrivesTheLossDownOnASeparableProblem)
{
    const int64_t batch = 16;
    const int64_t features = 8;
    const int64_t classes = 4;

    Mlp model = make_model(features, 24, classes);
    SA_ASSERT_OK_AND_ASSIGN(Train::TrainingGraph built,
                            Train::build_classifier(model, batch, features, classes));

    SA_ASSERT_OK_AND_ASSIGN(NN::ModelState state, NN::ModelState::create(*engine, model));
    TensorPool pool(*engine);
    SA_ASSERT_OK_AND_ASSIGN(std::vector<Optim::ParameterSlot> slots,
                            Train::bind_parameters(built, state, pool));
    SA_EXPECT_OK(NN::initialize(*engine, state));

    Optim::SGDOptions sgd;
    sgd.learning_rate = 0.2;
    sgd.momentum = 0.9;
    SA_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Optim::SGD> optimizer,
                            Optim::SGD::make(*engine, slots, sgd));

    Train::TrainerOptions options;
    options.epochs = 30;
    SA_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Train::Trainer> trainer,
                            Train::Trainer::make(*engine, built.graph, pool, built.spec,
                                                 *optimizer, options));

    Data::SyntheticOptions data_options;
    data_options.batch_size = batch;
    data_options.batches = 6;
    data_options.features = features;
    data_options.classes = classes;
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<Data::SyntheticClassification> dataset,
                            Data::SyntheticClassification::make(*engine, data_options));

    SA_ASSERT_OK_AND_ASSIGN(const Train::RunReport report, trainer->fit(*dataset));

    EXPECT_NEAR(report.first_loss, std::log(static_cast<double>(classes)), 1e-4);
    EXPECT_LT(report.last_loss, report.first_loss * 0.25) << report.to_string();
    EXPECT_GT(report.last_accuracy, 0.9) << report.to_string();
    EXPECT_EQ(report.compile_count, 1u);
}

TEST_F(Integration_Training, AdamWDrivesTheLossDownOnASeparableProblem)
{
    const int64_t batch = 16;
    const int64_t features = 8;
    const int64_t classes = 4;

    Mlp model = make_model(features, 24, classes);
    SA_ASSERT_OK_AND_ASSIGN(Train::TrainingGraph built,
                            Train::build_classifier(model, batch, features, classes));

    SA_ASSERT_OK_AND_ASSIGN(NN::ModelState state, NN::ModelState::create(*engine, model));
    TensorPool pool(*engine);
    SA_ASSERT_OK_AND_ASSIGN(std::vector<Optim::ParameterSlot> slots,
                            Train::bind_parameters(built, state, pool));
    SA_EXPECT_OK(NN::initialize(*engine, state));

    Optim::AdamWOptions adam;
    adam.learning_rate = 0.02;
    SA_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Optim::AdamW> optimizer,
                            Optim::AdamW::make(*engine, slots, adam));

    Train::TrainerOptions options;
    options.epochs = 30;
    SA_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Train::Trainer> trainer,
                            Train::Trainer::make(*engine, built.graph, pool, built.spec,
                                                 *optimizer, options));

    Data::SyntheticOptions data_options;
    data_options.batch_size = batch;
    data_options.batches = 6;
    data_options.features = features;
    data_options.classes = classes;
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<Data::SyntheticClassification> dataset,
                            Data::SyntheticClassification::make(*engine, data_options));

    SA_ASSERT_OK_AND_ASSIGN(const Train::RunReport report, trainer->fit(*dataset));

    EXPECT_LT(report.last_loss, report.first_loss * 0.25) << report.to_string();
    EXPECT_GT(report.last_accuracy, 0.9) << report.to_string();
    EXPECT_EQ(report.compile_count, 1u);
}

TEST_F(Integration_Training, AdamWsFirstStepMovesEveryParameterByAboutTheLearningRate)
{
    // Step one moves each coordinate by about lr; see tests/integration/README.md.
    const int64_t batch = 8;
    const int64_t features = 4;
    const int64_t classes = 3;

    NN::Linear layer{features, classes};
    SA_ASSERT_OK_AND_ASSIGN(Train::TrainingGraph built,
                            Train::build_classifier(layer, batch, features, classes));

    SA_ASSERT_OK_AND_ASSIGN(NN::ModelState state, NN::ModelState::create(*engine, layer));
    TensorPool pool(*engine);
    SA_ASSERT_OK_AND_ASSIGN(std::vector<Optim::ParameterSlot> slots,
                            Train::bind_parameters(built, state, pool));

    // No zeroed classifier here: a zero gradient would make the ratio 0/eps.
    NN::InitPolicy policy;
    policy.zero_final_layer = false;
    SA_EXPECT_OK(NN::initialize(*engine, state, policy));

    Optim::AdamWOptions adam;
    adam.learning_rate = 0.05;
    adam.weight_decay = 0.0;
    SA_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Optim::AdamW> optimizer,
                            Optim::AdamW::make(*engine, slots, adam));

    Train::TrainerOptions options;
    options.track_accuracy = false;
    SA_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Train::Trainer> trainer,
                            Train::Trainer::make(*engine, built.graph, pool, built.spec,
                                                 *optimizer, options));

    Data::SyntheticOptions data_options;
    data_options.batch_size = batch;
    data_options.batches = 1;
    data_options.features = features;
    data_options.classes = classes;
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<Data::SyntheticClassification> dataset,
                            Data::SyntheticClassification::make(*engine, data_options));

    const std::vector<double> before = SaTest::read_tensor(slots[0].parameter);
    SA_EXPECT_OK(trainer->run_step(dataset->batch(0)));
    const std::vector<double> after = SaTest::read_tensor(slots[0].parameter);
    const std::vector<double> gradient = SaTest::read_tensor(slots[0].gradient);

    ASSERT_EQ(before.size(), after.size());

    std::size_t checked = 0;
    for (std::size_t i = 0; i < before.size(); ++i)
    {
        // Coordinates where eps matters are skipped; see tests/integration/README.md.
        if (std::abs(gradient[i]) < 1e-4) continue;

        const double moved = (after[i] - before[i]) / -adam.learning_rate;
        const double expected = gradient[i] > 0.0 ? 1.0 : -1.0;
        EXPECT_NEAR(moved, expected, 2e-3) << "at " << i << " (gradient " << gradient[i] << ")";
        ++checked;
    }

    EXPECT_GT(checked, 0u) << "no coordinate had a large enough gradient to check";
}

TEST_F(Integration_Training, SgdWithoutMomentumIsExactlyMinusLearningRateTimesTheGradient)
{
    const int64_t batch = 8;
    const int64_t features = 4;
    const int64_t classes = 3;

    NN::Linear layer{features, classes};
    SA_ASSERT_OK_AND_ASSIGN(Train::TrainingGraph built,
                            Train::build_classifier(layer, batch, features, classes));

    SA_ASSERT_OK_AND_ASSIGN(NN::ModelState state, NN::ModelState::create(*engine, layer));
    TensorPool pool(*engine);
    SA_ASSERT_OK_AND_ASSIGN(std::vector<Optim::ParameterSlot> slots,
                            Train::bind_parameters(built, state, pool));

    NN::InitPolicy policy;
    policy.zero_final_layer = false;
    SA_EXPECT_OK(NN::initialize(*engine, state, policy));

    Optim::SGDOptions sgd;
    sgd.learning_rate = 0.1;
    SA_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Optim::SGD> optimizer,
                            Optim::SGD::make(*engine, slots, sgd));

    Train::TrainerOptions options;
    options.track_accuracy = false;
    SA_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Train::Trainer> trainer,
                            Train::Trainer::make(*engine, built.graph, pool, built.spec,
                                                 *optimizer, options));

    Data::SyntheticOptions data_options;
    data_options.batch_size = batch;
    data_options.batches = 1;
    data_options.features = features;
    data_options.classes = classes;
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<Data::SyntheticClassification> dataset,
                            Data::SyntheticClassification::make(*engine, data_options));

    const std::vector<double> before = SaTest::read_tensor(slots[0].parameter);
    SA_EXPECT_OK(trainer->run_step(dataset->batch(0)));
    const std::vector<double> after = SaTest::read_tensor(slots[0].parameter);
    const std::vector<double> gradient = SaTest::read_tensor(slots[0].gradient);

    for (std::size_t i = 0; i < before.size(); ++i)
        EXPECT_NEAR(after[i], before[i] - sgd.learning_rate * gradient[i], 1e-6) << "at " << i;
}

TEST_F(Integration_Training, ProfilingOnAnUnprofiledContextReportsNothingRatherThanZeroes)
{
    // The shared context records no timing; see tests/integration/README.md.
    const int64_t batch = 8;
    const int64_t features = 4;
    const int64_t classes = 3;

    Mlp model = make_model(features, 8, classes);
    SA_ASSERT_OK_AND_ASSIGN(Train::TrainingGraph built,
                            Train::build_classifier(model, batch, features, classes));

    SA_ASSERT_OK_AND_ASSIGN(NN::ModelState state, NN::ModelState::create(*engine, model));
    TensorPool pool(*engine);
    SA_ASSERT_OK_AND_ASSIGN(std::vector<Optim::ParameterSlot> slots,
                            Train::bind_parameters(built, state, pool));
    SA_EXPECT_OK(NN::initialize(*engine, state));

    SA_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Optim::SGD> optimizer,
                            Optim::SGD::make(*engine, slots, Optim::SGDOptions{}));

    Train::TrainerOptions options;
    options.epochs = 1;
    options.profile = true;
    SA_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Train::Trainer> trainer,
                            Train::Trainer::make(*engine, built.graph, pool, built.spec,
                                                 *optimizer, options));

    Data::SyntheticOptions data_options;
    data_options.batch_size = batch;
    data_options.batches = 2;
    data_options.features = features;
    data_options.classes = classes;
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<Data::SyntheticClassification> dataset,
                            Data::SyntheticClassification::make(*engine, data_options));

    SA_ASSERT_OK_AND_ASSIGN(const Train::RunReport report, trainer->fit(*dataset));

    EXPECT_EQ(report.steps, 2u);
    EXPECT_EQ(report.compile_count, 1u);
    EXPECT_NE(report.to_string().find("per-node timing"), std::string::npos);
}

TEST_F(Integration_Training, ADisconnectedParameterIsRejectedRatherThanSilentlySkipped)
{
    const int64_t batch = 4;
    const int64_t features = 4;
    const int64_t classes = 3;

    NN::Linear layer{features, classes};
    SA_ASSERT_OK_AND_ASSIGN(Train::TrainingGraph built,
                            Train::build_classifier(layer, batch, features, classes));

    // "ghost" is a parameter no gradient reaches; see tests/integration/README.md.
    built.parameters.push_back(Train::ParameterBinding{"ghost", layer.weight.id_in(built.graph),
                                                       Graph::K_NO_VALUE});

    SA_ASSERT_OK_AND_ASSIGN(NN::ModelState state, NN::ModelState::create(*engine, layer));
    TensorPool pool(*engine);
    SA_EXPECT_FAILURE(Train::bind_parameters(built, state, pool), StatusCode::InvalidDescription,
                      "'ghost'");
}

TEST_F(Integration_Training, TheSyntheticDatasetIsDeterministicAndBalanced)
{
    Data::SyntheticOptions options;
    options.batch_size = 8;
    options.batches = 3;
    options.features = 5;
    options.classes = 4;
    options.seed = 4242;

    // Each dataset gets a fresh Engine; see tests/integration/README.md.
    SushiBLAS::Engine first_engine(SaTest::TestEnvironment::context());
    SushiBLAS::Engine second_engine(SaTest::TestEnvironment::context());

    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<Data::SyntheticClassification> first,
                            Data::SyntheticClassification::make(first_engine, options));
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<Data::SyntheticClassification> second,
                            Data::SyntheticClassification::make(second_engine, options));

    ASSERT_EQ(first->batch_count(), 3u);
    EXPECT_EQ(first->name(), "synthetic-classification");

    for (std::size_t b = 0; b < first->batch_count(); ++b)
    {
        const std::vector<double> a = SaTest::read_tensor(first->batch(b).inputs);
        const std::vector<double> c = SaTest::read_tensor(second->batch(b).inputs);

        ASSERT_EQ(a.size(), c.size());
        for (std::size_t i = 0; i < a.size(); ++i)
            EXPECT_DOUBLE_EQ(a[i], c[i]) << "batch " << b << " element " << i;

        // Round-robin labels balance every batch; see tests/integration/README.md.
        std::vector<int> histogram(4, 0);
        for (int32_t label : first->batch(b).labels) ++histogram[static_cast<std::size_t>(label)];
        for (int count : histogram) EXPECT_EQ(count, 2);
    }
}

TEST_F(Integration_Training, RunningMeanMovesAcrossTwoStepsAsUpdatedStatisticSays)
{
    // Each step applies the host form of the update to the batch mean of that step.
    NormMlp model = make_norm_model();
    SA_ASSERT_OK_AND_ASSIGN(Train::TrainingGraph built,
                            Train::build_classifier(model, K_NORM_BATCH, K_NORM_FEATURES,
                                                    K_NORM_CLASSES));

    SA_ASSERT_OK_AND_ASSIGN(NN::ModelState state, NN::ModelState::create(*engine, model));
    TensorPool pool(*engine);
    SA_ASSERT_OK_AND_ASSIGN(std::vector<Optim::ParameterSlot> slots,
                            Train::bind_parameters(built, state, pool));
    SA_EXPECT_OK(NN::initialize(*engine, state));
    SA_ASSERT_OK_AND_ASSIGN(std::vector<Train::BufferUpdateSlot> updates,
                            Train::bind_buffer_updates(built.updates, state, pool));
    SA_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Optim::SGD> optimizer,
                            Optim::SGD::make(*engine, slots, Optim::SGDOptions{}));
    SA_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Train::Trainer> trainer,
                            Train::Trainer::make(*engine, built.graph, pool, built.spec,
                                                 *optimizer, Train::TrainerOptions{},
                                                 std::move(updates)));

    Data::SyntheticOptions data_options;
    data_options.batch_size = K_NORM_BATCH;
    data_options.batches = 2;
    data_options.features = K_NORM_FEATURES;
    data_options.classes = K_NORM_CLASSES;
    data_options.seed = K_DATA_SEED;
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<Data::SyntheticClassification> dataset,
                            Data::SyntheticClassification::make(*engine, data_options));

    const std::vector<double> means_first =
        channel_means(SaTest::read_tensor(dataset->batch(0).inputs));
    const std::vector<double> means_second =
        channel_means(SaTest::read_tensor(dataset->batch(1).inputs));

    SA_ASSERT_OK_AND_ASSIGN(SushiBLAS::Tensor* running,
                            state.tensor(running_mean_name(state)));

    SA_ASSERT_OK_AND_ASSIGN(const Train::StepMetrics first,
                            trainer->run_step(dataset->batch(0)));
    const std::vector<double> after_first = SaTest::read_tensor(*running);

    SA_ASSERT_OK_AND_ASSIGN(const Train::StepMetrics second,
                            trainer->run_step(dataset->batch(1)));
    const std::vector<double> after_second = SaTest::read_tensor(*running);

    EXPECT_GT(first.loss, 0.0);
    EXPECT_GT(second.loss, 0.0);

    const double momentum = NN::NormOptions{}.momentum;
    const std::size_t channels = static_cast<std::size_t>(K_NORM_CHANNELS);
    for (std::size_t channel = 0; channel < channels; ++channel)
    {
        const double expected_first =
            NN::updated_statistic(0.0, means_first[channel], momentum, 1.0);
        const double expected_second =
            NN::updated_statistic(expected_first, means_second[channel], momentum, 1.0);

        EXPECT_NEAR(after_first[channel], expected_first, K_TOLERANCE)
            << "channel " << channel << " after one step";
        EXPECT_NEAR(after_second[channel], expected_second, K_TOLERANCE)
            << "channel " << channel << " after two steps";
        EXPECT_GT(std::abs(after_second[channel] - after_first[channel]), K_MOVED)
            << "channel " << channel << " did not move on the second step";
    }
}

TEST_F(Integration_Training, APerStepOptimizerStillStepsWhenTheModelHoldsABatchNorm)
{
    NormMlp model = make_norm_model();
    SA_ASSERT_OK_AND_ASSIGN(Train::TrainingGraph built,
                            Train::build_classifier(model, K_NORM_BATCH, K_NORM_FEATURES,
                                                    K_NORM_CLASSES));

    SA_ASSERT_OK_AND_ASSIGN(NN::ModelState state, NN::ModelState::create(*engine, model));
    TensorPool pool(*engine);
    SA_ASSERT_OK_AND_ASSIGN(std::vector<Optim::ParameterSlot> slots,
                            Train::bind_parameters(built, state, pool));
    SA_EXPECT_OK(NN::initialize(*engine, state));
    SA_ASSERT_OK_AND_ASSIGN(std::vector<Train::BufferUpdateSlot> updates,
                            Train::bind_buffer_updates(built.updates, state, pool));
    SA_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Optim::SGD> optimizer,
                            Optim::SGD::make(*engine, slots, Optim::SGDOptions{}));

    Train::TrainerOptions options;
    options.optimizer_placement = Train::OptimizerPlacement::PerStep;
    SA_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Train::Trainer> trainer,
                            Train::Trainer::make(*engine, built.graph, pool, built.spec,
                                                 *optimizer, options, std::move(updates)));

    Data::SyntheticOptions data_options;
    data_options.batch_size = K_NORM_BATCH;
    data_options.batches = 1;
    data_options.features = K_NORM_FEATURES;
    data_options.classes = K_NORM_CLASSES;
    data_options.seed = K_DATA_SEED;
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<Data::SyntheticClassification> dataset,
                            Data::SyntheticClassification::make(*engine, data_options));

    std::vector<std::vector<double>> before;
    for (const Optim::ParameterSlot& slot : slots)
        before.push_back(SaTest::read_tensor(slot.parameter));

    SA_ASSERT_OK_AND_ASSIGN(const Train::StepMetrics metrics,
                            trainer->run_step(dataset->batch(0)));
    EXPECT_GT(metrics.loss, 0.0);
    EXPECT_FALSE(trainer->optimizer_is_in_plan());
    EXPECT_EQ(trainer->step_count(), 1u);

    // The classifier starts at zero, so the first step's gradient reaches its weight alone.
    bool found = false;
    for (std::size_t index = 0; index < slots.size(); ++index)
    {
        if (slots[index].name != "3.weight")
            continue;

        found = true;
        const std::vector<double> after = SaTest::read_tensor(slots[index].parameter);
        double largest = 0.0;

        for (std::size_t element = 0; element < after.size(); ++element)
            largest = std::max(largest, std::abs(after[element] - before[index][element]));

        EXPECT_GT(largest, K_MOVED) << "'3.weight' did not move under the per-step optimizer";
    }
    EXPECT_TRUE(found) << "the model has no parameter named '3.weight'";
}

TEST_F(Integration_Training, ABatchOfAnotherShapeIsRefusedWithBothShapes)
{
    const int64_t batch = 8;
    const int64_t features = 6;
    const int64_t classes = 5;

    Mlp model = make_model(features, 12, classes);
    SA_ASSERT_OK_AND_ASSIGN(Train::TrainingGraph built,
                            Train::build_classifier(model, batch, features, classes));

    SA_ASSERT_OK_AND_ASSIGN(NN::ModelState state, NN::ModelState::create(*engine, model));
    TensorPool pool(*engine);
    SA_ASSERT_OK_AND_ASSIGN(std::vector<Optim::ParameterSlot> slots,
                            Train::bind_parameters(built, state, pool));
    SA_EXPECT_OK(NN::initialize(*engine, state));

    SA_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Optim::SGD> optimizer,
                            Optim::SGD::make(*engine, slots, Optim::SGDOptions{}));
    SA_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Train::Trainer> trainer,
                            Train::Trainer::make(*engine, built.graph, pool, built.spec,
                                                 *optimizer));

    Data::SyntheticOptions data_options;
    data_options.batch_size = 4;
    data_options.batches = 1;
    data_options.features = features;
    data_options.classes = classes;
    data_options.seed = K_DATA_SEED;
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<Data::SyntheticClassification> wrong,
                            Data::SyntheticClassification::make(*engine, data_options));

    SA_EXPECT_FAILURE(trainer->run_step(wrong->batch(0)), StatusCode::Incompatible, "[4, 6]",
                      "[8, 6]");
}
