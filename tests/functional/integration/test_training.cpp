/****************************************************************/
/* test_training.cpp                                            */
/* SushiAI - https://github.com/SushiSystems/SushiAI            */
/* Copyright (c) 2026 Sushi Systems                             */
/* Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.    */
/* Commercial use requires a licence from Sushi Systems.        */
/****************************************************************/

// The end-to-end gate: a real MLP, real data, real convergence - and the
// compile-once guarantee held to account across every step of it.

#include <cmath>
#include <memory>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include "../../common/test_utils.hpp"

using namespace SushiAI;

// `Test` alone resolves to ::testing::Test inside a TEST_F body, where the
// fixture's base injects the name; the alias keeps the helpers reachable.
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
}

TEST_F(Integration_Training, InitialLossIsExactlyLogCWithAZeroedClassifier)
{
    // Zeroing the last layer makes every logit zero, so every class is equally
    // likely and the initial cross-entropy is ln(C) to the last digit. A
    // number that can be asserted beats a number that has to be eyeballed.
    const int64_t batch = 8;
    const int64_t features = 6;
    const int64_t classes = 5;

    Mlp model = make_model(features, 12, classes);
    Train::TrainingGraph built = Train::build_classifier(model, batch, features, classes);

    TensorPool pool(*engine);
    std::vector<Optim::ParameterSlot> slots = Train::bind_parameters(built, pool);
    NN::initialize(*engine, model, pool);

    Optim::SGD optimizer(*engine, slots, Optim::SGDOptions{});
    Train::TrainerOptions options;
    options.track_accuracy = false;
    Train::Trainer trainer(*engine, built.graph, pool, built.spec, optimizer, options);

    Data::SyntheticOptions data_options;
    data_options.batch_size = batch;
    data_options.batches = 1;
    data_options.features = features;
    data_options.classes = classes;
    const Data::SyntheticClassification dataset(*engine, data_options);

    const Train::StepMetrics metrics = trainer.run_step(dataset.batch(0));
    EXPECT_NEAR(metrics.loss, std::log(static_cast<double>(classes)), 1e-5);
}

TEST_F(Integration_Training, TheStepIsCompiledExactlyOnceAcrossManySteps)
{
    // The AI-1 headline gate. A loop that recompiles per step still converges,
    // it just pays graph analysis every step - a regression that hides for
    // months unless something counts.
    const int64_t batch = 16;
    const int64_t features = 8;
    const int64_t classes = 4;

    Mlp model = make_model(features, 16, classes);
    Train::TrainingGraph built = Train::build_classifier(model, batch, features, classes);

    TensorPool pool(*engine);
    std::vector<Optim::ParameterSlot> slots = Train::bind_parameters(built, pool);
    NN::initialize(*engine, model, pool);

    Optim::AdamW optimizer(*engine, slots, Optim::AdamWOptions{});
    Train::Trainer trainer(*engine, built.graph, pool, built.spec, optimizer);

    Data::SyntheticOptions data_options;
    data_options.batch_size = batch;
    data_options.batches = 8;
    data_options.features = features;
    data_options.classes = classes;
    const Data::SyntheticClassification dataset(*engine, data_options);

    EXPECT_EQ(trainer.compile_count(), 1u);

    Train::TrainerOptions options;
    options.epochs = 5;
    trainer.options() = options;

    const Train::RunReport report = trainer.fit(dataset);

    EXPECT_EQ(report.steps, 40u);
    EXPECT_EQ(report.compile_count, 1u);
    EXPECT_EQ(trainer.compile_count(), 1u);
    EXPECT_EQ(trainer.step_count(), 40u);
    EXPECT_GT(report.operations_per_step, 0u);
}

TEST_F(Integration_Training, TheStepAllocatesNothingInSteadyState)
{
    // Compile-once is only half the claim; the other half is that replaying
    // reuses the same buffers, so a long run does not creep.
    const int64_t batch = 8;
    const int64_t features = 6;
    const int64_t classes = 3;

    Mlp model = make_model(features, 8, classes);
    Train::TrainingGraph built = Train::build_classifier(model, batch, features, classes);

    TensorPool pool(*engine);
    std::vector<Optim::ParameterSlot> slots = Train::bind_parameters(built, pool);
    NN::initialize(*engine, model, pool);

    Optim::SGD optimizer(*engine, slots, Optim::SGDOptions{});
    Train::Trainer trainer(*engine, built.graph, pool, built.spec, optimizer);

    Data::SyntheticOptions data_options;
    data_options.batch_size = batch;
    data_options.batches = 4;
    data_options.features = features;
    data_options.classes = classes;
    const Data::SyntheticClassification dataset(*engine, data_options);

    trainer.run_step(dataset.batch(0));
    const std::size_t tensors_after_first = pool.tensor_count();
    const std::size_t bytes_after_first = pool.allocated_bytes();

    for (int i = 0; i < 12; ++i) trainer.run_step(dataset.batch(static_cast<std::size_t>(i % 4)));

    EXPECT_EQ(pool.tensor_count(), tensors_after_first);
    EXPECT_EQ(pool.allocated_bytes(), bytes_after_first);
}

TEST_F(Integration_Training, SgdDrivesTheLossDownOnASeparableProblem)
{
    const int64_t batch = 16;
    const int64_t features = 8;
    const int64_t classes = 4;

    Mlp model = make_model(features, 24, classes);
    Train::TrainingGraph built = Train::build_classifier(model, batch, features, classes);

    TensorPool pool(*engine);
    std::vector<Optim::ParameterSlot> slots = Train::bind_parameters(built, pool);
    NN::initialize(*engine, model, pool);

    Optim::SGDOptions sgd;
    sgd.learning_rate = 0.2;
    sgd.momentum = 0.9;
    Optim::SGD optimizer(*engine, slots, sgd);

    Train::TrainerOptions options;
    options.epochs = 30;
    Train::Trainer trainer(*engine, built.graph, pool, built.spec, optimizer, options);

    Data::SyntheticOptions data_options;
    data_options.batch_size = batch;
    data_options.batches = 6;
    data_options.features = features;
    data_options.classes = classes;
    const Data::SyntheticClassification dataset(*engine, data_options);

    const Train::RunReport report = trainer.fit(dataset);

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
    Train::TrainingGraph built = Train::build_classifier(model, batch, features, classes);

    TensorPool pool(*engine);
    std::vector<Optim::ParameterSlot> slots = Train::bind_parameters(built, pool);
    NN::initialize(*engine, model, pool);

    Optim::AdamWOptions adam;
    adam.learning_rate = 0.02;
    Optim::AdamW optimizer(*engine, slots, adam);

    Train::TrainerOptions options;
    options.epochs = 30;
    Train::Trainer trainer(*engine, built.graph, pool, built.spec, optimizer, options);

    Data::SyntheticOptions data_options;
    data_options.batch_size = batch;
    data_options.batches = 6;
    data_options.features = features;
    data_options.classes = classes;
    const Data::SyntheticClassification dataset(*engine, data_options);

    const Train::RunReport report = trainer.fit(dataset);

    EXPECT_LT(report.last_loss, report.first_loss * 0.25) << report.to_string();
    EXPECT_GT(report.last_accuracy, 0.9) << report.to_string();
    EXPECT_EQ(report.compile_count, 1u);
}

TEST_F(Integration_Training, AdamWsFirstStepMovesEveryParameterByAboutTheLearningRate)
{
    // The property the "eps after the bias correction" ordering exists to
    // preserve. On step one, m / (sqrt(v / bc2) + eps) is g / (|g| + eps), so
    // every coordinate moves by almost exactly the learning rate. Fold eps
    // inside the correction and it becomes 31.6x larger, visibly damping this
    // first step - which is the only place the bug is easy to see.
    const int64_t batch = 8;
    const int64_t features = 4;
    const int64_t classes = 3;

    NN::Linear layer{features, classes};
    Train::TrainingGraph built = Train::build_classifier(layer, batch, features, classes);

    TensorPool pool(*engine);
    std::vector<Optim::ParameterSlot> slots = Train::bind_parameters(built, pool);

    // No zeroed classifier here: a zero gradient would make the ratio 0/eps.
    NN::InitPolicy policy;
    policy.zero_final_layer = false;
    NN::initialize(*engine, layer, pool, policy);

    Optim::AdamWOptions adam;
    adam.learning_rate = 0.05;
    adam.weight_decay = 0.0;
    Optim::AdamW optimizer(*engine, slots, adam);

    Train::TrainerOptions options;
    options.track_accuracy = false;
    Train::Trainer trainer(*engine, built.graph, pool, built.spec, optimizer, options);

    Data::SyntheticOptions data_options;
    data_options.batch_size = batch;
    data_options.batches = 1;
    data_options.features = features;
    data_options.classes = classes;
    const Data::SyntheticClassification dataset(*engine, data_options);

    const std::vector<double> before = SaTest::read_tensor(slots[0].parameter);
    trainer.run_step(dataset.batch(0));
    const std::vector<double> after = SaTest::read_tensor(slots[0].parameter);
    const std::vector<double> gradient = SaTest::read_tensor(slots[0].gradient);

    ASSERT_EQ(before.size(), after.size());

    std::size_t checked = 0;
    for (std::size_t i = 0; i < before.size(); ++i)
    {
        // Skip coordinates whose gradient is small enough that eps genuinely
        // matters; the claim is about the ordinary case.
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
    Train::TrainingGraph built = Train::build_classifier(layer, batch, features, classes);

    TensorPool pool(*engine);
    std::vector<Optim::ParameterSlot> slots = Train::bind_parameters(built, pool);

    NN::InitPolicy policy;
    policy.zero_final_layer = false;
    NN::initialize(*engine, layer, pool, policy);

    Optim::SGDOptions sgd;
    sgd.learning_rate = 0.1;
    Optim::SGD optimizer(*engine, slots, sgd);

    Train::TrainerOptions options;
    options.track_accuracy = false;
    Train::Trainer trainer(*engine, built.graph, pool, built.spec, optimizer, options);

    Data::SyntheticOptions data_options;
    data_options.batch_size = batch;
    data_options.batches = 1;
    data_options.features = features;
    data_options.classes = classes;
    const Data::SyntheticClassification dataset(*engine, data_options);

    const std::vector<double> before = SaTest::read_tensor(slots[0].parameter);
    trainer.run_step(dataset.batch(0));
    const std::vector<double> after = SaTest::read_tensor(slots[0].parameter);
    const std::vector<double> gradient = SaTest::read_tensor(slots[0].gradient);

    for (std::size_t i = 0; i < before.size(); ++i)
        EXPECT_NEAR(after[i], before[i] - sgd.learning_rate * gradient[i], 1e-6) << "at " << i;
}

TEST_F(Integration_Training, ProfilingOnAnUnprofiledContextReportsNothingRatherThanZeroes)
{
    // The shared test context is built without RuntimeSettings::profiling, so
    // the runtime records no timing. The report has to say that, not print a
    // table of zeroes that reads like the work took no time.
    const int64_t batch = 8;
    const int64_t features = 4;
    const int64_t classes = 3;

    Mlp model = make_model(features, 8, classes);
    Train::TrainingGraph built = Train::build_classifier(model, batch, features, classes);

    TensorPool pool(*engine);
    std::vector<Optim::ParameterSlot> slots = Train::bind_parameters(built, pool);
    NN::initialize(*engine, model, pool);

    Optim::SGD optimizer(*engine, slots, Optim::SGDOptions{});

    Train::TrainerOptions options;
    options.epochs = 1;
    options.profile = true;
    Train::Trainer trainer(*engine, built.graph, pool, built.spec, optimizer, options);

    Data::SyntheticOptions data_options;
    data_options.batch_size = batch;
    data_options.batches = 2;
    data_options.features = features;
    data_options.classes = classes;
    const Data::SyntheticClassification dataset(*engine, data_options);

    const Train::RunReport report = trainer.fit(dataset);

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
    Train::TrainingGraph built = Train::build_classifier(layer, batch, features, classes);

    // Fake a parameter that no gradient reaches, standing in for a layer wired
    // into the model list but not into the forward pass.
    built.parameters.push_back(Train::ParameterBinding{"ghost", layer.weight.id(),
                                                       Graph::NO_VALUE});

    TensorPool pool(*engine);
    EXPECT_THROW((void)Train::bind_parameters(built, pool), Error);
}

TEST_F(Integration_Training, TheSyntheticDatasetIsDeterministicAndBalanced)
{
    Data::SyntheticOptions options;
    options.batch_size = 8;
    options.batches = 3;
    options.features = 5;
    options.classes = 4;
    options.seed = 4242;

    // A fresh Engine per dataset, deliberately. SushiBLAS's Engine::set_seed()
    // sets the seed but leaves the monotonic RNG offset counter alone, so a
    // second generator built on an engine that has already drawn gets a
    // different stream from the same seed. Reproducibility is therefore a
    // property of a fresh engine, which is exactly how the trainer uses it.
    SushiBLAS::Engine first_engine(SaTest::TestEnvironment::context());
    SushiBLAS::Engine second_engine(SaTest::TestEnvironment::context());

    const Data::SyntheticClassification first(first_engine, options);
    const Data::SyntheticClassification second(second_engine, options);

    ASSERT_EQ(first.batch_count(), 3u);
    EXPECT_EQ(first.name(), "synthetic-classification");

    for (std::size_t b = 0; b < first.batch_count(); ++b)
    {
        const std::vector<double> a = SaTest::read_tensor(first.batch(b).inputs);
        const std::vector<double> c = SaTest::read_tensor(second.batch(b).inputs);

        ASSERT_EQ(a.size(), c.size());
        for (std::size_t i = 0; i < a.size(); ++i)
            EXPECT_DOUBLE_EQ(a[i], c[i]) << "batch " << b << " element " << i;

        // Round-robin labels make every batch exactly balanced, so accuracy
        // reads correctly without class weighting.
        std::vector<int> histogram(4, 0);
        for (int32_t label : first.batch(b).labels) ++histogram[static_cast<std::size_t>(label)];
        for (int count : histogram) EXPECT_EQ(count, 2);
    }
}

TEST_F(Integration_Training, MnistFallsBackToSyntheticWhenTheFilesAreAbsent)
{
    Data::MnistOptions options;
    options.directory = "this-directory-does-not-exist";

    EXPECT_FALSE(Data::mnist_available(options));
    EXPECT_EQ(Data::try_load_mnist(*engine, options), nullptr);
    EXPECT_FALSE(Data::mnist_expected_paths(options).empty());
}
