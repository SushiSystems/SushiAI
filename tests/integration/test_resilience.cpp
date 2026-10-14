/**************************************************************************/
/* test_resilience.cpp                                                    */
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
 * @file test_resilience.cpp
 * @brief Tests loud failures, boundary extents, numerical extremes and resource discipline.
 * @author Mustafa Garip
 */

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <string_view>
#include <vector>
#include <gtest/gtest.h>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/data/roles.hpp>
#include <SushiAI/data/target.hpp>
#include <SushiAI/classifier_objective/classifier_objective.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/SushiAI.h>
#include <SushiAI/engine_fixture/engine_test.hpp>
#include <SushiAI/tensor_values/tensor_values.hpp>
#include <SushiAI/seeded_noise/host_noise.hpp>

using namespace SushiAI;

// SaTest stays reachable inside a TEST_F body; see tests/integration/README.md.
namespace SaTest = SushiAI::Test;
using Graph::CompiledStep;
using Graph::GraphBuilder;
using Graph::TensorPool;
using Graph::ValueID;

namespace
{
    class Integration_Resilience : public SaTest::EngineTest {};

    /** @brief Names the suite of the cases that expect the process to stop. */
    using Integration_ResilienceDeathTest = Integration_Resilience;

    using MLP = NN::Sequential<NN::Linear, NN::ReLU, NN::Linear>;

    /**
     * @class Runner
     * @brief Materialises a graph's leaves, compiles once, and replays.
     *
     * The same shape as test_forward.cpp's OneShot, kept local rather than
     * shared because the two files probe different things and a shared
     * fixture would couple their evolution.
     */
    class Runner
    {
        public:
            /**
             * @brief Builds a runner over @p graph and materialises its leaves.
             * @param engine Must outlive the runner.
             * @param graph Must outlive the runner.
             * @return The runner, or the failure of a leaf's allocation.
             */
            [[nodiscard]] static Result<std::unique_ptr<Runner>> make(SushiBLAS::Engine& engine,
                                                                      const Graph::Graph& graph)
            {
                std::unique_ptr<Runner> runner(new Runner(engine, graph));
                SA_TRY(runner->pool.materialise_leaves(graph));
                return runner;
            }

            /**
             * @brief Compiles the graph.
             * @pre Every leaf has been written.
             */
            [[nodiscard]] Status compile()
            {
                SA_TRY_ASSIGN(step_, CompiledStep::compile(*engine_, *graph_, pool));
                return Status();
            }

            /**
             * @brief Replays the compiled step.
             * @pre compile() has returned success.
             */
            [[nodiscard]] Status run() { return step_->run(); }

            TensorPool pool;

        private:
            /** @brief Stores the graph and the engine and builds an empty pool. */
            Runner(SushiBLAS::Engine& engine, const Graph::Graph& graph)
                : pool(engine), graph_(&graph), engine_(&engine) {}

            const Graph::Graph* graph_;
            SushiBLAS::Engine* engine_;
            std::unique_ptr<CompiledStep> step_;
    };

    /**
     * @class NoBatches
     * @brief A dataset with the right shapes and nothing in it.
     *
     * SyntheticClassification refuses to generate zero batches, so an empty
     * epoch cannot be reached through it - which is exactly why the trainer's
     * own guard needs a dataset that can.
     */
    class NoBatches final : public Data::Dataset
    {
        public:
            NoBatches(int64_t batch, int64_t features, int64_t classes)
                : batch_(batch), features_(features),
                  targets_{Data::TargetDescription{std::string(Data::K_ROLE_LABELS), Shape{},
                                                   DType::Int32, classes}} {}

            int64_t batch_size() const noexcept override { return batch_; }
            Shape sample_shape() const override { return Shape{features_}; }
            const std::vector<Data::TargetDescription>& targets() const noexcept override
            {
                return targets_;
            }
            std::size_t batch_count() const noexcept override { return 0; }
            std::string_view name() const noexcept override { return "no-batches"; }

            [[nodiscard]] Status read(std::size_t index, Data::Batch&) const override
            {
                SA_FATAL_IF(true, "Batch {} is out of range; the dataset has 0", index);
                return Status();
            }

        private:
            int64_t batch_ = 0;
            int64_t features_ = 0;
            std::vector<Data::TargetDescription> targets_;
    };

    /** @brief Every element of @p values is finite. */
    ::testing::AssertionResult all_finite(const std::vector<double>& values,
                                          std::string_view what)
    {
        for (std::size_t i = 0; i < values.size(); ++i)
            if (!std::isfinite(values[i]))
                return ::testing::AssertionFailure()
                    << what << "[" << i << "] is " << values[i]
                    << " - a non-finite value propagates into every weight on the next "
                       "update and the loss simply stops moving";

        return ::testing::AssertionSuccess();
    }

    /** @brief A ready-to-train MLP with its state, its pool and its parameter slots. */
    struct Fixture
    {
        MLP model;
        Train::TrainingGraph built;
        std::unique_ptr<NN::ModelState> state;
        std::unique_ptr<TensorPool> pool;
        std::vector<Optim::ParameterSlot> slots;
    };

    /** @brief Traces and binds an MLP of the given extents. */
    [[nodiscard]] Result<Fixture> make_fixture(SushiBLAS::Engine& engine, int64_t batch,
                                               int64_t features, int64_t hidden, int64_t classes)
    {
        Fixture fixture{MLP(NN::Linear{features, hidden}, NN::ReLU{},
                            NN::Linear{hidden, classes}), {}, nullptr, nullptr, {}};

        SA_TRY_ASSIGN(fixture.built,
                      Train::build_classifier(fixture.model, batch, features, classes));
        SA_TRY_ASSIGN(NN::ModelState created, NN::ModelState::create(engine, fixture.model));
        fixture.state = std::make_unique<NN::ModelState>(std::move(created));
        fixture.pool = std::make_unique<TensorPool>(engine);
        SA_TRY_ASSIGN(fixture.slots,
                      Train::bind_parameters(fixture.built, *fixture.state, *fixture.pool));
        SA_TRY(NN::initialize(engine, *fixture.state));
        return fixture;
    }

    /** @brief A synthetic problem of the given extents. */
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
}

TEST_F(Integration_Resilience, AnOptimizerNamesTheParameterWhoseGradientDoesNotMatch)
{
    SA_ASSERT_OK_AND_ASSIGN(SushiBLAS::Tensor parameter, allocate_tensor(*engine, Shape{4, 3}));
    SA_ASSERT_OK_AND_ASSIGN(SushiBLAS::Tensor gradient, allocate_tensor(*engine, Shape{4, 2}));

    std::vector<Optim::ParameterSlot> slots;
    slots.push_back(Optim::ParameterSlot{"0.weight", parameter, gradient});

    SA_EXPECT_FAILURE(Optim::SGD::make(*engine, slots, Optim::SGDOptions{}),
                      StatusCode::InvalidDescription,
                      "0.weight", "12 elements", "gradient has 8");
}

TEST_F(Integration_Resilience, AnOptimizerNamesBothDtypesWhenAGradientIsTheWrongType)
{
    SA_ASSERT_OK_AND_ASSIGN(SushiBLAS::Tensor parameter,
                            allocate_tensor(*engine, Shape{4}, DType::Float32));
    SA_ASSERT_OK_AND_ASSIGN(SushiBLAS::Tensor gradient,
                            allocate_tensor(*engine, Shape{4}, DType::Float64));

    std::vector<Optim::ParameterSlot> slots;
    slots.push_back(Optim::ParameterSlot{"0.bias", parameter, gradient});

    SA_EXPECT_FAILURE(Optim::SGD::make(*engine, slots, Optim::SGDOptions{}),
                      StatusCode::InvalidDescription, "0.bias", "float32", "float64");
}

TEST_F(Integration_Resilience, AnOptimizerRefusesADTypeItCannotServeAndExplainsWhy)
{
    // The half refusal must say why in its message; see tests/integration/README.md.
    SA_ASSERT_OK_AND_ASSIGN(SushiBLAS::Tensor complex_parameter,
                            allocate_tensor(*engine, Shape{4}, DType::Complex32));
    SA_ASSERT_OK_AND_ASSIGN(SushiBLAS::Tensor complex_gradient,
                            allocate_tensor(*engine, Shape{4}, DType::Complex32));

    std::vector<Optim::ParameterSlot> complex_slot;
    complex_slot.push_back(Optim::ParameterSlot{"0.weight", complex_parameter, complex_gradient});

    SA_EXPECT_FAILURE(Optim::SGD::make(*engine, complex_slot, Optim::SGDOptions{}),
                      StatusCode::InvalidDescription,
                      "0.weight", "complex32", "real floating tensors only");

    SA_ASSERT_OK_AND_ASSIGN(SushiBLAS::Tensor half_parameter,
                            allocate_tensor(*engine, Shape{4}, DType::Half));
    SA_ASSERT_OK_AND_ASSIGN(SushiBLAS::Tensor half_gradient,
                            allocate_tensor(*engine, Shape{4}, DType::Half));

    std::vector<Optim::ParameterSlot> half_slot;
    half_slot.push_back(Optim::ParameterSlot{"0.weight", half_parameter, half_gradient});

    SA_EXPECT_FAILURE(Optim::AdamW::make(*engine, half_slot, Optim::AdamWOptions{}),
                      StatusCode::InvalidDescription,
                      "0.weight", "half precision", "float32");
}

TEST_F(Integration_Resilience, AdamWNamesTheHyperparameterThatIsOutOfRange)
{
    SA_ASSERT_OK_AND_ASSIGN(SushiBLAS::Tensor parameter, allocate_tensor(*engine, Shape{4}));
    SA_ASSERT_OK_AND_ASSIGN(SushiBLAS::Tensor gradient, allocate_tensor(*engine, Shape{4}));

    std::vector<Optim::ParameterSlot> slots;
    slots.push_back(Optim::ParameterSlot{"0.weight", parameter, gradient});

    Optim::AdamWOptions bad_beta1;
    bad_beta1.beta1 = 1.0;
    SA_EXPECT_FAILURE(Optim::AdamW::make(*engine, slots, bad_beta1),
                      StatusCode::InvalidDescription, "beta1", "[0, 1)");

    Optim::AdamWOptions bad_beta2;
    bad_beta2.beta2 = -0.5;
    SA_EXPECT_FAILURE(Optim::AdamW::make(*engine, slots, bad_beta2),
                      StatusCode::InvalidDescription, "beta2", "[0, 1)");

    Optim::AdamWOptions bad_eps;
    bad_eps.eps = 0.0;
    SA_EXPECT_FAILURE(Optim::AdamW::make(*engine, slots, bad_eps),
                      StatusCode::InvalidDescription, "eps must be positive");
}

TEST_F(Integration_Resilience, ADisconnectedParameterNamesItselfAndSaysWhatToDo)
{
    NN::Linear layer{4, 3};
    SA_ASSERT_OK_AND_ASSIGN(Train::TrainingGraph built, Train::build_classifier(layer, 4, 4, 3));
    built.parameters.push_back(Train::ParameterBinding{"ghost", layer.weight.id_in(built.graph),
                                                       Graph::K_NO_VALUE});

    SA_ASSERT_OK_AND_ASSIGN(NN::ModelState state, NN::ModelState::create(*engine, layer));
    TensorPool pool(*engine);
    SA_EXPECT_FAILURE(Train::bind_parameters(built, state, pool), StatusCode::InvalidDescription,
                      "'ghost'", "receives no gradient", "no optimizer could ever move it");
}

TEST_F(Integration_Resilience, AnOptimizerStepBeforeAnyBackwardIsANoOpRatherThanGarbage)
{
    // An unwritten gradient must read as zero; see tests/integration/README.md.
    SA_ASSERT_OK_AND_ASSIGN(Fixture fixture, make_fixture(*engine, 8, 6, 12, 4));

    for (const Optim::ParameterSlot& slot : fixture.slots)
    {
        const std::vector<double> gradient = SaTest::read_tensor(slot.gradient);
        ASSERT_TRUE(all_finite(gradient, slot.name + ".grad"));

        for (std::size_t i = 0; i < gradient.size(); ++i)
            ASSERT_EQ(gradient[i], 0.0)
                << "'" << slot.name << "' gradient element " << i
                << " holds " << gradient[i] << " before any backward pass has run";
    }

    const std::vector<std::vector<double>> before = [&fixture]()
    {
        std::vector<std::vector<double>> snapshot;
        for (const Optim::ParameterSlot& slot : fixture.slots)
            snapshot.push_back(SaTest::read_tensor(slot.parameter));
        return snapshot;
    }();

    Optim::SGDOptions sgd;
    sgd.learning_rate = 0.1;
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<Optim::SGD> optimizer,
                            Optim::SGD::make(*engine, fixture.slots, sgd));
    SA_EXPECT_OK(optimizer->step());

    for (std::size_t p = 0; p < fixture.slots.size(); ++p)
    {
        const std::vector<double> after = SaTest::read_tensor(fixture.slots[p].parameter);
        ASSERT_EQ(after.size(), before[p].size());

        for (std::size_t i = 0; i < after.size(); ++i)
            EXPECT_EQ(after[i], before[p][i])
                << "'" << fixture.slots[p].name << "' moved on a step with no gradient, at " << i;
    }
}

TEST_F(Integration_Resilience, TheTrainerNamesTheBatchThatDoesNotFitTheCompiledStep)
{
    SA_ASSERT_OK_AND_ASSIGN(Fixture fixture, make_fixture(*engine, 8, 6, 10, 4));
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<Optim::SGD> optimizer,
                            Optim::SGD::make(*engine, fixture.slots, Optim::SGDOptions{}));
    const auto objective = SaTest::classifier_objective(fixture.built);
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<Train::Trainer> trainer,
                            Train::Trainer::make(*engine, fixture.built, *fixture.pool,
                                                 objective, *optimizer));

    // A wrong-sized batch can only be reported; see tests/integration/README.md.
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<Data::SyntheticClassification> wrong,
                            Data::SyntheticClassification::make(*engine, data_for(4, 1, 6, 4)));

    Data::Batch wrong_batch;
    ASSERT_TRUE(SaTest::status_is_ok(wrong->read(0, wrong_batch)));
    SA_EXPECT_FAILURE(trainer->run_step(wrong_batch), StatusCode::Incompatible,
                      "[4, 6]", "[8, 6]");
}

TEST_F(Integration_Resilience, FitNamesTheDatasetDimensionThatDisagrees)
{
    SA_ASSERT_OK_AND_ASSIGN(Fixture fixture, make_fixture(*engine, 8, 6, 10, 4));
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<Optim::SGD> optimizer,
                            Optim::SGD::make(*engine, fixture.slots, Optim::SGDOptions{}));
    const auto objective = SaTest::classifier_objective(fixture.built);
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<Train::Trainer> trainer,
                            Train::Trainer::make(*engine, fixture.built, *fixture.pool,
                                                 objective, *optimizer));

    const NoBatches empty(8, 6, 4);
    SA_EXPECT_FAILURE(trainer->fit(empty), StatusCode::InvalidDescription,
                      "The dataset holds no batches");

    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<Data::SyntheticClassification> narrow,
                            Data::SyntheticClassification::make(*engine, data_for(4, 1, 6, 4)));
    SA_EXPECT_FAILURE(trainer->fit(*narrow), StatusCode::Incompatible,
                      "batch size is 4", "compiled for 8");

    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<Data::SyntheticClassification> wrong_classes,
                            Data::SyntheticClassification::make(*engine, data_for(8, 1, 6, 5)));
    SA_EXPECT_FAILURE(trainer->fit(*wrong_classes), StatusCode::Incompatible,
                      "5 classes", "compiled for 4");
}

TEST_F(Integration_Resilience, TheSyntheticDatasetNamesEveryExtentWhenOneIsNotPositive)
{
    SA_EXPECT_FAILURE(Data::SyntheticClassification::make(*engine, data_for(8, 0, 6, 4)),
                      StatusCode::InvalidDescription, "positive extents", "batches=0");
    SA_EXPECT_FAILURE(Data::SyntheticClassification::make(*engine, data_for(0, 2, 6, 4)),
                      StatusCode::InvalidDescription, "positive extents", "batch=0");
    SA_EXPECT_FAILURE(Data::SyntheticClassification::make(*engine, data_for(8, 2, 6, 0)),
                      StatusCode::InvalidDescription, "positive extents", "classes=0");
}

TEST_F(Integration_Resilience, AllocationRefusesAnEmptyOrUnrepresentableTensor)
{
    const int64_t empty_dims[2] = {4, 0};
    SA_EXPECT_FAILURE(allocate_tensor(*engine, SushiRuntime::span<const int64_t>(empty_dims, 2)),
                      StatusCode::InvalidDescription, "Tensor extents must be positive", "got 0");

    // Overflow is detected before the multiply; see tests/integration/README.md.
    const int64_t overflowing[3] = {3'000'000'000LL, 3'000'000'000LL, 3'000'000'000LL};
    SA_EXPECT_FAILURE(allocate_tensor(*engine, SushiRuntime::span<const int64_t>(overflowing, 3)),
                      StatusCode::InvalidDescription, "more than", "elements", "3000000000");
}

TEST_F(Integration_Resilience, AModelTooLargeToAllocateNamesTheTensorRatherThanTheKernel)
{
    // A petabyte: the error names shape and dtype; see tests/integration/README.md.
    const int64_t enormous[2] = {1LL << 25, 1LL << 25};

    SA_EXPECT_FAILURE(allocate_tensor(*engine, SushiRuntime::span<const int64_t>(enormous, 2)),
                      StatusCode::OutOfMemory,
                      "Could not allocate the float32 tensor", "[33554432, 33554432]");
}

TEST_F(Integration_Resilience, ASingleSampleWithASingleFeatureTrains)
{
    // N = 1 and D = 1 with one Linear: every loop bound is at its minimum.
    const int64_t batch = 1;
    const int64_t features = 1;
    const int64_t classes = 3;

    NN::Linear layer{features, classes};
    SA_ASSERT_OK_AND_ASSIGN(Train::TrainingGraph built,
                            Train::build_classifier(layer, batch, features, classes));

    SA_ASSERT_OK_AND_ASSIGN(NN::ModelState state, NN::ModelState::create(*engine, layer));
    TensorPool pool(*engine);
    SA_ASSERT_OK_AND_ASSIGN(std::vector<Optim::ParameterSlot> slots,
                            Train::bind_parameters(built, state, pool));
    SA_EXPECT_OK(NN::initialize(*engine, state));

    Optim::SGDOptions sgd;
    sgd.learning_rate = 0.5;
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<Optim::SGD> optimizer,
                            Optim::SGD::make(*engine, slots, sgd));

    Train::TrainerOptions options;
    options.track_metrics = false;
    const auto objective = SaTest::classifier_objective(built);
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<Train::Trainer> trainer,
                            Train::Trainer::make(*engine, built, pool, objective,
                                                 *optimizer, options));

    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<Data::SyntheticClassification> dataset,
                            Data::SyntheticClassification::make(
                                *engine, data_for(batch, 2, features, classes)));

    Data::Batch dataset_batch;
    ASSERT_TRUE(SaTest::status_is_ok(dataset->read(0, dataset_batch)));
    SA_ASSERT_OK_AND_ASSIGN(const Train::StepMetrics first, trainer->run_step(dataset_batch));
    EXPECT_NEAR(first.loss, std::log(static_cast<double>(classes)), 1e-5);

    for (int step = 0; step < 8; ++step)
    {
        ASSERT_TRUE(SaTest::status_is_ok(
            dataset->read(static_cast<std::size_t>(step % 2), dataset_batch)));
        SA_ASSERT_OK_AND_ASSIGN(
            const Train::StepMetrics metrics,
            trainer->run_step(dataset_batch));
        ASSERT_TRUE(std::isfinite(metrics.loss)) << "at step " << step;
    }

    EXPECT_TRUE(all_finite(SaTest::read_tensor(slots[0].parameter), "weight"));
    EXPECT_TRUE(all_finite(SaTest::read_tensor(slots[0].gradient), "gradient"));
}

TEST_F(Integration_Resilience, ASingleClassIsDegenerateButNotUndefined)
{
    // With C = 1 loss and gradient are exactly zero; see tests/integration/README.md.
    const int64_t batch = 4;
    const int64_t features = 3;
    const int64_t classes = 1;

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

    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<Optim::AdamW> optimizer,
                            Optim::AdamW::make(*engine, slots, Optim::AdamWOptions{}));
    const auto objective = SaTest::classifier_objective(built);
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<Train::Trainer> trainer,
                            Train::Trainer::make(*engine, built, pool, objective,
                                                 *optimizer));

    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<Data::SyntheticClassification> dataset,
                            Data::SyntheticClassification::make(
                                *engine, data_for(batch, 1, features, classes)));

    Data::Batch dataset_batch;
    ASSERT_TRUE(SaTest::status_is_ok(dataset->read(0, dataset_batch)));
    SA_ASSERT_OK_AND_ASSIGN(const Train::StepMetrics metrics,
                            trainer->run_step(dataset_batch));

    EXPECT_NEAR(metrics.loss, 0.0, 1e-4);
    EXPECT_DOUBLE_EQ(metrics.metrics.value_or("accuracy", 0.0), 1.0);
    EXPECT_TRUE(all_finite(SaTest::read_tensor(slots[0].parameter), "weight"));

    const std::vector<double> gradient = SaTest::read_tensor(slots[0].gradient);
    EXPECT_TRUE(all_finite(gradient, "gradient"));
    for (std::size_t i = 0; i < gradient.size(); ++i)
        EXPECT_NEAR(gradient[i], 0.0, 1e-4) << "at " << i;
}

TEST_F(Integration_Resilience, MatmulIsExactAtTheRegisterBlockEdgeAndOnePastIt)
{
    // 64 fills a GEMM block; 65 leaves a remainder; see tests/integration/README.md.
    for (int64_t n : {int64_t{64}, int64_t{65}})
    {
        Graph::Graph graph;
        const GraphBuilder builder(graph);

        const ValueID x = builder.input(Shape{n, n}, "x");
        const ValueID w = builder.input(Shape{n, n}, "w");
        const ValueID y = builder.matmul(x, w, false, true);

        SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<Runner> run, Runner::make(*engine, graph));

        std::vector<double> x_values(static_cast<std::size_t>(n * n));
        std::vector<double> w_values(static_cast<std::size_t>(n * n));
        SaTest::HostNoise(0xB10Cu + static_cast<std::uint64_t>(n)).fill(x_values, 1.0);
        SaTest::HostNoise(0xB10Du + static_cast<std::uint64_t>(n)).fill(w_values, 1.0);

        SaTest::write_tensor(run->pool.tensor(x), x_values);
        SaTest::write_tensor(run->pool.tensor(w), w_values);

        ASSERT_TRUE(SaTest::status_is_ok(run->compile()));
        ASSERT_TRUE(SaTest::status_is_ok(run->run()));

        const std::vector<double> got = SaTest::read_tensor(run->pool.tensor(y));
        ASSERT_EQ(got.size(), static_cast<std::size_t>(n * n));

        // The reference uses the device's floats; see tests/integration/README.md.
        const std::vector<double> x_seen = SaTest::read_tensor(run->pool.tensor(x));
        const std::vector<double> w_seen = SaTest::read_tensor(run->pool.tensor(w));

        for (int64_t i = 0; i < n; ++i)
        {
            for (int64_t j = 0; j < n; ++j)
            {
                double expected = 0.0;
                for (int64_t k = 0; k < n; ++k)
                    expected += x_seen[static_cast<std::size_t>(i * n + k)]
                              * w_seen[static_cast<std::size_t>(j * n + k)];

                const double actual = got[static_cast<std::size_t>(i * n + j)];
                ASSERT_NEAR(actual, expected, 2e-3)
                    << "n=" << n << " at (" << i << ", " << j << ")";
            }
        }
    }
}

TEST_F(Integration_Resilience, TrainingIsFiniteAtTheRegisterBlockEdgeAndOnePastIt)
{
    for (int64_t n : {int64_t{64}, int64_t{65}})
    {
        SA_ASSERT_OK_AND_ASSIGN(Fixture fixture, make_fixture(*engine, n, n, n, 4));
        Optim::SGDOptions sgd;
        sgd.learning_rate = 0.1;
        SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<Optim::SGD> optimizer,
                                Optim::SGD::make(*engine, fixture.slots, sgd));

        Train::TrainerOptions options;
        options.track_metrics = false;
        const auto objective = SaTest::classifier_objective(fixture.built);
        SA_ASSERT_OK_AND_ASSIGN(
            const std::unique_ptr<Train::Trainer> trainer,
            Train::Trainer::make(*engine, fixture.built, *fixture.pool,
                                 objective, *optimizer, options));

        SA_ASSERT_OK_AND_ASSIGN(
            const std::unique_ptr<Data::SyntheticClassification> dataset,
            Data::SyntheticClassification::make(*engine, data_for(n, 2, n, 4)));

        for (std::size_t step = 0; step < 4; ++step)
        {
            Data::Batch dataset_batch;
            ASSERT_TRUE(SaTest::status_is_ok(dataset->read(step % 2, dataset_batch)));
            SA_ASSERT_OK_AND_ASSIGN(const Train::StepMetrics metrics,
                                    trainer->run_step(dataset_batch));
            ASSERT_TRUE(std::isfinite(metrics.loss)) << "n=" << n << " step " << step;
        }

        for (const Optim::ParameterSlot& slot : fixture.slots)
        {
            EXPECT_TRUE(all_finite(SaTest::read_tensor(slot.parameter), slot.name))
                << "n=" << n;
            EXPECT_TRUE(all_finite(SaTest::read_tensor(slot.gradient), slot.name + ".grad"))
                << "n=" << n;
        }
    }
}

namespace
{
    /**
     * @brief Log-softmax of @p logits, read back, for the stability checks.
     * @param engine The engine to run on.
     * @param rows Rows of the logit matrix.
     * @param columns Columns of the logit matrix.
     * @param logits The values to feed, row-major.
     * @return The [rows, columns] log-probabilities, or the failure of the compilation or the run.
     */
    [[nodiscard]] Result<std::vector<double>> log_softmax_of(SushiBLAS::Engine& engine,
                                                             int64_t rows, int64_t columns,
                                                             const std::vector<double>& logits)
    {
        Graph::Graph graph;
        const GraphBuilder builder(graph);

        const ValueID x = builder.input(Shape{rows, columns}, "logits");
        const ValueID y = builder.log_softmax(x);

        SA_TRY_ASSIGN(const std::unique_ptr<Runner> run, Runner::make(engine, graph));
        SaTest::write_tensor(run->pool.tensor(x), logits);
        SA_TRY(run->compile());
        SA_TRY(run->run());

        return SaTest::read_tensor(run->pool.tensor(y));
    }
}

TEST_F(Integration_Resilience, LogSoftmaxOfUniformLogitsIsMinusLogCAtAnyMagnitude)
{
    // A uniform row gives -log(C) at any offset; see tests/integration/README.md.
    const int64_t rows = 3;
    const int64_t columns = 4;
    const double expected = -std::log(static_cast<double>(columns));

    for (double magnitude : {0.0, 1.0, -1.0, 1e30, -1e30, 3.0e38})
    {
        const std::vector<double> logits(static_cast<std::size_t>(rows * columns), magnitude);
        SA_ASSERT_OK_AND_ASSIGN(const std::vector<double> got,
                                log_softmax_of(*engine, rows, columns, logits));

        ASSERT_TRUE(all_finite(got, "log_softmax")) << "at magnitude " << magnitude;
        for (std::size_t i = 0; i < got.size(); ++i)
            EXPECT_NEAR(got[i], expected, 1e-5) << "magnitude " << magnitude << " at " << i;
    }
}

TEST_F(Integration_Resilience, LogSoftmaxSurvivesBothExtremesInsideOneRow)
{
    // The winner's log-probability is exactly zero; see tests/integration/README.md.
    const int64_t rows = 2;
    const int64_t columns = 3;

    std::vector<double> logits = {
        1e30, -1e30, -1e30,
        -1e30, 1e30, -1e30,
    };

    SA_ASSERT_OK_AND_ASSIGN(const std::vector<double> got,
                            log_softmax_of(*engine, rows, columns, logits));

    ASSERT_TRUE(all_finite(got, "log_softmax"));
    EXPECT_NEAR(got[0], 0.0, 1e-6);
    EXPECT_NEAR(got[4], 0.0, 1e-6);
    EXPECT_LT(got[1], -1e29);
    EXPECT_LT(got[3], -1e29);
}

TEST_F(Integration_Resilience, LogSoftmaxIsExactAcrossRowsThatSpanTheExponentRange)
{
    // Each row is shifted by its own maximum; see tests/integration/README.md.
    const int64_t columns = 3;

    std::vector<double> logits = {
        60.0, 59.0, 58.0,   // the batch maximum
        0.0, -1.0, -2.0,    // 60 below it, still exact
    };

    SA_ASSERT_OK_AND_ASSIGN(const std::vector<double> got,
                            log_softmax_of(*engine, 2, columns, logits));
    ASSERT_TRUE(all_finite(got, "log_softmax"));

    // Equal spread gives equal log-probabilities; see tests/integration/README.md.
    for (std::size_t column = 0; column < 3; ++column)
        EXPECT_NEAR(got[column], got[column + 3], 1e-4) << "column " << column;

    const double denominator = 1.0 + std::exp(-1.0) + std::exp(-2.0);
    EXPECT_NEAR(got[0], -std::log(denominator), 1e-4);
}

TEST_F(Integration_Resilience, CrossEntropyIsFiniteOnPerfectlyConfidentPredictions)
{
    const int64_t rows = 2;
    const int64_t columns = 3;

    Graph::Graph graph;
    const GraphBuilder builder(graph);

    const ValueID logits = builder.input(Shape{rows, columns}, "logits");
    const ValueID targets = builder.input(Shape{rows, columns}, "targets");
    const ValueID loss = NN::CrossEntropyLoss{}.forward(builder, logits, targets);

    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<Runner> run, Runner::make(*engine, graph));

    // exp(80) overflows without the shift; see tests/integration/README.md.
    const std::vector<double> confident = {
        80.0, -80.0, -80.0,
        -80.0, 80.0, -80.0,
    };
    SaTest::write_tensor(run->pool.tensor(logits), confident);
    SaTest::write_tensor(run->pool.tensor(targets),
                         std::vector<double>{1.0, 0.0, 0.0, 0.0, 1.0, 0.0});

    ASSERT_TRUE(SaTest::status_is_ok(run->compile()));
    ASSERT_TRUE(SaTest::status_is_ok(run->run()));

    const double correct = SaTest::read_tensor(run->pool.tensor(loss))[0];
    EXPECT_TRUE(std::isfinite(correct));
    EXPECT_NEAR(correct, 0.0, 1e-5) << "a perfectly confident correct prediction costs nothing";

    // A wrong confident loss is large but finite; see tests/integration/README.md.
    SaTest::write_tensor(run->pool.tensor(targets),
                         std::vector<double>{0.0, 1.0, 0.0, 1.0, 0.0, 0.0});
    ASSERT_TRUE(SaTest::status_is_ok(run->run()));

    const double wrong = SaTest::read_tensor(run->pool.tensor(loss))[0];
    EXPECT_TRUE(std::isfinite(wrong)) << "loss on a confidently wrong prediction";
    EXPECT_NEAR(wrong, 160.0, 1e-2);
}

TEST_F(Integration_Resilience, AdamWsFirstStepWithAZeroGradientStaysFinite)
{
    // eps is added after the bias correction; see tests/integration/README.md.
    SA_ASSERT_OK_AND_ASSIGN(SushiBLAS::Tensor parameter, allocate_tensor(*engine, Shape{6}));
    SA_ASSERT_OK_AND_ASSIGN(SushiBLAS::Tensor gradient, allocate_tensor(*engine, Shape{6}));

    const std::vector<double> initial = {0.5, -1.5, 0.0, 2.0, -0.25, 100.0};
    SaTest::write_tensor(parameter, initial);
    SaTest::write_tensor(gradient, std::vector<double>(6, 0.0));

    std::vector<Optim::ParameterSlot> slots;
    slots.push_back(Optim::ParameterSlot{"p", parameter, gradient});

    Optim::AdamWOptions options;
    options.learning_rate = 0.1;
    options.weight_decay = 0.0;
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<Optim::AdamW> optimizer,
                            Optim::AdamW::make(*engine, slots, options));

    SA_EXPECT_OK(optimizer->step());

    const std::vector<double> after = SaTest::read_tensor(parameter);
    ASSERT_TRUE(all_finite(after, "parameter"));

    // With no decay and no gradient the parameter must be untouched.
    for (std::size_t i = 0; i < initial.size(); ++i)
        EXPECT_NEAR(after[i], initial[i], 1e-6) << "at " << i;

    // The bias corrections differ on step two; see tests/integration/README.md.
    SA_EXPECT_OK(optimizer->step());
    EXPECT_TRUE(all_finite(SaTest::read_tensor(parameter), "parameter after two steps"));
}

TEST_F(Integration_Resilience, BuildingAndDestroyingTrainersReachesASteadyState)
{
    // The runtime counts its live USM allocations; see tests/integration/README.md.
    const auto live_allocations = [this]()
    {
        return engine->runtime().advanced().live_allocation_count();
    };

    std::size_t baseline = 0;
    std::size_t baseline_bytes = 0;

    for (int iteration = 0; iteration < 5; ++iteration)
    {
        {
            SA_ASSERT_OK_AND_ASSIGN(Fixture fixture, make_fixture(*engine, 8, 6, 12, 4));
            SA_ASSERT_OK_AND_ASSIGN(
                const std::unique_ptr<Optim::AdamW> optimizer,
                Optim::AdamW::make(*engine, fixture.slots, Optim::AdamWOptions{}));

            Train::TrainerOptions options;
            options.track_metrics = false;
            const auto objective = SaTest::classifier_objective(fixture.built);
            SA_ASSERT_OK_AND_ASSIGN(
                const std::unique_ptr<Train::Trainer> trainer,
                Train::Trainer::make(*engine, fixture.built, *fixture.pool,
                                     objective, *optimizer, options));

            SA_ASSERT_OK_AND_ASSIGN(
                const std::unique_ptr<Data::SyntheticClassification> dataset,
                Data::SyntheticClassification::make(*engine, data_for(8, 2, 6, 4)));
            SA_EXPECT_OK(trainer->fit(*dataset));

            if (iteration == 0) baseline_bytes = fixture.pool->allocated_bytes();
            EXPECT_EQ(fixture.pool->allocated_bytes(), baseline_bytes)
                << "the same model allocated a different amount on iteration " << iteration;
        }

        engine->runtime().wait().value();

        if (iteration == 0)
            baseline = live_allocations();
        else
            EXPECT_EQ(live_allocations(), baseline)
                << "iteration " << iteration << " changed the number of live buffers; a "
                << "trainer's buffers must all die with it";
    }
}

TEST_F(Integration_ResilienceDeathTest, BatchNormUpdateStatsIsNotAnIROperationAndSaysSo)
{
    // BATCHNORM_UPDATE_STATS has no lowering; see tests/integration/README.md.
    Graph::Graph graph;
    const GraphBuilder builder(graph);

    const ValueID running = builder.parameter(Shape{4}, "running_mean");
    const ValueID batch = builder.input(Shape{4}, "batch_mean");
    const ValueID out = builder.like(running, "updated_running_mean");
    graph.add_node(Ops::K_BATCHNORM_UPDATE_STATS.id, {running, batch}, {out});
    graph.finalise();

    TensorPool pool(*engine);
    SA_EXPECT_OK(pool.adopt(Graph::MemoryPlan::build(graph)));
    SA_EXPECT_OK(pool.materialise_leaves(graph));

    SA_EXPECT_FATAL((void)CompiledStep::compile(*engine, graph, pool),
                    "No lowering for operation", "sushiai.batchnorm_update_stats");
}
