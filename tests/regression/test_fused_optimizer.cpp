/**************************************************************************/
/* test_fused_optimizer.cpp                                               */
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
 * @file test_fused_optimizer.cpp
 * @brief Tests that the fused optimizer update matches the separate operations byte for byte.
 * @author Mustafa Garip
 * @see tests/regression/README.md
 */

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <gtest/gtest.h>
#include <common/test_status.hpp>
#include <common/test_utils.hpp>

using namespace SushiAI;

// SaTest keeps SushiAI::Test reachable inside TEST_F; see tests/regression/README.md.
namespace SaTest = SushiAI::Test;
using Graph::TensorPool;
using Optim::UpdateKernel;

namespace
{
    class Regression_FusedOptimizer : public ::testing::Test {};

    /** @brief Names the suite of the cases that expect the process to stop. */
    using Regression_FusedOptimizerDeathTest = Regression_FusedOptimizer;

    // 133 is not a multiple of the vector width; see tests/regression/README.md.
    constexpr int64_t K_ELEMENTS = 133;

    constexpr int K_STEPS = 6;
    constexpr std::uint64_t K_PARAM_SEED = 0x5175'5348'4941'0001ull;
    constexpr std::uint64_t K_GRAD_SEED = 0x5175'5348'4941'0002ull;

    /**
     * @brief Returns the gradient for step @p step, identical across runs.
     * @see tests/regression/README.md
     */
    std::vector<double> gradient_for(int step)
    {
        SaTest::HostNoise noise(K_GRAD_SEED + static_cast<std::uint64_t>(step));

        std::vector<double> values(static_cast<std::size_t>(K_ELEMENTS));
        noise.fill(values, 0.75);

        values[0] = 0.0;
        values[1] = -0.0;
        values[2] = 1e-30;
        values[3] = 1e6;
        return values;
    }

    /** @brief The starting parameter values, identical across runs. */
    std::vector<double> initial_parameter()
    {
        SaTest::HostNoise noise(K_PARAM_SEED);
        std::vector<double> values(static_cast<std::size_t>(K_ELEMENTS));
        noise.fill(values, 2.0);
        return values;
    }

    /**
     * @brief Steps one optimizer over a fixed gradient sequence, standalone.
     * @param bytes Receives the parameter's bytes after the last step.
     * @see tests/regression/README.md
     */
    template<typename Make>
    void drive(Make make, UpdateKernel kernel, std::vector<unsigned char>& bytes)
    {
        SushiBLAS::Engine engine(SaTest::TestEnvironment::context());

        SA_ASSERT_OK_AND_ASSIGN(SushiBLAS::Tensor parameter,
                                allocate_tensor(engine, Shape{K_ELEMENTS}));
        SA_ASSERT_OK_AND_ASSIGN(SushiBLAS::Tensor gradient,
                                allocate_tensor(engine, Shape{K_ELEMENTS}));
        SaTest::write_tensor(parameter, initial_parameter());

        std::vector<Optim::ParameterSlot> slots{
            Optim::ParameterSlot{"p", parameter, gradient},
        };

        SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<Optim::Optimizer> optimizer,
                                make(engine, slots, kernel));

        for (int step = 0; step < K_STEPS; ++step)
        {
            SaTest::write_tensor(gradient, gradient_for(step));
            SA_EXPECT_OK(optimizer->step());
        }

        bytes = SaTest::read_tensor_bytes(parameter);
    }

    /** @brief Builds an AdamW with @p options and the requested kernel. */
    auto adamw_factory(Optim::AdamWOptions options)
    {
        return [options](SushiBLAS::Engine& engine,
                         const std::vector<Optim::ParameterSlot>& slots,
                         UpdateKernel kernel) -> Result<std::unique_ptr<Optim::Optimizer>>
        {
            Optim::AdamWOptions with_kernel = options;
            with_kernel.kernel = kernel;
            SA_TRY_ASSIGN(std::unique_ptr<Optim::AdamW> optimizer,
                          Optim::AdamW::make(engine, slots, with_kernel));
            return std::unique_ptr<Optim::Optimizer>(std::move(optimizer));
        };
    }

    /** @brief Builds an SGD with @p options and the requested kernel. */
    auto sgd_factory(Optim::SGDOptions options)
    {
        return [options](SushiBLAS::Engine& engine,
                         const std::vector<Optim::ParameterSlot>& slots,
                         UpdateKernel kernel) -> Result<std::unique_ptr<Optim::Optimizer>>
        {
            Optim::SGDOptions with_kernel = options;
            with_kernel.kernel = kernel;
            SA_TRY_ASSIGN(std::unique_ptr<Optim::SGD> optimizer,
                          Optim::SGD::make(engine, slots, with_kernel));
            return std::unique_ptr<Optim::Optimizer>(std::move(optimizer));
        };
    }

    /** @brief Asserts the two spellings produced identical bytes. */
    template<typename Make>
    void expect_kernels_agree(Make make, const std::string& what)
    {
        std::vector<unsigned char> fused;
        std::vector<unsigned char> separate;
        ASSERT_NO_FATAL_FAILURE(drive(make, UpdateKernel::Fused, fused));
        ASSERT_NO_FATAL_FAILURE(drive(make, UpdateKernel::Separate, separate));

        ASSERT_EQ(fused.size(), separate.size());
        EXPECT_EQ(SaTest::first_difference(fused, separate), fused.size())
            << what << ": the fused kernel and the separate operations disagree from byte "
            << SaTest::first_difference(fused, separate)
            << ". Find the expression that moved; do not widen this comparison.";
    }

    using Mlp = NN::Sequential<NN::Linear, NN::ReLU, NN::Linear>;

    constexpr int64_t K_BATCH = 8;
    constexpr int64_t K_FEATURES = 6;
    constexpr int64_t K_HIDDEN = 12;
    constexpr int64_t K_CLASSES = 4;
    constexpr int64_t K_BATCHES = 4;

    /** @brief The synthetic problem the training comparisons train on. */
    Data::SyntheticOptions data_options()
    {
        Data::SyntheticOptions options;
        options.batch_size = K_BATCH;
        options.batches = K_BATCHES;
        options.features = K_FEATURES;
        options.classes = K_CLASSES;
        options.seed = 0x0FED'0FED'0001ull;
        return options;
    }

    /**
     * @struct TrainedRun
     * @brief One whole training run, reduced to what a comparison needs.
     */
    struct TrainedRun
    {
        std::vector<std::vector<unsigned char>> parameters;
        std::vector<std::string> names;
        std::vector<double> losses;
        std::size_t compiles = 0;
        std::size_t operations_per_step = 0;
        bool optimizer_in_plan = false;
    };

    /**
     * @brief Trains the same MLP with AdamW, in the requested placement.
     * @param run Receives the run; a default-constructed TrainedRun.
     * @see tests/regression/README.md
     */
    void train(Train::OptimizerPlacement placement, UpdateKernel kernel, TrainedRun& run)
    {
        SushiBLAS::Engine engine(SaTest::TestEnvironment::context());

        Mlp model(NN::Linear{K_FEATURES, K_HIDDEN}, NN::ReLU{}, NN::Linear{K_HIDDEN, K_CLASSES});
        SA_ASSERT_OK_AND_ASSIGN(Train::TrainingGraph built,
                                Train::build_classifier(model, K_BATCH, K_FEATURES, K_CLASSES));

        SA_ASSERT_OK_AND_ASSIGN(NN::ModelState state, NN::ModelState::create(engine, model));
        TensorPool pool(engine);
        SA_ASSERT_OK_AND_ASSIGN(std::vector<Optim::ParameterSlot> slots,
                                Train::bind_parameters(built, state, pool));
        SA_EXPECT_OK(NN::initialize(engine, state));

        Optim::AdamWOptions adam;
        adam.learning_rate = 0.02;
        adam.kernel = kernel;
        SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<Optim::AdamW> optimizer,
                                Optim::AdamW::make(engine, slots, adam));

        Train::TrainerOptions options;
        options.track_accuracy = false;
        options.optimizer_placement = placement;
        SA_ASSERT_OK_AND_ASSIGN(
            const std::unique_ptr<Train::Trainer> trainer,
            Train::Trainer::make(engine, built.graph, pool, built.spec, *optimizer, options));

        SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<Data::SyntheticClassification> dataset,
                                Data::SyntheticClassification::make(engine, data_options()));

        run.optimizer_in_plan = trainer->optimizer_is_in_plan();
        SA_ASSERT_OK_AND_ASSIGN(const Train::RunReport report, trainer->fit(*dataset));
        run.operations_per_step = report.operations_per_step;

        for (int64_t epoch = 1; epoch < 3; ++epoch)
        {
            for (std::size_t batch = 0; batch < dataset->batch_count(); ++batch)
            {
                SA_ASSERT_OK_AND_ASSIGN(const Train::StepMetrics metrics,
                                        trainer->run_step(dataset->batch(batch)));
                run.losses.push_back(metrics.loss);
            }
        }

        for (const Optim::ParameterSlot& slot : slots)
        {
            run.names.push_back(slot.name);
            run.parameters.push_back(SaTest::read_tensor_bytes(slot.parameter));
        }

        run.compiles = trainer->compile_count();
    }

    /** @brief Asserts two runs agree byte for byte, naming where they do not. */
    void expect_runs_agree(const TrainedRun& a, const TrainedRun& b, const std::string& what)
    {
        ASSERT_EQ(a.names, b.names);
        ASSERT_EQ(a.losses.size(), b.losses.size());

        for (std::size_t step = 0; step < a.losses.size(); ++step)
            EXPECT_EQ(a.losses[step], b.losses[step])
                << what << ": the loss diverged at step " << step;

        for (std::size_t i = 0; i < a.parameters.size(); ++i)
            EXPECT_EQ(SaTest::first_difference(a.parameters[i], b.parameters[i]),
                      a.parameters[i].size())
                << what << ": parameter '" << a.names[i] << "' differs";
    }
}

TEST_F(Regression_FusedOptimizer, AdamWsFusedKernelMatchesTheSeparateOperationsExactly)
{
    Optim::AdamWOptions options;
    options.learning_rate = 0.02;
    options.weight_decay = 0.01;
    expect_kernels_agree(adamw_factory(options), "AdamW with decay");
}

/**
 * @brief Tests that AdamW without weight decay matches the separate operations byte for byte.
 * @see tests/regression/README.md
 */
TEST_F(Regression_FusedOptimizer, AdamWWithoutDecayTakesTheOtherBranchAndStillMatches)
{
    Optim::AdamWOptions options;
    options.learning_rate = 0.02;
    options.weight_decay = 0.0;
    expect_kernels_agree(adamw_factory(options), "AdamW without decay");
}

/**
 * @brief Tests that every SGD branch matches the separate operations byte for byte.
 * @see tests/regression/README.md
 */
TEST_F(Regression_FusedOptimizer, SGDsFusedKernelMatchesTheSeparateOperationsExactly)
{
    struct Case
    {
        const char* name;
        Optim::SGDOptions options;
    };

    std::vector<Case> cases;

    Optim::SGDOptions plain;
    plain.learning_rate = 0.1;
    cases.push_back({"plain", plain});

    Optim::SGDOptions decayed = plain;
    decayed.weight_decay = 0.05;
    cases.push_back({"coupled decay, no momentum", decayed});

    Optim::SGDOptions momentum = plain;
    momentum.momentum = 0.9;
    cases.push_back({"momentum", momentum});

    Optim::SGDOptions damped = momentum;
    damped.dampening = 0.2;
    cases.push_back({"momentum with dampening", damped});

    Optim::SGDOptions nesterov = momentum;
    nesterov.nesterov = true;
    cases.push_back({"nesterov", nesterov});

    Optim::SGDOptions everything = momentum;
    everything.weight_decay = 0.05;
    cases.push_back({"momentum and decay", everything});

    for (const Case& one : cases)
        expect_kernels_agree(sgd_factory(one.options), std::string("SGD, ") + one.name);
}

/**
 * @brief Tests that AdamW inside the compiled plan trains identically to AdamW outside it.
 * @see tests/regression/README.md
 */
TEST_F(Regression_FusedOptimizer, TheUpdateInThePlanTrainsIdenticallyToTheUpdateOutsideIt)
{
    TrainedRun folded;
    TrainedRun apart;
    ASSERT_NO_FATAL_FAILURE(
        train(Train::OptimizerPlacement::InPlan, UpdateKernel::Fused, folded));
    ASSERT_NO_FATAL_FAILURE(
        train(Train::OptimizerPlacement::PerStep, UpdateKernel::Fused, apart));

    ASSERT_TRUE(folded.optimizer_in_plan);
    ASSERT_FALSE(apart.optimizer_in_plan);
    ASSERT_FALSE(folded.losses.empty());

    expect_runs_agree(folded, apart, "the optimizer's placement");
}

/**
 * @brief Tests that the fused and separate spellings train a whole model identically.
 * @see tests/regression/README.md
 */
TEST_F(Regression_FusedOptimizer, TheSeparateSpellingOutsideThePlanTrainsIdenticallyToo)
{
    TrainedRun fused;
    TrainedRun separate;
    ASSERT_NO_FATAL_FAILURE(
        train(Train::OptimizerPlacement::PerStep, UpdateKernel::Fused, fused));
    ASSERT_NO_FATAL_FAILURE(
        train(Train::OptimizerPlacement::PerStep, UpdateKernel::Separate, separate));

    expect_runs_agree(fused, separate, "the update's spelling");
}

TEST_F(Regression_FusedOptimizer, TheStepCompilesOnceWithTheOptimizerInsideIt)
{
    TrainedRun folded;
    TrainedRun apart;
    ASSERT_NO_FATAL_FAILURE(
        train(Train::OptimizerPlacement::InPlan, UpdateKernel::Fused, folded));
    ASSERT_NO_FATAL_FAILURE(
        train(Train::OptimizerPlacement::PerStep, UpdateKernel::Fused, apart));

    EXPECT_EQ(folded.compiles, 1u);
    EXPECT_EQ(apart.compiles, 1u);

    // Each parameter adds one fused update to the plan; see tests/regression/README.md.
    EXPECT_EQ(folded.operations_per_step, apart.operations_per_step + folded.names.size())
        << "the folded plan does not hold one extra task per parameter";
}

/**
 * @brief Tests that the fused update records one task per parameter, against twelve for separate
 *        AdamW and four for separate SGD.
 * @see tests/regression/README.md
 */
TEST_F(Regression_FusedOptimizer, TheFusedUpdateIsOneTaskPerParameterWhereTheSeparateOneIsTwelve)
{
    constexpr std::size_t K_PARAMETERS = 3;

    const auto count = [](UpdateKernel kernel, auto make, std::size_t& tasks)
    {
        SushiBLAS::Engine engine(SaTest::TestEnvironment::context());

        std::vector<Optim::ParameterSlot> slots;
        std::vector<SushiBLAS::Tensor> owned;
        owned.reserve(K_PARAMETERS * 2);

        for (std::size_t i = 0; i < K_PARAMETERS; ++i)
        {
            SA_ASSERT_OK_AND_ASSIGN(SushiBLAS::Tensor parameter,
                                    allocate_tensor(engine, Shape{K_ELEMENTS}));
            SA_ASSERT_OK_AND_ASSIGN(SushiBLAS::Tensor gradient,
                                    allocate_tensor(engine, Shape{K_ELEMENTS}));
            owned.push_back(parameter);
            owned.push_back(gradient);
            SaTest::write_tensor(owned[owned.size() - 2], initial_parameter());
            SaTest::write_tensor(owned[owned.size() - 1], gradient_for(0));
            slots.push_back(Optim::ParameterSlot{
                "p" + std::to_string(i), owned[owned.size() - 2], owned[owned.size() - 1],
            });
        }

        SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<Optim::Optimizer> optimizer,
                                make(engine, slots, kernel));

        // The recorded tasks must be drained; see tests/regression/README.md.
        optimizer->before_replay();
        SA_ASSERT_OK_AND_ASSIGN(const std::size_t recorded,
                                optimizer->record_into_step(engine.get_recorder()));
        engine.execute();
        engine.get_context().wait_all();

        tasks = recorded;
    };

    Optim::AdamWOptions adam;
    adam.learning_rate = 0.02;
    adam.weight_decay = 0.01;

    std::size_t tasks = 0;
    ASSERT_NO_FATAL_FAILURE(count(UpdateKernel::Fused, adamw_factory(adam), tasks));
    EXPECT_EQ(tasks, K_PARAMETERS);
    ASSERT_NO_FATAL_FAILURE(count(UpdateKernel::Separate, adamw_factory(adam), tasks));
    EXPECT_EQ(tasks, K_PARAMETERS * 12);

    Optim::SGDOptions sgd;
    sgd.learning_rate = 0.1;
    sgd.momentum = 0.9;

    // Four is the floor for SGD, on its first step; see tests/regression/README.md.
    ASSERT_NO_FATAL_FAILURE(count(UpdateKernel::Fused, sgd_factory(sgd), tasks));
    EXPECT_EQ(tasks, K_PARAMETERS);
    ASSERT_NO_FATAL_FAILURE(count(UpdateKernel::Separate, sgd_factory(sgd), tasks));
    EXPECT_EQ(tasks, K_PARAMETERS * 4);
}

/**
 * @brief Tests that the trainer refuses to compile a separate-kernel AdamW into the step.
 * @see tests/regression/README.md
 */
TEST_F(Regression_FusedOptimizer, AskingToCompileAnUnfusedUpdateIsRefusedRatherThanIgnored)
{
    SushiBLAS::Engine engine(SaTest::TestEnvironment::context());

    Mlp model(NN::Linear{K_FEATURES, K_HIDDEN}, NN::ReLU{}, NN::Linear{K_HIDDEN, K_CLASSES});
    SA_ASSERT_OK_AND_ASSIGN(Train::TrainingGraph built,
                            Train::build_classifier(model, K_BATCH, K_FEATURES, K_CLASSES));

    SA_ASSERT_OK_AND_ASSIGN(NN::ModelState state, NN::ModelState::create(engine, model));
    TensorPool pool(engine);
    SA_ASSERT_OK_AND_ASSIGN(std::vector<Optim::ParameterSlot> slots,
                            Train::bind_parameters(built, state, pool));
    SA_EXPECT_OK(NN::initialize(engine, state));

    Optim::AdamWOptions adam;
    adam.kernel = UpdateKernel::Separate;
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<Optim::AdamW> optimizer,
                            Optim::AdamW::make(engine, slots, adam));

    EXPECT_FALSE(optimizer->is_replay_invariant());

    Train::TrainerOptions options;
    options.optimizer_placement = Train::OptimizerPlacement::InPlan;

    SA_EXPECT_FAILURE(Train::Trainer::make(engine, built.graph, pool, built.spec,
                                           *optimizer, options),
                      StatusCode::InvalidDescription,
                      "cannot be compiled into the step", "PerStep");
}

/**
 * @brief Tests that step() stops the process on an optimizer already recorded into a compiled step.
 * @see tests/regression/README.md
 */
TEST_F(Regression_FusedOptimizerDeathTest, SteppingAnOptimizerThatIsAlreadyInThePlanIsRefused)
{
    SushiBLAS::Engine engine(SaTest::TestEnvironment::context());

    Mlp model(NN::Linear{K_FEATURES, K_HIDDEN}, NN::ReLU{}, NN::Linear{K_HIDDEN, K_CLASSES});
    SA_ASSERT_OK_AND_ASSIGN(Train::TrainingGraph built,
                            Train::build_classifier(model, K_BATCH, K_FEATURES, K_CLASSES));

    SA_ASSERT_OK_AND_ASSIGN(NN::ModelState state, NN::ModelState::create(engine, model));
    TensorPool pool(engine);
    SA_ASSERT_OK_AND_ASSIGN(std::vector<Optim::ParameterSlot> slots,
                            Train::bind_parameters(built, state, pool));
    SA_EXPECT_OK(NN::initialize(engine, state));

    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<Optim::AdamW> optimizer,
                            Optim::AdamW::make(engine, slots, Optim::AdamWOptions{}));
    EXPECT_FALSE(optimizer->is_in_compiled_step());

    SA_ASSERT_OK_AND_ASSIGN(
        const std::unique_ptr<Train::Trainer> trainer,
        Train::Trainer::make(engine, built.graph, pool, built.spec, *optimizer));
    EXPECT_TRUE(optimizer->is_in_compiled_step());

    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<Data::SyntheticClassification> dataset,
                            Data::SyntheticClassification::make(engine, data_options()));
    SA_EXPECT_OK(trainer->run_step(dataset->batch(0)));

    SA_EXPECT_FATAL((void)optimizer->step(), "already recorded into a compiled step",
                    "twice per batch");
}

/**
 * @brief Tests that step() stops the process when momentum is switched on after construction.
 * @see tests/regression/README.md
 */
TEST_F(Regression_FusedOptimizerDeathTest,
       TurningMomentumOnAfterConstructionIsRefusedRatherThanIgnored)
{
    SushiBLAS::Engine engine(SaTest::TestEnvironment::context());

    SA_ASSERT_OK_AND_ASSIGN(SushiBLAS::Tensor parameter,
                            allocate_tensor(engine, Shape{K_ELEMENTS}));
    SA_ASSERT_OK_AND_ASSIGN(SushiBLAS::Tensor gradient,
                            allocate_tensor(engine, Shape{K_ELEMENTS}));
    SaTest::write_tensor(parameter, initial_parameter());
    SaTest::write_tensor(gradient, gradient_for(0));

    std::vector<Optim::ParameterSlot> slots{
        Optim::ParameterSlot{"p", parameter, gradient},
    };

    Optim::SGDOptions options;
    options.learning_rate = 0.1;
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<Optim::SGD> optimizer,
                            Optim::SGD::make(engine, slots, options));

    SA_EXPECT_OK(optimizer->step());
    optimizer->options().momentum = 0.9;

    SA_EXPECT_FATAL((void)optimizer->step(), "no momentum buffer", "0.9");
}
