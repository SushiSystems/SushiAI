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
 * @file test_fused_optimizer.cpp
 * @brief Tests that the fused optimizer update matches the separate operations byte for byte.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <gtest/gtest.h>
#include <SushiBLAS/engine/engine.hpp>
#include <SushiBLAS/tensor/tensor.hpp>
#include <SushiAI/classifier_objective/classifier_objective.hpp>
#include <SushiAI/data/dataset.hpp>
#include <SushiAI/engine_fixture/test_environment.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/graph/tensor_pool.hpp>
#include <SushiAI/nn/init.hpp>
#include <SushiAI/nn/layers.hpp>
#include <SushiAI/nn/model_state.hpp>
#include <SushiAI/optim/adamw.hpp>
#include <SushiAI/optim/optimizer.hpp>
#include <SushiAI/optim/sgd.hpp>
#include <SushiAI/seeded_noise/host_noise.hpp>
#include <SushiAI/shape/shape.hpp>
#include <SushiAI/tensor_values/tensor_values.hpp>
#include <SushiAI/train/trainer.hpp>
#include <SushiAI/train/training_graph.hpp>
#include <SushiAI/train/training_step.hpp>

using namespace SushiAI;

// SaTest keeps SushiAI::Test reachable inside TEST_F; see tests/layers/train/README.md.
namespace SaTest = SushiAI::Test;
using Graph::TensorPool;
using Optim::UpdateKernel;

namespace
{
    class Medium_FusedOptimizer : public ::testing::Test {};

    /** @brief Names the suite of the cases that expect the process to stop. */
    using Medium_FusedOptimizerDeathTest = Medium_FusedOptimizer;

    // 133 is not a multiple of the vector width; see tests/layers/train/README.md.
    constexpr int64_t K_ELEMENTS = 133;

    constexpr int K_STEPS = 6;
    constexpr std::uint64_t K_PARAM_SEED = 0x5175'5348'4941'0001ull;
    constexpr std::uint64_t K_GRAD_SEED = 0x5175'5348'4941'0002ull;

    /**
     * @brief Returns the gradient for step @p step, identical across runs.
     * @see tests/layers/train/README.md
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
     * @see tests/layers/train/README.md
     */
    template<typename Make>
    void drive(Make make, UpdateKernel kernel, std::vector<unsigned char>& bytes)
    {
        SushiBLAS::Engine engine(SaTest::TestEnvironment::runtime());
        SushiAI::RuntimeExecutionBackend backend(SaTest::TestEnvironment::runtime(), engine);

        SA_ASSERT_OK_AND_ASSIGN(SushiBLAS::Tensor parameter,
                                allocate_tensor(engine, Shape{K_ELEMENTS}));
        SA_ASSERT_OK_AND_ASSIGN(SushiBLAS::Tensor gradient,
                                allocate_tensor(engine, Shape{K_ELEMENTS}));
        SaTest::write_tensor(parameter, initial_parameter());

        std::vector<Optim::ParameterSlot> slots{
            Optim::ParameterSlot{"p", parameter, gradient},
        };

        SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<Optim::Optimizer> optimizer,
                                make(backend, slots, kernel));

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
        return [options](SushiAI::ExecutionBackend& backend,
                         const std::vector<Optim::ParameterSlot>& slots,
                         UpdateKernel kernel) -> Result<std::unique_ptr<Optim::Optimizer>>
        {
            Optim::AdamWOptions with_kernel = options;
            with_kernel.kernel = kernel;
            SA_TRY_ASSIGN(std::unique_ptr<Optim::AdamW> optimizer,
                          Optim::AdamW::make(backend, slots, with_kernel));
            return std::unique_ptr<Optim::Optimizer>(std::move(optimizer));
        };
    }

    /** @brief Builds an SGD with @p options and the requested kernel. */
    auto sgd_factory(Optim::SGDOptions options)
    {
        return [options](SushiAI::ExecutionBackend& backend,
                         const std::vector<Optim::ParameterSlot>& slots,
                         UpdateKernel kernel) -> Result<std::unique_ptr<Optim::Optimizer>>
        {
            Optim::SGDOptions with_kernel = options;
            with_kernel.kernel = kernel;
            SA_TRY_ASSIGN(std::unique_ptr<Optim::SGD> optimizer,
                          Optim::SGD::make(backend, slots, with_kernel));
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

    using MLP = NN::Sequential<NN::Linear, NN::ReLU, NN::Linear>;

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
        bool optimizer_in_plan = false;
    };

    /**
     * @brief Trains the same MLP with AdamW, in the requested placement.
     * @param run Receives the run; a default-constructed TrainedRun.
     * @see tests/layers/train/README.md
     */
    void train(Train::OptimizerPlacement placement, UpdateKernel kernel, TrainedRun& run)
    {
        SushiBLAS::Engine engine(SaTest::TestEnvironment::runtime());
        SushiAI::RuntimeExecutionBackend backend(SaTest::TestEnvironment::runtime(), engine);

        MLP model(NN::Linear{K_FEATURES, K_HIDDEN}, NN::ReLU{}, NN::Linear{K_HIDDEN, K_CLASSES});
        SA_ASSERT_OK_AND_ASSIGN(Train::TrainingGraph built,
                                Train::build_classifier(model, K_BATCH, K_FEATURES, K_CLASSES));

        SA_ASSERT_OK_AND_ASSIGN(NN::ModelState state, NN::ModelState::create(engine, model));
        TensorPool pool(engine);
        SA_ASSERT_OK_AND_ASSIGN(std::vector<Optim::ParameterSlot> slots,
                                Train::bind_parameters(backend, built, state, pool));
        SA_EXPECT_OK(NN::initialize(backend, state));

        Optim::AdamWOptions adam;
        adam.learning_rate = 0.02;
        adam.kernel = kernel;
        SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<Optim::AdamW> optimizer,
                                Optim::AdamW::make(backend, slots, adam));

        Train::TrainerOptions options;
        options.track_metrics = false;
        options.optimizer_placement = placement;
        const auto objective = SaTest::classifier_objective(built);
        SA_ASSERT_OK_AND_ASSIGN(
            const std::unique_ptr<Train::Trainer> trainer,
            Train::Trainer::make(backend, built, pool, objective, *optimizer, options));

        SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<Data::SyntheticClassification> dataset,
                                Data::SyntheticClassification::make(backend, data_options()));

        run.optimizer_in_plan = trainer->optimizer_is_in_plan();
        SA_ASSERT_OK_AND_ASSIGN(const Train::RunReport report, trainer->fit(*dataset));
        EXPECT_GT(report.steps, 0u);

        for (int64_t epoch = 1; epoch < 3; ++epoch)
        {
            for (std::size_t batch = 0; batch < dataset->batch_count(); ++batch)
            {
                Data::Batch dataset_batch;
                ASSERT_TRUE(SaTest::status_is_ok(dataset->read(batch, dataset_batch)));
                SA_ASSERT_OK_AND_ASSIGN(const Train::StepMetrics metrics,
                                        trainer->run_step(dataset_batch));
                run.losses.push_back(metrics.loss);
            }
        }

        for (const Optim::ParameterSlot& slot : slots)
        {
            run.names.push_back(slot.name);
            run.parameters.push_back(SaTest::read_tensor_bytes(slot.parameter));
        }
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
    /** @brief One AdamW row: a name and the weight decay of the run. */
    struct AdamWRow
    {
        const char* name;       /**< @brief Names the row in the test name. */
        double weight_decay;    /**< @brief Holds the decay; zero takes the branch without it. */
    };

    /** @brief One SGD row: a name and the options of the branch it runs. */
    struct SgdRow
    {
        const char* name;               /**< @brief Names the row in the test name. */
        Optim::SGDOptions options;      /**< @brief Holds the branch's options. */
    };

    /** @brief Returns the SGD row @p name with the rate 0.1 and the given branch options. */
    SgdRow sgd_row(const char* name, double weight_decay, double momentum, double dampening,
                   bool nesterov)
    {
        Optim::SGDOptions options;
        options.learning_rate = 0.1;
        options.weight_decay = weight_decay;
        options.momentum = momentum;
        options.dampening = dampening;
        options.nesterov = nesterov;
        return SgdRow{name, options};
    }

    /** @brief Runs the AdamW kernels of one decay row against each other. */
    class Medium_FusedOptimizerAdamW : public ::testing::TestWithParam<AdamWRow> {};

    /** @brief Runs the SGD kernels of one branch row against each other. */
    class Medium_FusedOptimizerSgd : public ::testing::TestWithParam<SgdRow> {};
}

/**
 * @brief Tests that AdamW matches the separate operations byte for byte, with and without decay.
 * @see tests/layers/train/README.md
 */
TEST_P(Medium_FusedOptimizerAdamW, FusedKernelMatchesTheSeparateOperationsExactly)
{
    Optim::AdamWOptions options;
    options.learning_rate = 0.02;
    options.weight_decay = GetParam().weight_decay;
    expect_kernels_agree(adamw_factory(options), GetParam().name);
}

INSTANTIATE_TEST_SUITE_P(Decay, Medium_FusedOptimizerAdamW,
                         ::testing::Values(AdamWRow{"WithDecay", 0.01},
                                           AdamWRow{"WithoutDecay", 0.0}),
                         [](const ::testing::TestParamInfo<AdamWRow>& info)
                         { return std::string(info.param.name); });

/**
 * @brief Tests that each SGD branch matches the separate operations byte for byte.
 * @see tests/layers/train/README.md
 */
TEST_P(Medium_FusedOptimizerSgd, FusedKernelMatchesTheSeparateOperationsExactly)
{
    expect_kernels_agree(sgd_factory(GetParam().options), GetParam().name);
}

INSTANTIATE_TEST_SUITE_P(Branches, Medium_FusedOptimizerSgd,
                         ::testing::Values(sgd_row("Plain", 0.0, 0.0, 0.0, false),
                                           sgd_row("CoupledDecayWithoutMomentum", 0.05, 0.0, 0.0,
                                                   false),
                                           sgd_row("Momentum", 0.0, 0.9, 0.0, false),
                                           sgd_row("MomentumWithDampening", 0.0, 0.9, 0.2, false),
                                           sgd_row("Nesterov", 0.0, 0.9, 0.0, true),
                                           sgd_row("MomentumAndDecay", 0.05, 0.9, 0.0, false)),
                         [](const ::testing::TestParamInfo<SgdRow>& info)
                         { return std::string(info.param.name); });

/**
 * @brief Tests that AdamW inside the compiled plan trains identically to AdamW outside it.
 * @see tests/layers/train/README.md
 */
TEST_F(Medium_FusedOptimizer, TheUpdateInThePlanTrainsIdenticallyToTheUpdateOutsideIt)
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
 * @brief Tests that step() stops the process on an optimizer already recorded into a compiled step.
 * @see tests/layers/train/README.md
 */
TEST_F(Medium_FusedOptimizerDeathTest, SteppingAnOptimizerThatIsAlreadyInThePlanIsRefused)
{
    SushiBLAS::Engine engine(SaTest::TestEnvironment::runtime());
    SushiAI::RuntimeExecutionBackend backend(SaTest::TestEnvironment::runtime(), engine);

    MLP model(NN::Linear{K_FEATURES, K_HIDDEN}, NN::ReLU{}, NN::Linear{K_HIDDEN, K_CLASSES});
    SA_ASSERT_OK_AND_ASSIGN(Train::TrainingGraph built,
                            Train::build_classifier(model, K_BATCH, K_FEATURES, K_CLASSES));

    SA_ASSERT_OK_AND_ASSIGN(NN::ModelState state, NN::ModelState::create(engine, model));
    TensorPool pool(engine);
    SA_ASSERT_OK_AND_ASSIGN(std::vector<Optim::ParameterSlot> slots,
                            Train::bind_parameters(backend, built, state, pool));
    SA_EXPECT_OK(NN::initialize(backend, state));

    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<Optim::AdamW> optimizer,
                            Optim::AdamW::make(backend, slots, Optim::AdamWOptions{}));
    EXPECT_FALSE(optimizer->is_in_compiled_step());

    const auto objective = SaTest::classifier_objective(built);
    SA_ASSERT_OK_AND_ASSIGN(
        const std::unique_ptr<Train::Trainer> trainer,
        Train::Trainer::make(backend, built, pool, objective, *optimizer));
    EXPECT_TRUE(optimizer->is_in_compiled_step());

    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<Data::SyntheticClassification> dataset,
                            Data::SyntheticClassification::make(backend, data_options()));
    Data::Batch dataset_batch;
    ASSERT_TRUE(SaTest::status_is_ok(dataset->read(0, dataset_batch)));
    SA_EXPECT_OK(trainer->run_step(dataset_batch));

    SA_EXPECT_FATAL((void)optimizer->step(), "already recorded into a compiled step",
                    "twice per batch");
}

/**
 * @brief Tests that step() stops the process when momentum is switched on after construction.
 * @see tests/layers/train/README.md
 */
TEST_F(Medium_FusedOptimizerDeathTest,
       TurningMomentumOnAfterConstructionIsRefusedRatherThanIgnored)
{
    SushiBLAS::Engine engine(SaTest::TestEnvironment::runtime());
    SushiAI::RuntimeExecutionBackend backend(SaTest::TestEnvironment::runtime(), engine);

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
                            Optim::SGD::make(backend, slots, options));

    SA_EXPECT_OK(optimizer->step());
    optimizer->options().momentum = 0.9;

    SA_EXPECT_FATAL((void)optimizer->step(), "no momentum buffer", "0.9");
}
