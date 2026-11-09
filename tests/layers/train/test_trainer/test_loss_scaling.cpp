/**************************************************************************/
/* test_loss_scaling.cpp                                                  */
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
 * @file test_loss_scaling.cpp
 * @brief Tests that loss scaling detects an fp16 overflow, skips the step and never recompiles.
 * @author Mustafa Garip
 */

#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include <SushiBLAS/engine/engine.hpp>
#include <SushiBLAS/tensor/tensor.hpp>
#include <SushiAI/classifier_objective/classifier_objective.hpp>
#include <SushiAI/data/dataset.hpp>
#include <SushiAI/dtype/dtype.hpp>
#include <SushiAI/engine_fixture/test_environment.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/graph/fusion.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/graph/precision.hpp>
#include <SushiAI/graph/tensor_pool.hpp>
#include <SushiAI/nn/init.hpp>
#include <SushiAI/nn/layers.hpp>
#include <SushiAI/nn/model_state.hpp>
#include <SushiAI/optim/adamw.hpp>
#include <SushiAI/optim/loss_scaler.hpp>
#include <SushiAI/optim/optimizer.hpp>
#include <SushiAI/seeded_noise/host_noise.hpp>
#include <SushiAI/shape/shape.hpp>
#include <SushiAI/tensor_values/tensor_values.hpp>
#include <SushiAI/train/trainer.hpp>
#include <SushiAI/train/training_graph.hpp>
#include <support/test_names.hpp>

using namespace SushiAI;
namespace SaTest = SushiAI::Test;
using Graph::MixedPrecisionPolicy;
using Graph::TensorPool;
using Graph::UniformPrecisionPolicy;
using Optim::LossScaleOptions;
using Optim::LossScaler;

namespace
{
    class Medium_LossScaling : public ::testing::Test {};

    /** @brief One direction of scale change: its name and the run that produces it. */
    struct DirectionRow
    {
        const char* name;        /**< @brief Names the row in the test name. */
        bool mixed;              /**< @brief Holds whether the run trains under fp16 mixed precision. */
        double initial;          /**< @brief Holds the starting scale. */
        uint64_t growth_interval;/**< @brief Holds the steps between two growths. */
        int64_t epochs;          /**< @brief Holds the epochs of the run. */
    };

    /** @brief Runs each direction of scale change and expects one compile. */
    class Medium_LossScalingDirection : public ::testing::TestWithParam<DirectionRow> {};

    using MLP = NN::Sequential<NN::Linear, NN::ReLU, NN::Linear>;

    constexpr int64_t K_BATCH = 16;
    constexpr int64_t K_FEATURES = 8;
    constexpr int64_t K_HIDDEN = 16;
    constexpr int64_t K_CLASSES = 4;
    constexpr int64_t K_BATCHES = 4;

    /**
     * @brief Holds 2^20, a loss scale high enough that the fp16 backward pass overflows.
     * @see tests/layers/train/README.md
     */
    constexpr double K_OVERFLOWING_SCALE = 1048576.0;

    /** @brief How many epochs the overflowing run gets to back off into a clean scale. */
    constexpr int64_t K_OVERFLOW_EPOCHS = 4;

    constexpr int64_t K_ELEMENTS = 67;
    constexpr std::uint64_t K_PARAM_SEED = 0x5175'5348'4941'0011ull;
    constexpr std::uint64_t K_GRAD_SEED = 0x5175'5348'4941'0012ull;

    /**
     * @struct ScaledRun
     * @brief What one whole training run under a scaler produced.
     */
    struct ScaledRun
    {
        std::size_t compiles = 0;
        std::uint64_t skipped = 0;
        std::uint64_t backoffs = 0;
        std::uint64_t growths = 0;
        double final_scale = 0.0;
        double last_loss = 0.0;
        std::uint64_t optimizer_steps = 0;
    };

    /**
     * @brief Trains the fixed model under @p precision with a scaler built from @p scale_options.
     * @return The run's figures, or the first failure of a library call.
     * @see tests/layers/train/README.md
     */
    [[nodiscard]] Result<ScaledRun> train_scaled(SushiAI::ExecutionBackend& backend,
                                                 const Graph::PrecisionPolicy* precision,
                                                 LossScaleOptions scale_options,
                                                 int64_t epochs)
    {
        MLP model(NN::Linear{K_FEATURES, K_HIDDEN}, NN::ReLU{}, NN::Linear{K_HIDDEN, K_CLASSES});

        SA_TRY_ASSIGN(Train::TrainingGraph built,
                      Train::build_classifier(model, K_BATCH, K_FEATURES, K_CLASSES, DType::Float32,
                                              Graph::FusionOptions{}, precision,
                                              /*loss_scaling=*/true));

        SA_TRY_ASSIGN(NN::ModelState state, NN::ModelState::create(backend.engine(), model));
        TensorPool pool(backend.engine());
        SA_TRY_ASSIGN(std::vector<Optim::ParameterSlot> slots,
                      Train::bind_parameters(backend, built, state, pool));
        SA_TRY(NN::initialize(backend, state));

        Optim::AdamWOptions adamw;
        adamw.learning_rate = 3e-3;
        SA_TRY_ASSIGN(std::unique_ptr<Optim::AdamW> optimizer,
                      Optim::AdamW::make(backend, slots, adamw));

        SA_TRY_ASSIGN(std::unique_ptr<LossScaler> scaler,
                      LossScaler::make(backend.engine(), pool.tensor(built.loss_scale),
                                       slots.size(),
                                       scale_options));

        // The scaler is set before the trainer records; see tests/layers/train/README.md.
        SA_TRY(optimizer->set_loss_scaler(scaler.get()));

        Train::TrainerOptions options;
        options.precision = precision;
        const auto objective = SaTest::classifier_objective(built);
        SA_TRY_ASSIGN(std::unique_ptr<Train::Trainer> trainer,
                      Train::Trainer::make(backend, built, pool, objective, *optimizer,
                                           options));

        Data::SyntheticOptions data;
        data.batch_size = K_BATCH;
        data.batches = K_BATCHES;
        data.features = K_FEATURES;
        data.classes = K_CLASSES;
        SA_TRY_ASSIGN(std::unique_ptr<Data::SyntheticClassification> dataset,
                      Data::SyntheticClassification::make(backend, data));

        ScaledRun result;
        for (int64_t epoch = 0; epoch < epochs; ++epoch)
            for (std::size_t batch = 0; batch < dataset->batch_count(); ++batch)
            {
                Data::Batch dataset_batch;
                SA_TRY(dataset->read(batch, dataset_batch));
                SA_TRY_ASSIGN(const Train::StepMetrics metrics,
                              trainer->run_step(dataset_batch));
                result.last_loss = metrics.loss;
            }

        result.compiles = trainer->compile_count();
        result.skipped = scaler->skipped_steps();
        result.backoffs = scaler->backoffs();
        result.growths = scaler->growths();
        result.final_scale = scaler->scale();
        result.optimizer_steps = optimizer->step_count();
        return result;
    }

    /** @brief A fixed, finite gradient, identical across runs. */
    std::vector<double> finite_gradient()
    {
        SaTest::HostNoise noise(K_GRAD_SEED);
        std::vector<double> values(static_cast<std::size_t>(K_ELEMENTS));
        noise.fill(values, 0.75);
        values[0] = 0.0;
        values[1] = -0.0;
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

    /** @brief True when every element of @p tensor is finite. */
    bool all_finite(const SushiBLAS::Tensor& tensor)
    {
        for (double value : SaTest::read_tensor(tensor))
            if (!std::isfinite(value))
                return false;

        return true;
    }
}

namespace
{
    /** @brief Runs the overflowing configuration and holds its figures for the cases below. */
    class Medium_LossScalingOverflow : public ::testing::Test
    {
        protected:
            /** @brief Trains under the mixed policy at a scale that overflows fp16, and keeps the run. */
            void SetUp() override
            {
                engine = std::make_unique<SushiBLAS::Engine>(SaTest::TestEnvironment::runtime());
                backend = std::make_unique<RuntimeExecutionBackend>(
                    SaTest::TestEnvironment::runtime(), *engine);

                LossScaleOptions options;
                options.initial = K_OVERFLOWING_SCALE;

                // The interval is out of reach, so growth cannot confuse the reading.
                options.growth_interval = 1000;

                SA_ASSERT_OK_AND_ASSIGN(
                    run, train_scaled(*backend, &mixed, options, /*epochs=*/K_OVERFLOW_EPOCHS));

                // Detection is Small_LossScaling's case; here an overflow is the precondition.
                ASSERT_GT(run->skipped, 0u) << "no step was thrown away, so nothing overflowed";
            }

            MixedPrecisionPolicy mixed;                         /**< @brief Holds the policy. */
            std::unique_ptr<SushiBLAS::Engine> engine;          /**< @brief Owns the device state. */
            std::unique_ptr<RuntimeExecutionBackend> backend;   /**< @brief Runs the steps. */
            std::optional<ScaledRun> run;                       /**< @brief Holds the run's figures. */
    };
}

/**
 * @brief Tests that the run recovers: not every step is thrown away and the loss stays finite.
 * @see tests/layers/train/README.md
 */
TEST_F(Medium_LossScalingOverflow, TheRunRecoversFromTheOverflow)
{
    // The backed-off scale must reach the device; see tests/layers/train/README.md.
    EXPECT_LT(run->skipped, static_cast<std::uint64_t>(K_OVERFLOW_EPOCHS * K_BATCHES))
        << "every step was thrown away, so the new scale never reached the graph";
    EXPECT_LT(run->final_scale, K_OVERFLOWING_SCALE);

    // A loop that went NaN reports NaN here; see tests/layers/train/README.md.
    EXPECT_TRUE(std::isfinite(run->last_loss));
}

/**
 * @brief Tests that a thrown-away step does not advance the optimizer's counter.
 * @see tests/layers/train/README.md
 */
TEST_F(Medium_LossScalingOverflow, ASkippedStepDoesNotAdvanceTheOptimizerCounter)
{
    EXPECT_EQ(run->optimizer_steps,
              static_cast<std::uint64_t>(K_OVERFLOW_EPOCHS * K_BATCHES) - run->skipped);
}

/**
 * @brief Tests that the same scale under the fp32 policy overflows nothing and skips nothing.
 * @see tests/layers/train/README.md
 */
TEST_F(Medium_LossScaling, TheSameScaleInFloat32OverflowsNothing)
{
    SushiBLAS::Engine engine(SaTest::TestEnvironment::runtime());
    SushiAI::RuntimeExecutionBackend backend(SaTest::TestEnvironment::runtime(), engine);

    LossScaleOptions options;
    options.initial = K_OVERFLOWING_SCALE;
    options.growth_interval = 1000;

    const UniformPrecisionPolicy fp32;
    SA_ASSERT_OK_AND_ASSIGN(const ScaledRun run,
                            train_scaled(backend, &fp32, options, /*epochs=*/2));

    EXPECT_EQ(run.skipped, 0u);
    EXPECT_EQ(run.backoffs, 0u);
    EXPECT_EQ(run.final_scale, K_OVERFLOWING_SCALE);
    EXPECT_EQ(run.optimizer_steps, 2u * static_cast<std::uint64_t>(K_BATCHES));
}

/**
 * @brief Tests that a skipped step leaves the parameter and both AdamW moments byte-identical.
 * @see tests/layers/train/README.md
 */
TEST_F(Medium_LossScaling, ASkippedStepLeavesTheParametersAndMomentsBitIdentical)
{
    SushiBLAS::Engine engine(SaTest::TestEnvironment::runtime());
    SushiAI::RuntimeExecutionBackend backend(SaTest::TestEnvironment::runtime(), engine);

    SA_ASSERT_OK_AND_ASSIGN(SushiBLAS::Tensor parameter,
                            allocate_tensor(engine, Shape{K_ELEMENTS}));
    SA_ASSERT_OK_AND_ASSIGN(SushiBLAS::Tensor gradient,
                            allocate_tensor(engine, Shape{K_ELEMENTS}));
    SA_ASSERT_OK_AND_ASSIGN(SushiBLAS::Tensor factor, allocate_tensor(engine, Shape{1}));
    SaTest::write_tensor(parameter, initial_parameter());

    std::vector<Optim::ParameterSlot> slots{Optim::ParameterSlot{"p", parameter, gradient}};

    Optim::AdamWOptions adamw;
    adamw.learning_rate = 1e-2;

    // A skip that still decays would show up through this non-zero decay.
    adamw.weight_decay = 1e-2;
    SA_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Optim::AdamW> optimizer,
                            Optim::AdamW::make(backend, slots, adamw));

    LossScaleOptions options;
    options.initial = 1024.0;
    SA_ASSERT_OK_AND_ASSIGN(std::unique_ptr<LossScaler> scaler,
                            LossScaler::make(engine, factor, slots.size(), options));
    SA_EXPECT_OK(optimizer->set_loss_scaler(scaler.get()));

    // The moments are non-zero before the poison; see tests/layers/train/README.md.
    SaTest::write_tensor(gradient, finite_gradient());
    SA_EXPECT_OK(optimizer->step());
    ASSERT_FALSE(scaler->last_step_skipped());
    ASSERT_EQ(optimizer->step_count(), 1u);

    const std::vector<unsigned char> theta_before = SaTest::read_tensor_bytes(parameter);
    const std::vector<unsigned char> m_before =
        SaTest::read_tensor_bytes(optimizer->first_moment(0));
    const std::vector<unsigned char> v_before =
        SaTest::read_tensor_bytes(optimizer->second_moment(0));

    // The gradient holds real infinities and a NaN; see tests/layers/train/README.md.
    std::vector<double> poisoned = finite_gradient();
    poisoned[5] = std::numeric_limits<double>::infinity();
    poisoned[9] = -std::numeric_limits<double>::infinity();
    poisoned[13] = std::numeric_limits<double>::quiet_NaN();
    SaTest::write_tensor(gradient, poisoned);

    SA_EXPECT_OK(optimizer->step());

    EXPECT_TRUE(scaler->last_step_skipped());
    EXPECT_EQ(scaler->scale(), 512.0);

    // The step counter did not move; see tests/layers/train/README.md.
    EXPECT_EQ(optimizer->step_count(), 1u);

    EXPECT_EQ(SaTest::read_tensor_bytes(parameter), theta_before);
    EXPECT_EQ(SaTest::read_tensor_bytes(optimizer->first_moment(0)), m_before)
        << "the first moment moved on a skipped step - this is the 0 * inf trap";
    EXPECT_EQ(SaTest::read_tensor_bytes(optimizer->second_moment(0)), v_before)
        << "the second moment moved on a skipped step";

    EXPECT_TRUE(all_finite(optimizer->first_moment(0)));
    EXPECT_TRUE(all_finite(optimizer->second_moment(0)));
    EXPECT_TRUE(all_finite(parameter));
}

/**
 * @brief Tests that a finite gradient under the same scaler moves the parameter and the first
 *        moment.
 * @see tests/layers/train/README.md
 */
TEST_F(Medium_LossScaling, ACleanStepUnderTheSameScalerStillMovesEverything)
{
    SushiBLAS::Engine engine(SaTest::TestEnvironment::runtime());
    SushiAI::RuntimeExecutionBackend backend(SaTest::TestEnvironment::runtime(), engine);

    SA_ASSERT_OK_AND_ASSIGN(SushiBLAS::Tensor parameter,
                            allocate_tensor(engine, Shape{K_ELEMENTS}));
    SA_ASSERT_OK_AND_ASSIGN(SushiBLAS::Tensor gradient,
                            allocate_tensor(engine, Shape{K_ELEMENTS}));
    SA_ASSERT_OK_AND_ASSIGN(SushiBLAS::Tensor factor, allocate_tensor(engine, Shape{1}));
    SaTest::write_tensor(parameter, initial_parameter());

    std::vector<Optim::ParameterSlot> slots{Optim::ParameterSlot{"p", parameter, gradient}};

    Optim::AdamWOptions adamw;
    adamw.learning_rate = 1e-2;
    SA_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Optim::AdamW> optimizer,
                            Optim::AdamW::make(backend, slots, adamw));

    SA_ASSERT_OK_AND_ASSIGN(std::unique_ptr<LossScaler> scaler,
                            LossScaler::make(engine, factor, slots.size(), LossScaleOptions{}));
    SA_EXPECT_OK(optimizer->set_loss_scaler(scaler.get()));

    SaTest::write_tensor(gradient, finite_gradient());
    SA_EXPECT_OK(optimizer->step());

    const std::vector<unsigned char> theta_before = SaTest::read_tensor_bytes(parameter);
    const std::vector<unsigned char> m_before =
        SaTest::read_tensor_bytes(optimizer->first_moment(0));

    SaTest::write_tensor(gradient, finite_gradient());
    SA_EXPECT_OK(optimizer->step());

    EXPECT_FALSE(scaler->last_step_skipped());
    EXPECT_EQ(optimizer->step_count(), 2u);
    EXPECT_NE(SaTest::read_tensor_bytes(parameter), theta_before);
    EXPECT_NE(SaTest::read_tensor_bytes(optimizer->first_moment(0)), m_before);
}

/**
 * @brief Tests that the step compiles once while the loss scale changes in the row's direction.
 * @see tests/layers/train/README.md
 */
TEST_P(Medium_LossScalingDirection, TheStepIsCompiledOnceThoughTheScaleChanges)
{
    SushiBLAS::Engine engine(SaTest::TestEnvironment::runtime());
    SushiAI::RuntimeExecutionBackend backend(SaTest::TestEnvironment::runtime(), engine);

    LossScaleOptions options;
    options.initial = GetParam().initial;
    options.growth_interval = GetParam().growth_interval;

    const MixedPrecisionPolicy mixed;
    const UniformPrecisionPolicy fp32;
    const Graph::PrecisionPolicy* precision = GetParam().mixed
        ? static_cast<const Graph::PrecisionPolicy*>(&mixed)
        : static_cast<const Graph::PrecisionPolicy*>(&fp32);
    SA_ASSERT_OK_AND_ASSIGN(const ScaledRun run,
                            train_scaled(backend, precision, options, GetParam().epochs));

    ASSERT_GT(run.growths + run.backoffs, 0u)
        << "the scale never moved, so this run does not test what it claims to";
    EXPECT_NE(run.final_scale, GetParam().initial);
    EXPECT_EQ(run.compiles, 1u);
}

INSTANTIATE_TEST_SUITE_P(Directions, Medium_LossScalingDirection,
                         ::testing::Values(DirectionRow{"Growing", false, 1024.0, 2, 2},
                                           DirectionRow{"BackingOff", true, K_OVERFLOWING_SCALE,
                                                        1000, 1}),
                         [](const ::testing::TestParamInfo<DirectionRow>& info)
                         { return ::SushiAI::Test::parameter_name(info.param.name); });

/**
 * @brief Tests that an AdamW with the separate kernel refuses a loss scaler.
 * @see tests/layers/train/README.md
 */
TEST_F(Medium_LossScaling, TheSeparateUpdateKernelRefusesAScaler)
{
    SushiBLAS::Engine engine(SaTest::TestEnvironment::runtime());
    SushiAI::RuntimeExecutionBackend backend(SaTest::TestEnvironment::runtime(), engine);

    SA_ASSERT_OK_AND_ASSIGN(SushiBLAS::Tensor parameter,
                            allocate_tensor(engine, Shape{K_ELEMENTS}));
    SA_ASSERT_OK_AND_ASSIGN(SushiBLAS::Tensor gradient,
                            allocate_tensor(engine, Shape{K_ELEMENTS}));
    SA_ASSERT_OK_AND_ASSIGN(SushiBLAS::Tensor factor, allocate_tensor(engine, Shape{1}));

    std::vector<Optim::ParameterSlot> slots{Optim::ParameterSlot{"p", parameter, gradient}};

    Optim::AdamWOptions adamw;
    adamw.kernel = Optim::UpdateKernel::Separate;
    SA_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Optim::AdamW> optimizer,
                            Optim::AdamW::make(backend, slots, adamw));

    SA_ASSERT_OK_AND_ASSIGN(std::unique_ptr<LossScaler> scaler,
                            LossScaler::make(engine, factor, slots.size(), LossScaleOptions{}));

    // The refusal names the fused kernel, which is the setting the user must change.
    SA_EXPECT_FAILURE(optimizer->set_loss_scaler(scaler.get()),
                      StatusCode::InvalidDescription, "UpdateKernel::Fused");
}
