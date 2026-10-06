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
 * @file test_loss_scaling.cpp
 * @brief Tests that loss scaling detects an fp16 overflow, skips the step and never recompiles.
 * @author Mustafa Garip
 * @see tests/functional/regression/README.md
 */

#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <vector>
#include <gtest/gtest.h>
#include "../../common/test_throws.hpp"
#include "../../common/test_utils.hpp"

using namespace SushiAI;

namespace SaTest = SushiAI::Test;
using Graph::MixedPrecisionPolicy;
using Graph::TensorPool;
using Graph::UniformPrecisionPolicy;
using Optim::LossScaleOptions;
using Optim::LossScaler;

namespace
{
    class Regression_LossScaling : public ::testing::Test {};

    using Mlp = NN::Sequential<NN::Linear, NN::ReLU, NN::Linear>;

    constexpr int64_t BATCH = 16;
    constexpr int64_t FEATURES = 8;
    constexpr int64_t HIDDEN = 16;
    constexpr int64_t CLASSES = 4;
    constexpr int64_t BATCHES = 4;

    /**
     * @brief Holds 2^20, a loss scale high enough that the fp16 backward pass overflows.
     * @see tests/functional/regression/README.md
     */
    constexpr double OVERFLOWING_SCALE = 1048576.0;

    /** @brief How many epochs the overflowing run gets to back off into a clean scale. */
    constexpr int64_t OVERFLOW_EPOCHS = 4;

    /** @brief The fixed scale the exactness case runs at, and its step count. */
    constexpr double EXACT_SCALE = 1024.0;
    constexpr int EXACT_STEPS = 5;

    constexpr int64_t ELEMENTS = 67;
    constexpr std::uint64_t PARAM_SEED = 0x5175'5348'4941'0011ull;
    constexpr std::uint64_t GRAD_SEED = 0x5175'5348'4941'0012ull;

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
     * @brief Trains the fixed model under @p precision with a scaler.
     *
     * Everything but the policy and the scaler's options is held constant, so
     * a difference between two calls is those two things and nothing else.
     */
    ScaledRun train_scaled(SushiBLAS::Engine& engine,
                           const Graph::IPrecisionPolicy* precision,
                           LossScaleOptions scale_options,
                           int64_t epochs)
    {
        Mlp model(NN::Linear{FEATURES, HIDDEN}, NN::ReLU{}, NN::Linear{HIDDEN, CLASSES});

        Train::TrainingGraph built = Train::build_classifier(
            model, BATCH, FEATURES, CLASSES, DType::FLOAT32,
            Graph::FusionOptions{}, precision, /*loss_scaling=*/true);

        TensorPool pool(engine);
        std::vector<Optim::ParameterSlot> slots = Train::bind_parameters(built, pool);
        NN::initialize(engine, model, pool);

        Optim::AdamWOptions adamw;
        adamw.learning_rate = 3e-3;
        Optim::AdamW optimizer(engine, slots, adamw);

        LossScaler scaler(engine, pool.tensor(built.loss_scale), slots.size(), scale_options);

        // The scaler is set before the trainer records; see tests/functional/regression/README.md.
        optimizer.set_loss_scaler(&scaler);

        Train::TrainerOptions options;
        options.precision = precision;
        Train::Trainer trainer(engine, built.graph, pool, built.spec, optimizer, options);

        Data::SyntheticOptions data;
        data.batch_size = BATCH;
        data.batches = BATCHES;
        data.features = FEATURES;
        data.classes = CLASSES;
        const Data::SyntheticClassification dataset(engine, data);

        ScaledRun result;
        for (int64_t epoch = 0; epoch < epochs; ++epoch)
            for (std::size_t batch = 0; batch < dataset.batch_count(); ++batch)
                result.last_loss = trainer.run_step(dataset.batch(batch)).loss;

        result.compiles = trainer.compile_count();
        result.skipped = scaler.skipped_steps();
        result.backoffs = scaler.backoffs();
        result.growths = scaler.growths();
        result.final_scale = scaler.scale();
        result.optimizer_steps = optimizer.step_count();
        return result;
    }

    /** @brief A fixed, finite gradient, identical across runs. */
    std::vector<double> finite_gradient()
    {
        SaTest::HostNoise noise(GRAD_SEED);
        std::vector<double> values(static_cast<std::size_t>(ELEMENTS));
        noise.fill(values, 0.75);
        values[0] = 0.0;
        values[1] = -0.0;
        return values;
    }

    /** @brief The starting parameter values, identical across runs. */
    std::vector<double> initial_parameter()
    {
        SaTest::HostNoise noise(PARAM_SEED);
        std::vector<double> values(static_cast<std::size_t>(ELEMENTS));
        noise.fill(values, 2.0);
        return values;
    }

    /** @brief True when every element of @p tensor is finite. */
    bool all_finite(const SushiBLAS::Tensor& tensor)
    {
        for (double value : SaTest::read_tensor(tensor))
            if (!std::isfinite(value)) return false;
        return true;
    }
}

/**
 * @brief Tests that an fp16 overflow is detected, backs the scale off and lets the run recover.
 * @see tests/functional/regression/README.md
 */
TEST_F(Regression_LossScaling, AnFp16OverflowIsDetectedAndBacksTheScaleOff)
{
    SushiBLAS::Engine engine(SaTest::TestEnvironment::context());

    LossScaleOptions options;
    options.initial = OVERFLOWING_SCALE;
    options.growth_interval = 1000;   // out of reach, so growth cannot confuse the reading

    const MixedPrecisionPolicy mixed;
    const ScaledRun run = train_scaled(engine, &mixed, options, /*epochs=*/OVERFLOW_EPOCHS);

    EXPECT_GT(run.skipped, 0u) << "no step was thrown away, so nothing overflowed - "
                                  "the detector is not seeing the infinities";
    EXPECT_GT(run.backoffs, 0u);
    EXPECT_LT(run.final_scale, OVERFLOWING_SCALE);
    EXPECT_EQ(run.growths, 0u);

    // The backed-off scale must reach the device; see tests/functional/regression/README.md.
    EXPECT_LT(run.skipped, static_cast<std::uint64_t>(OVERFLOW_EPOCHS * BATCHES))
        << "every step was thrown away, so the new scale never reached the graph";
    EXPECT_GT(run.optimizer_steps, 0u);

    // A thrown-away step does not advance the counter; see tests/functional/regression/README.md.
    EXPECT_EQ(run.optimizer_steps,
              static_cast<std::uint64_t>(OVERFLOW_EPOCHS * BATCHES) - run.skipped);

    // A loop that went NaN reports NaN here; see tests/functional/regression/README.md.
    EXPECT_TRUE(std::isfinite(run.last_loss));
}

/**
 * @brief Tests that the same scale under the fp32 policy overflows nothing and skips nothing.
 * @see tests/functional/regression/README.md
 */
TEST_F(Regression_LossScaling, TheSameScaleInFloat32OverflowsNothing)
{
    SushiBLAS::Engine engine(SaTest::TestEnvironment::context());

    LossScaleOptions options;
    options.initial = OVERFLOWING_SCALE;
    options.growth_interval = 1000;

    const UniformPrecisionPolicy fp32;
    const ScaledRun run = train_scaled(engine, &fp32, options, /*epochs=*/2);

    EXPECT_EQ(run.skipped, 0u);
    EXPECT_EQ(run.backoffs, 0u);
    EXPECT_EQ(run.final_scale, OVERFLOWING_SCALE);
    EXPECT_EQ(run.optimizer_steps, 2u * static_cast<std::uint64_t>(BATCHES));
}

/**
 * @brief Tests that a skipped step leaves the parameter and both AdamW moments byte-identical.
 * @see tests/functional/regression/README.md
 */
TEST_F(Regression_LossScaling, ASkippedStepLeavesTheParametersAndMomentsBitIdentical)
{
    SushiBLAS::Engine engine(SaTest::TestEnvironment::context());

    SushiBLAS::Tensor parameter = allocate_tensor(engine, Shape{ELEMENTS});
    SushiBLAS::Tensor gradient = allocate_tensor(engine, Shape{ELEMENTS});
    SushiBLAS::Tensor factor = allocate_tensor(engine, Shape{1});
    SaTest::write_tensor(parameter, initial_parameter());

    std::vector<Optim::ParameterSlot> slots{Optim::ParameterSlot{"p", parameter, gradient}};

    Optim::AdamWOptions adamw;
    adamw.learning_rate = 1e-2;
    adamw.weight_decay = 1e-2;   // so a "skip" that still decays would show up
    Optim::AdamW optimizer(engine, slots, adamw);

    LossScaleOptions options;
    options.initial = 1024.0;
    LossScaler scaler(engine, factor, slots.size(), options);
    optimizer.set_loss_scaler(&scaler);

    // The moments are non-zero before the poison; see tests/functional/regression/README.md.
    SaTest::write_tensor(gradient, finite_gradient());
    optimizer.step();
    ASSERT_FALSE(scaler.last_step_skipped());
    ASSERT_EQ(optimizer.step_count(), 1u);

    const std::vector<unsigned char> theta_before = SaTest::read_tensor_bytes(parameter);
    const std::vector<unsigned char> m_before =
        SaTest::read_tensor_bytes(optimizer.first_moment(0));
    const std::vector<unsigned char> v_before =
        SaTest::read_tensor_bytes(optimizer.second_moment(0));

    // The gradient holds real infinities and a NaN; see tests/functional/regression/README.md.
    std::vector<double> poisoned = finite_gradient();
    poisoned[5] = std::numeric_limits<double>::infinity();
    poisoned[9] = -std::numeric_limits<double>::infinity();
    poisoned[13] = std::numeric_limits<double>::quiet_NaN();
    SaTest::write_tensor(gradient, poisoned);

    optimizer.step();

    EXPECT_TRUE(scaler.last_step_skipped());
    EXPECT_EQ(scaler.scale(), 512.0);

    // The step counter did not move; see tests/functional/regression/README.md.
    EXPECT_EQ(optimizer.step_count(), 1u);

    EXPECT_EQ(SaTest::read_tensor_bytes(parameter), theta_before);
    EXPECT_EQ(SaTest::read_tensor_bytes(optimizer.first_moment(0)), m_before)
        << "the first moment moved on a skipped step - this is the 0 * inf trap";
    EXPECT_EQ(SaTest::read_tensor_bytes(optimizer.second_moment(0)), v_before)
        << "the second moment moved on a skipped step";

    EXPECT_TRUE(all_finite(optimizer.first_moment(0)));
    EXPECT_TRUE(all_finite(optimizer.second_moment(0)));
    EXPECT_TRUE(all_finite(parameter));
}

/**
 * @brief Tests that a finite gradient under the same scaler moves the parameter and the first
 *        moment.
 * @see tests/functional/regression/README.md
 */
TEST_F(Regression_LossScaling, ACleanStepUnderTheSameScalerStillMovesEverything)
{
    SushiBLAS::Engine engine(SaTest::TestEnvironment::context());

    SushiBLAS::Tensor parameter = allocate_tensor(engine, Shape{ELEMENTS});
    SushiBLAS::Tensor gradient = allocate_tensor(engine, Shape{ELEMENTS});
    SushiBLAS::Tensor factor = allocate_tensor(engine, Shape{1});
    SaTest::write_tensor(parameter, initial_parameter());

    std::vector<Optim::ParameterSlot> slots{Optim::ParameterSlot{"p", parameter, gradient}};

    Optim::AdamWOptions adamw;
    adamw.learning_rate = 1e-2;
    Optim::AdamW optimizer(engine, slots, adamw);

    LossScaler scaler(engine, factor, slots.size(), LossScaleOptions{});
    optimizer.set_loss_scaler(&scaler);

    SaTest::write_tensor(gradient, finite_gradient());
    optimizer.step();

    const std::vector<unsigned char> theta_before = SaTest::read_tensor_bytes(parameter);
    const std::vector<unsigned char> m_before =
        SaTest::read_tensor_bytes(optimizer.first_moment(0));

    SaTest::write_tensor(gradient, finite_gradient());
    optimizer.step();

    EXPECT_FALSE(scaler.last_step_skipped());
    EXPECT_EQ(optimizer.step_count(), 2u);
    EXPECT_NE(SaTest::read_tensor_bytes(parameter), theta_before);
    EXPECT_NE(SaTest::read_tensor_bytes(optimizer.first_moment(0)), m_before);
}

/**
 * @brief Tests that a run scaled by 1024 and unscaled matches the unscaled run bit for bit.
 * @see tests/functional/regression/README.md
 */
TEST_F(Regression_LossScaling, UnscalingByAPowerOfTwoIsExact)
{
    const auto drive = [](bool scaled) -> std::vector<unsigned char>
    {
        SushiBLAS::Engine engine(SaTest::TestEnvironment::context());

        SushiBLAS::Tensor parameter = allocate_tensor(engine, Shape{ELEMENTS});
        SushiBLAS::Tensor gradient = allocate_tensor(engine, Shape{ELEMENTS});
        SushiBLAS::Tensor factor = allocate_tensor(engine, Shape{1});
        SaTest::write_tensor(parameter, initial_parameter());

        std::vector<Optim::ParameterSlot> slots{Optim::ParameterSlot{"p", parameter, gradient}};

        Optim::SGDOptions sgd;
        sgd.learning_rate = 1e-2;
        sgd.momentum = 0.9;
        Optim::SGD optimizer(engine, slots, sgd);

        LossScaleOptions options;
        options.initial = EXACT_SCALE;
        options.growth_interval = 1000;  // out of reach: the scale must not move
        std::unique_ptr<LossScaler> scaler;

        if (scaled)
        {
            scaler = std::make_unique<LossScaler>(engine, factor, slots.size(), options);
            optimizer.set_loss_scaler(scaler.get());
        }

        for (int step = 0; step < EXACT_STEPS; ++step)
        {
            std::vector<double> values = finite_gradient();
            if (scaled)
                for (double& value : values) value *= EXACT_SCALE;

            SaTest::write_tensor(gradient, values);
            optimizer.step();
        }

        if (scaled) EXPECT_EQ(scaler->scale(), EXACT_SCALE);
        EXPECT_EQ(optimizer.step_count(), static_cast<std::uint64_t>(EXACT_STEPS));

        return SaTest::read_tensor_bytes(parameter);
    };

    EXPECT_EQ(drive(true), drive(false))
        << "a scaled run and an unscaled one did not agree bit for bit, so the "
           "unscale is not the exact inverse of the scale";
}

/**
 * @brief Tests that the step compiles once while the loss scale grows.
 * @see tests/functional/regression/README.md
 */
TEST_F(Regression_LossScaling, TheStepIsCompiledOnceThoughTheScaleChanges)
{
    SushiBLAS::Engine engine(SaTest::TestEnvironment::context());

    LossScaleOptions options;
    options.initial = 1024.0;
    options.growth_interval = 2;

    const UniformPrecisionPolicy fp32;
    const ScaledRun run = train_scaled(engine, &fp32, options, /*epochs=*/2);

    ASSERT_GT(run.growths, 0u) << "the scale never moved, so this run does not test "
                                  "what it claims to";
    EXPECT_GT(run.final_scale, 1024.0);
    EXPECT_EQ(run.compiles, 1u);
}

/**
 * @brief Tests that the step compiles once while the loss scale backs off.
 * @see tests/functional/regression/README.md
 */
TEST_F(Regression_LossScaling, ABackingOffScaleStillCompilesOnce)
{
    SushiBLAS::Engine engine(SaTest::TestEnvironment::context());

    LossScaleOptions options;
    options.initial = OVERFLOWING_SCALE;
    options.growth_interval = 1000;

    const MixedPrecisionPolicy mixed;
    const ScaledRun run = train_scaled(engine, &mixed, options, /*epochs=*/1);

    ASSERT_GT(run.backoffs, 0u);
    EXPECT_EQ(run.compiles, 1u);
}

/**
 * @brief Tests that an AdamW with the separate kernel refuses a loss scaler.
 * @see tests/functional/regression/README.md
 */
TEST_F(Regression_LossScaling, TheSeparateUpdateKernelRefusesAScaler)
{
    SushiBLAS::Engine engine(SaTest::TestEnvironment::context());

    SushiBLAS::Tensor parameter = allocate_tensor(engine, Shape{ELEMENTS});
    SushiBLAS::Tensor gradient = allocate_tensor(engine, Shape{ELEMENTS});
    SushiBLAS::Tensor factor = allocate_tensor(engine, Shape{1});

    std::vector<Optim::ParameterSlot> slots{Optim::ParameterSlot{"p", parameter, gradient}};

    Optim::AdamWOptions adamw;
    adamw.kernel = Optim::UpdateKernel::SEPARATE;
    Optim::AdamW optimizer(engine, slots, adamw);

    LossScaler scaler(engine, factor, slots.size(), LossScaleOptions{});

    SA_EXPECT_ERROR(optimizer.set_loss_scaler(&scaler), "FUSED");
}
