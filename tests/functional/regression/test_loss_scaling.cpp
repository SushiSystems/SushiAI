/****************************************************************/
/* test_loss_scaling.cpp                                        */
/* SushiAI - https://github.com/SushiSystems/SushiAI            */
/* Copyright (c) 2026 Sushi Systems                             */
/* Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.    */
/* Commercial use requires a licence from Sushi Systems.        */
/****************************************************************/

// Loss scaling is the part of mixed precision whose every failure is silent,
// which is what puts it in this tier.
//
//   - A scale that never backs off keeps overflowing, and every overflowed
//     step is thrown away, so the loop stops learning while reporting a
//     perfectly ordinary loss.
//   - A skip spelled as "multiply the step by zero" writes NaN into AdamW's
//     first moment, because 0 * inf is NaN and the gradient on a skipped step
//     is an infinity by definition. Nothing throws; the loss goes NaN a step
//     later, three layers from the cause. That is why the assertions below
//     read the *moments* and not only the parameters - a test that checked the
//     parameters alone would pass on exactly this bug.
//   - An unscale that is off by a rounding is invisible against any tolerance,
//     which is why the exactness case uses EXPECT_EQ.
//   - And a device-resident scale that someone moved into Attributes would
//     recompile the step on every backoff, converge identically, and cost the
//     compile-once guarantee the whole architecture is built on.
//
// ON CONSTRUCTING THE OVERFLOW
//
// This workload does not overflow on its own. Measured on this machine, fp16
// without loss scaling neither stalls nor diverges here: both arms converge to
// the same loss. So there is no natural overflow to wait for and a test that
// merely ran the demo with scaling on would prove nothing - it converged
// without scaling too. The overflow below is therefore *built*: the run starts
// at a scale of 2^20, which puts the fp32 logit gradient just past binary16's
// largest finite value of 65504. The cast into the backward matmul then
// produces genuine infinities, which is the real thing and not a simulation of
// it.

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
     * @brief 2^20: high enough that the fp16 backward pass genuinely overflows.
     *
     * The logit gradient is about 1/BATCH, so this puts it at roughly 65536 in
     * fp32 - just past binary16's largest finite 65504 - and the backward
     * matmul that sums over the batch takes it further still. Chosen to be
     * *recoverable* as well as overflowing: the run below has to reach a clean
     * scale within its step budget, because "the backoff never helped" and
     * "the backoff was never given enough steps" are the same observation and
     * only one of them is a defect.
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

        // Before the trainer, which records the update into the plan: the
        // screens have to be recorded ahead of the updates, and the optimizer
        // owns that ordering only once it knows about the scaler.
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

// ---------------------------------------------------------------------------
// The overflow itself
// ---------------------------------------------------------------------------

// The end-to-end case: a scale chosen to overflow binary16 produces real
// infinities in the fp32 gradient buffers, count_nonfinite sees them, and the
// state machine backs the scale off until it stops happening.
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

    // And the run *recovers*: the backed-off scale has to reach the device, or
    // every step overflows for ever and the loop learns nothing while
    // reporting a healthy loss. This is what pins the scale being a value the
    // host rewrites between replays rather than a constant - a scaler whose
    // before_replay() did nothing would still back off on the host and would
    // fail here.
    EXPECT_LT(run.skipped, static_cast<std::uint64_t>(OVERFLOW_EPOCHS * BATCHES))
        << "every step was thrown away, so the new scale never reached the graph";
    EXPECT_GT(run.optimizer_steps, 0u);

    // The counter does not advance through a thrown-away step: SGD's
    // first-step flag and AdamW's bias corrections are both derived from it.
    EXPECT_EQ(run.optimizer_steps,
              static_cast<std::uint64_t>(OVERFLOW_EPOCHS * BATCHES) - run.skipped);

    // And the run survives it. A loop that went NaN would report NaN here,
    // which no comparison against a threshold would catch.
    EXPECT_TRUE(std::isfinite(run.last_loss));
}

// The control. Without it, "steps were skipped" says nothing about fp16: an
// implementation that skipped every step would pass the case above. The same
// scale in the fp32 policy is nowhere near float32's range, so nothing may
// overflow and nothing may be thrown away.
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

// ---------------------------------------------------------------------------
// What a skipped step must leave behind
// ---------------------------------------------------------------------------

// The assertion the whole gate exists for. A skip spelled through the existing
// scalars - step size zero, beta1 one - looks like an identity and is not:
// `m = 1*m + 0*inf` is NaN, and the optimizer never recovers. So the moments
// are read as raw bytes before and after, not only the parameters.
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

    // One healthy step first, so the moments are non-zero and a NaN written
    // into them is distinguishable from the zero they started at.
    SaTest::write_tensor(gradient, finite_gradient());
    optimizer.step();
    ASSERT_FALSE(scaler.last_step_skipped());
    ASSERT_EQ(optimizer.step_count(), 1u);

    const std::vector<unsigned char> theta_before = SaTest::read_tensor_bytes(parameter);
    const std::vector<unsigned char> m_before =
        SaTest::read_tensor_bytes(optimizer.first_moment(0));
    const std::vector<unsigned char> v_before =
        SaTest::read_tensor_bytes(optimizer.second_moment(0));

    // A gradient that is genuinely non-finite, which is what an overflowed
    // fp16 backward pass hands the optimizer.
    std::vector<double> poisoned = finite_gradient();
    poisoned[5] = std::numeric_limits<double>::infinity();
    poisoned[9] = -std::numeric_limits<double>::infinity();
    poisoned[13] = std::numeric_limits<double>::quiet_NaN();
    SaTest::write_tensor(gradient, poisoned);

    optimizer.step();

    EXPECT_TRUE(scaler.last_step_skipped());
    EXPECT_EQ(scaler.scale(), 512.0);

    // The counter did not move, so the next step's bias corrections are the
    // ones this step would have used.
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

// The control for the case above: with a finite gradient the same machinery
// must move all three. Otherwise "nothing changed" would be satisfied by an
// optimizer that had stopped working altogether.
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

// ---------------------------------------------------------------------------
// Exactness
// ---------------------------------------------------------------------------

// Scaling by a power of two changes only the exponent field, so the round trip
// is lossless and the comparison is EXPECT_EQ. Asserting it to a tolerance
// would hide the defect this exists to catch: a scale that is not dyadic, or
// an unscale folded in after the moments rather than before them.
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

// ---------------------------------------------------------------------------
// The compile-once guarantee, with a scale that actually moves
// ---------------------------------------------------------------------------

// The reason the scale is a device-resident value and not an Attributes slot.
// An attribute is a compile-time constant baked into the recorded task, so
// every backoff and every growth would recompile the step - and the loop would
// converge exactly the same while paying graph analysis on every one of them.
// The run below is arranged so the scale really does change during it.
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

// The overflowing arm of the same guarantee: backoffs are the direction that
// happens under load, and they must not recompile either.
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

// ---------------------------------------------------------------------------
// What is refused
// ---------------------------------------------------------------------------

// The separate spelling has no branch to take, so it would unscale correctly
// and then apply every overflowed step anyway. Refused by name rather than
// downgraded silently.
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
