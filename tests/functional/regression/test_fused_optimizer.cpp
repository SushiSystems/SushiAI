/**************************************************************************/
/* test_fused_optimizer.cpp                                               */
/**************************************************************************/
/*                          This file is part of:                         */
/*                                 SushiAI                                */
/**************************************************************************/

// The gate on collapsing eleven launches per parameter into one.
//
// Fusing an optimizer update is a pure performance change that touches the
// only arithmetic in the library nothing downstream validates. Every failure
// mode it has is silent: a transposed scalar index, an epsilon folded inside
// the square root, a momentum buffer read a step late, a first-step branch
// that stopped being special - each of those still trains, still converges,
// and still reports a healthy loss. So the fused kernel is not tested against
// a tolerance or a fixture; it is tested against the separate-op sequence it
// replaced, byte for byte, from identical state.
//
// WHY BYTE FOR BYTE IS THE RIGHT BAR, AND NOT OPTIMISM
//
// Each element's update is independent and the operation sequence is
// unchanged, so the only thing that could move a bit is a difference in
// rounding - and the one plausible source of that, a compiler contracting
// `a * b + c` into an FMA inside one kernel where it could not across two, is
// ruled out by construction: the whole stack compiles with `-ffp-contract=off`
// through SushiRuntime's `sushiruntime_deterministic_fp` contract. Every
// expression in the fused functors is also written in the same operand order
// as the kernel it replaces (`x *= alpha` is `x * alpha`; `y += alpha * x` is
// `y + (alpha * x)`), and both paths read their scalars out of the same
// float32 device buffer so neither can pick up a differently rounded constant.
//
// If this ever fails, the answer is to find which expression moved - not to
// loosen the comparison.
//
// THE SECOND HALF: FOLDING THE UPDATE INTO THE COMPILED PLAN
//
// Fusing is what makes the update replay-invariant, and replay-invariance is
// what lets it be recorded into the same TaskGraph as the forward and backward
// passes. That is a separate change with its own silent failure mode - a
// dependency edge the runtime does not see would let the update race the
// backward pass that produces its gradients, and a race that usually resolves
// the right way looks exactly like training. So the folded and unfolded
// placements are compared byte for byte too, and compile_count() is pinned at
// one with the optimizer inside.

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include "../../common/test_throws.hpp"
#include "../../common/test_utils.hpp"

using namespace SushiAI;

// `Test` alone resolves to ::testing::Test inside a TEST_F body, where the
// fixture's base injects the name; the alias keeps the helpers reachable.
namespace SaTest = SushiAI::Test;
using Graph::TensorPool;
using Optim::UpdateKernel;

namespace
{
    class Regression_FusedOptimizer : public ::testing::Test {};

    // Deliberately not a multiple of the 128-bit vector width, so both the
    // wide pass and the scalar tail of Launch::zip_inplace run on every call.
    constexpr int64_t ELEMENTS = 133;

    constexpr int STEPS = 6;
    constexpr std::uint64_t PARAM_SEED = 0x5175'5348'4941'0001ull;
    constexpr std::uint64_t GRAD_SEED = 0x5175'5348'4941'0002ull;

    /**
     * @brief The gradient for step @p step, identical across runs.
     *
     * The first four entries are hand-placed rather than sampled: an exact
     * zero, a negative zero, a denormal-adjacent tiny value and a large one.
     * Zero is where AdamW's 0 / (sqrt(0) + eps) lives, negative zero is what
     * distinguishes `b = staged` from `b = 0*b + 1*staged` (the latter turns
     * -0.0 into +0.0 and would be invisible to any tolerance), and the two
     * extremes exercise the square root and the division away from unity.
     *
     * @param step Which step's gradient to produce.
     * @return ELEMENTS values.
     */
    std::vector<double> gradient_for(int step)
    {
        SaTest::HostNoise noise(GRAD_SEED + static_cast<std::uint64_t>(step));

        std::vector<double> values(static_cast<std::size_t>(ELEMENTS));
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
        SaTest::HostNoise noise(PARAM_SEED);
        std::vector<double> values(static_cast<std::size_t>(ELEMENTS));
        noise.fill(values, 2.0);
        return values;
    }

    /**
     * @brief Steps one optimizer over a fixed gradient sequence, standalone.
     *
     * No graph, no trainer, no device RNG: one parameter, one gradient
     * rewritten from the host between steps, and the optimizer's own step().
     * That leaves the update as the only thing in the measurement.
     *
     * @param make Builds the optimizer over the slot, for a given kernel.
     * @param kernel Which spelling of the update to use.
     * @return The parameter's raw bytes after STEPS updates.
     */
    template<typename Make>
    std::vector<unsigned char> drive(Make make, UpdateKernel kernel)
    {
        SushiBLAS::Engine engine(SaTest::TestEnvironment::context());

        SushiBLAS::Tensor parameter = allocate_tensor(engine, Shape{ELEMENTS});
        SushiBLAS::Tensor gradient = allocate_tensor(engine, Shape{ELEMENTS});
        SaTest::write_tensor(parameter, initial_parameter());

        std::vector<Optim::ParameterSlot> slots{
            Optim::ParameterSlot{"p", parameter, gradient},
        };

        const std::unique_ptr<Optim::Optimizer> optimizer = make(engine, slots, kernel);

        for (int step = 0; step < STEPS; ++step)
        {
            SaTest::write_tensor(gradient, gradient_for(step));
            optimizer->step();
        }

        return SaTest::read_tensor_bytes(parameter);
    }

    /** @brief Builds an AdamW with @p options and the requested kernel. */
    auto adamw_factory(Optim::AdamWOptions options)
    {
        return [options](SushiBLAS::Engine& engine,
                         const std::vector<Optim::ParameterSlot>& slots,
                         UpdateKernel kernel) -> std::unique_ptr<Optim::Optimizer>
        {
            Optim::AdamWOptions with_kernel = options;
            with_kernel.kernel = kernel;
            return std::make_unique<Optim::AdamW>(engine, slots, with_kernel);
        };
    }

    /** @brief Builds an SGD with @p options and the requested kernel. */
    auto sgd_factory(Optim::SGDOptions options)
    {
        return [options](SushiBLAS::Engine& engine,
                         const std::vector<Optim::ParameterSlot>& slots,
                         UpdateKernel kernel) -> std::unique_ptr<Optim::Optimizer>
        {
            Optim::SGDOptions with_kernel = options;
            with_kernel.kernel = kernel;
            return std::make_unique<Optim::SGD>(engine, slots, with_kernel);
        };
    }

    /** @brief Asserts the two spellings produced identical bytes. */
    template<typename Make>
    void expect_kernels_agree(Make make, const std::string& what)
    {
        const std::vector<unsigned char> fused = drive(make, UpdateKernel::FUSED);
        const std::vector<unsigned char> separate = drive(make, UpdateKernel::SEPARATE);

        ASSERT_EQ(fused.size(), separate.size());
        EXPECT_EQ(SaTest::first_difference(fused, separate), fused.size())
            << what << ": the fused kernel and the separate operations disagree from byte "
            << SaTest::first_difference(fused, separate)
            << ". Find the expression that moved; do not widen this comparison.";
    }

    // --- The training-loop half ------------------------------------------

    using Mlp = NN::Sequential<NN::Linear, NN::ReLU, NN::Linear>;

    constexpr int64_t BATCH = 8;
    constexpr int64_t FEATURES = 6;
    constexpr int64_t HIDDEN = 12;
    constexpr int64_t CLASSES = 4;
    constexpr int64_t BATCHES = 4;

    /** @brief The synthetic problem the training comparisons train on. */
    Data::SyntheticOptions data_options()
    {
        Data::SyntheticOptions options;
        options.batch_size = BATCH;
        options.batches = BATCHES;
        options.features = FEATURES;
        options.classes = CLASSES;
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
     *
     * A fresh Engine, because reproducibility is a property of one - see
     * dataset.hpp on the RNG offset counter.
     *
     * @param placement Whether the update joins the compiled plan.
     * @param kernel Which spelling of the update to use.
     * @return The parameters, the trajectory and the compile count.
     */
    TrainedRun train(Train::OptimizerPlacement placement, UpdateKernel kernel)
    {
        SushiBLAS::Engine engine(SaTest::TestEnvironment::context());

        Mlp model(NN::Linear{FEATURES, HIDDEN}, NN::ReLU{}, NN::Linear{HIDDEN, CLASSES});
        Train::TrainingGraph built = Train::build_classifier(model, BATCH, FEATURES, CLASSES);

        TensorPool pool(engine);
        std::vector<Optim::ParameterSlot> slots = Train::bind_parameters(built, pool);
        NN::initialize(engine, model, pool);

        Optim::AdamWOptions adam;
        adam.learning_rate = 0.02;
        adam.kernel = kernel;
        Optim::AdamW optimizer(engine, slots, adam);

        Train::TrainerOptions options;
        options.track_accuracy = false;
        options.optimizer_placement = placement;
        Train::Trainer trainer(engine, built.graph, pool, built.spec, optimizer, options);

        const Data::SyntheticClassification dataset(engine, data_options());

        TrainedRun run;
        run.optimizer_in_plan = trainer.optimizer_is_in_plan();
        run.operations_per_step = trainer.fit(dataset).operations_per_step;

        for (int64_t epoch = 1; epoch < 3; ++epoch)
            for (std::size_t batch = 0; batch < dataset.batch_count(); ++batch)
                run.losses.push_back(trainer.run_step(dataset.batch(batch)).loss);

        for (const Optim::ParameterSlot& slot : slots)
        {
            run.names.push_back(slot.name);
            run.parameters.push_back(SaTest::read_tensor_bytes(slot.parameter));
        }

        run.compiles = trainer.compile_count();
        return run;
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

// =========================================================================
// 1. One kernel computes what eleven did
// =========================================================================

TEST_F(Regression_FusedOptimizer, AdamWsFusedKernelMatchesTheSeparateOperationsExactly)
{
    Optim::AdamWOptions options;
    options.learning_rate = 0.02;
    options.weight_decay = 0.01;
    expect_kernels_agree(adamw_factory(options), "AdamW with decay");
}

TEST_F(Regression_FusedOptimizer, AdamWWithoutDecayTakesTheOtherBranchAndStillMatches)
{
    // weight_decay = 0 makes the decay factor exactly 1, which both paths
    // must skip rather than multiply by - `theta * 1.0f` is bit-identical
    // here, but only because it is, and the branch is what says so.
    Optim::AdamWOptions options;
    options.learning_rate = 0.02;
    options.weight_decay = 0.0;
    expect_kernels_agree(adamw_factory(options), "AdamW without decay");
}

TEST_F(Regression_FusedOptimizer, SGDsFusedKernelMatchesTheSeparateOperationsExactly)
{
    // Every branch of the recipe, because each one is a separate line of the
    // fused functor and a separate opportunity to have written it differently.
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

// =========================================================================
// 2. Folding the update into the plan changes no number
// =========================================================================

TEST_F(Regression_FusedOptimizer, TheUpdateInThePlanTrainsIdenticallyToTheUpdateOutsideIt)
{
    // If these diverge, the update is not ordered against the graph the way
    // the separate flush ordered it - which means the runtime is missing a
    // dependency edge, and the fix is that edge, never a tolerance.
    const TrainedRun folded = train(Train::OptimizerPlacement::IN_PLAN, UpdateKernel::FUSED);
    const TrainedRun apart = train(Train::OptimizerPlacement::PER_STEP, UpdateKernel::FUSED);

    ASSERT_TRUE(folded.optimizer_in_plan);
    ASSERT_FALSE(apart.optimizer_in_plan);
    ASSERT_FALSE(folded.losses.empty());

    expect_runs_agree(folded, apart, "the optimizer's placement");
}

TEST_F(Regression_FusedOptimizer, TheSeparateSpellingOutsideThePlanTrainsIdenticallyToo)
{
    // The end-to-end version of test 1: not one optimizer over a synthetic
    // gradient, but the whole pipeline, with gradients the backward pass
    // produced and a trajectory long enough for a one-ULP difference in an
    // early denominator to become a visible one.
    const TrainedRun fused = train(Train::OptimizerPlacement::PER_STEP, UpdateKernel::FUSED);
    const TrainedRun separate = train(Train::OptimizerPlacement::PER_STEP, UpdateKernel::SEPARATE);

    expect_runs_agree(fused, separate, "the update's spelling");
}

// =========================================================================
// 3. It really is one compile, and the update really is inside it
// =========================================================================

TEST_F(Regression_FusedOptimizer, TheStepCompilesOnceWithTheOptimizerInsideIt)
{
    const TrainedRun folded = train(Train::OptimizerPlacement::IN_PLAN, UpdateKernel::FUSED);
    const TrainedRun apart = train(Train::OptimizerPlacement::PER_STEP, UpdateKernel::FUSED);

    EXPECT_EQ(folded.compiles, 1u);
    EXPECT_EQ(apart.compiles, 1u);

    // Four parameters, so four fused updates joined the plan. Stated as a
    // difference rather than an absolute so a change to the graph itself does
    // not have to be re-recorded here.
    EXPECT_EQ(folded.operations_per_step, apart.operations_per_step + folded.names.size())
        << "the folded plan does not hold one extra task per parameter";
}

// =========================================================================
// 4. The launch count itself
// =========================================================================

TEST_F(Regression_FusedOptimizer, TheFusedUpdateIsOneTaskPerParameterWhereTheSeparateOneIsTwelve)
{
    // The headline claim, counted rather than timed - a wall clock says
    // different things on a loaded machine, but the number of tasks issued is
    // the same everywhere and is the thing that was actually changed.
    //
    // Twelve for AdamW with decay: the decay scal, two for m, three for v,
    // four for the denominator, and two for the step. Eleven without decay,
    // which is the figure the SushiBLAS seam's own documentation quotes.
    constexpr std::size_t PARAMETERS = 3;

    const auto count = [](UpdateKernel kernel, auto make) -> std::size_t
    {
        SushiBLAS::Engine engine(SaTest::TestEnvironment::context());

        std::vector<Optim::ParameterSlot> slots;
        std::vector<SushiBLAS::Tensor> owned;
        owned.reserve(PARAMETERS * 2);

        for (std::size_t i = 0; i < PARAMETERS; ++i)
        {
            owned.push_back(allocate_tensor(engine, Shape{ELEMENTS}));
            owned.push_back(allocate_tensor(engine, Shape{ELEMENTS}));
            SaTest::write_tensor(owned[owned.size() - 2], initial_parameter());
            SaTest::write_tensor(owned[owned.size() - 1], gradient_for(0));
            slots.push_back(Optim::ParameterSlot{
                "p" + std::to_string(i), owned[owned.size() - 2], owned[owned.size() - 1],
            });
        }

        const std::unique_ptr<Optim::Optimizer> optimizer = make(engine, slots, kernel);

        // The phases step() would run, with the count read out of the middle
        // one. The flush afterwards is not optional: the tasks are in the
        // engine's builder and something has to drain them.
        optimizer->before_replay();
        const std::size_t tasks = optimizer->record_into_step();
        engine.execute();
        engine.get_context().wait_all();

        return tasks;
    };

    Optim::AdamWOptions adam;
    adam.learning_rate = 0.02;
    adam.weight_decay = 0.01;

    EXPECT_EQ(count(UpdateKernel::FUSED, adamw_factory(adam)), PARAMETERS);
    EXPECT_EQ(count(UpdateKernel::SEPARATE, adamw_factory(adam)), PARAMETERS * 12);

    Optim::SGDOptions sgd;
    sgd.learning_rate = 0.1;
    sgd.momentum = 0.9;

    // SGD's first step is its cheapest (b_1 = g_1 is one copy rather than a
    // scal and an axpy), so four is the floor, not the typical cost.
    EXPECT_EQ(count(UpdateKernel::FUSED, sgd_factory(sgd)), PARAMETERS);
    EXPECT_EQ(count(UpdateKernel::SEPARATE, sgd_factory(sgd)), PARAMETERS * 4);
}

// =========================================================================
// 5. An update that cannot be replayed is refused, not downgraded
// =========================================================================

TEST_F(Regression_FusedOptimizer, AskingToCompileAnUnfusedUpdateIsRefusedRatherThanIgnored)
{
    // The separate spelling bakes this step's bias corrections into its
    // recorded tasks. Folded, it would replay step one's corrections for ever
    // - and still converge, which is exactly why this has to be an error and
    // not a quiet fallback.
    SushiBLAS::Engine engine(SaTest::TestEnvironment::context());

    Mlp model(NN::Linear{FEATURES, HIDDEN}, NN::ReLU{}, NN::Linear{HIDDEN, CLASSES});
    Train::TrainingGraph built = Train::build_classifier(model, BATCH, FEATURES, CLASSES);

    TensorPool pool(engine);
    std::vector<Optim::ParameterSlot> slots = Train::bind_parameters(built, pool);
    NN::initialize(engine, model, pool);

    Optim::AdamWOptions adam;
    adam.kernel = UpdateKernel::SEPARATE;
    Optim::AdamW optimizer(engine, slots, adam);

    EXPECT_FALSE(optimizer.is_replay_invariant());

    Train::TrainerOptions options;
    options.optimizer_placement = Train::OptimizerPlacement::IN_PLAN;

    SA_EXPECT_ERROR((void)Train::Trainer(engine, built.graph, pool, built.spec,
                                         optimizer, options),
                    "cannot be compiled into the step", "PER_STEP");
}

TEST_F(Regression_FusedOptimizer, SteppingAnOptimizerThatIsAlreadyInThePlanIsRefused)
{
    // Before the update joined the plan, `trainer.run_step()` called
    // `optimizer.step()` and a caller doing the same got a visible double
    // update. Now the update is inside the replay and invisible from the
    // outside, so the same mistake would quietly train at twice the learning
    // rate and produce a run that looks fine and is not.
    SushiBLAS::Engine engine(SaTest::TestEnvironment::context());

    Mlp model(NN::Linear{FEATURES, HIDDEN}, NN::ReLU{}, NN::Linear{HIDDEN, CLASSES});
    Train::TrainingGraph built = Train::build_classifier(model, BATCH, FEATURES, CLASSES);

    TensorPool pool(engine);
    std::vector<Optim::ParameterSlot> slots = Train::bind_parameters(built, pool);
    NN::initialize(engine, model, pool);

    Optim::AdamW optimizer(engine, slots, Optim::AdamWOptions{});
    EXPECT_FALSE(optimizer.is_in_compiled_step());

    Train::Trainer trainer(engine, built.graph, pool, built.spec, optimizer);
    EXPECT_TRUE(optimizer.is_in_compiled_step());

    const Data::SyntheticClassification dataset(engine, data_options());
    trainer.run_step(dataset.batch(0));

    SA_EXPECT_ERROR(optimizer.step(), "already recorded into a compiled step",
                    "twice per batch");
}

TEST_F(Regression_FusedOptimizer, TurningMomentumOnAfterConstructionIsRefusedRatherThanIgnored)
{
    // The momentum buffer is allocated at construction or not at all, so the
    // fused kernel's operand count is fixed there too. Before this check the
    // separate path dereferenced a null tensor and the fused path would simply
    // have trained without momentum.
    SushiBLAS::Engine engine(SaTest::TestEnvironment::context());

    SushiBLAS::Tensor parameter = allocate_tensor(engine, Shape{ELEMENTS});
    SushiBLAS::Tensor gradient = allocate_tensor(engine, Shape{ELEMENTS});
    SaTest::write_tensor(parameter, initial_parameter());
    SaTest::write_tensor(gradient, gradient_for(0));

    std::vector<Optim::ParameterSlot> slots{
        Optim::ParameterSlot{"p", parameter, gradient},
    };

    Optim::SGDOptions options;
    options.learning_rate = 0.1;
    Optim::SGD optimizer(engine, slots, options);

    optimizer.step();
    optimizer.options().momentum = 0.9;

    SA_EXPECT_ERROR(optimizer.step(), "no momentum buffer", "0.9");
}
