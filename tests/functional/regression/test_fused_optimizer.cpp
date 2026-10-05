/****************************************************************/
/* test_fused_optimizer.cpp                                     */
/* SushiAI - https://github.com/SushiSystems/SushiAI            */
/* Copyright (c) 2026 Sushi Systems                             */
/* Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.    */
/* Commercial use requires a licence from Sushi Systems.        */
/****************************************************************/

/**
 * @file test_fused_optimizer.cpp
 * @brief Tests that the fused optimizer update matches the separate operations byte for byte.
 * @author Mustafa Garip
 * @see tests/functional/regression/README.md
 */

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include "../../common/test_throws.hpp"
#include "../../common/test_utils.hpp"

using namespace SushiAI;

// SaTest keeps SushiAI::Test reachable inside TEST_F; see tests/functional/regression/README.md.
namespace SaTest = SushiAI::Test;
using Graph::TensorPool;
using Optim::UpdateKernel;

namespace
{
    class Regression_FusedOptimizer : public ::testing::Test {};

    // 133 is not a multiple of the vector width; see tests/functional/regression/README.md.
    constexpr int64_t ELEMENTS = 133;

    constexpr int STEPS = 6;
    constexpr std::uint64_t PARAM_SEED = 0x5175'5348'4941'0001ull;
    constexpr std::uint64_t GRAD_SEED = 0x5175'5348'4941'0002ull;

    /**
     * @brief Returns the gradient for step @p step, identical across runs.
     * @see tests/functional/regression/README.md
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
     * @see tests/functional/regression/README.md
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
     * @see tests/functional/regression/README.md
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

TEST_F(Regression_FusedOptimizer, AdamWsFusedKernelMatchesTheSeparateOperationsExactly)
{
    Optim::AdamWOptions options;
    options.learning_rate = 0.02;
    options.weight_decay = 0.01;
    expect_kernels_agree(adamw_factory(options), "AdamW with decay");
}

/**
 * @brief Tests that AdamW without weight decay matches the separate operations byte for byte.
 * @see tests/functional/regression/README.md
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
 * @see tests/functional/regression/README.md
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
 * @see tests/functional/regression/README.md
 */
TEST_F(Regression_FusedOptimizer, TheUpdateInThePlanTrainsIdenticallyToTheUpdateOutsideIt)
{
    const TrainedRun folded = train(Train::OptimizerPlacement::IN_PLAN, UpdateKernel::FUSED);
    const TrainedRun apart = train(Train::OptimizerPlacement::PER_STEP, UpdateKernel::FUSED);

    ASSERT_TRUE(folded.optimizer_in_plan);
    ASSERT_FALSE(apart.optimizer_in_plan);
    ASSERT_FALSE(folded.losses.empty());

    expect_runs_agree(folded, apart, "the optimizer's placement");
}

/**
 * @brief Tests that the fused and separate spellings train a whole model identically.
 * @see tests/functional/regression/README.md
 */
TEST_F(Regression_FusedOptimizer, TheSeparateSpellingOutsideThePlanTrainsIdenticallyToo)
{
    const TrainedRun fused = train(Train::OptimizerPlacement::PER_STEP, UpdateKernel::FUSED);
    const TrainedRun separate = train(Train::OptimizerPlacement::PER_STEP, UpdateKernel::SEPARATE);

    expect_runs_agree(fused, separate, "the update's spelling");
}

TEST_F(Regression_FusedOptimizer, TheStepCompilesOnceWithTheOptimizerInsideIt)
{
    const TrainedRun folded = train(Train::OptimizerPlacement::IN_PLAN, UpdateKernel::FUSED);
    const TrainedRun apart = train(Train::OptimizerPlacement::PER_STEP, UpdateKernel::FUSED);

    EXPECT_EQ(folded.compiles, 1u);
    EXPECT_EQ(apart.compiles, 1u);

    // Each parameter adds one fused update to the plan; see tests/functional/regression/README.md.
    EXPECT_EQ(folded.operations_per_step, apart.operations_per_step + folded.names.size())
        << "the folded plan does not hold one extra task per parameter";
}

/**
 * @brief Tests that the fused update records one task per parameter, against twelve for separate
 *        AdamW and four for separate SGD.
 * @see tests/functional/regression/README.md
 */
TEST_F(Regression_FusedOptimizer, TheFusedUpdateIsOneTaskPerParameterWhereTheSeparateOneIsTwelve)
{
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

        // The recorded tasks must be drained; see tests/functional/regression/README.md.
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

    // Four is the floor for SGD, on its first step; see tests/functional/regression/README.md.
    EXPECT_EQ(count(UpdateKernel::FUSED, sgd_factory(sgd)), PARAMETERS);
    EXPECT_EQ(count(UpdateKernel::SEPARATE, sgd_factory(sgd)), PARAMETERS * 4);
}

/**
 * @brief Tests that the trainer refuses to compile a separate-kernel AdamW into the step.
 * @see tests/functional/regression/README.md
 */
TEST_F(Regression_FusedOptimizer, AskingToCompileAnUnfusedUpdateIsRefusedRatherThanIgnored)
{
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

/**
 * @brief Tests that step() reports an error on an optimizer already recorded into a compiled step.
 * @see tests/functional/regression/README.md
 */
TEST_F(Regression_FusedOptimizer, SteppingAnOptimizerThatIsAlreadyInThePlanIsRefused)
{
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

/**
 * @brief Tests that step() reports an error when momentum is switched on after construction.
 * @see tests/functional/regression/README.md
 */
TEST_F(Regression_FusedOptimizer, TurningMomentumOnAfterConstructionIsRefusedRatherThanIgnored)
{
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
