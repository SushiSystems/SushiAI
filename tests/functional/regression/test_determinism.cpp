/**************************************************************************/
/* test_determinism.cpp                                                   */
/**************************************************************************/
/*                          This file is part of:                         */
/*                                 SushiAI                                */
/**************************************************************************/

// Bit-for-bit reproducibility, end to end.
//
// This is the guarantee the whole stack is arranged around and the one nothing
// proved. SushiRuntime promises sushiruntime_deterministic_fp; backward.cpp
// sums a fan-out through an ordered ACCUMULATE chain rather than float
// atomics, and says in its own header that an atomic sum "would break that
// quietly, showing up as a training run that cannot be reproduced rather than
// as a test failure". A claim whose failure mode is explicitly "not a test
// failure" needs a test, and it has to compare bits: EXPECT_NEAR passes
// happily on a run that reorders a reduction, which is exactly the change the
// claim forbids.
//
// Regression tier, because every failure here is silent. A model that trains
// to a slightly different place each run still converges, still reports good
// accuracy, and still looks completely healthy - right up to the day someone
// needs to reproduce a result.
//
// A fresh Engine per run is part of the contract, not a workaround: SushiBLAS's
// set_seed() leaves the engine's monotonic RNG offset counter alone, so two
// generators built on one engine draw different streams from one seed. That is
// documented in dataset.hpp and is why every comparison below builds its own.

#include <cstddef>
#include <cstdint>
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

namespace
{
    class Regression_Determinism : public ::testing::Test {};

    using Mlp = NN::Sequential<NN::Linear, NN::ReLU, NN::Linear>;

    constexpr int64_t BATCH = 8;
    constexpr int64_t FEATURES = 6;
    constexpr int64_t HIDDEN = 12;
    constexpr int64_t CLASSES = 4;
    constexpr int64_t BATCHES = 4;
    constexpr std::uint64_t DATA_SEED = 0x0D37E4'C0FFEEull;

    /** @brief The synthetic problem every run below trains on. */
    Data::SyntheticOptions data_options()
    {
        Data::SyntheticOptions options;
        options.batch_size = BATCH;
        options.batches = BATCHES;
        options.features = FEATURES;
        options.classes = CLASSES;
        options.seed = DATA_SEED;
        return options;
    }

    /**
     * @struct RunResult
     * @brief One whole training run, reduced to what a comparison needs.
     */
    struct RunResult
    {
        /** @brief Every parameter's raw bytes, in walk order. */
        std::vector<std::vector<unsigned char>> parameters;

        /** @brief Every parameter's name, so a mismatch says which one. */
        std::vector<std::string> names;

        /** @brief The per-step loss, in order. */
        std::vector<double> losses;
    };

    /**
     * @brief Trains an MLP from scratch on a fresh Engine and reports the result.
     *
     * Everything the run depends on is either fixed here or seeded: the model
     * shape, the initialisation seed (InitPolicy's default), the data seed and
     * the optimizer. Two calls therefore differ in nothing at all, which is
     * what makes a difference in the output a real finding.
     *
     * @param epochs How many passes to make.
     * @param use_adam Whether to use AdamW rather than SGD with momentum.
     * @return The parameters and the loss trajectory.
     */
    RunResult train_once(int64_t epochs, bool use_adam)
    {
        SushiBLAS::Engine engine(SaTest::TestEnvironment::context());

        Mlp model(NN::Linear{FEATURES, HIDDEN}, NN::ReLU{}, NN::Linear{HIDDEN, CLASSES});
        Train::TrainingGraph built = Train::build_classifier(model, BATCH, FEATURES, CLASSES);

        TensorPool pool(engine);
        std::vector<Optim::ParameterSlot> slots = Train::bind_parameters(built, pool);
        NN::initialize(engine, model, pool);

        const Data::SyntheticClassification dataset(engine, data_options());

        std::unique_ptr<Optim::Optimizer> optimizer;
        if (use_adam)
        {
            Optim::AdamWOptions adam;
            adam.learning_rate = 0.02;
            optimizer = std::make_unique<Optim::AdamW>(engine, slots, adam);
        }
        else
        {
            Optim::SGDOptions sgd;
            sgd.learning_rate = 0.15;
            sgd.momentum = 0.9;
            optimizer = std::make_unique<Optim::SGD>(engine, slots, sgd);
        }

        Train::TrainerOptions options;
        options.track_accuracy = false;
        Train::Trainer trainer(engine, built.graph, pool, built.spec, *optimizer, options);

        RunResult result;
        for (int64_t epoch = 0; epoch < epochs; ++epoch)
            for (std::size_t batch = 0; batch < dataset.batch_count(); ++batch)
                result.losses.push_back(trainer.run_step(dataset.batch(batch)).loss);

        for (const Optim::ParameterSlot& slot : slots)
        {
            result.names.push_back(slot.name);
            result.parameters.push_back(SaTest::read_tensor_bytes(slot.parameter));
        }

        return result;
    }

    /** @brief Asserts two runs agree byte for byte, naming where they do not. */
    void expect_identical(const RunResult& first, const RunResult& second)
    {
        ASSERT_EQ(first.names, second.names);
        ASSERT_EQ(first.losses.size(), second.losses.size());

        for (std::size_t step = 0; step < first.losses.size(); ++step)
        {
            // The loss is read out of device memory as a float; two identical
            // runs must produce the same float, so == is the right comparison
            // and EXPECT_NEAR would be an admission that they might not.
            EXPECT_EQ(first.losses[step], second.losses[step])
                << "the runs diverged at step " << step << " (" << first.losses[step]
                << " vs " << second.losses[step] << ")";
        }

        for (std::size_t i = 0; i < first.parameters.size(); ++i)
        {
            const std::size_t at = SaTest::first_difference(first.parameters[i],
                                                            second.parameters[i]);
            EXPECT_EQ(at, first.parameters[i].size())
                << "parameter '" << first.names[i] << "' differs from byte " << at
                << " onward - the run is not reproducible";
        }
    }
}

TEST_F(Regression_Determinism, TwoIdenticalTrainingRunsAgreeByteForByte)
{
    // Two epochs with momentum: enough steps for a reordered reduction to
    // accumulate a visible difference, and momentum means step N depends on
    // every step before it, so a single divergent bit cannot cancel out.
    const RunResult first = train_once(2, false);
    const RunResult second = train_once(2, false);

    ASSERT_FALSE(first.losses.empty());
    expect_identical(first, second);
}

TEST_F(Regression_Determinism, TwoIdenticalAdamWRunsAgreeByteForByte)
{
    // AdamW divides by a square root of an accumulated second moment, which is
    // where a one-ULP difference in the gradient stops staying one ULP.
    const RunResult first = train_once(2, true);
    const RunResult second = train_once(2, true);

    expect_identical(first, second);
}

TEST_F(Regression_Determinism, InitialisationFromOneSeedIsBitwiseIdentical)
{
    // The starting point has to be identical before anything downstream can
    // be, and this is the only part of the run that goes through the device
    // RNG rather than through arithmetic.
    const auto initialise = []()
    {
        SushiBLAS::Engine engine(SaTest::TestEnvironment::context());

        Mlp model(NN::Linear{FEATURES, HIDDEN}, NN::ReLU{}, NN::Linear{HIDDEN, CLASSES});
        Train::TrainingGraph built = Train::build_classifier(model, BATCH, FEATURES, CLASSES);

        TensorPool pool(engine);
        std::vector<Optim::ParameterSlot> slots = Train::bind_parameters(built, pool);
        NN::initialize(engine, model, pool);

        std::vector<std::vector<unsigned char>> weights;
        for (const Optim::ParameterSlot& slot : slots)
            weights.push_back(SaTest::read_tensor_bytes(slot.parameter));
        return weights;
    };

    const std::vector<std::vector<unsigned char>> first = initialise();
    const std::vector<std::vector<unsigned char>> second = initialise();

    ASSERT_EQ(first.size(), second.size());
    for (std::size_t i = 0; i < first.size(); ++i)
        EXPECT_EQ(SaTest::first_difference(first[i], second[i]), first[i].size())
            << "parameter " << i << " initialised differently from the same seed";
}

TEST_F(Regression_Determinism, TheSyntheticDatasetIsBitwiseIdentical)
{
    // The existing coverage compares through EXPECT_DOUBLE_EQ, which tolerates
    // four ULP and cannot see a sign difference on a zero. A dataset is either
    // reproducible or it is not, so this compares the bytes.
    SushiBLAS::Engine first_engine(SaTest::TestEnvironment::context());
    SushiBLAS::Engine second_engine(SaTest::TestEnvironment::context());

    const Data::SyntheticClassification first(first_engine, data_options());
    const Data::SyntheticClassification second(second_engine, data_options());

    ASSERT_EQ(first.batch_count(), second.batch_count());

    for (std::size_t b = 0; b < first.batch_count(); ++b)
    {
        const std::vector<unsigned char> a = SaTest::read_tensor_bytes(first.batch(b).inputs);
        const std::vector<unsigned char> c = SaTest::read_tensor_bytes(second.batch(b).inputs);
        EXPECT_EQ(SaTest::first_difference(a, c), a.size()) << "inputs of batch " << b;

        const std::vector<unsigned char> t = SaTest::read_tensor_bytes(first.batch(b).targets);
        const std::vector<unsigned char> u = SaTest::read_tensor_bytes(second.batch(b).targets);
        EXPECT_EQ(SaTest::first_difference(t, u), t.size()) << "targets of batch " << b;

        EXPECT_EQ(first.batch(b).labels, second.batch(b).labels) << "labels of batch " << b;
    }
}

TEST_F(Regression_Determinism, ReplayingOneCompiledStepOnFixedInputsIsBitwiseStable)
{
    // The narrowest form of the claim, with the optimizer and the RNG removed:
    // the same compiled plan, the same buffers, run twice. Anything the
    // runtime does non-deterministically - a work-stealing reduction, an
    // atomic accumulation, a scheduler-dependent order - shows up here with
    // nothing else to blame it on.
    SushiBLAS::Engine engine(SaTest::TestEnvironment::context());

    Mlp model(NN::Linear{FEATURES, HIDDEN}, NN::ReLU{}, NN::Linear{HIDDEN, CLASSES});
    Train::TrainingGraph built = Train::build_classifier(model, BATCH, FEATURES, CLASSES);

    TensorPool pool(engine);
    std::vector<Optim::ParameterSlot> slots = Train::bind_parameters(built, pool);
    NN::initialize(engine, model, pool);

    const Data::SyntheticClassification dataset(engine, data_options());

    Optim::SGDOptions frozen;
    frozen.learning_rate = 0.0;
    Optim::SGD optimizer(engine, slots, frozen);

    Train::TrainerOptions options;
    options.track_accuracy = false;
    Train::Trainer trainer(engine, built.graph, pool, built.spec, optimizer, options);

    // A zero learning rate keeps the parameters fixed, so every replay sees
    // exactly the state the one before it did.
    trainer.run_step(dataset.batch(0));

    std::vector<std::vector<unsigned char>> reference;
    for (const Optim::ParameterSlot& slot : slots)
        reference.push_back(SaTest::read_tensor_bytes(slot.gradient));

    for (int replay = 1; replay < 6; ++replay)
    {
        trainer.run_step(dataset.batch(0));

        for (std::size_t i = 0; i < slots.size(); ++i)
        {
            const std::vector<unsigned char> again = SaTest::read_tensor_bytes(slots[i].gradient);
            EXPECT_EQ(SaTest::first_difference(reference[i], again), reference[i].size())
                << "replay " << replay << " produced a different gradient for '"
                << slots[i].name << "'";
        }
    }
}
