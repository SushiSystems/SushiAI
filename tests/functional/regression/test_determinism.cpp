/**************************************************************************/
/* test_determinism.cpp                                                   */
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
 * @file test_determinism.cpp
 * @brief Tests that training, initialisation, the dataset and a replayed step repeat bit for bit.
 * @author Mustafa Garip
 * @see tests/functional/regression/README.md
 */

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include "../../common/test_status.hpp"
#include "../../common/test_utils.hpp"

using namespace SushiAI;

// SaTest keeps SushiAI::Test reachable inside TEST_F; see tests/functional/regression/README.md.
namespace SaTest = SushiAI::Test;
using Graph::TensorPool;

namespace
{
    class Regression_Determinism : public ::testing::Test {};

    using Mlp = NN::Sequential<NN::Linear, NN::ReLU, NN::Linear>;

    constexpr int64_t K_BATCH = 8;
    constexpr int64_t K_FEATURES = 6;
    constexpr int64_t K_HIDDEN = 12;
    constexpr int64_t K_CLASSES = 4;
    constexpr int64_t K_BATCHES = 4;
    constexpr std::uint64_t K_DATA_SEED = 0x0D37E4'C0FFEEull;

    /** @brief The synthetic problem every run below trains on. */
    Data::SyntheticOptions data_options()
    {
        Data::SyntheticOptions options;
        options.batch_size = K_BATCH;
        options.batches = K_BATCHES;
        options.features = K_FEATURES;
        options.classes = K_CLASSES;
        options.seed = K_DATA_SEED;
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
     * @return The run, or the first failure of a library call.
     * @see tests/functional/regression/README.md
     */
    [[nodiscard]] Result<RunResult> train_once(int64_t epochs, bool use_adam)
    {
        SushiBLAS::Engine engine(SaTest::TestEnvironment::context());

        Mlp model(NN::Linear{K_FEATURES, K_HIDDEN}, NN::ReLU{}, NN::Linear{K_HIDDEN, K_CLASSES});
        SA_TRY_ASSIGN(Train::TrainingGraph built,
                      Train::build_classifier(model, K_BATCH, K_FEATURES, K_CLASSES));

        SA_TRY_ASSIGN(NN::ModelState state, NN::ModelState::create(engine, model));
        TensorPool pool(engine);
        SA_TRY_ASSIGN(std::vector<Optim::ParameterSlot> slots,
                      Train::bind_parameters(built, state, pool));
        SA_TRY(NN::initialize(engine, state));

        SA_TRY_ASSIGN(const std::unique_ptr<Data::SyntheticClassification> dataset,
                      Data::SyntheticClassification::make(engine, data_options()));

        std::unique_ptr<Optim::Optimizer> optimizer;
        if (use_adam)
        {
            Optim::AdamWOptions adam;
            adam.learning_rate = 0.02;
            SA_TRY_ASSIGN(optimizer, Optim::AdamW::make(engine, slots, adam));
        }
        else
        {
            Optim::SGDOptions sgd;
            sgd.learning_rate = 0.15;
            sgd.momentum = 0.9;
            SA_TRY_ASSIGN(optimizer, Optim::SGD::make(engine, slots, sgd));
        }

        Train::TrainerOptions options;
        options.track_accuracy = false;
        SA_TRY_ASSIGN(const std::unique_ptr<Train::Trainer> trainer,
                      Train::Trainer::make(engine, built.graph, pool, built.spec, *optimizer,
                                           options));

        RunResult result;
        for (int64_t epoch = 0; epoch < epochs; ++epoch)
        {
            for (std::size_t batch = 0; batch < dataset->batch_count(); ++batch)
            {
                SA_TRY_ASSIGN(const Train::StepMetrics metrics,
                              trainer->run_step(dataset->batch(batch)));
                result.losses.push_back(metrics.loss);
            }
        }

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
            // Identical runs produce the same float; see tests/functional/regression/README.md.
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

/**
 * @brief Tests that two SGD-with-momentum runs of two epochs agree byte for byte.
 * @see tests/functional/regression/README.md
 */
TEST_F(Regression_Determinism, TwoIdenticalTrainingRunsAgreeByteForByte)
{
    SA_ASSERT_OK_AND_ASSIGN(const RunResult first, train_once(2, false));
    SA_ASSERT_OK_AND_ASSIGN(const RunResult second, train_once(2, false));

    ASSERT_FALSE(first.losses.empty());
    expect_identical(first, second);
}

/**
 * @brief Tests that two AdamW runs of two epochs agree byte for byte.
 * @see tests/functional/regression/README.md
 */
TEST_F(Regression_Determinism, TwoIdenticalAdamWRunsAgreeByteForByte)
{
    SA_ASSERT_OK_AND_ASSIGN(const RunResult first, train_once(2, true));
    SA_ASSERT_OK_AND_ASSIGN(const RunResult second, train_once(2, true));

    expect_identical(first, second);
}

/**
 * @brief Tests that two initialisations from one seed produce identical parameter bytes.
 * @see tests/functional/regression/README.md
 */
TEST_F(Regression_Determinism, InitialisationFromOneSeedIsBitwiseIdentical)
{
    using Weights = std::vector<std::vector<unsigned char>>;

    const auto initialise = []() -> Result<Weights>
    {
        SushiBLAS::Engine engine(SaTest::TestEnvironment::context());

        Mlp model(NN::Linear{K_FEATURES, K_HIDDEN}, NN::ReLU{}, NN::Linear{K_HIDDEN, K_CLASSES});
        SA_TRY_ASSIGN(Train::TrainingGraph built,
                      Train::build_classifier(model, K_BATCH, K_FEATURES, K_CLASSES));

        SA_TRY_ASSIGN(NN::ModelState state, NN::ModelState::create(engine, model));
        TensorPool pool(engine);
        SA_TRY_ASSIGN(std::vector<Optim::ParameterSlot> slots,
                      Train::bind_parameters(built, state, pool));
        SA_TRY(NN::initialize(engine, state));

        Weights weights;
        for (const Optim::ParameterSlot& slot : slots)
            weights.push_back(SaTest::read_tensor_bytes(slot.parameter));

        return weights;
    };

    SA_ASSERT_OK_AND_ASSIGN(const Weights first, initialise());
    SA_ASSERT_OK_AND_ASSIGN(const Weights second, initialise());

    ASSERT_EQ(first.size(), second.size());
    for (std::size_t i = 0; i < first.size(); ++i)
        EXPECT_EQ(SaTest::first_difference(first[i], second[i]), first[i].size())
            << "parameter " << i << " initialised differently from the same seed";
}

/**
 * @brief Tests that two synthetic datasets built from one seed hold identical bytes.
 * @see tests/functional/regression/README.md
 */
TEST_F(Regression_Determinism, TheSyntheticDatasetIsBitwiseIdentical)
{
    SushiBLAS::Engine first_engine(SaTest::TestEnvironment::context());
    SushiBLAS::Engine second_engine(SaTest::TestEnvironment::context());

    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<Data::SyntheticClassification> first,
                            Data::SyntheticClassification::make(first_engine, data_options()));
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<Data::SyntheticClassification> second,
                            Data::SyntheticClassification::make(second_engine, data_options()));

    ASSERT_EQ(first->batch_count(), second->batch_count());

    for (std::size_t b = 0; b < first->batch_count(); ++b)
    {
        const std::vector<unsigned char> a = SaTest::read_tensor_bytes(first->batch(b).inputs);
        const std::vector<unsigned char> c = SaTest::read_tensor_bytes(second->batch(b).inputs);
        EXPECT_EQ(SaTest::first_difference(a, c), a.size()) << "inputs of batch " << b;

        const std::vector<unsigned char> t = SaTest::read_tensor_bytes(first->batch(b).targets);
        const std::vector<unsigned char> u = SaTest::read_tensor_bytes(second->batch(b).targets);
        EXPECT_EQ(SaTest::first_difference(t, u), t.size()) << "targets of batch " << b;

        EXPECT_EQ(first->batch(b).labels, second->batch(b).labels) << "labels of batch " << b;
    }
}

/**
 * @brief Tests that replaying one compiled step on fixed inputs yields identical gradient bytes.
 * @see tests/functional/regression/README.md
 */
TEST_F(Regression_Determinism, ReplayingOneCompiledStepOnFixedInputsIsBitwiseStable)
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

    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<Data::SyntheticClassification> dataset,
                            Data::SyntheticClassification::make(engine, data_options()));

    Optim::SGDOptions frozen;
    frozen.learning_rate = 0.0;
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<Optim::SGD> optimizer,
                            Optim::SGD::make(engine, slots, frozen));

    Train::TrainerOptions options;
    options.track_accuracy = false;
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<Train::Trainer> trainer,
                            Train::Trainer::make(engine, built.graph, pool, built.spec,
                                                 *optimizer, options));

    // A zero learning rate keeps the parameters fixed; see tests/functional/regression/README.md.
    SA_EXPECT_OK(trainer->run_step(dataset->batch(0)));

    std::vector<std::vector<unsigned char>> reference;
    for (const Optim::ParameterSlot& slot : slots)
        reference.push_back(SaTest::read_tensor_bytes(slot.gradient));

    for (int replay = 1; replay < 6; ++replay)
    {
        SA_EXPECT_OK(trainer->run_step(dataset->batch(0)));

        for (std::size_t i = 0; i < slots.size(); ++i)
        {
            const std::vector<unsigned char> again = SaTest::read_tensor_bytes(slots[i].gradient);
            EXPECT_EQ(SaTest::first_difference(reference[i], again), reference[i].size())
                << "replay " << replay << " produced a different gradient for '"
                << slots[i].name << "'";
        }
    }
}
