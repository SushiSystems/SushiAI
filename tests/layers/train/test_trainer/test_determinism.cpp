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
 * @file test_determinism.cpp
 * @brief Tests that training, initialisation and a replayed step repeat bit for bit.
 * @author Mustafa Garip
 */

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include <SushiBLAS/engine/engine.hpp>
#include <SushiAI/classifier_objective/classifier_objective.hpp>
#include <SushiAI/data/dataset.hpp>
#include <SushiAI/engine_fixture/test_environment.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/graph/tensor_pool.hpp>
#include <SushiAI/nn/init.hpp>
#include <SushiAI/nn/layers.hpp>
#include <SushiAI/nn/model_state.hpp>
#include <SushiAI/optim/adamw.hpp>
#include <SushiAI/optim/optimizer.hpp>
#include <SushiAI/optim/sgd.hpp>
#include <SushiAI/tensor_values/tensor_values.hpp>
#include <SushiAI/train/trainer.hpp>
#include <SushiAI/train/training_graph.hpp>

using namespace SushiAI;

// SaTest keeps SushiAI::Test reachable inside TEST_F; see tests/layers/train/README.md.
namespace SaTest = SushiAI::Test;
using Graph::TensorPool;

namespace
{
    using MLP = NN::Sequential<NN::Linear, NN::ReLU, NN::Linear>;

    constexpr int64_t K_BATCH = 8;
    constexpr int64_t K_FEATURES = 6;
    constexpr int64_t K_HIDDEN = 12;
    constexpr int64_t K_CLASSES = 4;
    constexpr int64_t K_BATCHES = 4;
    constexpr int64_t K_EPOCHS = 2;
    constexpr std::uint64_t K_DATA_SEED = 0x0D37E4'C0FFEEull;

    /** @brief The optimizers a run is repeated with. */
    enum class Descent
    {
        SgdWithMomentum,
        AdamW
    };

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
     * @brief Owns a fresh Engine and the classifier bound on it, as every run here needs.
     *
     * Each rig builds its own Engine, and that is part of the contract: set_seed() leaves the
     * monotonic RNG offset counter alone, so two generators built on one engine draw different
     * streams from one seed.
     */
    class Rig
    {
        public:
            /** @brief Makes the engine, its backend and the pool over it. */
            Rig() : backend(SaTest::TestEnvironment::runtime(), engine), pool(engine) {}

            /** @brief Traces the classifier, binds its parameters and leaves them uninitialised. */
            [[nodiscard]] Status prepare()
            {
                SA_TRY_ASSIGN(built, Train::build_classifier(model, K_BATCH, K_FEATURES, K_CLASSES));
                SA_TRY_ASSIGN(state, NN::ModelState::create(engine, model));
                SA_TRY_ASSIGN(slots, Train::bind_parameters(backend, built, *state, pool));
                return Status();
            }

            /** @brief Draws the starting point of the parameters from the default seed. */
            [[nodiscard]] Status initialise()
            {
                return NN::initialize(backend, *state);
            }

            SushiBLAS::Engine engine{SaTest::TestEnvironment::runtime()};   /**< @brief Owns the device state. */
            SushiAI::RuntimeExecutionBackend backend;                       /**< @brief Runs the steps. */
            MLP model{NN::Linear{K_FEATURES, K_HIDDEN}, NN::ReLU{}, NN::Linear{K_HIDDEN, K_CLASSES}};
            Train::TrainingGraph built;                                     /**< @brief Holds the graph. */
            std::optional<NN::ModelState> state;                            /**< @brief Holds the parameters. */
            TensorPool pool;                                                /**< @brief Holds the values. */
            std::vector<Optim::ParameterSlot> slots;                        /**< @brief Pairs them with gradients. */
    };

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
     * @see tests/layers/train/README.md
     */
    [[nodiscard]] Result<RunResult> train_once(int64_t epochs, Descent descent)
    {
        Rig rig;
        SA_TRY(rig.prepare());
        SA_TRY(rig.initialise());

        SA_TRY_ASSIGN(const std::unique_ptr<Data::SyntheticClassification> dataset,
                      Data::SyntheticClassification::make(rig.backend, data_options()));

        std::unique_ptr<Optim::Optimizer> optimizer;
        if (descent == Descent::AdamW)
        {
            Optim::AdamWOptions adam;
            adam.learning_rate = 0.02;
            SA_TRY_ASSIGN(optimizer, Optim::AdamW::make(rig.backend, rig.slots, adam));
        }
        else
        {
            Optim::SGDOptions sgd;
            sgd.learning_rate = 0.15;
            sgd.momentum = 0.9;
            SA_TRY_ASSIGN(optimizer, Optim::SGD::make(rig.backend, rig.slots, sgd));
        }

        Train::TrainerOptions options;
        options.track_metrics = false;
        const auto objective = SaTest::classifier_objective(rig.built);
        SA_TRY_ASSIGN(const std::unique_ptr<Train::Trainer> trainer,
                      Train::Trainer::make(rig.backend, rig.built, rig.pool, objective,
                                           *optimizer, options));

        RunResult result;
        for (int64_t epoch = 0; epoch < epochs; ++epoch)
        {
            for (std::size_t batch = 0; batch < dataset->batch_count(); ++batch)
            {
                Data::Batch dataset_batch;
                SA_TRY(dataset->read(batch, dataset_batch));
                SA_TRY_ASSIGN(const Train::StepMetrics metrics,
                              trainer->run_step(dataset_batch));
                result.losses.push_back(metrics.loss);
            }
        }

        for (const Optim::ParameterSlot& slot : rig.slots)
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
            // Identical runs produce the same float; see tests/layers/train/README.md.
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

    /** @brief Runs the optimizer given as the row twice and compares the two runs. */
    class Medium_DeterminismRun : public ::testing::TestWithParam<Descent> {};

    /** @brief Holds two classifiers bound on two engines, ready to be initialised. */
    class Medium_DeterminismInitialisation : public ::testing::TestWithParam<std::size_t>
    {
        protected:
            /** @brief Prepares and initialises both rigs from the one default seed. */
            void SetUp() override
            {
                ASSERT_TRUE(SaTest::status_is_ok(first.prepare()));
                ASSERT_TRUE(SaTest::status_is_ok(second.prepare()));
                ASSERT_TRUE(SaTest::status_is_ok(first.initialise()));
                ASSERT_TRUE(SaTest::status_is_ok(second.initialise()));
            }

            Rig first;   /**< @brief The first classifier. */
            Rig second;  /**< @brief The second classifier. */
    };

    /** @brief Holds a classifier whose step runs once, with its gradient bytes kept. */
    class Medium_DeterminismReplay : public ::testing::TestWithParam<std::size_t>
    {
        protected:
            /** @brief Runs the step once on batch 0 and keeps the gradient of the row's slot. */
            void SetUp() override
            {
                ASSERT_TRUE(SaTest::status_is_ok(rig.prepare()));
                SA_ASSERT_OK_AND_ASSIGN(
                    dataset, Data::SyntheticClassification::make(rig.backend, data_options()));
                ASSERT_TRUE(SaTest::status_is_ok(rig.initialise()));

                // A zero learning rate keeps the parameters fixed; see tests/layers/train/README.md.
                Optim::SGDOptions frozen;
                frozen.learning_rate = 0.0;
                SA_ASSERT_OK_AND_ASSIGN(optimizer,
                                        Optim::SGD::make(rig.backend, rig.slots, frozen));

                Train::TrainerOptions options;
                options.track_metrics = false;
                objective = std::make_unique<Train::CrossEntropyObjective>(
                    SaTest::classifier_objective(rig.built));
                SA_ASSERT_OK_AND_ASSIGN(
                    trainer, Train::Trainer::make(rig.backend, rig.built, rig.pool, *objective,
                                                  *optimizer, options));

                ASSERT_TRUE(SaTest::status_is_ok(dataset->read(0, batch)));
                SA_ASSERT_OK_AND_ASSIGN(const Train::StepMetrics metrics, trainer->run_step(batch));
                EXPECT_TRUE(std::isfinite(metrics.loss));
                reference = SaTest::read_tensor_bytes(rig.slots[GetParam()].gradient);
            }

            /** @brief Replays the step and expects the row's gradient to keep its bytes. */
            void expect_replay_stable()
            {
                SA_EXPECT_OK(trainer->run_step(batch));

                const std::vector<unsigned char> again =
                    SaTest::read_tensor_bytes(rig.slots[GetParam()].gradient);
                EXPECT_EQ(SaTest::first_difference(reference, again), reference.size())
                    << "a replay produced a different gradient for '"
                    << rig.slots[GetParam()].name << "'";
            }

            Rig rig;                                                      /**< @brief The classifier. */
            std::unique_ptr<Data::SyntheticClassification> dataset;       /**< @brief Feeds it. */
            std::unique_ptr<Optim::SGD> optimizer;                        /**< @brief Leaves it fixed. */
            std::unique_ptr<Train::CrossEntropyObjective> objective;      /**< @brief Scores a step. */
            std::unique_ptr<Train::Trainer> trainer;                      /**< @brief Runs the steps. */
            Data::Batch batch;                                            /**< @brief Holds batch 0. */
            std::vector<unsigned char> reference;                         /**< @brief Holds the first gradient. */
    };

    /** @brief Names a row by its optimizer. */
    std::string descent_name(const ::testing::TestParamInfo<Descent>& info)
    {
        return info.param == Descent::AdamW ? "AdamW" : "SgdWithMomentum";
    }
}

/**
 * @brief Tests that two runs of two epochs with the row's optimizer agree byte for byte.
 * @see tests/layers/train/README.md
 */
TEST_P(Medium_DeterminismRun, TwoIdenticalRunsAgreeByteForByte)
{
    SA_ASSERT_OK_AND_ASSIGN(const RunResult first, train_once(K_EPOCHS, GetParam()));
    SA_ASSERT_OK_AND_ASSIGN(const RunResult second, train_once(K_EPOCHS, GetParam()));

    ASSERT_FALSE(first.losses.empty());
    expect_identical(first, second);
}

INSTANTIATE_TEST_SUITE_P(Optimizers, Medium_DeterminismRun,
                         ::testing::Values(Descent::SgdWithMomentum, Descent::AdamW),
                         descent_name);

/**
 * @brief Tests that two initialisations from one seed give the row's parameter the same bytes.
 * @see tests/layers/train/README.md
 */
TEST_P(Medium_DeterminismInitialisation, InitialisationFromOneSeedIsBitwiseIdentical)
{
    ASSERT_EQ(first.slots.size(), 4u);
    ASSERT_EQ(second.slots.size(), 4u);

    const std::vector<unsigned char> a = SaTest::read_tensor_bytes(first.slots[GetParam()].parameter);
    const std::vector<unsigned char> b = SaTest::read_tensor_bytes(second.slots[GetParam()].parameter);

    EXPECT_EQ(SaTest::first_difference(a, b), a.size())
        << "'" << first.slots[GetParam()].name << "' initialised differently from the same seed";
}

INSTANTIATE_TEST_SUITE_P(Parameters, Medium_DeterminismInitialisation,
                         ::testing::Values(0u, 1u, 2u, 3u));

/**
 * @brief Tests that five replays of one compiled step on fixed inputs keep the row's gradient bytes.
 * @see tests/layers/train/README.md
 */
TEST_P(Medium_DeterminismReplay, ReplayingOneCompiledStepOnFixedInputsIsBitwiseStable)
{
    expect_replay_stable();
    expect_replay_stable();
    expect_replay_stable();
    expect_replay_stable();
    expect_replay_stable();
}

INSTANTIATE_TEST_SUITE_P(Parameters, Medium_DeterminismReplay, ::testing::Values(0u, 1u, 2u, 3u));
