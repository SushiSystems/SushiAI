/**************************************************************************/
/* test_free_training_step.cpp                                            */
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
 * @file test_free_training_step.cpp
 * @brief Tests on an engine that a two-level anchor-free model trains on the leaves its
 *        host stage assigns, and that two runs of one seed end with equal weights.
 * @author Mustafa Garip
 */

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include <SushiBLAS/tensor/tensor.hpp>
#include <SushiBLAS/engine/engine.hpp>

#include <SushiAI/failure/result.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/graph/tensor_pool.hpp>
#include <SushiAI/optim/optimizer.hpp>
#include <SushiAI/shape/shape.hpp>
#include <SushiAI/train/trainer.hpp>
#include <SushiAI/train/training_graph.hpp>
#include <SushiAI/allocation/allocation.hpp>
#include <SushiAI/engine_fixture/engine_test.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/nn/conv2d.hpp>
#include <SushiAI/nn/init.hpp>
#include <SushiAI/nn/leaky_relu.hpp>
#include <SushiAI/nn/model_graph.hpp>
#include <SushiAI/nn/model_state.hpp>
#include <SushiAI/optim/sgd.hpp>
#include <SushiAI/profile/profile_config.hpp>
#include <SushiAI/seeded_noise/host_noise.hpp>
#include <SushiAI/tensor_values/tensor_values.hpp>
#include <SushiAI/train/buffer_update_recorder.hpp>
#include <SushiAI/train/objective_context.hpp>
#include <SushiAI/train/steps.hpp>
#include <SushiAI/train/training_step.hpp>
#include <SushiAI/vision/detect/free/free_task.hpp>
#include <SushiAI/vision/detect/free/head_contract.hpp>
#include <SushiAI/vision/detect/ground_truth.hpp>
#include <SushiAI/vision/training/free_detection_loss_host.hpp>
#include <SushiAI/vision/training/free_loss_gains.hpp>
#include <SushiAI/vision/training/free_model_facts.hpp>
#include <SushiAI/vision/training/free_objective.hpp>
#include <SushiAI/vision/training/free_training_step.hpp>

#if SA_ENABLE_PROFILER
#include <SushiProfiler/registry/registry.hpp>
#include <SushiProfiler/session/session.hpp>
#include <SushiAI/profile/timeline_binding.hpp>
#include <SushiAI/profile/timelines.hpp>
#include <SushiAI/profile_records/profile_records.hpp>
#endif

using namespace SushiAI;
// SaTest stays reachable inside a TEST_F body; see tests/vision/training/README.md.
namespace SaTest = SushiAI::Test;
using Graph::TensorPool;
using Vision::Detect::GroundTruthBatch;
using Vision::Detect::Free::detect_port_name;
using Vision::Detect::Free::K_DETECT_BOXES;
using Vision::Detect::Free::K_DETECT_ONE_TO_MANY;
using Vision::Detect::Free::K_DETECT_ONE_TO_ONE;
using Vision::Detect::Free::K_DETECT_SCORES;
using Vision::Training::FreeLossTerms;
using Vision::Training::FreeObjective;
using Vision::Training::FreeTrainingStep;

namespace
{
    class Medium_FreeTrainingStep : public SaTest::EngineTest {};

#if SA_ENABLE_PROFILER
    class Medium_FreeTrainingZones : public SaTest::EngineTest {};
#endif

    /** @brief Holds the images per step. */
    constexpr std::int64_t K_BATCH = 2;

    /** @brief Holds the side of the square network input in pixels. */
    constexpr std::int64_t K_NETWORK_SIDE = 8;

    /** @brief Holds the channels both stride-2 convolutions produce. */
    constexpr std::int64_t K_BODY_CHANNELS = 8;

    /** @brief Holds the classes of the score ports. */
    constexpr std::int64_t K_CLASSES = 2;

    /** @brief Holds the distribution bins per box side. */
    constexpr std::int64_t K_REG_MAX = 2;

    /** @brief Holds the step size of the plain SGD the cases train with. */
    constexpr double K_LEARNING_RATE = 0.01;

    /** @brief Seeds the input image. */
    constexpr std::uint64_t K_INPUT_SEED = 20261026ull;

    /** @brief Bounds the difference of two runs of one seed: none. */
    constexpr double K_SAME = 0.0;

    /** @brief Holds the least change of a weight that counts as having moved. */
    constexpr double K_MOVED = 1e-9;

    /** @brief Returns a 1 by 1 convolution with a bias onto @p channels. */
    NN::Conv2d port_convolution(std::int64_t channels)
    {
        NN::Conv2d convolution = NN::Conv2d::same_padded(K_BODY_CHANNELS, channels, 1, 1);
        convolution.use_bias = true;
        return convolution;
    }

    /**
     * @brief Builds two stride-2 convolutions with a box and a score port behind each.
     * @param one_to_one Adds a one-to-one copy of every port.
     * @return A model over [8, 8, 3] that publishes the one-to-many ports of level 0 on a
     *         4 by 4 grid, stride 2, and of level 1 on a 2 by 2 grid, stride 4.
     */
    [[nodiscard]] Result<NN::ModelGraph> build_two_level_model(bool one_to_one)
    {
        NN::ModelGraph model(Shape{K_NETWORK_SIDE, K_NETWORK_SIDE, 3});

        SA_TRY_ASSIGN(const NN::NodeRef stem,
                      model.add("stem", NN::Conv2d::same_padded(3, K_BODY_CHANNELS, 3, 2),
                                model.input()));
        SA_TRY_ASSIGN(const NN::NodeRef stem_act, model.add("stem_act", NN::LeakyReLU{0.1}, stem));
        SA_TRY_ASSIGN(const NN::NodeRef down,
                      model.add("down",
                                NN::Conv2d::same_padded(K_BODY_CHANNELS, K_BODY_CHANNELS, 3, 2),
                                stem_act));
        SA_TRY_ASSIGN(const NN::NodeRef down_act, model.add("down_act", NN::LeakyReLU{0.1}, down));

        const NN::NodeRef bodies[] = {stem_act, down_act};
        for (std::size_t level = 0; level < 2; ++level)
        {
            const std::string suffix = std::to_string(level);
            SA_TRY_ASSIGN(const NN::NodeRef boxes,
                          model.add("boxes" + suffix, port_convolution(4 * K_REG_MAX),
                                    bodies[level]));
            SA_TRY_ASSIGN(const NN::NodeRef scores,
                          model.add("scores" + suffix, port_convolution(K_CLASSES),
                                    bodies[level]));
            SA_TRY(model.output(detect_port_name(K_DETECT_ONE_TO_MANY, level, K_DETECT_BOXES),
                                boxes));
            SA_TRY(model.output(detect_port_name(K_DETECT_ONE_TO_MANY, level, K_DETECT_SCORES),
                                scores));
            if (!one_to_one)
                continue;

            SA_TRY_ASSIGN(const NN::NodeRef copy_boxes,
                          model.add("copy_boxes" + suffix, port_convolution(4 * K_REG_MAX),
                                    bodies[level]));
            SA_TRY_ASSIGN(const NN::NodeRef copy_scores,
                          model.add("copy_scores" + suffix, port_convolution(K_CLASSES),
                                    bodies[level]));
            SA_TRY(model.output(detect_port_name(K_DETECT_ONE_TO_ONE, level, K_DETECT_BOXES),
                                copy_boxes));
            SA_TRY(model.output(detect_port_name(K_DETECT_ONE_TO_ONE, level, K_DETECT_SCORES),
                                copy_scores));
        }

        return model;
    }

    /** @brief Returns three truths in pixel corners over two images, padded to two rows. */
    GroundTruthBatch make_truth()
    {
        GroundTruthBatch truth;
        truth.batch = K_BATCH;
        truth.max_boxes = 2;
        truth.labels = {0, 1, 1, 0};
        truth.valid = {1, 1, 1, 0};
        truth.boxes = {0.5f, 0.5f, 3.5f, 3.5f, 2.0f, 1.0f, 8.0f, 7.0f,
                       3.0f, 3.0f, 7.0f, 7.5f, 0.0f, 0.0f, 0.0f, 0.0f};
        return truth;
    }

    /** @brief Returns a [2, 8, 8, 3] tensor of seeded values in (-1, 1). */
    [[nodiscard]] Result<SushiBLAS::Tensor> seeded_inputs(SushiBLAS::Engine& engine)
    {
        SA_TRY_ASSIGN(SushiBLAS::Tensor inputs,
                      allocate_tensor(engine,
                                      Shape{K_BATCH, K_NETWORK_SIDE, K_NETWORK_SIDE, 3}));

        std::vector<double> values(static_cast<std::size_t>(inputs.num_elements));
        SaTest::HostNoise(K_INPUT_SEED).fill(values);
        SaTest::write_tensor(inputs, values);

        return inputs;
    }

    /** @brief Holds a two-level model, its training graph and the step over them. */
    struct TwoLevelCase
    {
        /** @brief Builds everything the step is made from and sets @c ready on success. */
        void prepare(SushiBLAS::Engine& engine, bool one_to_one = false)
        {
            const Shape batch_shape{K_BATCH, K_NETWORK_SIDE, K_NETWORK_SIDE, 3};

            SA_ASSERT_OK_AND_ASSIGN(model, build_two_level_model(one_to_one));
            SA_ASSERT_OK_AND_ASSIGN(const Train::TracedStep probe,
                                    Train::build_probe_graph(*model, batch_shape));
            SA_ASSERT_OK_AND_ASSIGN(const std::vector<Train::OutputShape> outputs,
                                    Train::list_output_shapes(probe.graph, probe.outputs));
            SA_ASSERT_OK_AND_ASSIGN(
                Vision::Training::FreeModelFacts facts,
                Vision::Training::derive_free_model_facts(
                    Shape{K_NETWORK_SIDE, K_NETWORK_SIDE, 3}, outputs));
            SA_ASSERT_OK_AND_ASSIGN(
                objective, FreeObjective::make(std::move(facts), Vision::Detect::Free::FreeTask{},
                                               K_BATCH, Vision::Training::FreeLossGains{}));

            SA_ASSERT_OK_AND_ASSIGN(built,
                                    Train::build_training_graph(*model, *objective, batch_shape));
            SA_ASSERT_OK_AND_ASSIGN(state, NN::ModelState::create(engine, *model));

            training_pool = std::make_unique<TensorPool>(engine);
            SA_ASSERT_OK_AND_ASSIGN(std::vector<Optim::ParameterSlot> slots,
                                    Train::bind_parameters(built, *state, *training_pool));

            // The last weight stays drawn, so every port starts away from zero.
            NN::InitPolicy policy;
            policy.zero_final_layer = false;
            SA_EXPECT_OK(NN::initialize(engine, *state, policy));

            SA_ASSERT_OK_AND_ASSIGN(
                updates, Train::bind_buffer_updates(built.updates, *state, *training_pool));

            Optim::SGDOptions options;
            options.learning_rate = K_LEARNING_RATE;
            SA_ASSERT_OK_AND_ASSIGN(optimizer, Optim::SGD::make(engine, slots, options));

            SA_ASSERT_OK_AND_ASSIGN(
                step, FreeTrainingStep::make(engine, *objective, built, *training_pool,
                                             *optimizer, std::move(updates)));
            ready = true;
        }

        /** @brief Returns every state entry's values, in the state's order. */
        std::vector<std::vector<double>> weights()
        {
            std::vector<std::vector<double>> values;
            for (const NN::StateEntry& entry : state->entries())
            {
                const Result<SushiBLAS::Tensor*> found = state->tensor(entry.name);
                SA_EXPECT_OK(found);
                values.push_back(found.has_value() ? SaTest::read_tensor(*found.value())
                                                   : std::vector<double>{});
            }
            return values;
        }

        /** @brief Returns the values of the leaf declared as @p role. */
        std::vector<double> leaf_values(std::string_view role)
        {
            return SaTest::read_tensor(training_pool->tensor(built.leaves.at(role)));
        }

        std::optional<NN::ModelGraph> model;          /**< @brief Holds the two-level model. */
        std::optional<FreeObjective> objective;       /**< @brief Holds what the step is given. */
        Train::TrainingGraph built;                   /**< @brief Holds the training graph. */
        std::optional<NN::ModelState> state;          /**< @brief Holds the one shared state. */
        std::unique_ptr<TensorPool> training_pool;    /**< @brief Holds the training values. */
        std::vector<Train::BufferUpdateSlot> updates; /**< @brief Holds the running updates. */
        std::unique_ptr<Optim::SGD> optimizer;        /**< @brief Holds the plain SGD. */
        std::unique_ptr<FreeTrainingStep> step;       /**< @brief Holds the step under test. */
        bool ready = false;                           /**< @brief Tells whether prepare() ended. */
    };

    /** @brief Runs two steps of @p model on the seeded inputs and the fixture truth. */
    void run_two_steps(TwoLevelCase& model, SushiBLAS::Engine& engine,
                       std::vector<FreeLossTerms>& terms)
    {
        SA_ASSERT_OK_AND_ASSIGN(const SushiBLAS::Tensor inputs, seeded_inputs(engine));
        const GroundTruthBatch truth = make_truth();
        for (int run = 0; run < 2; ++run)
        {
            SA_ASSERT_OK_AND_ASSIGN(const FreeLossTerms measured, model.step->run(inputs, truth));
            terms.push_back(measured);
        }
    }
} // namespace

/** @brief Tests that two runs of one seed end two steps with equal weights and terms. */
TEST_F(Medium_FreeTrainingStep, TwoRunsOfOneSeedEndWithEqualWeights)
{
    TwoLevelCase first;
    first.prepare(*engine);
    ASSERT_TRUE(first.ready);
    const std::vector<std::vector<double>> start = first.weights();
    std::vector<FreeLossTerms> first_terms;
    run_two_steps(first, *engine, first_terms);
    ASSERT_EQ(first_terms.size(), 2u);

    TwoLevelCase second;
    second.prepare(*engine);
    ASSERT_TRUE(second.ready);
    std::vector<FreeLossTerms> second_terms;
    run_two_steps(second, *engine, second_terms);
    ASSERT_EQ(second_terms.size(), 2u);

    const std::vector<std::vector<double>> left = first.weights();
    const std::vector<std::vector<double>> right = second.weights();
    ASSERT_EQ(left.size(), right.size());
    ASSERT_EQ(left.size(), start.size());

    double moved = 0.0;
    for (std::size_t entry = 0; entry < left.size(); ++entry)
    {
        ASSERT_EQ(left[entry].size(), right[entry].size()) << "entry " << entry;
        for (std::size_t i = 0; i < left[entry].size(); ++i)
        {
            EXPECT_NEAR(left[entry][i], right[entry][i], K_SAME)
                << "entry " << entry << " element " << i;
            moved = std::fmax(moved, std::fabs(left[entry][i] - start[entry][i]));
        }
    }
    EXPECT_GT(moved, K_MOVED);

    for (std::size_t run = 0; run < 2; ++run)
    {
        EXPECT_TRUE(std::isfinite(first_terms[run].total)) << "run " << run;
        EXPECT_GT(first_terms[run].total, 0.0) << "run " << run;
        EXPECT_NEAR(first_terms[run].total, second_terms[run].total, K_SAME) << "run " << run;
    }
    EXPECT_EQ(first.step->run_count(), 2u);
}

/** @brief Tests that the leaves hold what the stage assigned and the projection's bins. */
TEST_F(Medium_FreeTrainingStep, TheLeavesHoldTheStageAndTheProjection)
{
    TwoLevelCase model;
    model.prepare(*engine);
    ASSERT_TRUE(model.ready);
    std::vector<FreeLossTerms> terms;
    run_two_steps(model, *engine, terms);

    const std::vector<double> sum = model.leaf_values("free.target_scores_sum");
    ASSERT_EQ(sum.size(), 1u);
    EXPECT_NEAR(sum[0],
                static_cast<double>(model.step->stage().targets().target_scores_sum[0]), K_SAME);
    EXPECT_GE(sum[0], 1.0);

    const std::vector<double> projection =
        model.leaf_values(Vision::Training::K_FREE_PROJECTION_ROLE);
    ASSERT_EQ(projection.size(), static_cast<std::size_t>(K_REG_MAX));
    EXPECT_NEAR(projection[0], 0.0, K_SAME);
    EXPECT_NEAR(projection[1], 1.0, K_SAME);
}

/** @brief Tests that a one-to-one copy trains every port and reads both branch weights. */
TEST_F(Medium_FreeTrainingStep, AOneToOneCopyMovesEveryWeightAndReadsBothBranchWeights)
{
    TwoLevelCase model;
    model.prepare(*engine, true);
    ASSERT_TRUE(model.ready);
    EXPECT_FLOAT_EQ(model.step->one_to_many_weight(), 0.8f);
    model.step->set_one_to_many_weight(0.25f);

    const std::vector<std::vector<double>> start = model.weights();
    std::vector<FreeLossTerms> terms;
    run_two_steps(model, *engine, terms);
    const std::vector<std::vector<double>> end = model.weights();
    ASSERT_EQ(start.size(), end.size());

    for (std::size_t entry = 0; entry < end.size(); ++entry)
    {
        double moved = 0.0;
        for (std::size_t i = 0; i < end[entry].size(); ++i)
            moved = std::fmax(moved, std::fabs(end[entry][i] - start[entry][i]));
        EXPECT_GT(moved, 0.0) << model.state->entries()[entry].name;
    }

    const std::vector<double> many =
        model.leaf_values(Vision::Training::K_FREE_ONE_TO_MANY_WEIGHT_ROLE);
    const std::vector<double> one =
        model.leaf_values(Vision::Training::K_FREE_ONE_TO_ONE_WEIGHT_ROLE);
    ASSERT_EQ(many.size(), 1u);
    ASSERT_EQ(one.size(), 1u);
    EXPECT_NEAR(many[0], 0.25, K_SAME);
    EXPECT_NEAR(one[0], 0.75, K_SAME);
    EXPECT_GE(model.leaf_values("free.one_to_one.target_scores_sum")[0], 1.0);
    for (const FreeLossTerms& measured : terms)
        EXPECT_TRUE(std::isfinite(measured.total));
}

/** @brief Tests that a truth of another batch is refused before any replay. */
TEST_F(Medium_FreeTrainingStep, ATruthOfAnotherBatchIsRefused)
{
    TwoLevelCase model;
    model.prepare(*engine);
    ASSERT_TRUE(model.ready);
    SA_ASSERT_OK_AND_ASSIGN(const SushiBLAS::Tensor inputs, seeded_inputs(*engine));

    GroundTruthBatch truth = make_truth();
    truth.batch = 1;

    SA_EXPECT_FAILURE(model.step->run(inputs, truth), StatusCode::Incompatible, "images");
    EXPECT_EQ(model.step->run_count(), 0u);
}

/** @brief Tests that a label outside the score ports' classes is refused by row. */
TEST_F(Medium_FreeTrainingStep, ALabelOutsideTheClassesIsRefused)
{
    TwoLevelCase model;
    model.prepare(*engine);
    ASSERT_TRUE(model.ready);
    SA_ASSERT_OK_AND_ASSIGN(const SushiBLAS::Tensor inputs, seeded_inputs(*engine));

    GroundTruthBatch truth = make_truth();
    truth.labels[1] = K_CLASSES;

    SA_EXPECT_FAILURE(model.step->run(inputs, truth), StatusCode::InvalidDescription, "row 1");
}

#if SA_ENABLE_PROFILER
/** @brief Tests that one step's host phases and device run fall in the period that opened it. */
TEST_F(Medium_FreeTrainingZones, AStepPutsReadbackAssignAndLossInItsPeriod)
{
    if (SA_PROFILE_LEVEL < 1)
        GTEST_SKIP() << "the build compiles every zone away";
    TwoLevelCase model;
    model.prepare(*engine);
    ASSERT_TRUE(model.ready);
    SA_ASSERT_OK_AND_ASSIGN(const SushiBLAS::Tensor inputs, seeded_inputs(*engine));
    const GroundTruthBatch truth = make_truth();

    const SushiProfiler::Registry registry;
    SushiProfiler::SessionStart started = SushiProfiler::Session::start(
        SaTest::following_timeline("train.step", 0U, 1U), registry);
    ASSERT_TRUE(started.ok()) << started.error;
    SushiProfiler::Session& session = *started.session;
    {
        const Profile::TimelineBinding binding(session);
        Profile::mark_period(Profile::Timeline::TrainStep);
        SA_EXPECT_OK(model.step->run(inputs, truth));

        // The next mark ends the period; a period with no end mark is not kept.
        Profile::mark_period(Profile::Timeline::TrainStep);
    }
    const SushiProfiler::SessionStop stopped = session.stop();

    ASSERT_EQ(stopped.record.periods.size(), 1u);
    const std::map<std::string, std::string> groups =
        SaTest::zones_of(stopped.record, stopped.record.periods[0]);
    const std::map<std::string, std::string> expected = {
        {"copy.input", "copy"},     {"device.run", "device"}, {"host.assign", "host"},
        {"host.loss", "host"},      {"host.readback", "host"}};
    EXPECT_EQ(groups, expected);
}
#endif
