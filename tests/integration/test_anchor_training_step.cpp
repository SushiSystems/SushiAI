/**************************************************************************/
/* test_anchor_training_step.cpp                                          */
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
 * @file test_anchor_training_step.cpp
 * @brief Tests on an engine that a two-head model trains one step on the deltas of its heads.
 * @author Mustafa Garip
 */

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>
#include <gtest/gtest.h>
#include <SushiBLAS/tensor/tensor.hpp>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/core/allocation.hpp>
#include <SushiAI/nn/batch_norm2d.hpp>
#include <SushiAI/nn/conv2d.hpp>
#include <SushiAI/nn/init.hpp>
#include <SushiAI/nn/leaky_relu.hpp>
#include <SushiAI/nn/model_graph.hpp>
#include <SushiAI/nn/model_state.hpp>
#include <SushiAI/optim/sgd.hpp>
#include <SushiAI/train/buffer_update_recorder.hpp>
#include <SushiAI/train/steps.hpp>
#include <SushiAI/train/trainer.hpp>
#include <SushiAI/vision/detect/anchor/anchor_layout.hpp>
#include <SushiAI/vision/detect/anchor/head_cost.hpp>
#include <SushiAI/vision/detect/anchor/head_delta.hpp>
#include <SushiAI/vision/detect/anchor/head_options.hpp>
#include <SushiAI/vision/detect/ground_truth.hpp>
#include <SushiAI/vision/training/anchor_objective.hpp>
#include <SushiAI/vision/training/anchor_training_step.hpp>
#include <common/test_status.hpp>
#include <common/test_utils.hpp>

using namespace SushiAI;

// SaTest stays reachable inside a TEST_F body; see tests/integration/README.md.
namespace SaTest = SushiAI::Test;
using Graph::TensorPool;
using Vision::Detect::Anchor::AnchorLayout;
using Vision::Detect::Anchor::HeadCost;
using Vision::Detect::Anchor::HeadDelta;
using Vision::Detect::Anchor::NetworkSize;
using Vision::Detect::GroundTruthBatch;
using Vision::Training::AnchorHead;
using Vision::Training::AnchorObjective;
using Vision::Training::AnchorTrainingStep;

namespace
{
    class Integration_AnchorTrainingStep : public SaTest::EngineTest {};

    /** @brief Holds the images per step. */
    constexpr std::int64_t K_BATCH = 2;

    /** @brief Holds the side of the square network input in pixels. */
    constexpr std::int64_t K_NETWORK_SIDE = 8;

    /** @brief Holds the channels both normalised convolutions produce. */
    constexpr std::int64_t K_BODY_CHANNELS = 8;

    /** @brief Holds the classes of both heads. */
    constexpr std::int64_t K_CLASSES = 2;

    /** @brief Holds the channels of a head: two anchors of 5 + K_CLASSES entries. */
    constexpr std::int64_t K_HEAD_CHANNELS = 14;

    /** @brief Holds the positions of the 4 by 4 grid over both images. */
    constexpr std::int64_t K_FINE_POSITIONS = K_BATCH * 4 * 4;

    /** @brief Holds the step size of the plain SGD the cases train with. */
    constexpr double K_LEARNING_RATE = 0.1;

    /** @brief Seeds the input image. */
    constexpr std::uint64_t K_INPUT_SEED = 20261009ull;

    /** @brief Bounds the difference of a leaf from the delta it is a copy of. */
    constexpr double K_FED_TOLERANCE = 0.0;

    /** @brief Bounds, relative to the value, a double sum of some hundred float32 squares. */
    constexpr double K_COST_TOLERANCE = 1e-9;

    /** @brief Bounds the absolute error of a float32 value of order one after one update. */
    constexpr double K_UPDATE_TOLERANCE = 1e-5;

    /** @brief Holds the least change of a value that counts as having moved. */
    constexpr double K_MOVED = 1e-7;

    /** @brief Holds the least batch mean at which one blend and two blends differ visibly. */
    constexpr double K_VISIBLE_MEAN = 1e-2;

    /** @brief Holds the truths each head takes: A and C go to head 0, B to head 1. */
    constexpr std::int64_t K_ASSIGNMENTS[] = {2, 1};

    /** @brief Returns a 1 by 1 convolution with a bias onto the channels of a head. */
    NN::Conv2d head_convolution()
    {
        NN::Conv2d convolution = NN::Conv2d::same_padded(K_BODY_CHANNELS, K_HEAD_CHANNELS, 1, 1);
        convolution.use_bias = true;
        return convolution;
    }

    /**
     * @brief Builds two normalised stride-2 convolutions with a head behind each.
     * @return A model over [8, 8, 3] that publishes "head.0" on a 4 by 4 grid, "head.1" on a
     *         2 by 2 grid, and "stem", the first convolution before its normalisation.
     */
    [[nodiscard]] Result<NN::ModelGraph> build_two_head_model()
    {
        NN::ModelGraph model(Shape{K_NETWORK_SIDE, K_NETWORK_SIDE, 3});
        const NN::NormOptions norm{};

        SA_TRY_ASSIGN(const NN::NodeRef stem,
                      model.add("stem", NN::Conv2d::same_padded(3, K_BODY_CHANNELS, 3, 2),
                                model.input()));
        SA_TRY_ASSIGN(const NN::NodeRef stem_norm,
                      model.add("stem_norm", NN::BatchNorm2d(K_BODY_CHANNELS, norm), stem));
        SA_TRY_ASSIGN(const NN::NodeRef stem_act,
                      model.add("stem_act", NN::LeakyReLU{0.1}, stem_norm));
        SA_TRY_ASSIGN(const NN::NodeRef fine, model.add("fine", head_convolution(), stem_act));

        SA_TRY_ASSIGN(const NN::NodeRef down,
                      model.add("down",
                                NN::Conv2d::same_padded(K_BODY_CHANNELS, K_BODY_CHANNELS, 3, 2),
                                stem_act));
        SA_TRY_ASSIGN(const NN::NodeRef down_norm,
                      model.add("down_norm", NN::BatchNorm2d(K_BODY_CHANNELS, norm), down));
        SA_TRY_ASSIGN(const NN::NodeRef down_act,
                      model.add("down_act", NN::LeakyReLU{0.1}, down_norm));
        SA_TRY_ASSIGN(const NN::NodeRef coarse,
                      model.add("coarse", head_convolution(), down_act));

        SA_TRY(model.output("head.0", fine));
        SA_TRY(model.output("head.1", coarse));
        SA_TRY(model.output("stem", stem));

        return model;
    }

    /** @brief Returns four anchors as fractions of the input, two for each head. */
    AnchorLayout two_head_layout()
    {
        AnchorLayout layout;
        layout.anchors = {{0.2f, 0.2f}, {0.4f, 0.3f}, {0.5f, 0.6f}, {0.9f, 0.8f}};
        layout.head_anchors = {{0, 1}, {2, 3}};
        return layout;
    }

    /** @brief Returns the two heads under the names @p fine and @p coarse. */
    std::vector<AnchorHead> two_heads(std::string fine, std::string coarse)
    {
        std::vector<AnchorHead> heads(2);
        heads[0].output = std::move(fine);
        heads[0].shape = {K_BATCH, 4, 4};
        heads[0].options.num_classes = K_CLASSES;
        heads[1].output = std::move(coarse);
        heads[1].shape = {K_BATCH, 2, 2};
        heads[1].options.num_classes = K_CLASSES;
        return heads;
    }

    /**
     * @brief Returns three truths in pixel corners over two images, padded to two rows.
     *
     * A is 0.25 by 0.25 of the input and nearest anchor 0; B is 0.875 by 0.75 and nearest
     * anchor 3; C is 0.25 by 0.375 and nearest anchor 1. Image 1 holds C and a padding row.
     */
    GroundTruthBatch make_truth()
    {
        GroundTruthBatch truth;
        truth.batch = K_BATCH;
        truth.max_boxes = 2;
        truth.labels = {0, 1, 1, 0};
        truth.valid = {1, 1, 1, 0};
        truth.boxes = {1.0f, 1.0f, 3.0f, 3.0f, 1.0f, 1.0f, 8.0f, 7.0f,
                       4.0f, 4.0f, 6.0f, 7.0f, 0.0f, 0.0f, 0.0f, 0.0f};
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

    /** @brief Holds a two-head model, both of its graphs and the step over them. */
    struct TwoHeadCase
    {
        /**
         * @brief Builds everything the step is made from and sets @c ready on success.
         * @param heads The heads the objective describes.
         */
        void prepare(SushiBLAS::Engine& engine, std::vector<AnchorHead> heads)
        {
            const Shape batch_shape{K_BATCH, K_NETWORK_SIDE, K_NETWORK_SIDE, 3};

            SA_ASSERT_OK_AND_ASSIGN(model, build_two_head_model());
            SA_ASSERT_OK_AND_ASSIGN(
                objective, AnchorObjective::make(NetworkSize{K_NETWORK_SIDE, K_NETWORK_SIDE},
                                                 two_head_layout(), std::move(heads)));
            SA_ASSERT_OK_AND_ASSIGN(probe, Train::build_probe_graph(*model, batch_shape));

            // The graph is built with the names the model publishes, whatever @p heads says.
            SA_ASSERT_OK_AND_ASSIGN(
                const AnchorObjective trained,
                AnchorObjective::make(NetworkSize{K_NETWORK_SIDE, K_NETWORK_SIDE},
                                      two_head_layout(), two_heads("head.0", "head.1")));
            SA_ASSERT_OK_AND_ASSIGN(built,
                                    Train::build_training_graph(*model, trained, batch_shape));
            SA_ASSERT_OK_AND_ASSIGN(state, NN::ModelState::create(engine, *model));

            probe_pool = std::make_unique<TensorPool>(engine);
            training_pool = std::make_unique<TensorPool>(engine);
            SA_ASSERT_OK_AND_ASSIGN(std::vector<Optim::ParameterSlot> slots,
                                    Train::bind_parameters(built, *state, *training_pool));

            // The last weight stays drawn, so both heads start away from zero.
            NN::InitPolicy policy;
            policy.zero_final_layer = false;
            SA_EXPECT_OK(NN::initialize(engine, *state, policy));

            SA_ASSERT_OK_AND_ASSIGN(
                updates, Train::bind_buffer_updates(built.updates, *state, *training_pool));

            Optim::SGDOptions options;
            options.learning_rate = K_LEARNING_RATE;
            SA_ASSERT_OK_AND_ASSIGN(optimizer, Optim::SGD::make(engine, slots, options));

            ready = true;
        }

        /** @brief Makes the step over the prepared parts; @c step stays null on a failure. */
        void make_step(SushiBLAS::Engine& engine)
        {
            SA_ASSERT_OK_AND_ASSIGN(
                step, AnchorTrainingStep::make(engine, *objective, probe, *probe_pool, built,
                                               *training_pool, *state, *optimizer,
                                               std::move(updates)));
        }

        /** @brief Returns the float32 elements of the forward-only output @p name. */
        std::vector<float> forward_output(const std::string& name) const
        {
            const Result<const SushiBLAS::Tensor*> found = step->evaluation().output(name);
            SA_EXPECT_OK(found);
            if (!found.has_value())
                return {};

            const float* const values = found.value()->data_as<float>();
            return std::vector<float>(values, values + found.value()->num_elements);
        }

        /** @brief Returns the elements of the state entry @p name, or none when it is absent. */
        std::vector<double> state_values(const std::string& name)
        {
            const Result<SushiBLAS::Tensor*> found = state->tensor(name);
            SA_EXPECT_OK(found);
            return found.has_value() ? SaTest::read_tensor(*found.value())
                                     : std::vector<double>{};
        }

        std::optional<NN::ModelGraph> model;        /**< @brief Holds the two-head model. */
        std::optional<AnchorObjective> objective;   /**< @brief Holds what the step is given. */
        Train::TracedStep probe;                    /**< @brief Holds the forward-only graph. */
        Train::TrainingGraph built;                 /**< @brief Holds the training graph. */
        std::optional<NN::ModelState> state;        /**< @brief Holds the one shared state. */
        std::unique_ptr<TensorPool> probe_pool;     /**< @brief Holds the forward-only values. */
        std::unique_ptr<TensorPool> training_pool;  /**< @brief Holds the training values. */
        std::vector<Train::BufferUpdateSlot> updates; /**< @brief Holds the running updates. */
        std::unique_ptr<Optim::SGD> optimizer;      /**< @brief Holds the plain SGD. */
        std::unique_ptr<AnchorTrainingStep> step;   /**< @brief Holds the step under test. */
        bool ready = false;                         /**< @brief Tells whether prepare() ended. */
    };
} // namespace

/** @brief Tests that each leaf holds the delta of its forward-only head, element for element. */
TEST_F(Integration_AnchorTrainingStep, TheLeavesFedAreTheDeltasOfTheForwardOnlyHeads)
{
    TwoHeadCase model;
    model.prepare(*engine, two_heads("head.0", "head.1"));
    ASSERT_TRUE(model.ready);
    model.make_step(*engine);
    ASSERT_NE(model.step, nullptr);

    SA_ASSERT_OK_AND_ASSIGN(const SushiBLAS::Tensor inputs, seeded_inputs(*engine));
    const GroundTruthBatch truth = make_truth();

    SA_ASSERT_OK_AND_ASSIGN(const HeadCost cost, model.step->run(inputs, truth));

    double squares = 0.0;
    for (std::size_t head = 0; head < 2; ++head)
    {
        const AnchorHead& described = model.objective->heads()[head];
        const std::vector<float> forward = model.forward_output(described.output);

        HeadDelta expected;
        Vision::Detect::Anchor::compute_head_delta(
            SushiRuntime::span<const float>(forward.data(), forward.size()), described.shape,
            described.options, model.objective->layout(), static_cast<std::int64_t>(head),
            described.training, truth, model.objective->network(), expected);

        const std::vector<double> fed = SaTest::read_tensor(model.training_pool->tensor(
            model.built.leaves.at(model.objective->leaf_role(head))));

        EXPECT_EQ(expected.assignments, K_ASSIGNMENTS[head]) << "head " << head;
        ASSERT_EQ(fed.size(), expected.delta.size()) << "head " << head;
        for (std::size_t i = 0; i < fed.size(); ++i)
        {
            const double value = static_cast<double>(expected.delta[i]);
            EXPECT_NEAR(fed[i], value, K_FED_TOLERANCE) << "head " << head << " element " << i;
            squares += value * value;
        }
    }

    // A squared-error head stores the sum of its squared deltas, and the total divides by 2.
    EXPECT_GT(squares, 0.0);
    EXPECT_NEAR(cost.stored, squares, K_COST_TOLERANCE * squares);
    EXPECT_NEAR(cost.total, squares / static_cast<double>(K_BATCH), K_COST_TOLERANCE * squares);
}

/** @brief Tests that one step moves a head's bias by rate / batch times the delta's sum. */
TEST_F(Integration_AnchorTrainingStep, OneStepMovesAHeadBiasAlongItsDelta)
{
    TwoHeadCase model;
    model.prepare(*engine, two_heads("head.0", "head.1"));
    ASSERT_TRUE(model.ready);
    model.make_step(*engine);
    ASSERT_NE(model.step, nullptr);

    SA_ASSERT_OK_AND_ASSIGN(const SushiBLAS::Tensor inputs, seeded_inputs(*engine));
    const std::vector<double> before = model.state_values("fine.bias");
    ASSERT_EQ(before.size(), static_cast<std::size_t>(K_HEAD_CHANNELS));

    SA_EXPECT_OK(model.step->run(inputs, make_truth()));

    const std::vector<double> after = model.state_values("fine.bias");
    ASSERT_EQ(after.size(), before.size());
    const std::vector<float>& delta = model.step->deltas()[0].delta;
    ASSERT_EQ(delta.size(), static_cast<std::size_t>(K_FINE_POSITIONS * K_HEAD_CHANNELS));

    // The gradient at a bias is -(1 / batch) times its channel's deltas over every position.
    double largest_move = 0.0;
    for (std::size_t channel = 0; channel < before.size(); ++channel)
    {
        double sum = 0.0;
        for (std::size_t position = 0; position < K_FINE_POSITIONS; ++position)
            sum += static_cast<double>(delta[position * K_HEAD_CHANNELS + channel]);

        const double expected =
            before[channel] + K_LEARNING_RATE * sum / static_cast<double>(K_BATCH);
        EXPECT_NEAR(after[channel], expected, K_UPDATE_TOLERANCE) << "channel " << channel;
        largest_move = std::max(largest_move, std::abs(after[channel] - before[channel]));
    }
    EXPECT_GT(largest_move, K_MOVED) << "no bias of head.0 moved";
}

/** @brief Tests that the two passes of one step blend a running mean once, not twice. */
TEST_F(Integration_AnchorTrainingStep, OneStepBlendsARunningMeanOnce)
{
    TwoHeadCase model;
    model.prepare(*engine, two_heads("head.0", "head.1"));
    ASSERT_TRUE(model.ready);
    model.make_step(*engine);
    ASSERT_NE(model.step, nullptr);

    SA_ASSERT_OK_AND_ASSIGN(const SushiBLAS::Tensor inputs, seeded_inputs(*engine));
    const std::vector<double> before = model.state_values("stem_norm.running_mean");
    ASSERT_EQ(before.size(), static_cast<std::size_t>(K_BODY_CHANNELS));

    SA_EXPECT_OK(model.step->run(inputs, make_truth()));

    const std::vector<double> after = model.state_values("stem_norm.running_mean");
    ASSERT_EQ(after.size(), before.size());
    const std::vector<float> stem = model.forward_output("stem");
    ASSERT_EQ(stem.size(), static_cast<std::size_t>(K_FINE_POSITIONS * K_BODY_CHANNELS));

    const double momentum = NN::NormOptions{}.momentum;
    double largest_mean = 0.0;
    for (std::size_t channel = 0; channel < before.size(); ++channel)
    {
        double mean = 0.0;
        for (std::size_t position = 0; position < K_FINE_POSITIONS; ++position)
            mean += static_cast<double>(stem[position * K_BODY_CHANNELS + channel]);
        mean /= static_cast<double>(K_FINE_POSITIONS);

        const double expected = (1.0 - momentum) * before[channel] + momentum * mean;
        EXPECT_NEAR(after[channel], expected, K_UPDATE_TOLERANCE) << "channel " << channel;
        largest_mean = std::max(largest_mean, std::abs(mean));
    }

    // A second blend would add momentum * (1 - momentum) * mean, 0.09 of the mean.
    EXPECT_GT(largest_mean, K_VISIBLE_MEAN) << "the batch means are too small to tell";
}

/** @brief Tests that a truth of another batch is Incompatible, with both batch sizes. */
TEST_F(Integration_AnchorTrainingStep, ATruthOfAnotherBatchIsRefused)
{
    TwoHeadCase model;
    model.prepare(*engine, two_heads("head.0", "head.1"));
    ASSERT_TRUE(model.ready);
    model.make_step(*engine);
    ASSERT_NE(model.step, nullptr);

    SA_ASSERT_OK_AND_ASSIGN(const SushiBLAS::Tensor inputs, seeded_inputs(*engine));
    GroundTruthBatch truth;
    truth.batch = 3;

    SA_EXPECT_FAILURE(model.step->run(inputs, truth), StatusCode::Incompatible, "3 images",
                      "batch of 2");
}

/** @brief Tests that a valid row whose label names no class is InvalidDescription. */
TEST_F(Integration_AnchorTrainingStep, ALabelOutsideTheClassesIsRefused)
{
    TwoHeadCase model;
    model.prepare(*engine, two_heads("head.0", "head.1"));
    ASSERT_TRUE(model.ready);
    model.make_step(*engine);
    ASSERT_NE(model.step, nullptr);

    SA_ASSERT_OK_AND_ASSIGN(const SushiBLAS::Tensor inputs, seeded_inputs(*engine));
    GroundTruthBatch truth = make_truth();
    truth.labels[2] = K_CLASSES;

    SA_EXPECT_FAILURE(model.step->run(inputs, truth), StatusCode::InvalidDescription,
                      "row 2", "label 2", "head.0");
}

/** @brief Tests that an objective naming a head the model lacks is refused by that name. */
TEST_F(Integration_AnchorTrainingStep, AnObjectiveNamingAnUnpublishedHeadIsRefused)
{
    TwoHeadCase model;
    model.prepare(*engine, two_heads("head.0", "head.9"));
    ASSERT_TRUE(model.ready);

    SA_EXPECT_FAILURE(
        AnchorTrainingStep::make(*engine, *model.objective, model.probe, *model.probe_pool,
                                 model.built, *model.training_pool, *model.state,
                                 *model.optimizer, std::move(model.updates)),
        StatusCode::InvalidDescription, "head.9");
}
