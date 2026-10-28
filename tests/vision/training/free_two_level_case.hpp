/**************************************************************************/
/* free_two_level_case.hpp                                                */
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
 * @file free_two_level_case.hpp
 * @brief Declares the two-level anchor-free model the free step and loop tests share.
 * @author Mustafa Garip
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>
#include <gtest/gtest.h>
#include <SushiBLAS/engine/engine.hpp>
#include <SushiBLAS/tensor/tensor.hpp>
#include <SushiAI/allocation/allocation.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/graph/tensor_pool.hpp>
#include <SushiAI/nn/conv2d.hpp>
#include <SushiAI/nn/init.hpp>
#include <SushiAI/nn/leaky_relu.hpp>
#include <SushiAI/nn/model_graph.hpp>
#include <SushiAI/nn/model_state.hpp>
#include <SushiAI/optim/optimizer.hpp>
#include <SushiAI/optim/sgd.hpp>
#include <SushiAI/seeded_noise/host_noise.hpp>
#include <SushiAI/shape/shape.hpp>
#include <SushiAI/tensor_values/tensor_values.hpp>
#include <SushiAI/train/buffer_update_recorder.hpp>
#include <SushiAI/train/objective_context.hpp>
#include <SushiAI/train/steps.hpp>
#include <SushiAI/train/trainer.hpp>
#include <SushiAI/train/training_graph.hpp>
#include <SushiAI/vision/detect/free/free_task.hpp>
#include <SushiAI/vision/detect/free/head_contract.hpp>
#include <SushiAI/vision/detect/ground_truth.hpp>
#include <SushiAI/vision/training/free_loss_gains.hpp>
#include <SushiAI/vision/training/free_model_facts.hpp>
#include <SushiAI/vision/training/free_objective.hpp>
#include <SushiAI/vision/training/free_training_step.hpp>

/** @brief Holds the two-level anchor-free model the free step and loop tests share. */
namespace SushiAI::FreeTwoLevelTest
{
    /** @brief Holds the images per step. */
    inline constexpr std::int64_t K_BATCH = 2;

    /** @brief Holds the side of the square network input in pixels. */
    inline constexpr std::int64_t K_NETWORK_SIDE = 8;

    /** @brief Holds the channels both stride-2 convolutions produce. */
    inline constexpr std::int64_t K_BODY_CHANNELS = 8;

    /** @brief Holds the classes of the score ports. */
    inline constexpr std::int64_t K_CLASSES = 2;

    /** @brief Holds the distribution bins per box side. */
    inline constexpr std::int64_t K_REG_MAX = 2;

    /** @brief Holds the step size of the plain SGD the cases train with. */
    inline constexpr double K_LEARNING_RATE = 0.01;

    /** @brief Seeds the input image. */
    inline constexpr std::uint64_t K_INPUT_SEED = 20261026ull;

    /** @brief Returns a 1 by 1 convolution with a bias onto @p channels. */
    inline NN::Conv2d port_convolution(std::int64_t channels)
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
    [[nodiscard]] inline Result<NN::ModelGraph> build_two_level_model(bool one_to_one)
    {
        using Vision::Detect::Free::detect_port_name;
        using Vision::Detect::Free::K_DETECT_BOXES;
        using Vision::Detect::Free::K_DETECT_ONE_TO_MANY;
        using Vision::Detect::Free::K_DETECT_ONE_TO_ONE;
        using Vision::Detect::Free::K_DETECT_SCORES;

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
    inline Vision::Detect::GroundTruthBatch make_truth()
    {
        Vision::Detect::GroundTruthBatch truth;
        truth.batch = K_BATCH;
        truth.max_boxes = 2;
        truth.labels = {0, 1, 1, 0};
        truth.valid = {1, 1, 1, 0};
        truth.boxes = {0.5f, 0.5f, 3.5f, 3.5f, 2.0f, 1.0f, 8.0f, 7.0f,
                       3.0f, 3.0f, 7.0f, 7.5f, 0.0f, 0.0f, 0.0f, 0.0f};
        return truth;
    }

    /** @brief Returns a [2, 8, 8, 3] tensor of seeded values in (-1, 1). */
    [[nodiscard]] inline Result<SushiBLAS::Tensor> seeded_inputs(SushiBLAS::Engine& engine)
    {
        SA_TRY_ASSIGN(SushiBLAS::Tensor inputs,
                      allocate_tensor(engine,
                                      Shape{K_BATCH, K_NETWORK_SIDE, K_NETWORK_SIDE, 3}));

        std::vector<double> values(static_cast<std::size_t>(inputs.num_elements));
        Test::HostNoise(K_INPUT_SEED).fill(values);
        Test::write_tensor(inputs, values);

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
                objective,
                Vision::Training::FreeObjective::make(std::move(facts),
                                                      Vision::Detect::Free::FreeTask{}, K_BATCH,
                                                      Vision::Training::FreeLossGains{}));

            SA_ASSERT_OK_AND_ASSIGN(built,
                                    Train::build_training_graph(*model, *objective, batch_shape));
            SA_ASSERT_OK_AND_ASSIGN(state, NN::ModelState::create(engine, *model));

            training_pool = std::make_unique<Graph::TensorPool>(engine);
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
                step, Vision::Training::FreeTrainingStep::make(engine, *objective, built,
                                                               *training_pool, *optimizer,
                                                               std::move(updates)));
            ready = true;
        }

        /** @brief Releases what holds tensors of the engine; call it before the engine goes. */
        void release()
        {
            step.reset();
            optimizer.reset();
            training_pool.reset();
            state.reset();
        }

        /** @brief Returns every state entry's values, in the state's order. */
        std::vector<std::vector<double>> weights()
        {
            std::vector<std::vector<double>> values;
            for (const NN::StateEntry& entry : state->entries())
            {
                const Result<SushiBLAS::Tensor*> found = state->tensor(entry.name);
                SA_EXPECT_OK(found);
                values.push_back(found.has_value() ? Test::read_tensor(*found.value())
                                                   : std::vector<double>{});
            }
            return values;
        }

        /** @brief Returns the values of the leaf declared as @p role. */
        std::vector<double> leaf_values(std::string_view role)
        {
            return Test::read_tensor(training_pool->tensor(built.leaves.at(role)));
        }

        /** @brief Holds the two-level model. */
        std::optional<NN::ModelGraph> model;
        /** @brief Holds what the step is given. */
        std::optional<Vision::Training::FreeObjective> objective;
        /** @brief Holds the training graph. */
        Train::TrainingGraph built;
        /** @brief Holds the one shared state. */
        std::optional<NN::ModelState> state;
        /** @brief Holds the training values. */
        std::unique_ptr<Graph::TensorPool> training_pool;
        /** @brief Holds the running updates. */
        std::vector<Train::BufferUpdateSlot> updates;
        /** @brief Holds the plain SGD. */
        std::unique_ptr<Optim::SGD> optimizer;
        /** @brief Holds the step under test. */
        std::unique_ptr<Vision::Training::FreeTrainingStep> step;
        /** @brief Tells whether prepare() ended. */
        bool ready = false;
    };
} // namespace SushiAI::FreeTwoLevelTest
