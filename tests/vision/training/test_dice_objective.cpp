/**************************************************************************/
/* test_dice_objective.cpp                                                */
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
 * @file test_dice_objective.cpp
 * @brief Tests the Dice objective: its leaves, graph and factory, and its loss and gradient on an
 *        engine.
 * @author Mustafa Garip
 */

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <tuple>
#include <utility>
#include <vector>
#include <gtest/gtest.h>
#include <SushiBLAS/engine/engine.hpp>
#include <SushiBLAS/tensor/tensor.hpp>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/SushiAI.hpp>
#include <SushiAI/config/value.hpp>
#include <SushiAI/core/common.hpp>
#include <SushiAI/core/ops.hpp>
#include <SushiAI/core/result.hpp>
#include <SushiAI/core/shape.hpp>
#include <SushiAI/core/status.hpp>
#include <SushiAI/data/dataset.hpp>
#include <SushiAI/data/input.hpp>
#include <SushiAI/data/materialised_dataset.hpp>
#include <SushiAI/data/roles.hpp>
#include <SushiAI/data/target.hpp>
#include <SushiAI/engine_fixture/engine_test.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/graph/builder.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/graph/tensor_pool.hpp>
#include <SushiAI/nn/conv2d.hpp>
#include <SushiAI/nn/model_state.hpp>
#include <SushiAI/nn/module.hpp>
#include <SushiAI/optim/optimizer.hpp>
#include <SushiAI/optim/sgd.hpp>
#include <SushiAI/tensor_values/tensor_values.hpp>
#include <SushiAI/train/objective.hpp>
#include <SushiAI/train/objective_context.hpp>
#include <SushiAI/train/steps.hpp>
#include <SushiAI/train/sum_objective.hpp>
#include <SushiAI/train/trainer.hpp>
#include <SushiAI/train/training_graph.hpp>
#include <SushiAI/train/training_step.hpp>
#include <SushiAI/vision/training/dice_objective.hpp>
#include <SushiAI/vision/training/pixel_cross_entropy_objective.hpp>
#include <SushiAI/vision/training/pixel_extents.hpp>
#include <SushiAI/vision/training/pixel_losses.hpp>

using namespace SushiAI;
using Vision::Training::DiceObjective;
using Vision::Training::make_dice_objective;
using Vision::Training::make_pixel_cross_entropy_objective;
using Vision::Training::PixelCrossEntropyObjective;
using Vision::Training::PixelExtents;
// SaTest stays reachable inside a TEST_F body; see tests/vision/training/README.md.
namespace SaTest = SushiAI::Test;

namespace
{
    /** @brief Holds the extents of the logits the cases score: N, H, W and C. */
    constexpr PixelExtents K_EXTENTS{2, 4, 4, 3};

    /** @brief Holds the channels of the input the test model reads. */
    constexpr std::int64_t K_INPUT_CHANNELS = 5;

    /** @brief Holds the pixels of a batch, N * H * W. */
    constexpr std::int64_t K_PIXELS = 32;

    /** @brief Returns a biased 1x1 convolution from K_INPUT_CHANNELS to C channels. */
    NN::Conv2d make_model()
    {
        NN::Conv2d model;
        model.in_channels = K_INPUT_CHANNELS;
        model.out_channels = K_EXTENTS.classes;

        return model;
    }

    /** @brief Returns the shape of the batch the test model reads. */
    Shape input_shape()
    {
        return Shape{K_EXTENTS.batch, K_EXTENTS.height, K_EXTENTS.width, K_INPUT_CHANNELS};
    }

    /** @brief Counts the nodes of @p graph whose operation is @p op. */
    std::size_t count_nodes(const Graph::Graph& graph, OpID op)
    {
        std::size_t count = 0;
        for (const Graph::Node& node : graph.nodes())
        {
            if (node.op == op)
                ++count;
        }

        return count;
    }

} // namespace

/** @brief Tests the five leaves of the Dice objective, none of them [R, C]. */
TEST(Unit_DiceObjective, TheDiceDeclaresTwoIndexLeavesAndNoOneHot)
{
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<DiceObjective> objective,
                            DiceObjective::make(K_EXTENTS));
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    Train::ObjectiveLeaves leaves;

    objective->declare(builder, leaves);

    ASSERT_EQ(leaves.all().size(), 5u);
    EXPECT_FALSE(leaves.has("one_hot"));
    EXPECT_EQ(graph.value(leaves.at("score_indices")).shape, (Shape{K_PIXELS}));
    EXPECT_EQ(graph.value(leaves.at("score_indices")).dtype, DType::Int32);
    EXPECT_EQ(graph.value(leaves.at("class_indices")).shape, (Shape{K_PIXELS}));
    EXPECT_EQ(graph.value(leaves.at("class_indices")).dtype, DType::Int32);
    for (const auto& leaf : leaves.all())
        EXPECT_LE(graph.value(leaf.second).shape.num_elements(), K_PIXELS) << leaf.first;
    EXPECT_EQ(graph.value(leaves.at("kept")).shape, (Shape{K_PIXELS, 1}));
    EXPECT_EQ(graph.value(leaves.at("class_pixels")).shape, (Shape{3}));
    EXPECT_EQ(graph.value(leaves.at("class_weights")).shape, (Shape{3}));
    ASSERT_EQ(leaves.required().size(), 1u);
    EXPECT_EQ(leaves.required().front().role, Data::K_ROLE_MASK);
}

/** @brief Tests that the Dice loss is one element whose gradient crosses the scatter-add. */
TEST(Unit_DiceObjective, TheDiceLossIsDifferentiated)
{
    NN::Conv2d model = make_model();
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<DiceObjective> objective,
                            DiceObjective::make(K_EXTENTS));

    SA_ASSERT_OK_AND_ASSIGN(const Train::TrainingGraph built,
                            Train::build_training_graph(model, *objective, input_shape()));

    EXPECT_EQ(built.graph.value(built.spec.loss).shape.num_elements(), 1);
    EXPECT_EQ(built.spec.targets, Graph::K_NO_VALUE);

    // The forward pass emits one of each; each backward rule appends the other operation.
    EXPECT_EQ(count_nodes(built.graph, Ops::K_GATHER_ROWS.id), 2u);
    EXPECT_EQ(count_nodes(built.graph, Ops::K_SCATTER_ADD_ROWS.id), 2u);
    ASSERT_EQ(built.parameters.size(), 2u);
    EXPECT_NE(built.parameters[0].gradient, Graph::K_NO_VALUE);
    EXPECT_NE(built.parameters[1].gradient, Graph::K_NO_VALUE);
}

namespace
{
    /** @brief Holds the extents of the logits: one 2 by 3 map of four classes. */
    constexpr PixelExtents K_HOST_EXTENTS{1, 2, 3, 4};

    /** @brief Holds the pixels of the map, R. */
    constexpr std::size_t K_HOST_PIXELS = 6;

    /** @brief Holds the class count, C. */
    constexpr std::size_t K_CLASSES = 4;

    /** @brief Holds the mask value of a pixel of no class. */
    constexpr std::int32_t K_IGNORE = 255;

    /** @brief Bounds the absolute error of a float32 loss or gradient of order one. */
    constexpr double K_TOLERANCE = 1e-5;

    /** @brief Holds the step of the central difference a host gradient is taken with. */
    constexpr double K_DIFFERENCE_STEP = 1e-4;

    /** @brief Names a host loss: the R * C logits and the R labels give one number. */
    using HostLoss =
        std::function<double(const std::vector<double>&, const std::vector<std::int32_t>&)>;

    /** @brief Returns the class of each pixel: two are ignored and class 3 is absent. */
    std::vector<std::int32_t> mask_labels()
    {
        return {0, 2, K_IGNORE, 1, 1, K_IGNORE};
    }

    /** @brief Returns the R * C logits: element i holds ((7 i) mod 11) / 4 - 1. */
    std::vector<double> make_logits()
    {
        std::vector<double> logits;
        for (std::size_t index = 0; index < K_HOST_PIXELS * K_CLASSES; ++index)
            logits.push_back(static_cast<double>((7 * index) % 11) / 4.0 - 1.0);

        return logits;
    }

    /** @brief Returns the softmax of the C logits of pixel @p pixel. */
    std::vector<double> probabilities_of(const std::vector<double>& logits, std::size_t pixel)
    {
        std::vector<double> probabilities(K_CLASSES);
        double total = 0.0;
        for (std::size_t column = 0; column < K_CLASSES; ++column)
        {
            probabilities[column] = std::exp(logits[pixel * K_CLASSES + column]);
            total += probabilities[column];
        }
        for (double& probability : probabilities)
            probability /= total;

        return probabilities;
    }

    /** @brief Returns the mean over the present classes of 1 - 2 I / (P + Y + eps). */
    double dice_on_host(const std::vector<double>& logits,
                        const std::vector<std::int32_t>& labels)
    {
        std::vector<double> hits(K_CLASSES, 0.0);
        std::vector<double> mass(K_CLASSES, 0.0);
        std::vector<double> pixels(K_CLASSES, 0.0);
        for (std::size_t pixel = 0; pixel < K_HOST_PIXELS; ++pixel)
        {
            if (labels[pixel] == K_IGNORE)
                continue;

            const std::vector<double> probabilities = probabilities_of(logits, pixel);
            const std::size_t label = static_cast<std::size_t>(labels[pixel]);
            hits[label] += probabilities[label];
            pixels[label] += 1.0;
            for (std::size_t column = 0; column < K_CLASSES; ++column)
                mass[column] += probabilities[column];
        }

        double total = 0.0;
        double present = 0.0;
        for (std::size_t column = 0; column < K_CLASSES; ++column)
        {
            if (pixels[column] == 0.0)
                continue;

            total += 1.0 - 2.0 * hits[column]
                         / (mass[column] + pixels[column] + Vision::Training::K_DICE_EPSILON);
            present += 1.0;
        }

        return total / present;
    }

    /** @brief Returns the central difference of @p loss at every logit. */
    std::vector<double> gradient_on_host(const HostLoss& loss, const std::vector<double>& logits,
                                         const std::vector<std::int32_t>& labels)
    {
        std::vector<double> gradient(logits.size());
        std::vector<double> moved = logits;
        for (std::size_t index = 0; index < logits.size(); ++index)
        {
            moved[index] = logits[index] + K_DIFFERENCE_STEP;
            const double above = loss(moved, labels);
            moved[index] = logits[index] - K_DIFFERENCE_STEP;
            const double below = loss(moved, labels);
            moved[index] = logits[index];
            gradient[index] = (above - below) / (2.0 * K_DIFFERENCE_STEP);
        }

        return gradient;
    }

    /** @brief Adds a learnable map to its input, so the gradient of the map is the logits'. */
    struct LogitMap
    {
        /** @brief Holds the map, of the shape of the input. */
        NN::Parameter logits{};

        /** @brief Lists the one parameter. */
        static constexpr auto fields()
        {
            return std::tuple{NN::field("logits", &LogitMap::logits)};
        }

        /** @brief Traces the sum of @p x and the map. */
        Graph::ValueID forward(const Graph::GraphBuilder& builder, Graph::ValueID x)
        {
            // By value: declaring the parameter grows the value table.
            const Shape shape = builder.graph().value(x).shape;
            const Graph::ValueID map =
                logits.declare(builder, shape, NN::ParameterRole::Weight, 1, 1, "logits");

            return builder.add(x, map);
        }

        /** @brief Returns @p input, which the sum keeps. */
        Shape output_shape(const Shape& input) const { return input; }
    };

    /** @brief Holds what one step left on the host. */
    struct Found
    {
        double loss = 0.0;            /**< The loss the step computed. */
        std::vector<double> gradient; /**< The gradient of the loss at every logit. */
    };

    /** @brief Returns a batch of zeros whose mask role holds @p labels. */
    Result<Data::Batch> make_batch(SushiBLAS::Engine& engine,
                                   const std::vector<std::int32_t>& labels)
    {
        const Data::InputDescription input{
            Shape{K_HOST_EXTENTS.height, K_HOST_EXTENTS.width, K_HOST_EXTENTS.classes}, DType::Float32};
        const std::vector<Data::TargetDescription> roles{
            Data::TargetDescription{std::string(Data::K_ROLE_MASK),
                                    Shape{K_HOST_EXTENTS.height, K_HOST_EXTENTS.width}, DType::Int32,
                                    K_HOST_EXTENTS.classes}};
        SA_TRY_ASSIGN(Data::Batch batch,
                      Data::allocate_batch(engine, K_HOST_EXTENTS.batch, input, roles));

        SaTest::write_tensor(batch.inputs, std::vector<double>(K_HOST_PIXELS * K_CLASSES, 0.0));
        for (std::size_t pixel = 0; pixel < labels.size(); ++pixel)
            batch.targets.front().values.data_as<std::int32_t>()[pixel] = labels[pixel];

        return batch;
    }

    /**
     * @brief Runs one step of @p objective over a LogitMap that holds @p logits.
     * @return The loss and the gradient at the logits, or the first failure; the rate of the
     *         optimizer is zero, so the map is read as it was written.
     */
    Result<Found> run_step(SushiBLAS::Engine& engine, const Train::Objective& objective,
                           const std::vector<double>& logits,
                           const std::vector<std::int32_t>& labels)
    {
        LogitMap model;
        const Shape batch_shape{K_HOST_EXTENTS.batch, K_HOST_EXTENTS.height, K_HOST_EXTENTS.width,
                                K_HOST_EXTENTS.classes};
        SA_TRY_ASSIGN(Train::TrainingGraph built,
                      Train::build_training_graph(model, objective, batch_shape));
        SA_TRY_ASSIGN(NN::ModelState state, NN::ModelState::create(engine, model));
        Graph::TensorPool pool(engine);
        SA_TRY_ASSIGN(std::vector<Optim::ParameterSlot> slots,
                      Train::bind_parameters(built, state, pool));
        SA_FAIL_IF(slots.size() != 1, StatusCode::InvalidDescription,
                   "the map has one parameter; found {} slots", slots.size());

        SA_TRY_ASSIGN(SushiBLAS::Tensor* const map, state.tensor("logits"));
        SaTest::write_tensor(*map, logits);

        Optim::SGDOptions sgd;
        sgd.learning_rate = 0.0;
        SA_TRY_ASSIGN(const std::unique_ptr<Optim::SGD> optimizer,
                      Optim::SGD::make(engine, slots, sgd));
        SA_TRY_ASSIGN(const std::unique_ptr<Train::TrainingStep> step,
                      Train::TrainingStep::make(engine, built.graph, pool, *optimizer, {}));

        SA_TRY_ASSIGN(const Data::Batch batch, make_batch(engine, labels));
        SaTest::write_tensor(pool.tensor(built.spec.inputs),
                             std::vector<double>(K_HOST_PIXELS * K_CLASSES, 0.0));
        SA_TRY(objective.feed(batch, built.leaves, pool));
        SA_TRY(step->run());

        Found found;
        found.loss = SaTest::read_tensor(pool.tensor(built.spec.loss)).front();
        found.gradient = SaTest::read_tensor(slots.front().gradient);

        return found;
    }

    /** @brief Provides the engine the state and the step are built on. */
    class Integration_DiceObjective : public SaTest::EngineTest
    {
    };

} // namespace

/** @brief Tests the Dice loss and its gradient at the logits against central differences. */
TEST_F(Integration_DiceObjective, TheDiceLossAndItsGradientEqualTheHostComputation)
{
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<Vision::Training::DiceObjective> objective,
                            Vision::Training::DiceObjective::make(K_HOST_EXTENTS, K_IGNORE));
    const std::vector<double> logits = make_logits();
    const std::vector<std::int32_t> labels = mask_labels();

    SA_ASSERT_OK_AND_ASSIGN(const Found found, run_step(*engine, *objective, logits, labels));

    EXPECT_NEAR(found.loss, dice_on_host(logits, labels), K_TOLERANCE);

    const std::vector<double> expected = gradient_on_host(dice_on_host, logits, labels);
    ASSERT_EQ(found.gradient.size(), expected.size());
    for (std::size_t index = 0; index < expected.size(); ++index)
        EXPECT_NEAR(found.gradient[index], expected[index], K_TOLERANCE) << "element " << index;
}
