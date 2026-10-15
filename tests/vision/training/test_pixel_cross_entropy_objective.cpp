/**************************************************************************/
/* test_pixel_cross_entropy_objective.cpp                                 */
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
 * @file test_pixel_cross_entropy_objective.cpp
 * @brief Tests the per-pixel cross-entropy objective: its leaves, graph and factory, and its loss
 *        and gradient on an engine.
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
#include <SushiAI/nn/batch_norm2d.hpp>
#include <SushiAI/nn/conv2d.hpp>
#include <SushiAI/nn/init.hpp>
#include <SushiAI/nn/model_state.hpp>
#include <SushiAI/nn/module.hpp>
#include <SushiAI/nn/trace.hpp>
#include <SushiAI/optim/optimizer.hpp>
#include <SushiAI/optim/sgd.hpp>
#include <SushiAI/tensor_values/tensor_values.hpp>
#include <SushiAI/train/buffer_update_recorder.hpp>
#include <SushiAI/train/metrics.hpp>
#include <SushiAI/train/objective.hpp>
#include <SushiAI/train/objective_context.hpp>
#include <SushiAI/train/steps.hpp>
#include <SushiAI/train/sum_objective.hpp>
#include <SushiAI/train/trainer.hpp>
#include <SushiAI/train/training_graph.hpp>
#include <SushiAI/train/training_step.hpp>
#include <SushiAI/vision/blocks/conv_norm.hpp>
#include <SushiAI/vision/blocks/segment_head.hpp>
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

    /** @brief Names the objective block in a diagnostic. */
    constexpr std::string_view K_CONTEXT = "the objective block";

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

    /** @brief Returns what a factory is given for the test model over @p classes classes. */
    Train::ObjectiveContext make_context(std::int64_t classes)
    {
        Train::ObjectiveContext given;
        given.batch = K_EXTENTS.batch;
        given.classes = classes;
        given.input_sample = Shape{K_EXTENTS.height, K_EXTENTS.width, K_INPUT_CHANNELS};
        given.outputs.push_back(Train::OutputShape{
            "output",
            Shape{K_EXTENTS.batch, K_EXTENTS.height, K_EXTENTS.width, K_EXTENTS.classes}});

        return given;
    }

    /** @brief Parses @p text as the keys of an objective block. */
    Config::Value keys(std::string_view text)
    {
        return Config::parse_json(text, "<test>");
    }

} // namespace

/** @brief Tests that make() keeps what it was given and refuses an extent below 1. */
TEST(Medium_PixelCrossEntropyObjective, MakeKeepsItsExtentsAndRefusesNone)
{
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<PixelCrossEntropyObjective> objective,
                            PixelCrossEntropyObjective::make(K_EXTENTS, 7));

    EXPECT_EQ(objective->extents().pixels(), K_PIXELS);
    EXPECT_EQ(objective->extents().classes, 3);
    EXPECT_EQ(objective->ignore_index(), 7);
    EXPECT_EQ(objective->name(), "pixel_cross_entropy");

    SA_EXPECT_FAILURE(PixelCrossEntropyObjective::make(PixelExtents{2, 0, 4, 3}),
                      StatusCode::InvalidDescription, "pixel_cross_entropy", "at least 1");
    SA_EXPECT_FAILURE(DiceObjective::make(PixelExtents{2, 4, 4, 0}),
                      StatusCode::InvalidDescription, "dice", "at least 1");
}

/** @brief Tests the ignore index that names a class and the element type that is no real. */
TEST(Medium_PixelCrossEntropyObjective, MakeRefusesAnIgnoreIndexThatIsAClass)
{
    SA_EXPECT_FAILURE(PixelCrossEntropyObjective::make(K_EXTENTS, 2),
                      StatusCode::InvalidDescription, "ignore index 2", "3 classes");
    SA_EXPECT_OK(PixelCrossEntropyObjective::make(K_EXTENTS, 3));
    SA_EXPECT_OK(PixelCrossEntropyObjective::make(K_EXTENTS, -1));
    SA_EXPECT_FAILURE(DiceObjective::make(K_EXTENTS, 0), StatusCode::InvalidDescription,
                      "ignore index 0");
    SA_EXPECT_FAILURE(PixelCrossEntropyObjective::make(K_EXTENTS, 255, DType::Int32),
                      StatusCode::InvalidDescription, "float32");
}

/** @brief Tests the two leaves of the cross entropy and the mask role it reads. */
TEST(Medium_PixelCrossEntropyObjective, TheCrossEntropyDeclaresIndicesAndWeights)
{
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<PixelCrossEntropyObjective> objective,
                            PixelCrossEntropyObjective::make(K_EXTENTS));
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    Train::ObjectiveLeaves leaves;

    objective->declare(builder, leaves);

    ASSERT_EQ(leaves.all().size(), 2u);
    EXPECT_EQ(graph.value(leaves.at("targets")).shape, (Shape{K_PIXELS}));
    EXPECT_EQ(graph.value(leaves.at("targets")).dtype, DType::Int32);
    EXPECT_EQ(graph.value(leaves.at("weights")).shape, (Shape{K_PIXELS, 1}));
    EXPECT_EQ(graph.value(leaves.at("weights")).dtype, DType::Float32);

    ASSERT_EQ(leaves.required().size(), 1u);
    const Data::TargetDescription& mask = leaves.required().front();
    EXPECT_EQ(mask.role, Data::K_ROLE_MASK);
    EXPECT_EQ(mask.sample_shape, (Shape{4, 4}));
    EXPECT_EQ(mask.dtype, DType::Int32);
    EXPECT_EQ(mask.classes, 0);
}

/** @brief Tests that the cross entropy is one element built from one gather and no one-hot. */
TEST(Medium_PixelCrossEntropyObjective, TheCrossEntropyGathersOneLogProbabilityPerPixel)
{
    NN::Conv2d model = make_model();
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<PixelCrossEntropyObjective> objective,
                            PixelCrossEntropyObjective::make(K_EXTENTS));

    SA_ASSERT_OK_AND_ASSIGN(const Train::TrainingGraph built,
                            Train::build_training_graph(model, *objective, input_shape()));

    EXPECT_EQ(built.graph.value(built.spec.loss).shape.num_elements(), 1);
    EXPECT_EQ(built.graph.value(built.spec.targets).dtype, DType::Int32);
    EXPECT_EQ(count_nodes(built.graph, Ops::K_NLL_LOSS.id), 0u);
    EXPECT_EQ(count_nodes(built.graph, Ops::K_LOG_SOFTMAX.id), 1u);
    EXPECT_EQ(count_nodes(built.graph, Ops::K_GATHER_ROWS.id), 1u);
    ASSERT_EQ(built.parameters.size(), 2u);
    EXPECT_NE(built.parameters[0].gradient, Graph::K_NO_VALUE);
    EXPECT_NE(built.parameters[1].gradient, Graph::K_NO_VALUE);
}

/** @brief Tests that both factories read N, H, W and C from the model's one output. */
TEST(Medium_PixelCrossEntropyObjective, TheFactoriesReadTheModelsOutput)
{
    SA_ASSERT_OK_AND_ASSIGN(
        const std::unique_ptr<Train::Objective> cross_entropy,
        make_pixel_cross_entropy_objective(make_context(3), keys("{}"), K_CONTEXT));
    SA_ASSERT_OK_AND_ASSIGN(
        const std::unique_ptr<Train::Objective> dice,
        make_dice_objective(make_context(0), keys(R"({"ignore_index": 9})"), K_CONTEXT));

    EXPECT_EQ(cross_entropy->name(), "pixel_cross_entropy");
    const auto* const made = dynamic_cast<const PixelCrossEntropyObjective*>(cross_entropy.get());
    ASSERT_NE(made, nullptr);
    EXPECT_EQ(made->extents().pixels(), K_PIXELS);
    EXPECT_EQ(made->ignore_index(), 255);

    const auto* const made_dice = dynamic_cast<const DiceObjective*>(dice.get());
    ASSERT_NE(made_dice, nullptr);
    EXPECT_EQ(made_dice->ignore_index(), 9);
}

/** @brief Tests the keys a factory refuses: an unknown one, a real and an index of a class. */
TEST(Medium_PixelCrossEntropyObjective, TheFactoriesRefuseABadSection)
{
    SA_EXPECT_FAILURE(
        make_pixel_cross_entropy_objective(make_context(3), keys(R"({"weight": 1})"), K_CONTEXT),
        StatusCode::InvalidDescription, "the objective block", "'weight'", "'ignore_index'");
    SA_EXPECT_FAILURE(
        make_dice_objective(make_context(3), keys(R"({"ignore_index": 0.5})"), K_CONTEXT),
        StatusCode::InvalidDescription, "'ignore_index'", "whole number");
    SA_EXPECT_FAILURE(
        make_pixel_cross_entropy_objective(make_context(3), keys(R"({"ignore_index": 1})"),
                                           K_CONTEXT),
        StatusCode::InvalidDescription, "ignore index 1");
}

/** @brief Tests the models a factory refuses, and data of more classes than the model scores. */
TEST(Medium_PixelCrossEntropyObjective, TheFactoriesRefuseAModelThatDoesNotFit)
{
    Train::ObjectiveContext two_outputs = make_context(3);
    two_outputs.outputs.push_back(Train::OutputShape{"second", Shape{2, 4, 4, 3}});
    Train::ObjectiveContext flat = make_context(3);
    flat.outputs.front().shape = Shape{2, 3};
    Train::ObjectiveContext half_size = make_context(3);
    half_size.outputs.front().shape = Shape{2, 2, 2, 3};

    SA_EXPECT_FAILURE(make_pixel_cross_entropy_objective(two_outputs, keys("{}"), K_CONTEXT),
                      StatusCode::InvalidDescription, "one [N, H, W, classes] output",
                      "publishes 2");
    SA_EXPECT_FAILURE(make_dice_objective(flat, keys("{}"), K_CONTEXT),
                      StatusCode::InvalidDescription, "'output'", "[2, 3]");
    SA_EXPECT_FAILURE(make_pixel_cross_entropy_objective(half_size, keys("{}"), K_CONTEXT),
                      StatusCode::InvalidDescription, "[2, 2, 2, 3]", "[4, 4, 5]",
                      "input size");
    SA_EXPECT_FAILURE(make_dice_objective(make_context(4), keys("{}"), K_CONTEXT),
                      StatusCode::InvalidDescription, "holds 4 classes", "scores 3");
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

    /** @brief Holds the pixels of mask_labels() that are not K_IGNORE. */
    constexpr double K_KEPT = 4.0;

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

    /** @brief Returns the mean of -log p(label) over the pixels that are not K_IGNORE. */
    double cross_entropy_on_host(const std::vector<double>& logits,
                                 const std::vector<std::int32_t>& labels)
    {
        double total = 0.0;
        double kept = 0.0;
        for (std::size_t pixel = 0; pixel < K_HOST_PIXELS; ++pixel)
        {
            if (labels[pixel] == K_IGNORE)
                continue;

            const std::size_t label = static_cast<std::size_t>(labels[pixel]);
            total -= std::log(probabilities_of(logits, pixel)[label]);
            kept += 1.0;
        }

        return total / kept;
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
    class Medium_PixelCrossEntropyObjectiveEndToEnd : public SaTest::EngineTest
    {
    };

} // namespace

/** @brief Tests the cross entropy and its gradient at the logits, two pixels ignored. */
TEST_F(Medium_PixelCrossEntropyObjectiveEndToEnd, TheCrossEntropyAndItsGradientEqualTheHostComputation)
{
    SA_ASSERT_OK_AND_ASSIGN(
        const std::unique_ptr<Vision::Training::PixelCrossEntropyObjective> objective,
        Vision::Training::PixelCrossEntropyObjective::make(K_HOST_EXTENTS, K_IGNORE));
    const std::vector<double> logits = make_logits();
    const std::vector<std::int32_t> labels = mask_labels();

    SA_ASSERT_OK_AND_ASSIGN(const Found found, run_step(*engine, *objective, logits, labels));

    EXPECT_NEAR(found.loss, cross_entropy_on_host(logits, labels), K_TOLERANCE);

    // The gradient is (softmax - one_hot) / kept on a kept pixel and zero on an ignored one.
    ASSERT_EQ(found.gradient.size(), K_HOST_PIXELS * K_CLASSES);
    for (std::size_t pixel = 0; pixel < K_HOST_PIXELS; ++pixel)
    {
        const std::vector<double> probabilities = probabilities_of(logits, pixel);
        for (std::size_t column = 0; column < K_CLASSES; ++column)
        {
            const bool is_kept = labels[pixel] != K_IGNORE;
            const double hit = labels[pixel] == static_cast<std::int32_t>(column) ? 1.0 : 0.0;
            const double expected = is_kept ? (probabilities[column] - hit) / K_KEPT : 0.0;
            EXPECT_NEAR(found.gradient[pixel * K_CLASSES + column], expected, K_TOLERANCE)
                << "pixel " << pixel << ", class " << column;
        }
    }
}

/** @brief Tests that a sum feeds both terms: the loss and its gradient are the weighted sums. */
TEST_F(Medium_PixelCrossEntropyObjectiveEndToEnd, AWeightedSumOfBothLossesEqualsTheHostComputation)
{
    // The weight of the Dice term, as a training file would state it.
    constexpr double K_DICE_WEIGHT = 0.5;

    SA_ASSERT_OK_AND_ASSIGN(
        std::unique_ptr<Vision::Training::PixelCrossEntropyObjective> cross_entropy,
        Vision::Training::PixelCrossEntropyObjective::make(K_HOST_EXTENTS, K_IGNORE));
    SA_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Vision::Training::DiceObjective> dice,
                            Vision::Training::DiceObjective::make(K_HOST_EXTENTS, K_IGNORE));
    std::vector<Train::SumTerm> terms;
    terms.push_back(Train::SumTerm{std::move(cross_entropy), 1.0});
    terms.push_back(Train::SumTerm{std::move(dice), K_DICE_WEIGHT});
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<Train::SumObjective> objective,
                            Train::SumObjective::make(std::move(terms)));
    const std::vector<double> logits = make_logits();
    const std::vector<std::int32_t> labels = mask_labels();

    SA_ASSERT_OK_AND_ASSIGN(const Found found, run_step(*engine, *objective, logits, labels));

    const HostLoss both = [](const std::vector<double>& scores,
                             const std::vector<std::int32_t>& classes)
    {
        return cross_entropy_on_host(scores, classes)
             + K_DICE_WEIGHT * dice_on_host(scores, classes);
    };
    EXPECT_NEAR(found.loss, both(logits, labels), K_TOLERANCE);

    const std::vector<double> expected = gradient_on_host(both, logits, labels);
    ASSERT_EQ(found.gradient.size(), expected.size());
    for (std::size_t index = 0; index < expected.size(); ++index)
        EXPECT_NEAR(found.gradient[index], expected[index], K_TOLERANCE) << "element " << index;
}

/** @brief Tests that a mask of ignored pixels alone gives a loss of zero and no gradient. */
TEST_F(Medium_PixelCrossEntropyObjectiveEndToEnd, AMaskWithNoKeptPixelGivesZero)
{
    SA_ASSERT_OK_AND_ASSIGN(
        const std::unique_ptr<Vision::Training::PixelCrossEntropyObjective> objective,
        Vision::Training::PixelCrossEntropyObjective::make(K_HOST_EXTENTS, K_IGNORE));
    const std::vector<std::int32_t> labels(K_HOST_PIXELS, K_IGNORE);

    SA_ASSERT_OK_AND_ASSIGN(const Found found,
                            run_step(*engine, *objective, make_logits(), labels));

    EXPECT_NEAR(found.loss, 0.0, K_TOLERANCE);
    for (const double element : found.gradient)
        EXPECT_NEAR(element, 0.0, K_TOLERANCE);
}

/** @brief Tests that a label at the class count stops the feed and names its pixel. */
TEST_F(Medium_PixelCrossEntropyObjectiveEndToEnd, ALabelOutsideTheClassesStopsTheFeed)
{
    SA_ASSERT_OK_AND_ASSIGN(
        const std::unique_ptr<Vision::Training::PixelCrossEntropyObjective> objective,
        Vision::Training::PixelCrossEntropyObjective::make(K_HOST_EXTENTS, K_IGNORE));

    SA_EXPECT_FAILURE(run_step(*engine, *objective, make_logits(), {0, 2, 4, 1, 1, 0}),
                      StatusCode::Incompatible, "holds 4 at pixel 2", "below 4");
}

namespace
{
    /** @brief Holds the samples of the batch, N. */
    constexpr std::int64_t K_BATCH = 2;

    /** @brief Holds the height and the width of an image and of its mask. */
    constexpr std::int64_t K_SIDE = 16;

    /** @brief Holds the channels of an image. */
    constexpr std::int64_t K_CHANNELS = 3;

    /** @brief Holds the class count. */
    constexpr std::int64_t K_OVERFIT_CLASSES = 2;

    /** @brief Holds the width of the two convolutions and of the head. */
    constexpr std::int64_t K_WIDTH = 8;

    /** @brief Holds the stride of the first convolution, which the head's upsample undoes. */
    constexpr std::int64_t K_STRIDE = 2;

    /** @brief Holds the passes fit() makes over the one batch; no run has confirmed it. */
    constexpr std::int64_t K_EPOCHS = 80;

    /** @brief Holds the rate of the SGD; no run has confirmed it. */
    constexpr double K_RATE = 0.1;

    /** @brief Holds the fraction of the first loss the last must be below; unconfirmed. */
    constexpr double K_LOSS_FRACTION = 0.5;

    /** @brief Holds the mean IoU the last step must exceed; no run has confirmed it. */
    constexpr double K_MEAN_IOU = 0.8;

    /** @brief Holds the epsilon and the momentum of every batch normalisation. */
    constexpr NN::NormOptions K_NORM{1e-5, 0.1};

    /** @brief Holds a strided convolution, a second one and a head back at the input size. */
    struct TinySegmenter
    {
        /** @brief Holds the 3x3 convolution that halves H and W. */
        Vision::Blocks::Conv stem;

        /** @brief Holds the 3x3 convolution that keeps H and W. */
        Vision::Blocks::Conv body;

        /** @brief Holds the head, which upsamples by the stride of the stem. */
        Vision::Blocks::SegmentHead head;

        /** @brief Builds the three blocks over K_CHANNELS, K_WIDTH and K_OVERFIT_CLASSES. */
        TinySegmenter()
            : stem(K_CHANNELS, K_WIDTH, 3, K_STRIDE, K_NORM),
              body(K_WIDTH, K_WIDTH, 3, 1, K_NORM),
              head(K_WIDTH, K_WIDTH, K_OVERFIT_CLASSES, K_STRIDE, K_NORM)
        {

        }

        /** @brief Lists the three children. */
        static constexpr auto fields()
        {
            return std::tuple{
                NN::child("stem", &TinySegmenter::stem),
                NN::child("body", &TinySegmenter::body),
                NN::child("head", &TinySegmenter::head),
            };
        }

        /** @brief Traces the stem, the body and the head. */
        Graph::ValueID forward(const NN::Trace& trace, Graph::ValueID x)
        {
            x = NN::trace_layer(stem, trace, x);
            x = NN::trace_layer(body, trace, x);

            return NN::trace_layer(head, trace, x);
        }

        /** @brief Computes the per-sample shape forward() produces. */
        Shape output_shape(const Shape& input) const
        {
            return head.output_shape(body.output_shape(stem.output_shape(input)));
        }
    };

    /** @brief Returns the description of one K_SIDE by K_SIDE float32 image. */
    Data::InputDescription image_input()
    {
        return Data::InputDescription{Shape{K_SIDE, K_SIDE, K_CHANNELS}, DType::Float32};
    }

    /** @brief Returns the mask role: int32 [K_SIDE, K_SIDE] over K_OVERFIT_CLASSES classes. */
    std::vector<Data::TargetDescription> mask_roles()
    {
        return {Data::TargetDescription{std::string(Data::K_ROLE_MASK), Shape{K_SIDE, K_SIDE},
                                        DType::Int32, K_OVERFIT_CLASSES}};
    }

    /**
     * @brief Returns the class of a pixel: sample 0 is split left and right, sample 1 above
     *        and below, both at K_SIDE / 2, which the 2 by 2 blocks of the head can state.
     */
    std::int32_t class_of(std::int64_t sample, std::int64_t row, std::int64_t column)
    {
        const std::int64_t position = sample == 0 ? column : row;

        return position < K_SIDE / 2 ? 0 : 1;
    }

    /**
     * @brief Returns a dataset of one batch: channel 0 of an image is 1 where the class is 1,
     *        channel 1 is 0.5, and the two corners of each mask are K_IGNORE.
     */
    Result<Data::MaterialisedDataset> make_dataset(SushiBLAS::Engine& engine)
    {
        SA_TRY_ASSIGN(Data::Batch batch,
                      Data::allocate_batch(engine, K_BATCH, image_input(), mask_roles()));

        std::vector<double> pixels;
        std::int32_t* const mask = batch.targets.front().values.data_as<std::int32_t>();
        std::size_t pixel = 0;
        for (std::int64_t sample = 0; sample < K_BATCH; ++sample)
        {
            for (std::int64_t row = 0; row < K_SIDE; ++row)
            {
                for (std::int64_t column = 0; column < K_SIDE; ++column, ++pixel)
                {
                    const std::int32_t label = class_of(sample, row, column);
                    const bool is_corner = row == column && (row == 0 || row == K_SIDE - 1);
                    mask[pixel] = is_corner ? K_IGNORE : label;
                    pixels.push_back(static_cast<double>(label));
                    pixels.push_back(0.5);
                    pixels.push_back(0.0);
                }
            }
        }
        SaTest::write_tensor(batch.inputs, pixels);

        std::vector<Data::Batch> batches;
        batches.push_back(std::move(batch));

        return Data::MaterialisedDataset(std::move(batches), K_BATCH, image_input(), mask_roles(),
                                         "halves");
    }

    /** @brief Provides the engine the state, the dataset and the step are built on. */
    class Medium_SegmentOverfit : public SaTest::EngineTest
    {
    };
} // namespace

/** @brief Tests that the loss falls below half its start and the mean IoU rises above 0.8. */
TEST_F(Medium_SegmentOverfit, TwoConvolutionsAndAHeadFitOneBatch)
{
    TinySegmenter model;
    SA_ASSERT_OK_AND_ASSIGN(
        const std::unique_ptr<Vision::Training::PixelCrossEntropyObjective> objective,
        Vision::Training::PixelCrossEntropyObjective::make(
            Vision::Training::PixelExtents{K_BATCH, K_SIDE, K_SIDE, K_OVERFIT_CLASSES}, K_IGNORE));
    SA_ASSERT_OK_AND_ASSIGN(
        Train::TrainingGraph built,
        Train::build_training_graph(model, *objective,
                                    Shape{K_BATCH, K_SIDE, K_SIDE, K_CHANNELS}));

    SA_ASSERT_OK_AND_ASSIGN(NN::ModelState state, NN::ModelState::create(*engine, model));
    Graph::TensorPool pool(*engine);
    SA_ASSERT_OK_AND_ASSIGN(std::vector<Optim::ParameterSlot> slots,
                            Train::bind_parameters(built, state, pool));
    SA_EXPECT_OK(NN::initialize(*engine, state));
    SA_ASSERT_OK_AND_ASSIGN(std::vector<Train::BufferUpdateSlot> updates,
                            Train::bind_buffer_updates(built.updates, state, pool));

    Optim::SGDOptions sgd;
    sgd.learning_rate = K_RATE;
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<Optim::SGD> optimizer,
                            Optim::SGD::make(*engine, slots, sgd));
    Train::TrainerOptions options;
    options.epochs = K_EPOCHS;
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<Train::Trainer> trainer,
                            Train::Trainer::make(*engine, built, pool, *objective, *optimizer,
                                                 options, std::move(updates)));
    SA_ASSERT_OK_AND_ASSIGN(const Data::MaterialisedDataset dataset, make_dataset(*engine));

    SA_ASSERT_OK_AND_ASSIGN(const Train::RunReport report, trainer->fit(dataset));

    EXPECT_EQ(report.steps, static_cast<std::size_t>(K_EPOCHS));
    EXPECT_EQ(report.compile_count, 1u);
    EXPECT_LT(report.last_loss, K_LOSS_FRACTION * report.first_loss) << report.to_string();

    const Train::Metric* const mean_iou = report.last_metrics.find("mean_iou");
    const Train::Metric* const accuracy = report.last_metrics.find("pixel_accuracy");
    ASSERT_NE(mean_iou, nullptr);
    ASSERT_NE(accuracy, nullptr);
    EXPECT_GT(mean_iou->value, K_MEAN_IOU) << report.to_string();

    // With two classes the accuracy is at least the IoU of each, so at least their mean.
    EXPECT_GE(accuracy->value, mean_iou->value) << report.to_string();
}
