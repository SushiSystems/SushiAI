/**************************************************************************/
/* test_pixel_objectives.cpp                                              */
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
 * @file test_pixel_objectives.cpp
 * @brief Tests the value and the gradient of the two per-pixel losses against a host computation.
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
#include <SushiAI/core/common.hpp>
#include <SushiAI/core/result.hpp>
#include <SushiAI/core/shape.hpp>
#include <SushiAI/core/status.hpp>
#include <SushiAI/data/dataset.hpp>
#include <SushiAI/data/input.hpp>
#include <SushiAI/data/materialised_dataset.hpp>
#include <SushiAI/data/roles.hpp>
#include <SushiAI/data/target.hpp>
#include <SushiAI/graph/builder.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/graph/tensor_pool.hpp>
#include <SushiAI/nn/model_state.hpp>
#include <SushiAI/nn/module.hpp>
#include <SushiAI/optim/optimizer.hpp>
#include <SushiAI/optim/sgd.hpp>
#include <SushiAI/train/objective.hpp>
#include <SushiAI/train/steps.hpp>
#include <SushiAI/train/sum_objective.hpp>
#include <SushiAI/train/trainer.hpp>
#include <SushiAI/train/training_graph.hpp>
#include <SushiAI/train/training_step.hpp>
#include <SushiAI/vision/training/dice_objective.hpp>
#include <SushiAI/vision/training/pixel_cross_entropy_objective.hpp>
#include <SushiAI/vision/training/pixel_extents.hpp>
#include <SushiAI/vision/training/pixel_losses.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/SushiAI.h>
#include <SushiAI/engine_fixture/engine_test.hpp>
#include <SushiAI/tensor_values/tensor_values.hpp>

using namespace SushiAI;
using Vision::Training::PixelExtents;

// SaTest stays reachable inside a TEST_F body; see tests/integration/README.md.
namespace SaTest = SushiAI::Test;

namespace
{
    /** @brief Holds the extents of the logits: one 2 by 3 map of four classes. */
    constexpr PixelExtents K_EXTENTS{1, 2, 3, 4};

    /** @brief Holds the pixels of the map, R. */
    constexpr std::size_t K_PIXELS = 6;

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
        for (std::size_t index = 0; index < K_PIXELS * K_CLASSES; ++index)
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
        for (std::size_t pixel = 0; pixel < K_PIXELS; ++pixel)
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
        for (std::size_t pixel = 0; pixel < K_PIXELS; ++pixel)
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
            Shape{K_EXTENTS.height, K_EXTENTS.width, K_EXTENTS.classes}, DType::Float32};
        const std::vector<Data::TargetDescription> roles{
            Data::TargetDescription{std::string(Data::K_ROLE_MASK),
                                    Shape{K_EXTENTS.height, K_EXTENTS.width}, DType::Int32,
                                    K_EXTENTS.classes}};
        SA_TRY_ASSIGN(Data::Batch batch,
                      Data::allocate_batch(engine, K_EXTENTS.batch, input, roles));

        SaTest::write_tensor(batch.inputs, std::vector<double>(K_PIXELS * K_CLASSES, 0.0));
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
        const Shape batch_shape{K_EXTENTS.batch, K_EXTENTS.height, K_EXTENTS.width,
                                K_EXTENTS.classes};
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
                             std::vector<double>(K_PIXELS * K_CLASSES, 0.0));
        SA_TRY(objective.feed(batch, built.leaves, pool));
        SA_TRY(step->run());

        Found found;
        found.loss = SaTest::read_tensor(pool.tensor(built.spec.loss)).front();
        found.gradient = SaTest::read_tensor(slots.front().gradient);

        return found;
    }

    /** @brief Provides the engine the state and the step are built on. */
    class Integration_PixelObjectives : public SaTest::EngineTest
    {
    };
} // namespace

/** @brief Tests the cross entropy and its gradient at the logits, two pixels ignored. */
TEST_F(Integration_PixelObjectives, TheCrossEntropyAndItsGradientEqualTheHostComputation)
{
    SA_ASSERT_OK_AND_ASSIGN(
        const std::unique_ptr<Vision::Training::PixelCrossEntropyObjective> objective,
        Vision::Training::PixelCrossEntropyObjective::make(K_EXTENTS, K_IGNORE));
    const std::vector<double> logits = make_logits();
    const std::vector<std::int32_t> labels = mask_labels();

    SA_ASSERT_OK_AND_ASSIGN(const Found found, run_step(*engine, *objective, logits, labels));

    EXPECT_NEAR(found.loss, cross_entropy_on_host(logits, labels), K_TOLERANCE);

    // The gradient is (softmax - one_hot) / kept on a kept pixel and zero on an ignored one.
    ASSERT_EQ(found.gradient.size(), K_PIXELS * K_CLASSES);
    for (std::size_t pixel = 0; pixel < K_PIXELS; ++pixel)
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

/** @brief Tests the Dice loss and its gradient at the logits against central differences. */
TEST_F(Integration_PixelObjectives, TheDiceLossAndItsGradientEqualTheHostComputation)
{
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<Vision::Training::DiceObjective> objective,
                            Vision::Training::DiceObjective::make(K_EXTENTS, K_IGNORE));
    const std::vector<double> logits = make_logits();
    const std::vector<std::int32_t> labels = mask_labels();

    SA_ASSERT_OK_AND_ASSIGN(const Found found, run_step(*engine, *objective, logits, labels));

    EXPECT_NEAR(found.loss, dice_on_host(logits, labels), K_TOLERANCE);

    const std::vector<double> expected = gradient_on_host(dice_on_host, logits, labels);
    ASSERT_EQ(found.gradient.size(), expected.size());
    for (std::size_t index = 0; index < expected.size(); ++index)
        EXPECT_NEAR(found.gradient[index], expected[index], K_TOLERANCE) << "element " << index;
}

/** @brief Tests that a sum feeds both terms: the loss and its gradient are the weighted sums. */
TEST_F(Integration_PixelObjectives, AWeightedSumOfBothLossesEqualsTheHostComputation)
{
    // The weight of the Dice term, as a training file would state it.
    constexpr double K_DICE_WEIGHT = 0.5;

    SA_ASSERT_OK_AND_ASSIGN(
        std::unique_ptr<Vision::Training::PixelCrossEntropyObjective> cross_entropy,
        Vision::Training::PixelCrossEntropyObjective::make(K_EXTENTS, K_IGNORE));
    SA_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Vision::Training::DiceObjective> dice,
                            Vision::Training::DiceObjective::make(K_EXTENTS, K_IGNORE));
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
TEST_F(Integration_PixelObjectives, AMaskWithNoKeptPixelGivesZero)
{
    SA_ASSERT_OK_AND_ASSIGN(
        const std::unique_ptr<Vision::Training::PixelCrossEntropyObjective> objective,
        Vision::Training::PixelCrossEntropyObjective::make(K_EXTENTS, K_IGNORE));
    const std::vector<std::int32_t> labels(K_PIXELS, K_IGNORE);

    SA_ASSERT_OK_AND_ASSIGN(const Found found,
                            run_step(*engine, *objective, make_logits(), labels));

    EXPECT_NEAR(found.loss, 0.0, K_TOLERANCE);
    for (const double element : found.gradient)
        EXPECT_NEAR(element, 0.0, K_TOLERANCE);
}

/** @brief Tests that a label at the class count stops the feed and names its pixel. */
TEST_F(Integration_PixelObjectives, ALabelOutsideTheClassesStopsTheFeed)
{
    SA_ASSERT_OK_AND_ASSIGN(
        const std::unique_ptr<Vision::Training::PixelCrossEntropyObjective> objective,
        Vision::Training::PixelCrossEntropyObjective::make(K_EXTENTS, K_IGNORE));

    SA_EXPECT_FAILURE(run_step(*engine, *objective, make_logits(), {0, 2, 4, 1, 1, 0}),
                      StatusCode::Incompatible, "holds 4 at pixel 2", "below 4");
}
