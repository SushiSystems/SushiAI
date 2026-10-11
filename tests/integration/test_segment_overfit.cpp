/**************************************************************************/
/* test_segment_overfit.cpp                                               */
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
 * @file test_segment_overfit.cpp
 * @brief Tests that two convolutions and a SegmentHead fit one batch through Train::Trainer.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <tuple>
#include <utility>
#include <vector>
#include <gtest/gtest.h>
#include <SushiBLAS/engine/engine.hpp>
#include <SushiBLAS/tensor/tensor.hpp>
#include <SushiAI/core/common.hpp>
#include <SushiAI/core/result.hpp>
#include <SushiAI/core/shape.hpp>
#include <SushiAI/core/status.hpp>
#include <SushiAI/data/dataset.hpp>
#include <SushiAI/data/input.hpp>
#include <SushiAI/data/materialised_dataset.hpp>
#include <SushiAI/data/roles.hpp>
#include <SushiAI/data/target.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/graph/tensor_pool.hpp>
#include <SushiAI/nn/batch_norm2d.hpp>
#include <SushiAI/nn/init.hpp>
#include <SushiAI/nn/model_state.hpp>
#include <SushiAI/nn/module.hpp>
#include <SushiAI/nn/trace.hpp>
#include <SushiAI/optim/optimizer.hpp>
#include <SushiAI/optim/sgd.hpp>
#include <SushiAI/train/buffer_update_recorder.hpp>
#include <SushiAI/train/metrics.hpp>
#include <SushiAI/train/steps.hpp>
#include <SushiAI/train/trainer.hpp>
#include <SushiAI/train/training_graph.hpp>
#include <SushiAI/vision/blocks/conv_norm.hpp>
#include <SushiAI/vision/blocks/segment_head.hpp>
#include <SushiAI/vision/training/pixel_cross_entropy_objective.hpp>
#include <SushiAI/vision/training/pixel_extents.hpp>
#include <common/test_status.hpp>
#include <common/test_utils.hpp>

using namespace SushiAI;

// SaTest stays reachable inside a TEST_F body; see tests/integration/README.md.
namespace SaTest = SushiAI::Test;

namespace
{
    /** @brief Holds the samples of the batch, N. */
    constexpr std::int64_t K_BATCH = 2;

    /** @brief Holds the height and the width of an image and of its mask. */
    constexpr std::int64_t K_SIDE = 16;

    /** @brief Holds the channels of an image. */
    constexpr std::int64_t K_CHANNELS = 3;

    /** @brief Holds the class count. */
    constexpr std::int64_t K_CLASSES = 2;

    /** @brief Holds the width of the two convolutions and of the head. */
    constexpr std::int64_t K_WIDTH = 8;

    /** @brief Holds the stride of the first convolution, which the head's upsample undoes. */
    constexpr std::int64_t K_STRIDE = 2;

    /** @brief Holds the mask value of a pixel of no class. */
    constexpr std::int32_t K_IGNORE = 255;

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

        /** @brief Builds the three blocks over K_CHANNELS, K_WIDTH and K_CLASSES. */
        TinySegmenter()
            : stem(K_CHANNELS, K_WIDTH, 3, K_STRIDE, K_NORM),
              body(K_WIDTH, K_WIDTH, 3, 1, K_NORM),
              head(K_WIDTH, K_WIDTH, K_CLASSES, K_STRIDE, K_NORM)
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

    /** @brief Returns the mask role: int32 [K_SIDE, K_SIDE] over K_CLASSES classes. */
    std::vector<Data::TargetDescription> mask_roles()
    {
        return {Data::TargetDescription{std::string(Data::K_ROLE_MASK), Shape{K_SIDE, K_SIDE},
                                        DType::Int32, K_CLASSES}};
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
    class Integration_SegmentOverfit : public SaTest::EngineTest
    {
    };
} // namespace

/** @brief Tests that the loss falls below half its start and the mean IoU rises above 0.8. */
TEST_F(Integration_SegmentOverfit, TwoConvolutionsAndAHeadFitOneBatch)
{
    TinySegmenter model;
    SA_ASSERT_OK_AND_ASSIGN(
        const std::unique_ptr<Vision::Training::PixelCrossEntropyObjective> objective,
        Vision::Training::PixelCrossEntropyObjective::make(
            Vision::Training::PixelExtents{K_BATCH, K_SIDE, K_SIDE, K_CLASSES}, K_IGNORE));
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
