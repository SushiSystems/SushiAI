/**************************************************************************/
/* test_yolov7.cpp                                                        */
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
 * @file test_yolov7.cpp
 * @brief Tests the head deltas of YOLOv7 and one training step of it under batch statistics against
 *        what Darknet dumped of both.
 * @author Mustafa Garip
 */

#include <gtest/gtest.h>
#include <SushiAI/vision/darknet_training_cases/darknet_training.hpp>
#include <SushiAI/vision/detect/anchor/head_training_options.hpp>

namespace SaTest = SushiAI::Test;

namespace
{
    /**
     * @brief States what yolov7.cfg and its two training runs hold (REPORT 21 and 22).
     *
     * Of the inference-mode runs only the `*_plain_pool*` ones are valid (REPORT 25 and 28).
     * `yolov7_train_step` is a training-mode run, which the defect does not reach.
     * A delta is held to 1e-6 absolute (PLAN 4.7), a box entry to 1e-4 besides (REPORT 11).
     * A step bound the 2026-10-13 run exceeded is 1.5 times its largest difference (README).
     */
    SaTest::DarknetTrainingCase describe_model()
    {
        SaTest::DarknetTrainingCase training;
        training.stem = "yolov7";
        training.host_run = "yolov7_plain_pool_thresh_0p01";
        training.step_run = "yolov7_train_step";
        training.box_loss = SushiAI::Vision::Detect::Anchor::BoxLoss::CompleteIoU;
        training.objectness_target = SushiAI::Vision::Detect::Anchor::ObjectnessTarget::BestIoU;
        training.bounds.host_delta = {1e-6, 1e-4, 1e-6};
        training.bounds.host_cost_relative = 1e-5;
        training.bounds.head_absolute = 7.9e-3;
        training.bounds.head_relative = 1e-4;
        training.bounds.yolo_absolute = 7.9e-3;
        training.bounds.step_delta = {1.5e-3, 1.5e-4, 1.6e-3};
        training.bounds.step_cost_relative = 1e-3;
        training.bounds.head_update_share = 1e-3;
        training.bounds.update_relative = 1e-3;
        training.bounds.rolling_absolute = 1e-5;
        training.overfit_steps = 200;
        training.overfit_cost_share = 0.1;
        return training;
    }

    /** @brief Runs the training suite of YOLOv7 on an engine; skipped without its files. */
    class Large_DarknetYolov7Training : public SaTest::DarknetTrainingTest
    {
        protected:
            /** @brief Hands the fixture what this file states of the model. */
            Large_DarknetYolov7Training() : SaTest::DarknetTrainingTest(describe_model())
            {

            }
    };
} // namespace

/**
 * @brief Tests that the options read from the cfg are the box loss and the objectness target
 *        this file states, and the weights and thresholds the dump's descriptions hold.
 */
TEST_F(Large_DarknetYolov7Training, TheTrainingOptionsMatchTheDumpedSettings)
{
    SaTest::check_training_settings(training);
}

/**
 * @brief Tests that the delta computed from each dumped head of one image has Darknet's
 *        assignments exactly, its delta within the host bounds and its stored cost to 1e-5.
 */
TEST_F(Large_DarknetYolov7Training, TheDeltaOfADumpedHeadMatchesDarknet)
{
    SaTest::check_host_deltas(training);
}

/**
 * @brief Tests that the delta computed from each dumped head of the two-image step equals the
 *        step's dumped delta within the host bounds, and its stored cost to 1e-5.
 */
TEST_F(Large_DarknetYolov7Training, TheDeltaOfATwoImageHeadMatchesDarknet)
{
    SaTest::check_two_image_deltas(training);
}

/**
 * @brief Tests that each head of the Separate model under batch statistics, and the [yolo]
 *        output computed from it, equal the dump within the head and the [yolo] bounds.
 */
TEST_F(Large_DarknetYolov7Training, TheHeadsUnderBatchStatisticsMatchDarknet)
{
    SaTest::check_step_heads(*engine, training);
}

/**
 * @brief Tests that the deltas the step fed equal the dumped ones within the step bounds, and
 *        that the summed cost and its mean over the heads equal the layer and network costs.
 *
 * The step bounds do not cover a slot that crosses the ignore threshold.
 */
TEST_F(Large_DarknetYolov7Training, TheDeltasOfTheStepMatchDarknet)
{
    SaTest::check_step_deltas(*engine, training);
}

/**
 * @brief Tests that the weight and bias gradients of the convolution before each head, times
 *        minus the batch, equal the dumped updates within the head update share of the
 *        dump's largest magnitude plus 1e-3 of the entry.
 */
TEST_F(Large_DarknetYolov7Training, TheUpdatesOfTheHeadConvolutionsMatchDarknet)
{
    SaTest::check_head_updates(*engine, training);
}

/**
 * @brief Tests that the weight, the scale and the shift of the first convolution each have a
 *        finite gradient that is not zero after one step.
 *
 * The dumped updates of that layer are not the gradient and are not compared; see
 * tests/vision/importers/README.md.
 */
TEST_F(Large_DarknetYolov7Training, TheFirstConvolutionReceivesAGradient)
{
    SaTest::check_first_gradients(*engine, training);
}

/**
 * @brief Tests that the rolling mean and variance of the first batch norm equal the dump
 *        before the step and are within 1e-5 of it after one blend.
 */
TEST_F(Large_DarknetYolov7Training, TheRollingStatisticsAfterOneStepMatchDarknet)
{
    SaTest::check_rolling_statistics(*engine, training);
}

/**
 * @brief Tests that 200 steps on the batch and truth of the step run bring the stored cost of
 *        the Separate model below 0.1 of the first step's.
 */
TEST_F(Large_DarknetYolov7Training, TheCostOfOneBatchFallsUnderATenth)
{
    SaTest::check_one_batch_overfit(*engine, training);
}

/**
 * @brief Tests that the memory plans of the probe graph and the training graph overlap live
 *        ranges; prints each graph's nodes and the bytes its plan asks for.
 */
TEST_F(Large_DarknetYolov7Training, ThePlansOfTheStepOverlapLiveRanges)
{
    SaTest::check_training_plan(training);
}
