/**************************************************************************/
/* test_yolov4_tiny.cpp                                                   */
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
 * @file test_yolov4_tiny.cpp
 * @brief Tests the head deltas of YOLOv4-tiny and one training step of it under batch statistics
 *        against what Darknet dumped of both.
 * @author Mustafa Garip
 */

#include <cstdint>
#include <string_view>
#include <gtest/gtest.h>
#include <SushiAI/vision/darknet_training_cases/darknet_training.hpp>
#include <SushiAI/vision/detect/anchor/head_training_options.hpp>
#include <support/tolerance.hpp>

namespace SaTest = SushiAI::Test;

namespace
{
    /** @brief Returns @p count times the float unit roundoff. */
    constexpr double units(double count)
    {
        return count * TestSupport::unit_roundoff<float>();
    }

    /**
     * @brief Holds the share of its first cost that the cost of 200 steps on one batch
     *        must fall under: a tenth, the learning check this suite names.
     */
    constexpr double K_OVERFIT_COST_SHARE = 0.1;

    /**
     * @brief States what yolov4-tiny.cfg and its two training runs hold (REPORT 21 and 22).
     *
     * The model has no stride-1 pool (REPORT section 25), so `yolov4_tiny` is valid.
     * `yolov4_tiny_train_step` is a training-mode run, which the defect does not reach.
     * Bounds are multiples of the float unit roundoff u, each the smallest power of two not
     * tighter than the earlier decimal bound (PLAN 4.7, REPORT 11, importers README).
     */
    SaTest::DarknetTrainingCase describe_model()
    {
        SaTest::DarknetTrainingCase training;
        training.stem = "yolov4-tiny";
        training.host_run = "yolov4_tiny";
        training.step_run = "yolov4_tiny_train_step";
        training.box_loss = SushiAI::Vision::Detect::Anchor::BoxLoss::CompleteIoU;
        training.objectness_target = SushiAI::Vision::Detect::Anchor::ObjectnessTarget::One;
        training.bounds.host_delta = {units(32), units(2048), units(32)};
        training.bounds.host_cost_relative = units(256);
        training.bounds.head_absolute = units(2097152);
        training.bounds.head_relative = units(2048);
        training.bounds.yolo_absolute = units(262144);
        training.bounds.step_delta = {units(32768), units(4096), units(65536)};
        training.bounds.step_cost_relative = units(32768);
        training.bounds.head_update_share = units(262144);
        training.bounds.update_relative = units(32768);
        training.bounds.rolling_absolute = units(256);
        training.overfit_steps = 200;
        training.overfit_cost_share = K_OVERFIT_COST_SHARE;
        return training;
    }

    /** @brief Holds the images of the step whose training graph is planned. */
    constexpr std::int64_t K_PLAN_BATCH = 2;

    /**
     * @brief Holds a compact cfg with the layer kinds and the head keys of YOLOv4-tiny.
     *
     * The input is 64 by 64 with one class, so the training graph is planned on the
     * host from this text alone; it is not the yolov4-tiny.cfg of Darknet.
     */
    constexpr std::string_view K_PLAN_CFG =
        "[net]\n"
        "batch=2\nsubdivisions=1\nwidth=64\nheight=64\nchannels=3\nmomentum=0.949\n"
        "decay=0.0005\nlearning_rate=0.001\nburn_in=1000\nmax_batches=500500\n"
        "policy=steps\nsteps=400000,450000\nscales=.1,.1\n"
        "[convolutional]\nbatch_normalize=1\nfilters=8\n"
        "size=3\nstride=2\npad=1\nactivation=leaky\n"
        "[convolutional]\nbatch_normalize=1\nfilters=16\n"
        "size=3\nstride=2\npad=1\nactivation=leaky\n"
        "[convolutional]\nbatch_normalize=1\nfilters=16\n"
        "size=3\nstride=1\npad=1\nactivation=leaky\n"
        "[route]\nlayers=-1\ngroups=2\ngroup_id=1\n"
        "[convolutional]\nbatch_normalize=1\nfilters=8\n"
        "size=3\nstride=1\npad=1\nactivation=leaky\n"
        "[convolutional]\nbatch_normalize=1\nfilters=8\n"
        "size=3\nstride=1\npad=1\nactivation=leaky\n"
        "[route]\nlayers=-1,-2\n"
        "[convolutional]\nbatch_normalize=1\nfilters=16\n"
        "size=1\nstride=1\npad=1\nactivation=leaky\n"
        "[route]\nlayers=-6,-1\n"
        "[maxpool]\nsize=2\nstride=2\n"
        "[convolutional]\nbatch_normalize=1\nfilters=32\n"
        "size=3\nstride=1\npad=1\nactivation=leaky\n"
        "[convolutional]\nbatch_normalize=1\nfilters=16\n"
        "size=1\nstride=1\npad=1\nactivation=leaky\n"
        "[convolutional]\nbatch_normalize=1\nfilters=32\n"
        "size=3\nstride=1\npad=1\nactivation=leaky\n"
        "[convolutional]\nfilters=18\n"
        "size=1\nstride=1\npad=1\nactivation=linear\n"
        "[yolo]\nmask=3,4,5\nanchors=4,5, 8,10, 12,16, 20,24, 28,36, 44,52\nclasses=1\nnum=6\n"
        "scale_x_y=1.05\ncls_normalizer=1.0\niou_normalizer=0.07\niou_loss=ciou\n"
        "ignore_thresh=.7\ntruth_thresh=1\nnms_kind=greedynms\nbeta_nms=0.6\n"
        "[route]\nlayers=-4\n"
        "[convolutional]\nbatch_normalize=1\nfilters=8\n"
        "size=1\nstride=1\npad=1\nactivation=leaky\n"
        "[upsample]\nstride=2\n"
        "[route]\nlayers=-1,7\n"
        "[convolutional]\nbatch_normalize=1\nfilters=16\n"
        "size=3\nstride=1\npad=1\nactivation=leaky\n"
        "[convolutional]\nfilters=18\n"
        "size=1\nstride=1\npad=1\nactivation=linear\n"
        "[yolo]\nmask=0,1,2\nanchors=4,5, 8,10, 12,16, 20,24, 28,36, 44,52\nclasses=1\nnum=6\n"
        "scale_x_y=1.05\ncls_normalizer=1.0\niou_normalizer=0.07\niou_loss=ciou\n"
        "ignore_thresh=.7\ntruth_thresh=1\nnms_kind=greedynms\nbeta_nms=0.6\n";

    /** @brief Runs the training suite of YOLOv4-tiny on an engine; skipped without its files. */
    class Medium_DarknetYolov4TinyTraining : public SaTest::DarknetTrainingTest
    {
        protected:
            /** @brief Hands the fixture what this file states of the model. */
            Medium_DarknetYolov4TinyTraining() : SaTest::DarknetTrainingTest(describe_model())
            {

            }
    };
} // namespace

/**
 * @brief Tests that the options read from the cfg are the box loss and the objectness target
 *        this file states, and the weights and thresholds the dump's descriptions hold.
 */
TEST_F(Medium_DarknetYolov4TinyTraining, TheTrainingOptionsMatchTheDumpedSettings)
{
    SaTest::check_training_settings(training);
}

/**
 * @brief Tests that the delta computed from each dumped head of one image has Darknet's
 *        assignments exactly, its delta within the host bounds and its stored cost to 256 u.
 */
TEST_F(Medium_DarknetYolov4TinyTraining, TheDeltaOfADumpedHeadMatchesDarknet)
{
    SaTest::check_host_deltas(training);
}

/**
 * @brief Tests that the delta computed from each dumped head of the two-image step equals the
 *        step's dumped delta within the host bounds, and its stored cost to 256 u.
 */
TEST_F(Medium_DarknetYolov4TinyTraining, TheDeltaOfATwoImageHeadMatchesDarknet)
{
    SaTest::check_two_image_deltas(training);
}

/**
 * @brief Tests that each head of the Separate model under batch statistics, and the [yolo]
 *        output computed from it, equal the dump within the head and the [yolo] bounds.
 */
TEST_F(Medium_DarknetYolov4TinyTraining, TheHeadsUnderBatchStatisticsMatchDarknet)
{
    SaTest::check_step_heads(backend(), training);
}

/**
 * @brief Tests that the deltas the step fed equal the dumped ones within the step bounds, and
 *        that the summed cost and its mean over the heads equal the layer and network costs.
 *
 * The step bounds do not cover a slot that crosses the ignore threshold.
 */
TEST_F(Medium_DarknetYolov4TinyTraining, TheDeltasOfTheStepMatchDarknet)
{
    SaTest::check_step_deltas(backend(), training);
}

/**
 * @brief Tests that the weight and bias gradients of the convolution before each head, times
 *        minus the batch, equal the dumped updates within the head update share of the
 *        dump's largest magnitude plus 32768 u of the entry.
 */
TEST_F(Medium_DarknetYolov4TinyTraining, TheUpdatesOfTheHeadConvolutionsMatchDarknet)
{
    SaTest::check_head_updates(backend(), training);
}

/**
 * @brief Tests that the weight, the scale and the shift of the first convolution each have a
 *        finite gradient that is not zero after one step.
 *
 * The dumped updates of that layer are not the gradient and are not compared; see
 * tests/vision/importers/README.md.
 */
TEST_F(Medium_DarknetYolov4TinyTraining, TheFirstConvolutionReceivesAGradient)
{
    SaTest::check_first_gradients(backend(), training);
}

/**
 * @brief Tests that the rolling mean and variance of the first batch norm equal the dump
 *        before the step and are within 256 u of it after one blend.
 */
TEST_F(Medium_DarknetYolov4TinyTraining, TheRollingStatisticsAfterOneStepMatchDarknet)
{
    SaTest::check_rolling_statistics(backend(), training);
}

/**
 * @brief Tests that 200 steps on the batch and truth of the step run bring the stored cost of
 *        the Separate model below 0.1 of the first step's.
 */
TEST_F(Medium_DarknetYolov4TinyTraining, TheCostOfOneBatchFallsUnderATenth)
{
    SaTest::check_one_batch_overfit(backend(), training);
}

/**
 * @brief Tests that the memory plan of the training graph of a compact YOLOv4-tiny cfg
 *        overlaps live ranges; prints the graph's nodes and the bytes its plan asks for.
 */
TEST(Small_DarknetYolov4TinyTrainingPlan, ThePlansOfTheStepOverlapLiveRanges)
{
    SaTest::check_training_plan("yolov4-tiny", K_PLAN_CFG, K_PLAN_BATCH);
}

/**
 * @brief Tests that the memory plan of the step changes no head, no delta and no gradient.
 */
TEST_F(Medium_DarknetYolov4TinyTraining, ThePlanOfTheStepChangesNoValue)
{
    SaTest::check_plan_changes_no_value(backend(), training);
}
