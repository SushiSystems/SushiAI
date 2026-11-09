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
 * @file test_yolov7.cpp
 * @brief Tests the head deltas of YOLOv7 and one training step of it under batch statistics against
 *        what Darknet dumped of both.
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
     * @brief States what yolov7.cfg and its two training runs hold (REPORT 21 and 22).
     *
     * Of the inference-mode runs only the `*_plain_pool*` ones are valid (REPORT 25 and 28).
     * `yolov7_train_step` is a training-mode run, which the defect does not reach.
     * Bounds are multiples of the float unit roundoff u, each the smallest power of two not
     * tighter than the earlier decimal bound (PLAN 4.7, REPORT 11, importers README).
     */
    SaTest::DarknetTrainingCase describe_model()
    {
        SaTest::DarknetTrainingCase training;
        training.stem = "yolov7";
        training.host_run = "yolov7_plain_pool_thresh_0p01";
        training.step_run = "yolov7_train_step";
        training.box_loss = SushiAI::Vision::Detect::Anchor::BoxLoss::CompleteIoU;
        training.objectness_target = SushiAI::Vision::Detect::Anchor::ObjectnessTarget::BestIoU;
        training.bounds.host_delta = {units(32), units(2048), units(32)};
        training.bounds.host_cost_relative = units(256);
        training.bounds.head_absolute = units(262144);
        training.bounds.head_relative = units(2048);
        training.bounds.yolo_absolute = units(262144);
        training.bounds.step_delta = {units(32768), units(4096), units(32768)};
        training.bounds.step_cost_relative = units(32768);
        training.bounds.head_update_share = units(32768);
        training.bounds.update_relative = units(32768);
        training.bounds.rolling_absolute = units(256);
        training.overfit_steps = 200;
        training.overfit_cost_share = K_OVERFIT_COST_SHARE;
        return training;
    }

    /** @brief Holds the images of the step whose training graph is planned. */
    constexpr std::int64_t K_PLAN_BATCH = 2;

    /**
     * @brief Holds a compact cfg with the layer kinds and the head keys of YOLOv7.
     *
     * The input is 64 by 64 with one class, so the training graph is planned on the
     * host from this text alone; it is not the yolov7.cfg of Darknet.
     */
    constexpr std::string_view K_PLAN_CFG =
        "[net]\n"
        "batch=2\nsubdivisions=1\nwidth=64\nheight=64\nchannels=3\nmomentum=0.949\n"
        "decay=0.0005\nlearning_rate=0.001\nburn_in=1000\nmax_batches=500500\n"
        "policy=steps\nsteps=400000,450000\nscales=.1,.1\n"
        "[convolutional]\nbatch_normalize=1\nfilters=8\n"
        "size=3\nstride=2\npad=1\nactivation=swish\n"
        "[convolutional]\nbatch_normalize=1\nfilters=16\n"
        "size=3\nstride=2\npad=1\nactivation=swish\n"
        "[convolutional]\nbatch_normalize=1\nfilters=8\n"
        "size=1\nstride=1\npad=1\nactivation=swish\n"
        "[route]\nlayers=-2\n"
        "[convolutional]\nbatch_normalize=1\nfilters=8\n"
        "size=1\nstride=1\npad=1\nactivation=swish\n"
        "[convolutional]\nbatch_normalize=1\nfilters=8\n"
        "size=3\nstride=1\npad=1\nactivation=swish\n"
        "[convolutional]\nbatch_normalize=1\nfilters=8\n"
        "size=3\nstride=1\npad=1\nactivation=swish\n"
        "[route]\nlayers=-1,-3,-5,-6\n"
        "[convolutional]\nbatch_normalize=1\nfilters=16\n"
        "size=1\nstride=1\npad=1\nactivation=swish\n"
        "[maxpool]\nsize=2\nstride=2\n"
        "[convolutional]\nbatch_normalize=1\nfilters=16\n"
        "size=1\nstride=1\npad=1\nactivation=swish\n"
        "[route]\nlayers=-2\n"
        "[convolutional]\nbatch_normalize=1\nfilters=16\n"
        "size=1\nstride=1\npad=1\nactivation=swish\n"
        "[convolutional]\nbatch_normalize=1\nfilters=16\n"
        "size=3\nstride=1\npad=1\nactivation=swish\n"
        "[route]\nlayers=-1,-2,-4\n"
        "[convolutional]\nbatch_normalize=1\nfilters=32\n"
        "size=1\nstride=1\npad=1\nactivation=swish\n"
        "[convolutional]\nfilters=18\n"
        "size=1\nstride=1\npad=1\nactivation=linear\n"
        "[yolo]\nmask=3,4,5\nanchors=4,5, 8,10, 12,16, 20,24, 28,36, 44,52\nclasses=1\nnum=6\n"
        "scale_x_y=2.0\nobjectness_smooth=1\nignore_thresh=0.7\ntruth_thresh=1\n"
        "iou_thresh=0.2\niou_normalizer=0.05\ncls_normalizer=0.5\nobj_normalizer=1.0\n"
        "iou_loss=ciou\nnms_kind=diounms\nbeta_nms=0.6\nnew_coords=1\n"
        "max_delta=2\n"
        "[route]\nlayers=-3\n"
        "[convolutional]\nbatch_normalize=1\nfilters=8\n"
        "size=1\nstride=1\npad=1\nactivation=swish\n"
        "[upsample]\nstride=2\n"
        "[route]\nlayers=-1,8\n"
        "[convolutional]\nbatch_normalize=1\nfilters=16\n"
        "size=3\nstride=1\npad=1\nactivation=swish\n"
        "[convolutional]\nfilters=18\n"
        "size=1\nstride=1\npad=1\nactivation=linear\n"
        "[yolo]\nmask=0,1,2\nanchors=4,5, 8,10, 12,16, 20,24, 28,36, 44,52\nclasses=1\nnum=6\n"
        "scale_x_y=2.0\nobjectness_smooth=1\nignore_thresh=0.7\ntruth_thresh=1\n"
        "iou_thresh=0.2\niou_normalizer=0.05\ncls_normalizer=0.5\nobj_normalizer=1.0\n"
        "iou_loss=ciou\nnms_kind=diounms\nbeta_nms=0.6\nnew_coords=1\n"
        "max_delta=2\n";

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
 *        assignments exactly, its delta within the host bounds and its stored cost to 256 u.
 */
TEST_F(Large_DarknetYolov7Training, TheDeltaOfADumpedHeadMatchesDarknet)
{
    SaTest::check_host_deltas(training);
}

/**
 * @brief Tests that the delta computed from each dumped head of the two-image step equals the
 *        step's dumped delta within the host bounds, and its stored cost to 256 u.
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
    SaTest::check_step_heads(backend(), training);
}

/**
 * @brief Tests that the deltas the step fed equal the dumped ones within the step bounds, and
 *        that the summed cost and its mean over the heads equal the layer and network costs.
 *
 * The step bounds do not cover a slot that crosses the ignore threshold.
 */
TEST_F(Large_DarknetYolov7Training, TheDeltasOfTheStepMatchDarknet)
{
    SaTest::check_step_deltas(backend(), training);
}

/**
 * @brief Tests that the weight and bias gradients of the convolution before each head, times
 *        minus the batch, equal the dumped updates within the head update share of the
 *        dump's largest magnitude plus 32768 u of the entry.
 */
TEST_F(Large_DarknetYolov7Training, TheUpdatesOfTheHeadConvolutionsMatchDarknet)
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
TEST_F(Large_DarknetYolov7Training, TheFirstConvolutionReceivesAGradient)
{
    SaTest::check_first_gradients(backend(), training);
}

/**
 * @brief Tests that the rolling mean and variance of the first batch norm equal the dump
 *        before the step and are within 256 u of it after one blend.
 */
TEST_F(Large_DarknetYolov7Training, TheRollingStatisticsAfterOneStepMatchDarknet)
{
    SaTest::check_rolling_statistics(backend(), training);
}

/**
 * @brief Tests that 200 steps on the batch and truth of the step run bring the stored cost of
 *        the Separate model below 0.1 of the first step's.
 */
TEST_F(Large_DarknetYolov7Training, TheCostOfOneBatchFallsUnderATenth)
{
    SaTest::check_one_batch_overfit(backend(), training);
}

/**
 * @brief Tests that the memory plan of the training graph of a compact YOLOv7 cfg
 *        overlaps live ranges; prints the graph's nodes and the bytes its plan asks for.
 */
TEST(Small_DarknetYolov7TrainingPlan, ThePlansOfTheStepOverlapLiveRanges)
{
    SaTest::check_training_plan("yolov7", K_PLAN_CFG, K_PLAN_BATCH);
}
