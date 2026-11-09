/**************************************************************************/
/* test_yolov3_tiny_forward.cpp                                           */
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
 * @file test_yolov3_tiny_forward.cpp
 * @brief Tests the forward pass of the imported YOLOv3-tiny against the Darknet reference tensors:
 *        the heads, the [yolo] outputs, the detection rows and the Separate form.
 * @author Mustafa Garip
 */

#include <gtest/gtest.h>
#include <SushiAI/vision/darknet_forward_cases/darknet_forward.hpp>
#include <SushiAI/vision/detect/anchor/head_options.hpp>
#include <SushiAI/vision/detect/suppression_criterion.hpp>
#include <support/tolerance.hpp>

namespace SaTest = SushiAI::Test;

namespace
{
    /**
     * @brief States what yolov3-tiny.cfg and its runs hold (REPORT section 6).
     *
     * The bounds are those of SPEC 6.3, as multiples of the float unit roundoff u: a head value
     * within 32768 u absolute plus 2048 u relative, a box entry and a score of a row within
     * 2048 u, a decision named within 32 u of a threshold. Each multiple is the smallest power of
     * two not tighter than the bound SPEC 6.3 states (1e-3, 1e-4, 1e-4 and 1e-6).
     */
    SaTest::DarknetParityCase describe_model()
    {
        SaTest::DarknetParityCase parity;
        parity.stem = "yolov3-tiny";
        parity.grids = {13, 26};
        parity.scales = {1.0f, 1.0f};
        parity.run_rows = {0, 16};
        parity.classes = 80;
        parity.slots = 3;
        parity.suppression = 0.45f;
        parity.coding = SushiAI::Vision::Detect::Anchor::HeadCoding::Logit;
        parity.criterion = SushiAI::Vision::Detect::SuppressionCriterion::IoU;
        parity.bounds.head_absolute = 32768.0 * TestSupport::unit_roundoff<float>();
        parity.bounds.head_relative = 2048.0 * TestSupport::unit_roundoff<float>();
        parity.bounds.row_absolute = 2048.0 * TestSupport::unit_roundoff<float>();
        parity.bounds.threshold_margin = 32.0 * TestSupport::unit_roundoff<float>();
        return parity;
    }

    /** @brief Runs the parity suite of YOLOv3-tiny on an engine; skipped when a file is missing. */
    class Medium_DarknetYolov3TinyForward : public SaTest::DarknetParityTest
    {
        protected:
            /** @brief Hands the fixture what this file states of the model. */
            Medium_DarknetYolov3TinyForward() : SaTest::DarknetParityTest(describe_model())
            {

            }
    };
} // namespace

/**
 * @brief Tests that both published heads of the folded model equal the raw heads Darknet
 *        dumped, in both runs, within 32768 u absolute plus 2048 u relative.
 */
TEST_F(Medium_DarknetYolov3TinyForward, TheFoldedHeadsMatchTheDumpedHeads)
{
    SaTest::check_folded_heads(backend(), parity);
}

/**
 * @brief Tests that the [yolo] output computed on the host from our heads, the logistic of
 *        every entry but the sizes, equals Darknet's layer output within the head tolerance.
 *
 * The logistic has a slope of at most a quarter, so the bound of the head holds after it.
 */
TEST_F(Medium_DarknetYolov3TinyForward, TheHostYOLOOutputsMatchTheDumpedLayers)
{
    SaTest::check_yolo_outputs(backend(), parity);
}

/**
 * @brief Tests that decode, threshold and suppression over our heads give Darknet's rows: none
 *        at 0.25 and the 16 of REPORT section 6 at 0.01, both with suppression at 0.45 on the IoU.
 *
 * Both sides are sorted by class, then by score. A row matches when its class is equal and its
 * box entries and score are within 2048 u. Every decision within 32 u of a threshold is printed.
 */
TEST_F(Medium_DarknetYolov3TinyForward, TheDetectionsMatchTheDumpedRows)
{
    SaTest::check_detections(backend(), parity);
}

/**
 * @brief Tests that the Separate model, with its own batch-norm rows, computes the heads of the
 *        Folded model, and prints the largest difference at each head.
 *
 * The bound is the head tolerance, 32768 u absolute plus 2048 u relative to the folded value.
 */
TEST_F(Medium_DarknetYolov3TinyForward, TheSeparateFormComputesTheFoldedHeads)
{
    SaTest::check_separate_form(backend(), parity);
}
