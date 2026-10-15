/**************************************************************************/
/* test_yolov4_tiny_forward.cpp                                           */
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
 * @file test_yolov4_tiny_forward.cpp
 * @brief Tests the forward pass of the imported YOLOv4-tiny against the Darknet reference tensors:
 *        the heads, the [yolo] outputs, the detection rows and the Separate form.
 * @author Mustafa Garip
 */

#include <gtest/gtest.h>
#include <SushiAI/vision/darknet_forward_cases/darknet_forward.hpp>
#include <SushiAI/vision/detect/anchor/head_options.hpp>
#include <SushiAI/vision/detect/suppression_criterion.hpp>

namespace SaTest = SushiAI::Test;

namespace
{
    /**
     * @brief States what yolov4-tiny.cfg and its runs hold (REPORT sections 12 and 22).
     *
     * The bounds are those of SPEC 6.3: a head value within 1e-3 absolute plus 1e-4 relative, a
     * box entry and a score of a row within 1e-4, a decision named within 1e-6 of a threshold.
     */
    SaTest::DarknetParityCase describe_model()
    {
        SaTest::DarknetParityCase parity;
        parity.stem = "yolov4-tiny";
        parity.grids = {13, 26};
        parity.scales = {1.05f, 1.05f};
        parity.run_rows = {0, 16};
        parity.classes = 80;
        parity.slots = 3;
        parity.suppression = 0.45f;
        parity.coding = SushiAI::Vision::Detect::Anchor::HeadCoding::Logit;
        parity.criterion = SushiAI::Vision::Detect::SuppressionCriterion::DistanceIoU;
        parity.distance_exponent = 0.6f;
        parity.bounds.head_absolute = 1e-3;
        parity.bounds.head_relative = 1e-4;
        parity.bounds.row_absolute = 1e-4;
        parity.bounds.threshold_margin = 1e-6;
        return parity;
    }

    /** @brief Runs the parity suite of YOLOv4-tiny on an engine; skipped when a file is missing. */
    class Medium_DarknetYolov4TinyForward : public SaTest::DarknetParityTest
    {
        protected:
            /** @brief Hands the fixture what this file states of the model. */
            Medium_DarknetYolov4TinyForward() : SaTest::DarknetParityTest(describe_model())
            {

            }
    };
} // namespace

/**
 * @brief Tests that both published heads of the folded model equal the raw heads Darknet
 *        dumped, in both runs, within 1e-3 absolute plus 1e-4 relative.
 */
TEST_F(Medium_DarknetYolov4TinyForward, TheFoldedHeadsMatchTheDumpedHeads)
{
    SaTest::check_folded_heads(*engine, parity);
}

/**
 * @brief Tests that the [yolo] output computed on the host from our heads, the logistic of
 *        every entry but the sizes with x and y stretched by the scale_x_y of 1.05, equals
 *        Darknet's layer output within the head tolerance.
 *
 * The stretched logistic has a slope of at most 1.05 / 4, so the bound of the head holds after it.
 */
TEST_F(Medium_DarknetYolov4TinyForward, TheHostYOLOOutputsMatchTheDumpedLayers)
{
    SaTest::check_yolo_outputs(*engine, parity);
}

/**
 * @brief Tests that decode, threshold and suppression over our heads give Darknet's rows: none
 *        at 0.25 and 16 at 0.01, as REPORT section 12 states, both with suppression at 0.45 on
 *        the IoU less the distance term to the power 0.6.
 *
 * Both sides are sorted by class, then by score. A row matches when its class is equal and its
 * box entries and score are within 1e-4. Every decision within 1e-6 of a threshold is printed.
 */
TEST_F(Medium_DarknetYolov4TinyForward, TheDetectionsMatchTheDumpedRows)
{
    SaTest::check_detections(*engine, parity);
}

/**
 * @brief Tests that the Separate model, with its own batch-norm rows, computes the heads of the
 *        Folded model, and prints the largest difference at each head.
 *
 * The bound is the head tolerance, 1e-3 absolute plus 1e-4 relative to the folded value.
 */
TEST_F(Medium_DarknetYolov4TinyForward, TheSeparateFormComputesTheFoldedHeads)
{
    SaTest::check_separate_form(*engine, parity);
}
