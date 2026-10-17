/**************************************************************************/
/* test_darknet_forward.cpp                                               */
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
 * @file test_darknet_forward.cpp
 * @brief Tests the helpers of the Darknet forward cases that need no model and no dump.
 * @author Mustafa Garip
 */

#include <cmath>
#include <cstdint>
#include <limits>
#include <set>
#include <string>
#include <string_view>
#include <vector>
#include <gtest/gtest.h>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/vision/darknet_forward_cases/darknet_forward.hpp>
#include <SushiAI/vision/detect/anchor/head_options.hpp>

using namespace SushiAI;
namespace SaTest = SushiAI::Test;
using Vision::Detect::Anchor::HeadCoding;
using Vision::Detect::Anchor::HeadOptions;

namespace
{
    /** @brief Bounds the error of a logistic and a stretch the test works out by hand. */
    constexpr double K_YOLO_TOLERANCE = 1e-6;

    /** @brief Names a model no reference data directory holds. */
    constexpr std::string_view K_ABSENT_STEM = "sushiai_no_such_model";

    /** @brief Returns head options of one class under @p coding and @p scale. */
    HeadOptions one_class(HeadCoding coding, float scale)
    {
        HeadOptions options;
        options.num_classes = 1;
        options.coding = coding;
        options.scale_xy = scale;
        return options;
    }
} // namespace

/** @brief Tests that the parity table holds six distinct models, each with a run and heads. */
TEST(Small_DarknetForward, TheParityTableHoldsSixModelsWithRunsAndHeads)
{
    const std::vector<SaTest::DarknetParityModel> models = SaTest::list_darknet_parity_models();

    ASSERT_EQ(models.size(), 6u);
    std::set<std::string_view> stems;
    for (const SaTest::DarknetParityModel& model : models)
    {
        EXPECT_TRUE(stems.insert(model.stem).second) << model.stem;
        EXPECT_GT(model.side, 0) << model.stem;
        EXPECT_FALSE(model.runs.empty()) << model.stem;
        EXPECT_FALSE(model.yolo_layers.empty()) << model.stem;
    }
}

/** @brief Tests that a lookup by stem returns the row of that stem. */
TEST(Small_DarknetForward, ALookupReturnsTheRowOfItsStem)
{
    const SaTest::DarknetParityModel tiny = SaTest::find_darknet_parity_model("yolov3-tiny");

    EXPECT_EQ(tiny.stem, "yolov3-tiny");
    EXPECT_EQ(tiny.side, 416);
    EXPECT_EQ(tiny.yolo_layers, (std::vector<std::int64_t>{16, 23}));
    EXPECT_EQ(SaTest::find_darknet_parity_model("yolov4").side, 608);
}

/** @brief Tests that a layer name pads the index to three digits and keeps a longer one. */
TEST(Small_DarknetForward, ALayerNamePadsItsIndexToThreeDigits)
{
    EXPECT_EQ(SaTest::name_darknet_layer(16, "yolo"), "layer_016_yolo");
    EXPECT_EQ(SaTest::name_darknet_layer(0, "yolo_input"), "layer_000_yolo_input");
    EXPECT_EQ(SaTest::name_darknet_layer(1234, "convolutional"), "layer_1234_convolutional");
    EXPECT_EQ(SaTest::name_observed_layer(7), "layer.7");
}

/** @brief Tests that the gap counts the values beyond the bound and keeps the largest. */
TEST(Small_DarknetForward, TheGapCountsValuesBeyondTheBoundAndKeepsTheLargest)
{
    const std::vector<float> reference = {1.0f, 2.0f, 100.0f};
    const std::vector<float> actual = {1.0f, 2.5f, 101.0f};

    const SaTest::ToleranceGap absolute = SaTest::measure_gap(actual, reference, 0.1, 0.0);
    EXPECT_EQ(absolute.outside, 2u);
    EXPECT_EQ(absolute.at, 2u);

    const SaTest::ToleranceGap relative = SaTest::measure_gap(actual, reference, 0.6, 0.02);
    EXPECT_EQ(relative.outside, 0u);
    EXPECT_NEAR(relative.largest, 1.0, K_YOLO_TOLERANCE);
}

/** @brief Tests that a NaN is outside every bound and is reported as the largest gap. */
TEST(Small_DarknetForward, ANaNIsOutsideEveryBound)
{
    const std::vector<float> reference = {1.0f, 2.0f};
    const std::vector<float> actual = {1.5f, std::numeric_limits<float>::quiet_NaN()};

    const SaTest::ToleranceGap gap = SaTest::measure_gap(actual, reference, 1.0, 1.0);

    EXPECT_EQ(gap.outside, 1u);
    EXPECT_EQ(gap.at, 1u);
}

/** @brief Tests that the tolerance check fails on a size mismatch and on a value outside. */
TEST(Small_DarknetForward, TheToleranceCheckFailsOnASizeMismatchOrAValueOutside)
{
    const std::vector<float> reference = {1.0f, 2.0f};

    EXPECT_TRUE(SaTest::within_tolerance("same", reference, reference, 0.0, 0.0));
    EXPECT_FALSE(SaTest::within_tolerance("short", std::vector<float>{1.0f}, reference, 1.0, 0.0));
    EXPECT_FALSE(SaTest::within_tolerance("far", std::vector<float>{1.0f, 3.0f}, reference, 0.5,
                                          0.0));
}

/** @brief Tests that a logit head takes the logistic of all but the sizes and stretches x, y. */
TEST(Small_DarknetForward, ALogitHeadIsActivatedAndStretched)
{
    const float three = static_cast<float>(std::log(3.0));
    const std::vector<float> head = {three, 0.0f, 0.3f, -0.2f, 0.0f, three};

    const std::vector<float> output =
        SaTest::compute_yolo_output(head, one_class(HeadCoding::Logit, 2.0f));

    ASSERT_EQ(output.size(), head.size());
    EXPECT_NEAR(output[0], 1.0, K_YOLO_TOLERANCE);
    EXPECT_NEAR(output[1], 0.5, K_YOLO_TOLERANCE);
    EXPECT_NEAR(output[2], 0.3, K_YOLO_TOLERANCE);
    EXPECT_NEAR(output[3], -0.2, K_YOLO_TOLERANCE);
    EXPECT_NEAR(output[4], 0.5, K_YOLO_TOLERANCE);
    EXPECT_NEAR(output[5], 0.75, K_YOLO_TOLERANCE);
}

/** @brief Tests that an activated head is only stretched, at x and y of every slot. */
TEST(Small_DarknetForward, AnActivatedHeadIsOnlyStretched)
{
    const std::vector<float> head = {0.75f, 0.25f, 0.3f, 0.2f, 0.9f, 0.1f,
                                     0.5f,  0.0f,  0.4f, 0.6f, 0.8f, 0.7f};

    const std::vector<float> output =
        SaTest::compute_yolo_output(head, one_class(HeadCoding::Activated, 2.0f));

    const std::vector<double> expected = {1.0, 0.0, 0.3, 0.2, 0.9, 0.1,
                                          0.5, -0.5, 0.4, 0.6, 0.8, 0.7};
    ASSERT_EQ(output.size(), expected.size());
    for (std::size_t i = 0; i < expected.size(); ++i)
        EXPECT_NEAR(output[i], expected[i], K_YOLO_TOLERANCE) << i;
}

/** @brief Tests that an output is found by its name and an unpublished one is null. */
TEST(Small_DarknetForward, AnOutputIsFoundByItsName)
{
    SaTest::DarknetForwardRun run;
    run.outputs.emplace(SaTest::name_observed_layer(3), std::vector<float>{1.0f, 2.0f});

    const std::vector<float>* found = SaTest::find_output(run, "layer.3");
    ASSERT_NE(found, nullptr);
    EXPECT_EQ(found->size(), 2u);
    EXPECT_EQ(SaTest::find_output(run, "layer.4"), nullptr);
}

/** @brief Tests that the files of a model no reference holds are reported and refused. */
TEST(Small_DarknetForward, AModelWithoutFilesIsReportedAndRefused)
{
    const std::vector<std::string> names = {"input"};

    EXPECT_NE(SaTest::find_missing_darknet_file(K_ABSENT_STEM, "run", names).find(K_ABSENT_STEM),
              std::string::npos);
    SA_EXPECT_FAILURE(SaTest::load_darknet_config_file(K_ABSENT_STEM), StatusCode::FileNotFound,
                      "sushiai_no_such_model.cfg");
}
