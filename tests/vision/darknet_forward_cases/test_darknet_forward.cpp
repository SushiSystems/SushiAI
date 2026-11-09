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
 * @file test_darknet_forward.cpp
 * @brief Tests the helpers of the Darknet forward cases that need no model and no dump.
 * @author Mustafa Garip
 */

#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <string_view>
#include <vector>
#include <gtest/gtest.h>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/vision/darknet_forward_cases/darknet_forward.hpp>
#include <SushiAI/vision/detect/anchor/head_options.hpp>
#include <SushiAI/vision/importers/darknet_config.hpp>
#include <support/tolerance.hpp>

using namespace SushiAI;
namespace SaTest = SushiAI::Test;
using Vision::Detect::Anchor::HeadCoding;
using Vision::Detect::Anchor::HeadOptions;

namespace
{
    /**
     * @brief Holds the absolute bound on the error of a logistic and a stretch the test works out
     *        by hand: 32 times the float unit roundoff, the smallest power of two not tighter than
     *        the earlier 1e-6.
     */
    constexpr double K_YOLO_TOLERANCE = 32.0 * TestSupport::unit_roundoff<float>();

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

    /** @brief Returns whether each output value is within K_YOLO_TOLERANCE of its expected one. */
    ::testing::AssertionResult is_near_each(const std::vector<float>& output,
                                            const std::vector<double>& expected)
    {
        if (output.size() != expected.size())
            return ::testing::AssertionFailure()
                   << output.size() << " values, expected " << expected.size();

        for (std::size_t i = 0; i < expected.size(); ++i)
        {
            const ::testing::AssertionResult near = TestSupport::is_within(
                TestSupport::measure_error(output[i], expected[i], K_YOLO_TOLERANCE, 1.0));
            if (!near)
                return ::testing::AssertionFailure()
                       << "value " << i << " is " << output[i] << ", expected " << expected[i]
                       << ": " << near.message();
        }

        return ::testing::AssertionSuccess();
    }

    /** @brief Returns the stem of every row of the parity table. */
    std::vector<std::string> parity_stems()
    {
        std::vector<std::string> stems;
        for (const SaTest::DarknetParityModel& model : SaTest::list_darknet_parity_models())
            stems.emplace_back(model.stem);

        return stems;
    }

    /** @brief Returns the value a section spells for @p key, or an empty text. */
    std::string option_of(const Vision::Importers::DarknetSection& section, std::string_view key)
    {
        for (const Vision::Importers::DarknetOption& option : section.options)
            if (option.key == key)
                return option.value;

        return std::string();
    }

    /** @brief Returns the indices of the [yolo] sections of a cfg, in file order. */
    std::vector<std::int64_t> yolo_indices_of(const Vision::Importers::DarknetConfig& config)
    {
        std::vector<std::int64_t> indices;
        for (std::size_t layer = 0; layer < config.layers.size(); ++layer)
            if (config.layers[layer].type == "yolo")
                indices.push_back(static_cast<std::int64_t>(layer));

        return indices;
    }

    /** @brief Runs a check over the stem of each row of the parity table. */
    class Small_DarknetForwardLookup : public ::testing::TestWithParam<std::string>
    {
    };

    /** @brief Runs a check of a table row against its cfg file, for each stem of the table. */
    class Medium_DarknetForwardTable : public ::testing::TestWithParam<std::string>
    {
    };

    /** @brief Names a test after the stem it checks, in a form gtest accepts. */
    std::string stem_name(const ::testing::TestParamInfo<std::string>& info)
    {
        std::string name = info.param;
        for (char& letter : name)
            if (letter == '-')
                letter = '_';

        return name;
    }
} // namespace

/** @brief Tests that a lookup by stem returns the row of that stem. */
TEST_P(Small_DarknetForwardLookup, ALookupReturnsTheRowOfItsStem)
{
    const std::string stem = GetParam();

    const SaTest::DarknetParityModel row = SaTest::find_darknet_parity_model(stem);

    EXPECT_EQ(row.stem, stem);
    EXPECT_GT(row.side, 0);
    EXPECT_FALSE(row.yolo_layers.empty());
}

INSTANTIATE_TEST_SUITE_P(Stems, Small_DarknetForwardLookup, ::testing::ValuesIn(parity_stems()),
                         stem_name);

/** @brief Tests that a table row names the side and the [yolo] layers its cfg file spells. */
TEST_P(Medium_DarknetForwardTable, ARowNamesTheSideAndTheYoloLayersItsCfgSpells)
{
    const std::string stem = GetParam();
    const std::string missing = SaTest::find_missing_darknet_file(stem, "", {});
    if (!missing.empty())
        GTEST_SKIP() << stem << ": " << missing;

    const SaTest::DarknetParityModel row = SaTest::find_darknet_parity_model(stem);

    SA_ASSERT_OK_AND_ASSIGN(const Vision::Importers::DarknetConfig config,
                            SaTest::load_darknet_config_file(stem));
    EXPECT_EQ(std::to_string(row.side), option_of(config.net, "width"));
    EXPECT_EQ(row.yolo_layers, yolo_indices_of(config));
}

INSTANTIATE_TEST_SUITE_P(Stems, Medium_DarknetForwardTable, ::testing::ValuesIn(parity_stems()),
                         stem_name);

/** @brief Tests that a layer name pads the index to three digits and keeps a longer one. */
TEST(Small_DarknetForward, ALayerNamePadsItsIndexToThreeDigits)
{
    EXPECT_EQ(SaTest::name_darknet_layer(16, "yolo"), "layer_016_yolo");
    EXPECT_EQ(SaTest::name_darknet_layer(0, "yolo_input"), "layer_000_yolo_input");
    EXPECT_EQ(SaTest::name_darknet_layer(1234, "convolutional"), "layer_1234_convolutional");
}

/** @brief Tests that an observed layer is named by its index. */
TEST(Small_DarknetForward, AnObservedLayerIsNamedByItsIndex)
{
    EXPECT_EQ(SaTest::name_observed_layer(7), "layer.7");
}

/** @brief Tests that the gap under an absolute bound counts the values beyond it. */
TEST(Small_DarknetForward, TheGapCountsValuesBeyondAnAbsoluteBound)
{
    const std::vector<float> reference = {1.0f, 2.0f, 100.0f};
    const std::vector<float> actual = {1.0f, 2.5f, 101.0f};

    const SaTest::ToleranceGap gap = SaTest::measure_gap(actual, reference, 0.1, 0.0);

    EXPECT_EQ(gap.outside, 2u);
    EXPECT_EQ(gap.at, 2u);
}

/** @brief Tests that the gap under an absolute plus relative bound keeps the largest gap. */
TEST(Small_DarknetForward, TheGapKeepsTheLargestWithinAnAbsolutePlusRelativeBound)
{
    const std::vector<float> reference = {1.0f, 2.0f, 100.0f};
    const std::vector<float> actual = {1.0f, 2.5f, 101.0f};

    const SaTest::ToleranceGap gap = SaTest::measure_gap(actual, reference, 0.6, 0.02);

    EXPECT_EQ(gap.outside, 0u);
    EXPECT_EQ(gap.largest, 1.0);
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

/** @brief Tests that the tolerance check passes two equal spans. */
TEST(Small_DarknetForward, TheToleranceCheckPassesTwoEqualSpans)
{
    const std::vector<float> reference = {1.0f, 2.0f};

    EXPECT_TRUE(SaTest::within_tolerance("same", reference, reference, 0.0, 0.0));
}

/** @brief Tests that the tolerance check fails on a size mismatch. */
TEST(Small_DarknetForward, TheToleranceCheckFailsOnASizeMismatch)
{
    const std::vector<float> reference = {1.0f, 2.0f};

    EXPECT_FALSE(SaTest::within_tolerance("short", std::vector<float>{1.0f}, reference, 1.0, 0.0));
}

/** @brief Tests that the tolerance check fails on a value outside the bound. */
TEST(Small_DarknetForward, TheToleranceCheckFailsOnAValueOutsideTheBound)
{
    const std::vector<float> reference = {1.0f, 2.0f};

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

    EXPECT_TRUE(is_near_each(output, {1.0, 0.5, 0.3, -0.2, 0.5, 0.75}));
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
    EXPECT_TRUE(is_near_each(output, expected));
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

/** @brief Tests that the files of a model no reference holds are reported by stem. */
TEST(Small_DarknetForward, AModelWithoutFilesIsReported)
{
    const std::vector<std::string> names = {"input"};

    EXPECT_NE(SaTest::find_missing_darknet_file(K_ABSENT_STEM, "run", names).find(K_ABSENT_STEM),
              std::string::npos);
}

/** @brief Tests that the load of a stem with no cfg file is FileNotFound, naming the file. */
TEST(Medium_DarknetForwardFiles, ALoadOfAStemWithoutACfgIsFileNotFound)
{
    SA_EXPECT_FAILURE(SaTest::load_darknet_config_file(K_ABSENT_STEM), StatusCode::FileNotFound,
                      "sushiai_no_such_model.cfg");
}
