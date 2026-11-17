/**************************************************************************/
/* test_detection_report.cpp                                              */
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
 * @file test_detection_report.cpp
 * @brief Tests the batch loop, the printout and the checks every detection report shares.
 * @author Mustafa Garip
 */

#include "detection_report.hpp"

#include <cstddef>
#include <memory>
#include <string>
#include <string_view>
#include <vector>
#include <gtest/gtest.h>
#include <SushiLog/text/value_text.hpp>
#include <SushiAI/data/dataset.hpp>
#include <SushiAI/data/roles.hpp>
#include <SushiAI/data/target.hpp>
#include <SushiAI/engine_fixture/engine_test.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/shape/shape.hpp>
#include <SushiAI/vision/detect/detection_row.hpp>

#include "prepared_inference_harness.hpp"
#include "tiny_yolo_files.hpp"

using namespace SushiAI;
// SaTest stays reachable inside a TEST_F body; see tests/applications/programs/README.md.
namespace SaTest = SushiAI::Test;
namespace Prepared = SushiAI::Apps::PreparedTest;
namespace TinyYolo = SushiAI::Apps::TinyYoloTest;

namespace
{
    /** @brief Gives each case an engine on the shared test runtime. */
    class Medium_DetectionReport : public SaTest::EngineTest {};

    /** @brief Holds the images one batch of the fixture reads. */
    constexpr std::size_t K_BATCH_IMAGES = 2;

    /** @brief Holds the batches of the fixture: its four images in batches of two. */
    constexpr std::size_t K_BATCHES = TinyYolo::K_IMAGES / K_BATCH_IMAGES;

    /** @brief Holds the score a plan keeps a detection above. */
    constexpr float K_KEPT_ABOVE = 0.5f;

    /** @brief Holds the score a plan counts a detection for the precision above. */
    constexpr float K_COUNTED_ABOVE = 0.001f;

    /** @brief Holds what run_detection_report returned and the text it printed. */
    struct PrintedReport
    {
        /** @brief What the report returned. */
        Status status;

        /** @brief What the report wrote to standard output. */
        std::string printed;
    };

    /**
     * @struct ScriptedDecoder
     * @brief Hands the detections it holds to every batch and counts its calls.
     */
    struct ScriptedDecoder
    {
        /** @brief The detections given to every batch: one flat row list per image. */
        std::vector<std::vector<float>> images;

        /** @brief The call, counted from 1, that fails; 0 fails none. */
        std::size_t failing_call = 0;

        /** @brief How many times the decoder was called. */
        std::size_t calls = 0;

        /** @brief Gives the scripted detections to the batch, or fails on the failing call. */
        Status operator()(std::vector<std::vector<float>>& detections)
        {
            ++calls;
            SA_FAIL_IF(calls == failing_call, StatusCode::DeviceFailed,
                       "the decoder stopped at call {}", calls);
            detections = images;
            return Status();
        }
    };

    /** @brief Returns one detection row: the box, the score and the class. */
    std::vector<float> detection(float x1, float y1, float x2, float y2, float score,
                                 float class_index)
    {
        std::vector<float> row(Vision::Detect::K_DETECTION_VALUES);
        row[0] = x1;
        row[1] = y1;
        row[2] = x2;
        row[3] = y2;
        row[Vision::Detect::K_DETECTION_SCORE] = score;
        row[Vision::Detect::K_DETECTION_CLASS] = class_index;
        return row;
    }

    /** @brief Returns the rows of @p first followed by the rows of @p second. */
    std::vector<float> rows_of(std::vector<float> first, const std::vector<float>& second)
    {
        first.insert(first.end(), second.begin(), second.end());
        return first;
    }

    /** @brief Returns the plan of the fixture: two images a batch, kept above K_KEPT_ABOVE. */
    Apps::DetectionReportPlan fixture_plan(bool map_to_source)
    {
        Apps::DetectionReportPlan plan;
        plan.images = K_BATCH_IMAGES;
        plan.score_threshold = K_KEPT_ABOVE;
        plan.confidence_floor = K_COUNTED_ABOVE;
        plan.map_to_source = map_to_source;
        return plan;
    }

    /** @brief Wraps @p decoder as the callable a report takes, without copying it. */
    Apps::DecodeDetections decode_with(ScriptedDecoder& decoder)
    {
        return [&decoder](std::vector<std::vector<float>>& detections)
        {
            return decoder(detections);
        };
    }

    /** @brief Runs the report of @p plan over @p harness and captures what it prints. */
    PrintedReport run_report(Prepared::InferenceHarness& harness,
                             const Apps::DetectionReportPlan& plan, ScriptedDecoder& decoder)
    {
        testing::internal::CaptureStdout();
        const Status status = Apps::run_detection_report(*harness.prepared, plan,
                                                         decode_with(decoder));
        return PrintedReport{status, testing::internal::GetCapturedStdout()};
    }

    /** @brief Builds the harness with the role hidden, or stops the process with the refusal. */
    std::unique_ptr<Prepared::InferenceHarness> built_harness(SushiAI::ExecutionBackend& backend,
                                                              std::string_view hidden_role = {})
    {
        return SaTest::require_value(Prepared::make_inference_harness(backend, hidden_role));
    }

    /** @brief Returns the line a box of four corners prints as, after its class and score. */
    std::string box_text(float x1, float y1, float x2, float y2)
    {
        return "box " + SushiLog::right(SushiLog::fixed(x1, 2), 8) + " "
               + SushiLog::right(SushiLog::fixed(y1, 2), 8) + " "
               + SushiLog::right(SushiLog::fixed(x2, 2), 8) + " "
               + SushiLog::right(SushiLog::fixed(y2, 2), 8);
    }
} // namespace

/** @brief Tests that the legend states both thresholds and the frame the boxes are measured in. */
TEST(Small_DetectionReport, TheLegendStatesTheThresholdsAndTheFrameOfTheBoxes)
{
    testing::internal::CaptureStdout();
    Apps::print_detection_legend(0.5f, 0.25f, "16 by 16 input");
    const std::string printed = testing::internal::GetCapturedStdout();

    EXPECT_EQ(printed,
              "Detections : score above 0.5, suppression above 0.25; boxes are x1 y1 x2 y2 "
              "in pixels of the 16 by 16 input\n\n");
}

/** @brief Tests that an input that is not height, width, depth is refused naming the model kind. */
TEST_F(Medium_DetectionReport, AnInputThatIsNotHeightWidthDepthIsRefused)
{
    const auto harness = built_harness(backend());

    SA_EXPECT_FAILURE(Apps::check_input_rank(*harness->prepared, Shape{16, 16}, "an anchor model"),
                      StatusCode::InvalidDescription, "model.json", "an anchor model",
                      "[height, width, channels]");
}

/** @brief Tests that an input of height, width and depth passes the check. */
TEST_F(Medium_DetectionReport, AnInputOfHeightWidthDepthPasses)
{
    const auto harness = built_harness(backend());

    SA_EXPECT_OK(Apps::check_input_rank(*harness->prepared, Shape{16, 16, 3}, "an anchor model"));
}

/** @brief Tests that a score threshold below the confidence floor is refused naming both. */
TEST_F(Medium_DetectionReport, AScoreThresholdBelowTheConfidenceFloorIsRefused)
{
    const auto harness = built_harness(backend());

    SA_EXPECT_FAILURE(Apps::check_score_floor(*harness->prepared, 0.25f, 0.5f),
                      StatusCode::InvalidDescription, "score threshold 0.25",
                      "confidence floor 0.5");
}

/** @brief Tests that a score threshold equal to the confidence floor passes the check. */
TEST_F(Medium_DetectionReport, AScoreThresholdEqualToTheConfidenceFloorPasses)
{
    const auto harness = built_harness(backend());

    SA_EXPECT_OK(Apps::check_score_floor(*harness->prepared, 0.5f, 0.5f));
}

/** @brief Tests that data which publishes the fit role is described as the source image of each file. */
TEST_F(Medium_DetectionReport, BoxesAreMeasuredInTheSourceImageWhenTheDataPublishesFit)
{
    const auto harness = built_harness(backend());

    EXPECT_EQ(Apps::describe_box_frame(*harness->prepared, Shape{16, 16, 3}),
              "source image of each file");
}

/** @brief Tests that data without the fit role is described as the network input, width by height. */
TEST_F(Medium_DetectionReport, BoxesAreMeasuredInTheInputWhenTheDataHidesFit)
{
    const auto harness = built_harness(backend(), Data::K_ROLE_FIT);

    EXPECT_EQ(Apps::describe_box_frame(*harness->prepared, Shape{20, 30, 3}), "30 by 20 input");
}

/** @brief Tests that the decoder is called once for each batch of the data. */
TEST_F(Medium_DetectionReport, TheDecoderRunsOncePerBatch)
{
    const auto harness = built_harness(backend());
    ScriptedDecoder decoder;
    decoder.images = {{}, {}};

    const PrintedReport report = run_report(*harness, fixture_plan(false), decoder);

    SA_EXPECT_OK(report.status);
    EXPECT_EQ(decoder.calls, K_BATCHES);
}

/** @brief Tests that each image prints the detections above the score threshold and no other. */
TEST_F(Medium_DetectionReport, EachImagePrintsOnlyTheDetectionsAboveTheScoreThreshold)
{
    const auto harness = built_harness(backend());
    ScriptedDecoder decoder;
    decoder.images = {rows_of(detection(1, 2, 3, 4, 0.9f, 1), detection(5, 6, 7, 8, 0.1f, 0)),
                      detection(1, 2, 3, 4, 0.4f, 0)};

    const PrintedReport report = run_report(*harness, fixture_plan(false), decoder);

    SA_EXPECT_OK(report.status);
    EXPECT_NE(report.printed.find("image 0: 1 detections"), std::string::npos) << report.printed;
    EXPECT_NE(report.printed.find("image 1: 0 detections"), std::string::npos) << report.printed;
    EXPECT_NE(report.printed.find("image 2: 1 detections"), std::string::npos) << report.printed;
    EXPECT_NE(report.printed.find("image 3: 0 detections"), std::string::npos) << report.printed;
    EXPECT_EQ(report.printed.find("score 0.1000"), std::string::npos) << report.printed;
    EXPECT_EQ(report.printed.find("score 0.4000"), std::string::npos) << report.printed;
}

/** @brief Tests that a printed detection states its class, its score and its box as decoded. */
TEST_F(Medium_DetectionReport, ADetectionPrintsItsClassItsScoreAndItsBox)
{
    const auto harness = built_harness(backend());
    ScriptedDecoder decoder;
    decoder.images = {detection(10, 20, 30, 40, 0.9f, 1), {}};

    const PrintedReport report = run_report(*harness, fixture_plan(false), decoder);

    EXPECT_NE(report.printed.find("class   1  score 0.9000  " + box_text(10, 20, 30, 40)),
              std::string::npos)
        << report.printed;
}

/** @brief Tests that the total counts the detections printed over the images of every batch. */
TEST_F(Medium_DetectionReport, TheTotalCountsEveryPrintedDetectionOverEveryImage)
{
    const auto harness = built_harness(backend());
    ScriptedDecoder decoder;
    decoder.images = {rows_of(detection(1, 2, 3, 4, 0.9f, 1), detection(5, 6, 7, 8, 0.8f, 0)),
                      {}};

    const PrintedReport report = run_report(*harness, fixture_plan(false), decoder);

    EXPECT_NE(report.printed.find("4 detections over 4 images."), std::string::npos)
        << report.printed;
}

/** @brief Tests that a plan which maps to the source clips a box to the source image of its file. */
TEST_F(Medium_DetectionReport, AMappingPlanClipsTheBoxToTheSourceImageOfItsFile)
{
    const auto harness = built_harness(backend());
    Data::Batch batch;
    SA_EXPECT_OK(harness->prepared->dataset.read(0, batch));
    const Data::BatchTarget* const fit = Data::find_target(batch.targets, Data::K_ROLE_FIT);
    ASSERT_NE(fit, nullptr);
    const float source_height = fit->values.data_as<float>()[0];
    const float source_width = fit->values.data_as<float>()[1];
    ScriptedDecoder decoder;
    decoder.images = {detection(-1e6f, -1e6f, 1e6f, 1e6f, 0.9f, 0), {}};

    const PrintedReport report = run_report(*harness, fixture_plan(true), decoder);

    EXPECT_NE(report.printed.find(box_text(0, 0, source_width, source_height)), std::string::npos)
        << report.printed;
}

/** @brief Tests that a plan which does not map prints the box as the decoder gave it. */
TEST_F(Medium_DetectionReport, AnUnmappedPlanPrintsTheBoxAsDecoded)
{
    const auto harness = built_harness(backend());
    ScriptedDecoder decoder;
    decoder.images = {detection(-1e6f, -1e6f, 1e6f, 1e6f, 0.9f, 0), {}};

    const PrintedReport report = run_report(*harness, fixture_plan(false), decoder);

    EXPECT_NE(report.printed.find(box_text(-1e6f, -1e6f, 1e6f, 1e6f)), std::string::npos)
        << report.printed;
}

/** @brief Tests that data with boxes ends the report with the mean average precision over its images. */
TEST_F(Medium_DetectionReport, DataWithBoxesEndsWithTheMeanAveragePrecision)
{
    const auto harness = built_harness(backend());
    ScriptedDecoder decoder;
    decoder.images = {{}, {}};

    const PrintedReport report = run_report(*harness, fixture_plan(false), decoder);

    EXPECT_NE(report.printed.find("Mean average precision at IoU 0.5"), std::string::npos)
        << report.printed;
    EXPECT_NE(report.printed.find("boxes of 4 images"), std::string::npos) << report.printed;
}

/** @brief Tests that data without a boxes role states that the precision is not computed. */
TEST_F(Medium_DetectionReport, DataWithoutBoxesStatesThatNothingWasScored)
{
    const auto harness = built_harness(backend(), Data::K_ROLE_BOXES);
    ScriptedDecoder decoder;
    decoder.images = {{}, {}};

    const PrintedReport report = run_report(*harness, fixture_plan(false), decoder);

    SA_EXPECT_OK(report.status);
    EXPECT_NE(report.printed.find("Mean average precision : not computed"), std::string::npos)
        << report.printed;
    EXPECT_EQ(report.printed.find("Mean average precision at IoU"), std::string::npos)
        << report.printed;
}

/** @brief Tests that a report asked for detections only scores nothing although the data has boxes. */
TEST_F(Medium_DetectionReport, AReportOfDetectionsOnlyScoresNothing)
{
    const auto harness = built_harness(backend());
    harness->prepared->comparison = Apps::Comparison::DetectionsOnly;
    ScriptedDecoder decoder;
    decoder.images = {{}, {}};

    const PrintedReport report = run_report(*harness, fixture_plan(false), decoder);

    EXPECT_NE(report.printed.find("Mean average precision : not computed"), std::string::npos)
        << report.printed;
}

/** @brief Tests that a failing decoder stops the report with its failure before the total prints. */
TEST_F(Medium_DetectionReport, AFailingDecoderStopsTheReportWithItsFailure)
{
    const auto harness = built_harness(backend());
    ScriptedDecoder decoder;
    decoder.images = {{}, {}};
    decoder.failing_call = 2;

    const PrintedReport report = run_report(*harness, fixture_plan(false), decoder);

    SA_EXPECT_FAILURE(report.status, StatusCode::DeviceFailed, "the decoder stopped at call 2");
    EXPECT_EQ(decoder.calls, 2u);
    EXPECT_EQ(report.printed.find("detections over"), std::string::npos) << report.printed;
}
