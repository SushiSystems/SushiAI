/**************************************************************************/
/* test_detection_reports.cpp                                             */
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
 * @file test_detection_reports.cpp
 * @brief Tests the anchor, anchor-free and YOLOX detection reports over a prepared inference.
 * @author Mustafa Garip
 */

#include "detection_reports.hpp"

#include <memory>
#include <string>
#include <string_view>
#include <gtest/gtest.h>
#include <SushiAI/config/value.hpp>
#include <SushiAI/engine_fixture/engine_test.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/shape/shape.hpp>

#include "prepared_inference_harness.hpp"

using namespace SushiAI;
// SaTest stays reachable inside a TEST_F body; see tests/applications/programs/README.md.
namespace SaTest = SushiAI::Test;
namespace Prepared = SushiAI::Apps::PreparedTest;

namespace
{
    /** @brief Gives each case an engine on the shared test runtime. */
    class Medium_DetectionReports : public SaTest::EngineTest {};

    /** @brief The task section of an anchor-free model, which asks for no assignment. */
    constexpr std::string_view K_FREE_TASK = R"({"type": "anchor_free_detection",
        "score_threshold": 0.5, "suppression": {"criterion": "iou", "threshold": 0.5}})";

    /** @brief The task section of a YOLOX model with a score threshold of 0.5. */
    constexpr std::string_view K_YOLOX_TASK = R"({"type": "yolox_detection", "score_threshold": 0.5,
        "suppression": {"criterion": "iou", "threshold": 0.5}})";

    /** @brief The task section of the fixture, whose one head names an output no model publishes. */
    constexpr std::string_view K_ANCHOR_TASK_WITH_A_MISSING_HEAD = R"({
        "type": "anchor_detection", "classes": 2, "anchors": [[0.25, 0.25], [0.5, 0.5]],
        "heads": [{"output": "head.9", "anchors": [0, 1], "coding": "logit", "scale_xy": 1.0}],
        "suppression": {"criterion": "iou"}})";

    /** @brief Holds what a report returned and the text it printed. */
    struct PrintedReport
    {
        /** @brief What the report returned. */
        Status status;

        /** @brief What the report wrote to standard output. */
        std::string printed;
    };

    /** @brief Builds the harness over the fixture, or stops the process with the refusal. */
    std::unique_ptr<Prepared::InferenceHarness> built_harness(SushiAI::ExecutionBackend& backend)
    {
        return SaTest::require_value(Prepared::make_inference_harness(backend));
    }

    /** @brief Parses @p text as a task section, or stops the process with the refusal. */
    Config::Value task_section(std::string_view text)
    {
        return SaTest::require_value(Config::parse_json(text, "<test>"));
    }

    /** @brief Runs the anchor report and captures what it prints. */
    PrintedReport anchor_report(Prepared::InferenceHarness& harness)
    {
        testing::internal::CaptureStdout();
        const Status status = Apps::report_anchor_detections(*harness.prepared);
        return PrintedReport{status, testing::internal::GetCapturedStdout()};
    }
} // namespace

/** @brief Tests that the anchor report names its heads and classes and totals the images. */
TEST_F(Medium_DetectionReports, TheAnchorReportNamesItsHeadsAndClassesAndTotalsEveryImage)
{
    const auto harness = built_harness(backend());

    const PrintedReport report = anchor_report(*harness);

    SA_EXPECT_OK(report.status);
    EXPECT_NE(report.printed.find("1 heads, 2 classes"), std::string::npos) << report.printed;
    EXPECT_NE(report.printed.find("detections over 4 images."), std::string::npos)
        << report.printed;
}

/** @brief Tests that the anchor report scores its detections against the labels of the split. */
TEST_F(Medium_DetectionReports, TheAnchorReportEndsWithTheMeanAveragePrecision)
{
    const auto harness = built_harness(backend());

    const PrintedReport report = anchor_report(*harness);

    EXPECT_NE(report.printed.find("Mean average precision at IoU 0.5"), std::string::npos)
        << report.printed;
}

/** @brief Tests that the anchor report measures its boxes in the source image of each file. */
TEST_F(Medium_DetectionReports, TheAnchorReportMeasuresItsBoxesInTheSourceImageOfEachFile)
{
    const auto harness = built_harness(backend());

    const PrintedReport report = anchor_report(*harness);

    EXPECT_NE(report.printed.find("in pixels of the source image of each file"),
              std::string::npos)
        << report.printed;
}

/** @brief Tests that the thresholds of the evaluation file replace the defaults of the report. */
TEST_F(Medium_DetectionReports, TheAnchorReportReadsItsThresholdsFromTheEvaluationFile)
{
    const auto harness = built_harness(backend());
    harness->prepared->detection.score_threshold = 0.75f;
    harness->prepared->detection.suppression_threshold = 0.3f;

    const PrintedReport report = anchor_report(*harness);

    EXPECT_NE(report.printed.find("score above 0.75, suppression above 0.3;"), std::string::npos)
        << report.printed;
}

/** @brief Tests that the anchor report keeps its default thresholds when the file states none. */
TEST_F(Medium_DetectionReports, TheAnchorReportKeepsItsDefaultThresholdsWhenTheFileStatesNone)
{
    const auto harness = built_harness(backend());

    const PrintedReport report = anchor_report(*harness);

    EXPECT_NE(report.printed.find("score above 0.25, suppression above 0.45;"), std::string::npos)
        << report.printed;
}

/** @brief Tests that a score threshold below the confidence floor is refused before anything prints. */
TEST_F(Medium_DetectionReports, TheAnchorReportRefusesAScoreThresholdBelowTheConfidenceFloor)
{
    const auto harness = built_harness(backend());
    harness->prepared->detection.score_threshold = 0.1f;
    harness->prepared->detection.confidence_floor = 0.5f;

    const PrintedReport report = anchor_report(*harness);

    SA_EXPECT_FAILURE(report.status, StatusCode::InvalidDescription, "confidence floor 0.5");
    EXPECT_TRUE(report.printed.empty()) << report.printed;
}

/** @brief Tests that an input that is not height, width, depth is refused before anything prints. */
TEST_F(Medium_DetectionReports, TheAnchorReportRefusesAnInputThatIsNotHeightWidthDepth)
{
    const auto harness = built_harness(backend());
    harness->model.input_shape = Shape{16, 16};

    const PrintedReport report = anchor_report(*harness);

    SA_EXPECT_FAILURE(report.status, StatusCode::InvalidDescription, "an anchor model");
    EXPECT_TRUE(report.printed.empty()) << report.printed;
}

/** @brief Tests that a head naming an output the model does not publish stops the report early. */
TEST_F(Medium_DetectionReports, TheAnchorReportRefusesAHeadThatNamesNoOutputOfTheModel)
{
    const auto harness = built_harness(backend());
    harness->model.task = task_section(K_ANCHOR_TASK_WITH_A_MISSING_HEAD);

    const PrintedReport report = anchor_report(*harness);

    EXPECT_FALSE(report.status.is_ok());
    EXPECT_TRUE(report.printed.empty()) << report.printed;
}

/** @brief Tests that the anchor-free report refuses a task section of another type. */
TEST_F(Medium_DetectionReports, TheAnchorFreeReportRefusesATaskSectionOfAnotherType)
{
    const auto harness = built_harness(backend());

    SA_EXPECT_FAILURE(Apps::report_free_detections(*harness->prepared),
                      StatusCode::InvalidDescription, "'type'", "'anchor_free_detection'");
}

/** @brief Tests that the anchor-free report refuses an input that is not height, width, depth. */
TEST_F(Medium_DetectionReports, TheAnchorFreeReportRefusesAnInputThatIsNotHeightWidthDepth)
{
    const auto harness = built_harness(backend());
    harness->model.task = task_section(K_FREE_TASK);
    harness->model.input_shape = Shape{16, 16};

    SA_EXPECT_FAILURE(Apps::report_free_detections(*harness->prepared),
                      StatusCode::InvalidDescription, "an anchor-free model");
}

/** @brief Tests that the YOLOX report refuses a task section of another type. */
TEST_F(Medium_DetectionReports, TheYoloxReportRefusesATaskSectionOfAnotherType)
{
    const auto harness = built_harness(backend());

    SA_EXPECT_FAILURE(Apps::report_yolox_detections(*harness->prepared),
                      StatusCode::InvalidDescription, "'type'", "'yolox_detection'");
}

/** @brief Tests that the YOLOX report refuses an input that is not height, width, depth. */
TEST_F(Medium_DetectionReports, TheYoloxReportRefusesAnInputThatIsNotHeightWidthDepth)
{
    const auto harness = built_harness(backend());
    harness->model.task = task_section(K_YOLOX_TASK);
    harness->model.input_shape = Shape{16, 16};

    SA_EXPECT_FAILURE(Apps::report_yolox_detections(*harness->prepared),
                      StatusCode::InvalidDescription, "a YOLOX model");
}

/** @brief Tests that the YOLOX report refuses a score threshold below the confidence floor. */
TEST_F(Medium_DetectionReports, TheYoloxReportRefusesAScoreThresholdBelowTheConfidenceFloor)
{
    const auto harness = built_harness(backend());
    harness->model.task = task_section(K_YOLOX_TASK);
    harness->prepared->detection.confidence_floor = 0.75f;

    SA_EXPECT_FAILURE(Apps::report_yolox_detections(*harness->prepared),
                      StatusCode::InvalidDescription, "confidence floor 0.75");
}
