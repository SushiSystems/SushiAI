/**************************************************************************/
/* test_yolox_task.cpp                                                    */
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
 * @file test_yolox_task.cpp
 * @brief Tests that a YOLOX task and its task section convert both ways and name a bad key.
 * @author Mustafa Garip
 */

#include <cstdint>
#include <cstring>
#include <string>
#include <string_view>

#include <gtest/gtest.h>

#include <SushiAI/config/json_writer.hpp>
#include <SushiAI/config/value.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/vision/detect/suppression_criterion.hpp>
#include <SushiAI/vision/detect/yolox/yolox_task.hpp>

namespace
{
    using SushiAI::Test::require_value;
    using SushiAI::StatusCode;
    using SushiAI::Config::Value;
    using SushiAI::Vision::Detect::K_DEFAULT_DISTANCE_EXPONENT;
    using SushiAI::Vision::Detect::SuppressionCriterion;
    using SushiAI::Vision::Detect::Yolox::YoloxTask;
    using SushiAI::Vision::Detect::Yolox::read_yolox_task;
    using SushiAI::Vision::Detect::Yolox::write_yolox_task;

    /** @brief Holds a full section in the writer's key order; every real is a short binary. */
    constexpr std::string_view K_FULL_SECTION = R"({
        "type": "yolox_detection",
        "score_threshold": 0.25,
        "suppression": {"criterion": "distance_iou", "threshold": 0.625,
                        "distance_exponent": 0.5}
    })";

    /** @brief Parses @p text as a task section. */
    Value section_from(std::string_view text)
    {
        return require_value(SushiAI::Config::parse_json(text, "<test>"));
    }

    /** @brief Reports whether @p left and @p right are the same float, bit for bit. */
    bool same_bits(float left, float right)
    {
        std::uint32_t left_bits = 0;
        std::uint32_t right_bits = 0;
        std::memcpy(&left_bits, &left, sizeof(left_bits));
        std::memcpy(&right_bits, &right, sizeof(right_bits));
        return left_bits == right_bits;
    }

    /** @brief Returns a task whose floats no short decimal states. */
    YoloxTask make_task()
    {
        YoloxTask task;
        task.score_threshold = 0.001f;
        task.suppression.criterion = SuppressionCriterion::DistanceIoU;
        task.suppression.threshold = 0.65f;
        task.suppression.distance_exponent = 0.6f;
        return task;
    }
} // namespace

/** @brief Tests that read_yolox_task(write_yolox_task(task)) returns every field bit for bit. */
TEST(Small_DetectYoloxTask, ATaskRoundTripsThroughItsSection)
{
    const YoloxTask task = make_task();
    SA_ASSERT_OK_AND_ASSIGN(const Value section, write_yolox_task(task));
    SA_ASSERT_OK_AND_ASSIGN(const std::string text, SushiAI::Config::write_json(section));
    SA_ASSERT_OK_AND_ASSIGN(const YoloxTask read, read_yolox_task(section_from(text)));

    EXPECT_TRUE(same_bits(read.score_threshold, task.score_threshold));
    EXPECT_EQ(read.suppression.criterion, task.suppression.criterion);
    EXPECT_TRUE(same_bits(read.suppression.threshold, task.suppression.threshold));
    EXPECT_TRUE(same_bits(read.suppression.distance_exponent, task.suppression.distance_exponent));
}

/** @brief Tests that a full section reads field by field. */
TEST(Small_DetectYoloxTask, ASectionReadsFieldByField)
{
    SA_ASSERT_OK_AND_ASSIGN(const YoloxTask task, read_yolox_task(section_from(K_FULL_SECTION)));

    EXPECT_EQ(task.score_threshold, 0.25f);
    EXPECT_EQ(task.suppression.criterion, SuppressionCriterion::DistanceIoU);
    EXPECT_EQ(task.suppression.threshold, 0.625f);
    EXPECT_EQ(task.suppression.distance_exponent, 0.5f);
}

/** @brief Tests that a task read from a full section writes back to the same JSON. */
TEST(Small_DetectYoloxTask, ASectionWritesBackToTheSameJson)
{
    const Value section = section_from(K_FULL_SECTION);
    SA_ASSERT_OK_AND_ASSIGN(const YoloxTask task, read_yolox_task(section));

    SA_ASSERT_OK_AND_ASSIGN(const Value written, write_yolox_task(task));
    SA_ASSERT_OK_AND_ASSIGN(const std::string found, SushiAI::Config::write_json(written));
    SA_ASSERT_OK_AND_ASSIGN(const std::string wanted, SushiAI::Config::write_json(section));
    EXPECT_EQ(found, wanted);
}

/** @brief Tests that an IoU section keeps the default distance exponent. */
TEST(Small_DetectYoloxTask, AnIoUSectionKeepsTheDefaultDistanceExponent)
{
    SA_ASSERT_OK_AND_ASSIGN(const YoloxTask task, read_yolox_task(section_from(
        R"({"type": "yolox_detection", "score_threshold": 0.5,
            "suppression": {"criterion": "iou", "threshold": 0.5}})")));

    EXPECT_EQ(task.suppression.criterion, SuppressionCriterion::IoU);
    EXPECT_EQ(task.suppression.distance_exponent, K_DEFAULT_DISTANCE_EXPONENT);
}

/** @brief Tests that another type is refused, naming the key and the type it wants. */
TEST(Small_DetectYoloxTask, RefusesAnotherType)
{
    SA_EXPECT_FAILURE(read_yolox_task(section_from(R"({"type": "anchor_free_detection"})")),
                      StatusCode::InvalidDescription, "'type'", "'yolox_detection'");
}

/** @brief Tests that an unknown key is refused by name. */
TEST(Small_DetectYoloxTask, RefusesAnUnknownKey)
{
    SA_EXPECT_FAILURE(read_yolox_task(section_from(R"({"type": "yolox_detection",
                                                       "strides": [8, 16]})")),
                      StatusCode::InvalidDescription, "'strides'");
}

/** @brief Tests that an unknown key of the suppression object is refused by name. */
TEST(Small_DetectYoloxTask, RefusesAnUnknownSuppressionKey)
{
    SA_EXPECT_FAILURE(read_yolox_task(section_from(R"({"type": "yolox_detection",
        "score_threshold": 0.5,
        "suppression": {"criterion": "iou", "threshold": 0.5, "class_agnostic": true}})")),
                      StatusCode::InvalidDescription, "'class_agnostic'", "'suppression'");
}

/** @brief Tests that a section without its score threshold is refused by name. */
TEST(Small_DetectYoloxTask, RefusesAMissingScoreThreshold)
{
    SA_EXPECT_FAILURE(read_yolox_task(section_from(R"({"type": "yolox_detection",
        "suppression": {"criterion": "iou", "threshold": 0.5}})")),
                      StatusCode::InvalidDescription, "'score_threshold'");
}

/** @brief Tests that a suppression without its threshold is refused by name. */
TEST(Small_DetectYoloxTask, RefusesAMissingThreshold)
{
    SA_EXPECT_FAILURE(read_yolox_task(section_from(R"({"type": "yolox_detection",
        "score_threshold": 0.5, "suppression": {"criterion": "iou"}})")),
                      StatusCode::InvalidDescription, "'threshold'");
}

/** @brief Tests that a score threshold out of range is refused, naming the key and the value. */
TEST(Small_DetectYoloxTask, RefusesAScoreThresholdOutOfRange)
{
    SA_EXPECT_FAILURE(read_yolox_task(section_from(R"({"type": "yolox_detection",
        "score_threshold": 1.5, "suppression": {"criterion": "iou", "threshold": 0.5}})")),
                      StatusCode::InvalidDescription, "'score_threshold'", "1.5");
}

/** @brief Tests that writing a task whose score threshold is negative is refused. */
TEST(Small_DetectYoloxTask, RefusesANegativeScoreThresholdOnWrite)
{
    YoloxTask negative = make_task();
    negative.score_threshold = -0.5f;

    SA_EXPECT_FAILURE(write_yolox_task(negative), StatusCode::InvalidDescription,
                      "'score_threshold'");
}
