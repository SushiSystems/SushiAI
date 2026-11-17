/**************************************************************************/
/* test_suppression_section.cpp                                           */
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
 * @file test_suppression_section.cpp
 * @brief Tests the reader and the writer of a task section's "suppression" object.
 * @author Mustafa Garip
 */

#include <string>
#include <string_view>

#include <gtest/gtest.h>

#include <SushiAI/config/json_writer.hpp>
#include <SushiAI/config/value.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/vision/detect/suppression_section.hpp>

namespace
{
    using SushiAI::Test::require_value;
    using SushiAI::StatusCode;
    using SushiAI::Config::Value;
    using SushiAI::Vision::Detect::SuppressionCriterion;
    using SushiAI::Vision::Detect::SuppressionSection;
    using SushiAI::Vision::Detect::read_suppression_section;
    using SushiAI::Vision::Detect::write_suppression_section;

    /** @brief Names the task a refusal starts with. */
    constexpr std::string_view K_WHERE = "task";

    /** @brief Parses @p text as a task section. */
    Value section_from(std::string_view text)
    {
        return require_value(SushiAI::Config::parse_json(text, "<test>"));
    }
} // namespace

/** @brief Tests that a distance_iou object reads its three fields. */
TEST(Small_DetectSuppressionSection, ADistanceIoUObjectReadsItsThreeFields)
{
    SA_ASSERT_OK_AND_ASSIGN(const SuppressionSection section, read_suppression_section(
        section_from(R"({"suppression": {"criterion": "distance_iou", "threshold": 0.625,
                                         "distance_exponent": 0.5}})"), K_WHERE));

    EXPECT_EQ(section.criterion, SuppressionCriterion::DistanceIoU);
    EXPECT_EQ(section.threshold, 0.625f);
    EXPECT_EQ(section.distance_exponent, 0.5f);
}

/** @brief Tests that an iou object is written without the exponent it does not read. */
TEST(Small_DetectSuppressionSection, AnIoUObjectIsWrittenWithoutTheExponent)
{
    SuppressionSection section;
    section.criterion = SuppressionCriterion::IoU;
    section.threshold = 0.5f;

    SA_ASSERT_OK_AND_ASSIGN(const Value written, write_suppression_section(section, K_WHERE));
    SA_ASSERT_OK_AND_ASSIGN(const std::string text, SushiAI::Config::write_json(written));
    EXPECT_EQ(text.find("distance_exponent"), std::string::npos);
}

/** @brief Tests that an absent "suppression" member is refused, naming it. */
TEST(Small_DetectSuppressionSection, RefusesAnAbsentObject)
{
    SA_EXPECT_FAILURE(read_suppression_section(section_from(R"({})"), K_WHERE),
                      StatusCode::InvalidDescription, "'suppression'");
}

/** @brief Tests that an unknown criterion is refused, naming the value and the choices. */
TEST(Small_DetectSuppressionSection, RefusesAnUnknownCriterion)
{
    SA_EXPECT_FAILURE(read_suppression_section(section_from(
        R"({"suppression": {"criterion": "giou", "threshold": 0.5}})"), K_WHERE),
                      StatusCode::InvalidDescription, "'criterion'", "'giou'", "'distance_iou'");
}

/** @brief Tests that a threshold above 1 is refused, naming the key and the value. */
TEST(Small_DetectSuppressionSection, RefusesAThresholdAboveOne)
{
    SA_EXPECT_FAILURE(read_suppression_section(section_from(
        R"({"suppression": {"criterion": "iou", "threshold": 1.5}})"), K_WHERE),
                      StatusCode::InvalidDescription, "'threshold'", "1.5");
}

/** @brief Tests that an exponent under the iou criterion is refused. */
TEST(Small_DetectSuppressionSection, RefusesAnExponentOnIoU)
{
    SA_EXPECT_FAILURE(read_suppression_section(section_from(
        R"({"suppression": {"criterion": "iou", "threshold": 0.5, "distance_exponent": 0.6}})"),
                                               K_WHERE),
                      StatusCode::InvalidDescription, "'distance_exponent'", "'iou'");
}

/** @brief Tests that the exponent is required under distance_iou. */
TEST(Small_DetectSuppressionSection, RefusesAMissingExponentOnDistanceIoU)
{
    SA_EXPECT_FAILURE(read_suppression_section(section_from(
        R"({"suppression": {"criterion": "distance_iou", "threshold": 0.5}})"), K_WHERE),
                      StatusCode::InvalidDescription, "'distance_exponent'");
}
