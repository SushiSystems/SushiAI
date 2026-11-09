/**************************************************************************/
/* test_free_task.cpp                                                     */
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
 * @file test_free_task.cpp
 * @brief Tests that an anchor-free task and its task section convert both ways without loss.
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
#include <SushiAI/vision/detect/free/assigner_config.hpp>
#include <SushiAI/vision/detect/free/free_task.hpp>
#include <SushiAI/vision/detect/suppression_criterion.hpp>

namespace
{
    using SushiAI::Test::require_value;
    using SushiAI::StatusCode;
    using SushiAI::Config::Value;
    using SushiAI::Vision::Detect::K_DEFAULT_DISTANCE_EXPONENT;
    using SushiAI::Vision::Detect::SuppressionCriterion;
    using SushiAI::Vision::Detect::Free::AssignerConfig;
    using SushiAI::Vision::Detect::Free::FreeTask;
    using SushiAI::Vision::Detect::Free::read_free_task;
    using SushiAI::Vision::Detect::Free::write_free_task;

    /** @brief Holds a full section in the writer's key order; every real is a short binary. */
    constexpr std::string_view K_FULL_SECTION = R"({
        "type": "anchor_free_detection",
        "score_threshold": 0.25,
        "suppression": {"criterion": "distance_iou", "threshold": 0.625,
                        "distance_exponent": 0.5},
        "assignment": {
            "one_to_many": {"top_k": 10, "top_k2": 10, "alpha": 0.5, "beta": 6.0},
            "one_to_one": {"top_k": 7, "top_k2": 1, "alpha": 0.5, "beta": 6.0}
        }
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

    /** @brief Expects the section's fields of @p found and @p wanted to be equal. */
    void expect_same_branch(const AssignerConfig& found, const AssignerConfig& wanted)
    {
        EXPECT_EQ(found.topk, wanted.topk);
        EXPECT_EQ(found.topk2, wanted.topk2);
        EXPECT_TRUE(same_bits(found.alpha, wanted.alpha));
        EXPECT_TRUE(same_bits(found.beta, wanted.beta));
    }

    /** @brief Expects every field the section holds to be equal, floats bit for bit. */
    void expect_same_task(const FreeTask& found, const FreeTask& wanted)
    {
        EXPECT_TRUE(same_bits(found.score_threshold, wanted.score_threshold));
        EXPECT_TRUE(same_bits(found.suppression_threshold, wanted.suppression_threshold));
        EXPECT_EQ(found.criterion, wanted.criterion);
        EXPECT_TRUE(same_bits(found.distance_exponent, wanted.distance_exponent));
        expect_same_branch(found.one_to_many, wanted.one_to_many);
        expect_same_branch(found.one_to_one, wanted.one_to_one);
    }

    /** @brief Reads a section that states no "assignment" and an IoU suppression. */
    SushiAI::Result<FreeTask> read_without_assignment()
    {
        return read_free_task(section_from(R"({"type": "anchor_free_detection",
                                               "score_threshold": 0.5,
                                               "suppression": {"criterion": "iou",
                                                               "threshold": 0.5}})"));
    }

    /** @brief Returns a task whose floats no short decimal states. */
    FreeTask make_task()
    {
        FreeTask task;
        task.score_threshold = 0.001f;
        task.suppression_threshold = 0.7f;
        task.criterion = SuppressionCriterion::DistanceIoU;
        task.distance_exponent = 0.6f;
        task.one_to_many.topk = 13;
        task.one_to_many.topk2 = 11;
        task.one_to_many.alpha = 0.3f;
        task.one_to_many.beta = 5.9f;
        return task;
    }
} // namespace

/** @brief Tests that read_free_task(write_free_task(task)) returns every field bit for bit. */
TEST(Small_DetectFreeTask, ATaskRoundTripsThroughItsSection)
{
    const FreeTask task = make_task();
    SA_ASSERT_OK_AND_ASSIGN(const Value section, write_free_task(task));
    SA_ASSERT_OK_AND_ASSIGN(const std::string text, SushiAI::Config::write_json(section));
    SA_ASSERT_OK_AND_ASSIGN(const FreeTask read, read_free_task(section_from(text)));
    expect_same_task(read, task);
}

/** @brief Tests that a full section reads field by field. */
TEST(Small_DetectFreeTask, ASectionReadsFieldByField)
{
    SA_ASSERT_OK_AND_ASSIGN(const FreeTask task, read_free_task(section_from(K_FULL_SECTION)));

    EXPECT_EQ(task.score_threshold, 0.25f);
    EXPECT_EQ(task.suppression_threshold, 0.625f);
    EXPECT_EQ(task.criterion, SuppressionCriterion::DistanceIoU);
    EXPECT_EQ(task.distance_exponent, 0.5f);
    EXPECT_EQ(task.one_to_many.topk, 10);
    EXPECT_EQ(task.one_to_many.topk2, 10);
    EXPECT_EQ(task.one_to_one.topk, 7);
    EXPECT_EQ(task.one_to_one.topk2, 1);
    EXPECT_EQ(task.one_to_one.alpha, 0.5f);
    EXPECT_EQ(task.one_to_one.beta, 6.0f);
}

/** @brief Tests that a task read from a full section writes back to the same JSON. */
TEST(Small_DetectFreeTask, ASectionWritesBackToTheSameJson)
{
    const Value section = section_from(K_FULL_SECTION);
    SA_ASSERT_OK_AND_ASSIGN(const FreeTask task, read_free_task(section));

    SA_ASSERT_OK_AND_ASSIGN(const Value written, write_free_task(task));
    SA_ASSERT_OK_AND_ASSIGN(const std::string found, SushiAI::Config::write_json(written));
    SA_ASSERT_OK_AND_ASSIGN(const std::string wanted, SushiAI::Config::write_json(section));
    EXPECT_EQ(found, wanted);
}

/** @brief Tests that without "assignment" both branches keep the presets. */
TEST(Small_DetectFreeTask, AnAbsentAssignmentKeepsTheBranchPresets)
{
    SA_ASSERT_OK_AND_ASSIGN(const FreeTask task, read_without_assignment());

    const FreeTask defaults;
    expect_same_branch(task.one_to_many, defaults.one_to_many);
    expect_same_branch(task.one_to_one, defaults.one_to_one);
    EXPECT_EQ(task.one_to_one.topk, 7);
    EXPECT_EQ(task.one_to_one.topk2, 1);
}

/** @brief Tests that an IoU section without "distance_exponent" keeps the default exponent. */
TEST(Small_DetectFreeTask, AnIoUSectionKeepsTheDefaultDistanceExponent)
{
    SA_ASSERT_OK_AND_ASSIGN(const FreeTask task, read_without_assignment());

    EXPECT_EQ(task.distance_exponent, K_DEFAULT_DISTANCE_EXPONENT);
}

/** @brief Tests that another type is refused, naming the key and the type it wants. */
TEST(Small_DetectFreeTask, RefusesAnotherType)
{
    SA_EXPECT_FAILURE(read_free_task(section_from(R"({"type": "anchor_detection"})")),
                      StatusCode::InvalidDescription, "'type'", "'anchor_free_detection'");
}

/** @brief Tests that an unknown key is refused by name. */
TEST(Small_DetectFreeTask, RefusesAnUnknownKey)
{
    SA_EXPECT_FAILURE(read_free_task(section_from(R"({"type": "anchor_free_detection",
                                                      "strides": [8, 16]})")),
                      StatusCode::InvalidDescription, "'strides'");
}

/** @brief Tests that a suppression without its threshold is refused by name. */
TEST(Small_DetectFreeTask, RefusesAMissingThreshold)
{
    SA_EXPECT_FAILURE(read_free_task(section_from(R"({"type": "anchor_free_detection",
        "score_threshold": 0.5, "suppression": {"criterion": "iou"}})")),
                      StatusCode::InvalidDescription, "'threshold'");
}

/** @brief Tests that a branch without top_k2 is refused by name. */
TEST(Small_DetectFreeTask, RefusesAMissingTopK2)
{
    SA_EXPECT_FAILURE(read_free_task(section_from(R"({"type": "anchor_free_detection",
        "score_threshold": 0.5, "suppression": {"criterion": "iou", "threshold": 0.5},
        "assignment": {"one_to_many": {"top_k": 10, "alpha": 0.5, "beta": 6.0}}})")),
                      StatusCode::InvalidDescription, "'top_k2'");
}

/** @brief Tests that a score threshold out of range is refused, naming the key and the value. */
TEST(Small_DetectFreeTask, RefusesAScoreThresholdOutOfRange)
{
    SA_EXPECT_FAILURE(read_free_task(section_from(R"({"type": "anchor_free_detection",
        "score_threshold": 1.5, "suppression": {"criterion": "iou", "threshold": 0.5}})")),
                      StatusCode::InvalidDescription, "'score_threshold'", "1.5");
}

/** @brief Tests that a distance exponent on the IoU criterion is refused. */
TEST(Small_DetectFreeTask, RefusesADistanceExponentOnIoU)
{
    SA_EXPECT_FAILURE(read_free_task(section_from(R"({"type": "anchor_free_detection",
        "score_threshold": 0.5,
        "suppression": {"criterion": "iou", "threshold": 0.5, "distance_exponent": 0.6}})")),
                      StatusCode::InvalidDescription, "'distance_exponent'", "'iou'");
}

/** @brief Tests that a top_k2 above top_k is refused, naming the key and the value. */
TEST(Small_DetectFreeTask, RefusesATopK2AboveTopK)
{
    SA_EXPECT_FAILURE(read_free_task(section_from(R"({"type": "anchor_free_detection",
        "score_threshold": 0.5, "suppression": {"criterion": "iou", "threshold": 0.5},
        "assignment": {"one_to_one": {"top_k": 7, "top_k2": 8, "alpha": 0.5, "beta": 6.0}}})")),
                      StatusCode::InvalidDescription, "'top_k2'", "8");
}

/** @brief Tests that writing a task with a negative beta is refused, naming the key. */
TEST(Small_DetectFreeTask, RefusesANegativeBetaOnWrite)
{
    FreeTask negative = make_task();
    negative.one_to_one.beta = -1.0f;

    SA_EXPECT_FAILURE(write_free_task(negative), StatusCode::InvalidDescription, "'beta'");
}
