/**************************************************************************/
/* test_assignment_section.cpp                                            */
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
 * @file test_assignment_section.cpp
 * @brief Tests the "assignment" member of an anchor task section, read and written.
 * @author Mustafa Garip
 */

#include <string>
#include <string_view>
#include <gtest/gtest.h>
#include <SushiAI/config/json_writer.hpp>
#include <SushiAI/config/value.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/vision/detect/anchor/anchor_task.hpp>
#include <SushiAI/vision/detect/anchor/assignment_section.hpp>

namespace
{
    using SushiAI::Test::require_value;
    using SushiAI::StatusCode;
    using SushiAI::Config::Value;
    using SushiAI::Vision::Detect::Anchor::AnchorTask;
    using SushiAI::Vision::Detect::Anchor::AssignmentRuleKind;
    using SushiAI::Vision::Detect::Anchor::read_anchor_task;
    using SushiAI::Vision::Detect::Anchor::write_anchor_task;

    /** @brief Returns a two-head section whose "assignment" member is @p assignment, if any. */
    Value section_with(std::string_view assignment)
    {
        std::string text = R"({"type": "anchor_detection", "classes": 3,
            "anchors": [[0.125, 0.25], [0.25, 0.5], [0.5, 0.25], [0.75, 0.5]],
            "heads": [{"output": "a", "anchors": [2, 3], "coding": "logit", "scale_xy": 2.0},
                      {"output": "b", "anchors": [0, 1], "coding": "logit", "scale_xy": 2.0}],
            "suppression": {"criterion": "iou"})";
        if (!assignment.empty())
            text += R"(, "assignment": )" + std::string(assignment);
        return require_value(SushiAI::Config::parse_json(text + "}", "<test>"));
    }
} // namespace

/** @brief Tests that a section without the member selects the best-shape rule. */
TEST(Small_DetectAnchorAssignmentSection, ASectionWithoutTheMemberSelectsBestShape)
{
    SA_ASSERT_OK_AND_ASSIGN(const AnchorTask task, read_anchor_task(section_with("")));
    EXPECT_EQ(task.assignment.rule, AssignmentRuleKind::BestShape);

    SA_ASSERT_OK_AND_ASSIGN(const Value written, write_anchor_task(task));
    EXPECT_FALSE(written.has("assignment"));
}

/** @brief Tests that the shape-ratio rule reads its defaults when the keys are left out. */
TEST(Small_DetectAnchorAssignmentSection, ShapeRatioReadsItsDefaults)
{
    SA_ASSERT_OK_AND_ASSIGN(const AnchorTask task,
                            read_anchor_task(section_with(R"({"rule": "shape_ratio"})")));
    EXPECT_EQ(task.assignment.rule, AssignmentRuleKind::ShapeRatio);
    EXPECT_EQ(task.assignment.shape_ratio.ratio_limit, 4.0f);
    EXPECT_EQ(task.assignment.shape_ratio.neighbour_offset, 0.5f);
}

/** @brief Tests that a shape-ratio section is written back with the same JSON text. */
TEST(Small_DetectAnchorAssignmentSection, AShapeRatioSectionRoundTrips)
{
    const Value section = section_with(
        R"({"rule": "shape_ratio", "ratio_limit": 3.5, "neighbour_offset": 0.25})");
    SA_ASSERT_OK_AND_ASSIGN(const AnchorTask task, read_anchor_task(section));
    EXPECT_EQ(task.assignment.shape_ratio.ratio_limit, 3.5f);
    EXPECT_EQ(task.assignment.shape_ratio.neighbour_offset, 0.25f);

    SA_ASSERT_OK_AND_ASSIGN(const Value written, write_anchor_task(task));
    SA_ASSERT_OK_AND_ASSIGN(const std::string written_text, SushiAI::Config::write_json(written));
    SA_ASSERT_OK_AND_ASSIGN(const std::string section_text, SushiAI::Config::write_json(section));
    EXPECT_EQ(written_text, section_text);
}

/** @brief Tests that an unknown rule, an unknown key and bad values are refused by name. */
TEST(Small_DetectAnchorAssignmentSection, RefusesBadMembersNamingTheKey)
{
    SA_EXPECT_FAILURE(read_anchor_task(section_with(R"({"rule": "nearest"})")),
                      StatusCode::InvalidDescription, "'rule'", "'nearest'", "'shape_ratio'");
    SA_EXPECT_FAILURE(read_anchor_task(section_with(R"({"rule": "shape_ratio", "g": 0.5})")),
                      StatusCode::InvalidDescription, "'g'");
    SA_EXPECT_FAILURE(
        read_anchor_task(section_with(R"({"rule": "shape_ratio", "ratio_limit": 0.5})")),
        StatusCode::InvalidDescription, "'ratio_limit'");
    SA_EXPECT_FAILURE(
        read_anchor_task(section_with(R"({"rule": "shape_ratio", "neighbour_offset": 1.0})")),
        StatusCode::InvalidDescription, "'neighbour_offset'");
    SA_EXPECT_FAILURE(
        read_anchor_task(section_with(R"({"rule": "best_shape", "ratio_limit": 4.0})")),
        StatusCode::InvalidDescription, "'ratio_limit'", "'best_shape'");
}
