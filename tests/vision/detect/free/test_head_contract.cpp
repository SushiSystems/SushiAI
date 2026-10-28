/**************************************************************************/
/* test_head_contract.cpp                                                 */
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
 * @file test_head_contract.cpp
 * @brief Tests the port names and the class bias an anchor-free head and its training share.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <string>
#include <string_view>
#include <gtest/gtest.h>
#include <SushiAI/vision/detect/free/head_contract.hpp>

namespace
{
    using namespace SushiAI::Vision::Detect::Free;

    /** @brief Bounds the error of a class bias against the value computed apart, in nats. */
    constexpr double K_BIAS_TOLERANCE = 1e-12;
} // namespace

/** @brief Tests that the copy, role and branch names equal the literals checkpoints carry. */
TEST(Small_DetectFreeHeadContract, TheNamesEqualTheLiteralsOfTheCheckpoints)
{
    EXPECT_EQ(K_DETECT_ONE_TO_MANY, "one2many");
    EXPECT_EQ(K_DETECT_ONE_TO_ONE, "one2one");
    EXPECT_EQ(K_DETECT_BOXES, "boxes");
    EXPECT_EQ(K_DETECT_SCORES, "scores");
    EXPECT_EQ(K_DETECT_BOX_BRANCH, "cv2");
    EXPECT_EQ(K_DETECT_CLASS_BRANCH, "cv3");
    EXPECT_EQ(K_DETECT_ONE_TO_ONE_BOX_BRANCH, "one2one_cv2");
    EXPECT_EQ(K_DETECT_ONE_TO_ONE_CLASS_BRANCH, "one2one_cv3");
}

/** @brief Tests that the bias constants are the ones the class bias formula reads. */
TEST(Small_DetectFreeHeadContract, TheBiasConstantsAreTheReferenceValues)
{
    EXPECT_EQ(K_DETECT_LAST_CONV_INDEX, std::size_t{2});
    EXPECT_EQ(K_DETECT_BOX_BIAS, 1.0);
    EXPECT_EQ(K_DETECT_OBJECTS_PER_IMAGE, 5.0);
    EXPECT_EQ(K_DETECT_REFERENCE_IMAGE_SIDE, 640.0);
}

/** @brief Tests that a port name joins the copy, the level and the role with dots. */
TEST(Small_DetectFreeHeadContract, ThePortNameJoinsCopyLevelAndRole)
{
    EXPECT_EQ(detect_port_name(K_DETECT_ONE_TO_MANY, 0, K_DETECT_BOXES), "one2many.level0.boxes");
    EXPECT_EQ(detect_port_name(K_DETECT_ONE_TO_MANY, 2, K_DETECT_SCORES),
              "one2many.level2.scores");
    EXPECT_EQ(detect_port_name(K_DETECT_ONE_TO_ONE, 12, K_DETECT_SCORES),
              "one2one.level12.scores");
}

/** @brief Tests that a port name takes any copy and role, not only the published ones. */
TEST(Small_DetectFreeHeadContract, ThePortNameKeepsWhatItIsGiven)
{
    EXPECT_EQ(detect_port_name("copy", 3, "role"), "copy.level3.role");
    EXPECT_EQ(detect_port_name("", 1, ""), ".level1.");
}

/** @brief Tests the class bias log(5 / classes / (640 / stride)^2) against values from outside. */
TEST(Small_DetectFreeHeadContract, TheClassBiasFollowsTheStrideAndTheClassCount)
{
    EXPECT_NEAR(detect_class_bias(80, 8.0), -11.536641991587544, K_BIAS_TOLERANCE);
    EXPECT_NEAR(detect_class_bias(80, 16.0), -10.150347630467653, K_BIAS_TOLERANCE);
    EXPECT_NEAR(detect_class_bias(80, 32.0), -8.764053269347762, K_BIAS_TOLERANCE);
    EXPECT_NEAR(detect_class_bias(1, 32.0), -4.382026634673881, K_BIAS_TOLERANCE);
    EXPECT_NEAR(detect_class_bias(20, 64.0), -5.991464547107982, K_BIAS_TOLERANCE);
}

/** @brief Tests that a larger stride and fewer classes raise the bias, which is below zero. */
TEST(Small_DetectFreeHeadContract, TheClassBiasRisesWithTheStrideAndFallsWithTheClasses)
{
    EXPECT_LT(detect_class_bias(80, 8.0), detect_class_bias(80, 16.0));
    EXPECT_LT(detect_class_bias(80, 16.0), detect_class_bias(80, 32.0));
    EXPECT_LT(detect_class_bias(80, 32.0), detect_class_bias(1, 32.0));
    EXPECT_LT(detect_class_bias(1, 32.0), 0.0);
}
