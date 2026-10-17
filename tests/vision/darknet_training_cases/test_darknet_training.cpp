/**************************************************************************/
/* test_darknet_training.cpp                                              */
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
 * @file test_darknet_training.cpp
 * @brief Tests the delta measure of the Darknet training cases and its loaders without a dump.
 * @author Mustafa Garip
 */

#include <cstdint>
#include <string_view>
#include <vector>
#include <gtest/gtest.h>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/vision/darknet_training_cases/darknet_training.hpp>

using namespace SushiAI;
namespace SaTest = SushiAI::Test;

namespace
{
    /** @brief Holds the entries of an anchor slot of one class: four box entries and two more. */
    constexpr std::int64_t K_SLOT = 6;

    /** @brief Names a run folder no reference data directory holds. */
    constexpr std::string_view K_ABSENT_RUN = "sushiai_no_such_run";

    /** @brief Returns bounds of 0.1 plus 1 % on a box entry and 0.01 on any other. */
    SaTest::DarknetDeltaBounds bounds()
    {
        SaTest::DarknetDeltaBounds made;
        made.box_absolute = 0.1;
        made.box_relative = 0.01;
        made.other_absolute = 0.01;
        return made;
    }

    /** @brief Returns one slot of reference deltas: box 1, 2, 3, 100, then 0.5 and 0.25. */
    std::vector<float> reference_slot()
    {
        return {1.0f, 2.0f, 3.0f, 100.0f, 0.5f, 0.25f};
    }
} // namespace

/** @brief Tests that a box entry is held to its relative bound and passes inside it. */
TEST(Small_DarknetTrainingCases, ABoxEntryIsHeldToItsRelativeBound)
{
    std::vector<float> actual = reference_slot();
    actual[3] = 101.0f;

    EXPECT_TRUE(SaTest::delta_within("box", actual, reference_slot(), K_SLOT, bounds()));

    actual[3] = 101.5f;
    EXPECT_FALSE(SaTest::delta_within("box", actual, reference_slot(), K_SLOT, bounds()));
}

/** @brief Tests that an objectness or class entry is held to the absolute bound alone. */
TEST(Small_DarknetTrainingCases, AnotherEntryIsHeldToTheAbsoluteBoundAlone)
{
    std::vector<float> actual = reference_slot();
    actual[4] = 0.505f;

    EXPECT_TRUE(SaTest::delta_within("objectness", actual, reference_slot(), K_SLOT, bounds()));

    actual[5] = 0.3f;
    EXPECT_FALSE(SaTest::delta_within("class", actual, reference_slot(), K_SLOT, bounds()));
}

/** @brief Tests that spans of other sizes or not of whole slots fail before any measure. */
TEST(Small_DarknetTrainingCases, SizesThatHoldNoWholeSlotFail)
{
    const std::vector<float> slot = reference_slot();
    const std::vector<float> shorter(slot.begin(), slot.end() - 1);

    EXPECT_FALSE(SaTest::delta_within("sizes", shorter, slot, K_SLOT, bounds()));
    EXPECT_FALSE(SaTest::delta_within("slots", shorter, shorter, K_SLOT, bounds()));
    EXPECT_FALSE(SaTest::delta_within("zero", slot, slot, 0, bounds()));
}

/** @brief Tests that the loaders of a run no reference holds fail naming the missing file. */
TEST(Small_DarknetTrainingCases, TheLoadersOfAnAbsentRunAreNotFound)
{
    SA_EXPECT_FAILURE(SaTest::load_darknet_maps(K_ABSENT_RUN, "layer_015_yolo_input"),
                      StatusCode::FileNotFound, "layer_015_yolo_input");
    SA_EXPECT_FAILURE(SaTest::load_darknet_head_dump(K_ABSENT_RUN, 16), StatusCode::FileNotFound,
                      "layer_015_yolo_input");
}
