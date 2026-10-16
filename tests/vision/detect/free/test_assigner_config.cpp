/**************************************************************************/
/* test_assigner_config.cpp                                               */
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
 * @file test_assigner_config.cpp
 * @brief Tests that both assigner presets carry the class count and refuse a count below one.
 * @author Mustafa Garip
 */

#include <vector>
#include <gtest/gtest.h>
#include <SushiAI/failure_expectations/throw_expectations.hpp>
#include <SushiAI/vision/detect/free/assigner_config.hpp>

namespace
{
    using SushiAI::Vision::Detect::Free::one_to_many_assigner_config;
    using SushiAI::Vision::Detect::Free::one_to_one_assigner_config;

    /** @brief Holds the strides of a three-level model. */
    const std::vector<float> K_STRIDES{8.0f, 16.0f, 32.0f};
} // namespace

/** @brief Tests that both presets hold the class count they are given. */
TEST(Small_DetectAssignerConfig, BothPresetsCarryTheClassCount)
{
    EXPECT_EQ(one_to_many_assigner_config(3, K_STRIDES).num_classes, 3);
    EXPECT_EQ(one_to_one_assigner_config(3, K_STRIDES).num_classes, 3);
}

/** @brief Tests that both presets refuse a class count of zero or below. */
TEST(Small_DetectAssignerConfig, BothPresetsRefuseAClassCountBelowOne)
{
    SA_EXPECT_ERROR(static_cast<void>(one_to_many_assigner_config(0, K_STRIDES)),
                    "positive class count", "got 0");
    SA_EXPECT_ERROR(static_cast<void>(one_to_one_assigner_config(-2, K_STRIDES)),
                    "positive class count", "got -2");
}
