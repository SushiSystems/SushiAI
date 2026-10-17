/**************************************************************************/
/* test_pixel_extents.cpp                                                 */
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
 * @file test_pixel_extents.cpp
 * @brief Tests the pixel extents a per-pixel objective reads.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <gtest/gtest.h>
#include <SushiAI/config/value.hpp>
#include <SushiAI/ops/ops.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/shape/shape.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/data/roles.hpp>
#include <SushiAI/data/target.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/graph/builder.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/nn/conv2d.hpp>
#include <SushiAI/train/objective.hpp>
#include <SushiAI/train/objective_context.hpp>
#include <SushiAI/train/steps.hpp>
#include <SushiAI/train/training_graph.hpp>
#include <SushiAI/vision/training/dice_objective.hpp>
#include <SushiAI/vision/training/pixel_cross_entropy_objective.hpp>
#include <SushiAI/vision/training/pixel_extents.hpp>

using namespace SushiAI;
using Vision::Training::DiceObjective;
using Vision::Training::make_dice_objective;
using Vision::Training::make_pixel_cross_entropy_objective;
using Vision::Training::PixelCrossEntropyObjective;
using Vision::Training::PixelExtents;

/** @brief Tests the largest batch of 512 by 512 pixels over 150 classes INT32 indexes: 54. */
TEST(Small_PixelExtents, MakeRefusesWhatInt32CannotIndex)
{
    // 54 * 512 * 512 * 150 is 2 123 366 400; one image more is 2 162 688 000.
    SA_EXPECT_OK(PixelCrossEntropyObjective::make(PixelExtents{54, 512, 512, 150}));
    SA_EXPECT_FAILURE(PixelCrossEntropyObjective::make(PixelExtents{55, 512, 512, 150}),
                      StatusCode::InvalidDescription, "55 * 512 * 512 * 150", "2147483647");

    // The Dice objective gathers one probability per pixel, so the same bound holds for it.
    SA_EXPECT_OK(DiceObjective::make(PixelExtents{54, 512, 512, 150}));
    SA_EXPECT_FAILURE(DiceObjective::make(PixelExtents{55, 512, 512, 150}),
                      StatusCode::InvalidDescription, "55 * 512 * 512 * 150", "2147483647");
}
