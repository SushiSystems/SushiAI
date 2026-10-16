/**************************************************************************/
/* test_prediction_shape.cpp                                              */
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
 * @file test_prediction_shape.cpp
 * @brief Tests the default extents of a prediction and the order its fields take.
 * @author Mustafa Garip
 */

#include <gtest/gtest.h>
#include <SushiAI/vision/detect/prediction_shape.hpp>

namespace
{
    using SushiAI::Vision::Detect::PredictionShape;
} // namespace

/** @brief Tests that a default shape has no images, no anchors and no classes. */
TEST(Small_DetectPredictionShape, ADefaultShapeHasNoExtents)
{
    const PredictionShape shape;

    EXPECT_EQ(shape.batch, 0);
    EXPECT_EQ(shape.anchors, 0);
    EXPECT_EQ(shape.num_classes, 0);
}

/** @brief Tests that a braced shape names the images, then the anchors, then the classes. */
TEST(Small_DetectPredictionShape, ABracedShapeNamesImagesThenAnchorsThenClasses)
{
    const PredictionShape shape{2, 8400, 80};

    EXPECT_EQ(shape.batch, 2);
    EXPECT_EQ(shape.anchors, 8400);
    EXPECT_EQ(shape.num_classes, 80);
}
