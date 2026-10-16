/**************************************************************************/
/* test_fitted_sample.cpp                                                 */
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
 * @file test_fitted_sample.cpp
 * @brief Tests what a fitted sample holds before a pipeline fills it, and a tabular one.
 * @author Mustafa Garip
 */

#include <gtest/gtest.h>
#include <SushiAI/vision/datasets/fitted_sample.hpp>
#include <SushiAI/vision/datasets/value_table.hpp>

using namespace SushiAI::Vision::Datasets;

/** @brief Tests that a default fitted sample holds no pixel, no label and no table value. */
TEST(Small_DatasetsFittedSample, ADefaultSampleHoldsNothing)
{
    const FittedSample sample;

    EXPECT_EQ(sample.image.byte_count(), 0u);
    EXPECT_EQ(sample.file, 0u);
    EXPECT_FALSE(sample.class_index.has_value());
    EXPECT_TRUE(sample.objects.is_empty());
    EXPECT_EQ(sample.mask.byte_count(), 0u);
    EXPECT_TRUE(sample.features.empty());
    EXPECT_TRUE(sample.targets.empty());
}

/** @brief Tests that a tabular sample views the row of its table without copying it. */
TEST(Small_DatasetsFittedSample, ATabularSampleViewsTheRowOfItsTable)
{
    ValueTable table;
    table.feature_count = 2;
    table.target_count = 1;
    table.values = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};

    FittedSample sample;
    sample.file = 1;
    sample.features = table.features(1);
    sample.targets = table.targets(1);

    EXPECT_EQ(sample.image.byte_count(), 0u);
    ASSERT_EQ(sample.features.size(), 2u);
    ASSERT_EQ(sample.targets.size(), 1u);
    EXPECT_EQ(sample.features.data(), table.values.data() + 3);
    EXPECT_EQ(sample.targets.data(), table.values.data() + 5);
}
