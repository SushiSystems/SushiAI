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
 * @file test_fitted_sample.cpp
 * @brief Tests what a fitted sample holds before a pipeline fills it, and a tabular one.
 * @author Mustafa Garip
 */

#include <gtest/gtest.h>
#include <SushiAI/vision/datasets/fitted_sample.hpp>
#include <SushiAI/vision/datasets/value_table.hpp>

using namespace SushiAI::Vision::Datasets;

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
