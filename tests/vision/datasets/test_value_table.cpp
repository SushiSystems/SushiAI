/**************************************************************************/
/* test_value_table.cpp                                                   */
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
 * @file test_value_table.cpp
 * @brief Tests the row arithmetic and the row views of the table a tabular list keeps.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <vector>
#include <gtest/gtest.h>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/vision/datasets/value_table.hpp>

using namespace SushiAI::Vision::Datasets;

namespace
{
    /** @brief Returns a table of two features and one target holding three rows. */
    ValueTable three_rows()
    {
        ValueTable table;
        table.columns = {"x", "y", "label"};
        table.feature_count = 2;
        table.target_count = 1;
        table.values = {1.0f, 2.0f, 10.0f, 3.0f, 4.0f, 20.0f, 5.0f, 6.0f, 30.0f};
        return table;
    }

    /** @brief Copies the values a span views. */
    std::vector<float> copy_of(SushiRuntime::span<const float> values)
    {
        return std::vector<float>(values.begin(), values.end());
    }
} // namespace

/** @brief Tests that a default table, the table of a list of images, holds no row. */
TEST(Small_DatasetsValueTable, ADefaultTableHoldsNoRow)
{
    const ValueTable table;

    EXPECT_EQ(table.row_width(), 0u);
    EXPECT_EQ(table.row_count(), 0u);
}

/** @brief Tests that a row is the features and then the targets, and the count follows. */
TEST(Small_DatasetsValueTable, TheRowWidthIsTheFeaturesPlusTheTargets)
{
    const ValueTable table = three_rows();

    EXPECT_EQ(table.row_width(), 3u);
    EXPECT_EQ(table.row_count(), 3u);
}

/** @brief Tests that values short of a whole row are not counted as a row. */
TEST(Small_DatasetsValueTable, APartialRowIsNotCounted)
{
    ValueTable table = three_rows();
    table.values.push_back(7.0f);

    EXPECT_EQ(table.row_count(), 3u);
}

/** @brief Tests that the views of a row hold its features and its targets, in place. */
TEST(Small_DatasetsValueTable, TheViewsOfARowSplitItAtTheFeatureCount)
{
    const ValueTable table = three_rows();

    EXPECT_EQ(copy_of(table.features(1)), (std::vector<float>{3.0f, 4.0f}));
    EXPECT_EQ(copy_of(table.targets(1)), (std::vector<float>{20.0f}));
    EXPECT_EQ(table.features(2).data(), table.values.data() + 6);
    EXPECT_EQ(table.targets(2).data(), table.values.data() + 8);
}

/** @brief Tests that a table of targets alone gives empty features and whole-row targets. */
TEST(Small_DatasetsValueTable, ATableWithoutFeaturesViewsOnlyTargets)
{
    ValueTable table;
    table.target_count = 2;
    table.values = {1.0f, 2.0f, 3.0f, 4.0f};

    EXPECT_EQ(table.row_count(), 2u);
    EXPECT_TRUE(table.features(1).empty());
    EXPECT_EQ(copy_of(table.targets(1)), (std::vector<float>{3.0f, 4.0f}));
}
