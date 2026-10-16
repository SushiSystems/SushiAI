/**************************************************************************/
/* test_sample.cpp                                                        */
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
 * @file test_sample.cpp
 * @brief Tests what a sample record and a sample list hold before a reader fills them.
 * @author Mustafa Garip
 */

#include <cstdint>
#include <gtest/gtest.h>
#include <SushiAI/vision/datasets/sample.hpp>

using namespace SushiAI::Vision::Datasets;

/** @brief Tests that a default sample names file 0 and carries no label of any layout. */
TEST(Small_DatasetsSample, ADefaultSampleCarriesNoLabel)
{
    const Sample sample;

    EXPECT_EQ(sample.file, 0u);
    EXPECT_FALSE(sample.stored.has_value());
    EXPECT_FALSE(sample.class_index.has_value());
    EXPECT_TRUE(sample.objects.is_empty());
    EXPECT_FALSE(sample.mask.has_value());
}

/** @brief Tests that a default list holds no file, no sample, no class and no table row. */
TEST(Small_DatasetsSample, ADefaultListIsEmpty)
{
    const SampleList list;

    EXPECT_TRUE(list.files.empty());
    EXPECT_TRUE(list.samples.empty());
    EXPECT_TRUE(list.class_names.empty());
    EXPECT_EQ(list.class_count, 0);
    EXPECT_EQ(list.table.row_count(), 0u);
}

/** @brief Tests that a sample names its pixel file and its mask file by index into the list. */
TEST(Small_DatasetsSample, SamplesReferToTheFilesOfTheirListByIndex)
{
    SampleList list;
    list.files = {"images/a.png", "masks/a.png"};
    Sample sample;
    sample.file = 0;
    sample.mask = 1u;
    list.samples.push_back(sample);

    ASSERT_EQ(list.samples.size(), 1u);
    EXPECT_EQ(list.files[list.samples[0].file], "images/a.png");
    EXPECT_EQ(list.files[*list.samples[0].mask], "masks/a.png");
}
