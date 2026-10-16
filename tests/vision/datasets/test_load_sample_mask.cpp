/**************************************************************************/
/* test_load_sample_mask.cpp                                              */
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
 * @file test_load_sample_mask.cpp
 * @brief Tests that the loader of a sample's mask keeps the channels a mask file stores.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstdint>
#include <vector>
#include <gtest/gtest.h>
#include "datasets_test_support.hpp"
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/vision/datasets/load_sample_mask.hpp>
#include <SushiAI/vision/datasets/sample.hpp>
#include <SushiAI/vision/imaging/image.hpp>

using namespace SushiAI::Vision;
using namespace SushiAI::Vision::Datasets;

/** @brief Tests that a mask file of three channels decodes to those channels, unchanged. */
TEST(Medium_DatasetsLoadSampleMask, AColorMaskKeepsTheChannelsItStores)
{
    const Support::ScratchDirectory scratch("sushiai_test_datasets_load_color_mask");
    Imaging::Image stored(Imaging::ImageSize{1, 2}, 3);
    const std::vector<std::uint8_t> bytes = {1, 2, 3, 40, 50, 60};
    for (std::size_t i = 0; i < bytes.size(); ++i)
        stored.data()[i] = bytes[i];

    SampleList list;
    list.files.push_back(scratch.path() / "image.jpg");
    list.files.push_back(Support::write_raw_image(scratch, "mask.saim", stored));
    list.samples.emplace_back();
    list.samples[0].mask = 1u;

    Imaging::Image mask;
    SA_EXPECT_OK(load_sample_mask(list, 0, mask));

    EXPECT_EQ(mask.size(), (Imaging::ImageSize{1, 2}));
    EXPECT_EQ(mask.channels(), 3);
    EXPECT_EQ(std::vector<std::uint8_t>(mask.data(), mask.data() + mask.byte_count()), bytes);
}
