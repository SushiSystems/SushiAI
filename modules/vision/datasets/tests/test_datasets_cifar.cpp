/**************************************************************************/
/* test_datasets_cifar.cpp                                                */
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
 * @file test_datasets_cifar.cpp
 * @brief Tests the reader of the CIFAR-10 and CIFAR-100 binary batch files.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include <SushiAI/core/result.hpp>
#include <SushiAI/core/status.hpp>
#include <SushiAI/vision/datasets/cifar.hpp>
#include <SushiAI/vision/datasets/sample.hpp>
#include <SushiAI/vision/datasets/stored_pixels.hpp>
#include <SushiAI/vision/imaging/image.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>

#include "datasets_test_support.hpp"

using namespace SushiAI::Vision::Datasets;
using SushiAI::StatusCode;
using SushiAI::Vision::Imaging::Image;

namespace
{
    /** @brief Holds the byte count of one plane of a CIFAR image. */
    constexpr std::size_t K_PLANE_BYTES = 32 * 32;

    /**
     * @brief Returns one record: the label bytes, then a plane of each of three levels.
     * @param levels The value of every byte of the red, the green and the blue plane.
     */
    std::vector<std::uint8_t> record(const std::vector<std::uint8_t>& labels,
                                     const std::vector<std::uint8_t>& levels)
    {
        std::vector<std::uint8_t> bytes = labels;
        for (std::uint8_t level : levels)
            bytes.insert(bytes.end(), K_PLANE_BYTES, level);

        return bytes;
    }

    /** @brief Returns the bytes of two records, one after the other. */
    std::vector<std::uint8_t> joined(std::vector<std::uint8_t> first,
                                     const std::vector<std::uint8_t>& second)
    {
        first.insert(first.end(), second.begin(), second.end());
        return first;
    }

    /** @brief Returns options that read one variant and split of a directory. */
    CIFAROptions options_of(const Support::ScratchDirectory& scratch, CIFARVariant variant,
                            const char* split)
    {
        CIFAROptions options;
        options.root = scratch.path();
        options.variant = variant;
        options.split = split;
        return options;
    }
} // namespace

/** @brief Tests that the training split of CIFAR-10 lists its five files in order. */
TEST(Unit_DatasetsCifar, TheCIFAR10TrainingSplitListsItsFiveFilesInOrder)
{
    const Support::ScratchDirectory scratch("sushiai_test_datasets_cifar10_train");
    for (std::uint8_t batch = 1; batch <= 5; ++batch)
    {
        const std::string name = "data_batch_" + std::to_string(batch) + ".bin";
        scratch.write(name.c_str(), record({batch}, {1, 2, 3}));
    }

    SA_ASSERT_OK_AND_ASSIGN(const SampleList list,
                            read_cifar(options_of(scratch, CIFARVariant::CIFAR10, "train")));

    EXPECT_EQ(list.class_count, 10);
    ASSERT_EQ(list.files.size(), 5u);
    ASSERT_EQ(list.samples.size(), 5u);

    for (std::size_t i = 0; i < list.samples.size(); ++i)
    {
        EXPECT_EQ(list.files[i].filename().generic_string(),
                  "data_batch_" + std::to_string(i + 1) + ".bin");
        EXPECT_EQ(list.samples[i].file, i);
        EXPECT_EQ(list.samples[i].class_index, static_cast<std::int32_t>(i + 1));
    }
}

/** @brief Tests that a record's sample refers to its planes after the label byte. */
TEST(Unit_DatasetsCifar, ASampleRefersToThePlanesAfterItsLabel)
{
    const Support::ScratchDirectory scratch("sushiai_test_datasets_cifar10_test");
    scratch.write("test_batch.bin", joined(record({3}, {10, 20, 30}), record({9}, {40, 50, 60})));

    SA_ASSERT_OK_AND_ASSIGN(const SampleList list,
                            read_cifar(options_of(scratch, CIFARVariant::CIFAR10, "test")));

    ASSERT_EQ(list.samples.size(), 2u);
    EXPECT_EQ(list.samples[0].class_index, 3);
    EXPECT_EQ(list.samples[1].class_index, 9);

    const Sample& second = list.samples[1];
    ASSERT_TRUE(second.stored.has_value());
    EXPECT_EQ(second.stored->offset, 3073u + 1u);
    EXPECT_EQ(second.stored->size.height, K_CIFAR_SIDE);
    EXPECT_EQ(second.stored->size.width, K_CIFAR_SIDE);
    EXPECT_EQ(second.stored->channels, 3);
    EXPECT_EQ(second.stored->layout, PlaneLayout::Planar);

    Image image;
    SA_EXPECT_OK(read_stored_pixels(list.files[second.file], *second.stored, image));

    ASSERT_EQ(image.byte_count(), 3 * K_PLANE_BYTES);
    for (std::size_t pixel = 0; pixel < K_PLANE_BYTES; pixel += 341)
    {
        EXPECT_EQ(image.data()[pixel * 3], 40) << "pixel " << pixel;
        EXPECT_EQ(image.data()[pixel * 3 + 1], 50) << "pixel " << pixel;
        EXPECT_EQ(image.data()[pixel * 3 + 2], 60) << "pixel " << pixel;
    }
}

/** @brief Tests that CIFAR-100 takes the second label byte as fine and the first as coarse. */
TEST(Unit_DatasetsCifar, CIFAR100TakesTheFineOrTheCoarseLabel)
{
    const Support::ScratchDirectory scratch("sushiai_test_datasets_cifar100");
    scratch.write("train.bin", record({19, 87}, {1, 2, 3}));

    SA_ASSERT_OK_AND_ASSIGN(const SampleList fine,
                            read_cifar(options_of(scratch, CIFARVariant::CIFAR100Fine, "train")));
    SA_ASSERT_OK_AND_ASSIGN(
        const SampleList coarse,
        read_cifar(options_of(scratch, CIFARVariant::CIFAR100Coarse, "train")));

    EXPECT_EQ(fine.class_count, 100);
    ASSERT_EQ(fine.samples.size(), 1u);
    EXPECT_EQ(fine.samples[0].class_index, 87);
    ASSERT_TRUE(fine.samples[0].stored.has_value());
    EXPECT_EQ(fine.samples[0].stored->offset, 2u);

    EXPECT_EQ(coarse.class_count, 20);
    ASSERT_EQ(coarse.samples.size(), 1u);
    EXPECT_EQ(coarse.samples[0].class_index, 19);
}

/** @brief Tests that a label at or above the class count names the file and the record. */
TEST(Unit_DatasetsCifar, ALabelOutsideTheClassesNamesItsRecord)
{
    const Support::ScratchDirectory scratch("sushiai_test_datasets_cifar_label");
    scratch.write("test_batch.bin", joined(record({3}, {0, 0, 0}), record({10}, {0, 0, 0})));
    scratch.write("test.bin", record({20, 5}, {0, 0, 0}));

    SA_EXPECT_FAILURE(read_cifar(options_of(scratch, CIFARVariant::CIFAR10, "test")),
                      StatusCode::InvalidFile, "test_batch.bin", "record 2 of 2", "label 10");
    SA_EXPECT_FAILURE(read_cifar(options_of(scratch, CIFARVariant::CIFAR100Coarse, "test")),
                      StatusCode::InvalidFile, "test.bin", "record 1 of 1", "label 20");
}

/** @brief Tests that a file that is empty or ends inside a record is refused. */
TEST(Unit_DatasetsCifar, AFileThatIsNotWholeRecordsIsRefused)
{
    const Support::ScratchDirectory scratch("sushiai_test_datasets_cifar_size");

    std::vector<std::uint8_t> cut = record({3}, {0, 0, 0});
    cut.pop_back();
    scratch.write("test_batch.bin", cut);
    SA_EXPECT_FAILURE(read_cifar(options_of(scratch, CIFARVariant::CIFAR10, "test")),
                      StatusCode::InvalidFile, "test_batch.bin", "3072 bytes", "3073 bytes");

    scratch.write("test_batch.bin", {});
    SA_EXPECT_FAILURE(read_cifar(options_of(scratch, CIFARVariant::CIFAR10, "test")),
                      StatusCode::InvalidFile, "test_batch.bin", "0 bytes");
}

/** @brief Tests that an absent file of the split is reported with its path. */
TEST(Unit_DatasetsCifar, AnAbsentFileIsNotFound)
{
    const Support::ScratchDirectory scratch("sushiai_test_datasets_cifar_absent");
    scratch.write("data_batch_1.bin", record({0}, {0, 0, 0}));

    SA_EXPECT_FAILURE(read_cifar(options_of(scratch, CIFARVariant::CIFAR10, "train")),
                      StatusCode::FileNotFound, "data_batch_2.bin");
}

/** @brief Tests that a split other than train and test is refused. */
TEST(Unit_DatasetsCifar, AnotherSplitIsRefused)
{
    const Support::ScratchDirectory scratch("sushiai_test_datasets_cifar_split");

    SA_EXPECT_FAILURE(read_cifar(options_of(scratch, CIFARVariant::CIFAR10, "val")),
                      StatusCode::InvalidDescription, "'val'");
}
