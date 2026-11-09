/**************************************************************************/
/* test_cifar.cpp                                                         */
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
 * @file test_cifar.cpp
 * @brief Tests the reader of the CIFAR-10 and CIFAR-100 binary batch files.
 * @author Mustafa Garip
 */
#include <cstddef>
#include <cstdint>
#include <sstream>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include "datasets_test_support.hpp"
#include <SushiAI/failure/result.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/vision/datasets/byte_source.hpp>
#include <SushiAI/vision/datasets/cifar.hpp>
#include <SushiAI/vision/datasets/sample.hpp>
#include <SushiAI/vision/datasets/stored_pixels.hpp>
#include <SushiAI/vision/imaging/image.hpp>

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

    /** @brief Returns the bytes of an image of K_PLANE_BYTES pixels, each holding the levels. */
    std::vector<std::uint8_t> interleaved(const std::vector<std::uint8_t>& levels)
    {
        std::vector<std::uint8_t> bytes;
        for (std::size_t pixel = 0; pixel < K_PLANE_BYTES; ++pixel)
            bytes.insert(bytes.end(), levels.begin(), levels.end());

        return bytes;
    }

    /** @brief Returns options that read one variant and split of the folder data. */
    CIFAROptions options_of(CIFARVariant variant, const char* split)
    {
        CIFAROptions options;
        options.root = "data";
        options.variant = variant;
        options.split = split;
        return options;
    }

    /** @brief Returns a source holding the bytes as the file data/<name>. */
    MemorySource source_of(const char* name, const std::vector<std::uint8_t>& bytes)
    {
        MemorySource source;
        source.add_file(std::string("data/") + name, Support::text_of(bytes));
        return source;
    }

    /** @brief Returns a source holding the two records of the CIFAR-10 test batch. */
    MemorySource test_batch()
    {
        return source_of("test_batch.bin", joined(record({3}, {10, 20, 30}),
                                                  record({9}, {40, 50, 60})));
    }

    /** @brief Holds a file whose label lies outside the classes and where the message points. */
    struct LabelRow
    {
        std::string name;                 /**< The row's name. */
        CIFARVariant variant;             /**< The variant read. */
        std::string file;                 /**< The file name in the split. */
        std::vector<std::uint8_t> bytes;  /**< The whole file. */
        std::string record;               /**< The record the message names. */
        std::string label;                /**< The label the message names. */
    };

    /** @brief Holds a file that is not whole records and what the message says. */
    struct SizeRow
    {
        std::string name;                 /**< The row's name. */
        std::vector<std::uint8_t> bytes;  /**< The whole file. */
        std::string size;                 /**< The size the message names. */
        std::string expected;             /**< The size a record has; empty when not named. */
    };

    /** @brief Runs read_cifar over each file with a label outside the classes. */
    class Small_DatasetsCifarLabel : public ::testing::TestWithParam<LabelRow>
    {
    };

    /** @brief Runs read_cifar over each file that is not whole records. */
    class Small_DatasetsCifarSize : public ::testing::TestWithParam<SizeRow>
    {
    };

    /** @brief Names a test after its label row. */
    std::string label_row_name(const ::testing::TestParamInfo<LabelRow>& info)
    {
        return info.param.name;
    }

    /** @brief Names a test after its size row. */
    std::string size_row_name(const ::testing::TestParamInfo<SizeRow>& info)
    {
        return info.param.name;
    }

    /** @brief Returns the bytes of a record cut one byte short. */
    std::vector<std::uint8_t> cut_record()
    {
        std::vector<std::uint8_t> cut = record({3}, {0, 0, 0});
        cut.pop_back();
        return cut;
    }
} // namespace

/** @brief Tests that the training split of CIFAR-10 lists its five files in order. */
TEST(Small_DatasetsCifar, TheCIFAR10TrainingSplitListsItsFiveFilesInOrder)
{
    MemorySource source;
    for (std::uint8_t batch = 1; batch <= 5; ++batch)
        source.add_file("data/data_batch_" + std::to_string(batch) + ".bin",
                        Support::text_of(record({batch}, {1, 2, 3})));

    SA_ASSERT_OK_AND_ASSIGN(const SampleList list,
                            read_cifar(options_of(CIFARVariant::CIFAR10, "train"), source));

    std::vector<std::string> names;
    std::vector<std::size_t> files;
    std::vector<std::int32_t> classes;
    for (const std::filesystem::path& file : list.files)
        names.push_back(file.filename().generic_string());
    for (const Sample& sample : list.samples)
    {
        files.push_back(sample.file);
        classes.push_back(sample.class_index.value_or(-1));
    }

    EXPECT_EQ(list.class_count, 10);
    EXPECT_EQ(names, (std::vector<std::string>{"data_batch_1.bin", "data_batch_2.bin",
                                               "data_batch_3.bin", "data_batch_4.bin",
                                               "data_batch_5.bin"}));
    EXPECT_EQ(files, (std::vector<std::size_t>{0, 1, 2, 3, 4}));
    EXPECT_EQ(classes, (std::vector<std::int32_t>{1, 2, 3, 4, 5}));
}

/** @brief Tests that a record's sample refers to its planes after the label byte. */
TEST(Small_DatasetsCifar, ASampleRefersToThePlanesAfterItsLabel)
{
    SA_ASSERT_OK_AND_ASSIGN(const SampleList list,
                            read_cifar(options_of(CIFARVariant::CIFAR10, "test"), test_batch()));

    ASSERT_EQ(list.samples.size(), 2u);
    EXPECT_EQ(list.samples[0].class_index, 3);
    EXPECT_EQ(list.samples[1].class_index, 9);

    const Sample& second = list.samples[1];
    ASSERT_TRUE(second.stored.has_value());
    EXPECT_EQ(second.stored->offset, 3074u);
    EXPECT_EQ(second.stored->size.height, K_CIFAR_SIDE);
    EXPECT_EQ(second.stored->size.width, K_CIFAR_SIDE);
    EXPECT_EQ(second.stored->channels, 3);
    EXPECT_EQ(second.stored->layout, PlaneLayout::Planar);
}

/** @brief Tests that the stored planes of a sample read back as one interleaved image. */
TEST(Small_DatasetsCifar, TheStoredPlanesReadBackAsOneInterleavedImage)
{
    SA_ASSERT_OK_AND_ASSIGN(const SampleList list,
                            read_cifar(options_of(CIFARVariant::CIFAR10, "test"), test_batch()));
    ASSERT_EQ(list.samples.size(), 2u);
    std::istringstream file(Support::text_of(joined(record({3}, {10, 20, 30}),
                                                    record({9}, {40, 50, 60}))),
                            std::ios::binary);

    Image image;
    SA_EXPECT_OK(read_stored_pixels(file, *list.samples[1].stored, image));

    EXPECT_EQ(std::vector<std::uint8_t>(image.data(), image.data() + image.byte_count()),
              interleaved({40, 50, 60}));
}

/** @brief Tests that CIFAR-100 takes the second label byte as the fine label. */
TEST(Small_DatasetsCifar, CIFAR100TakesTheFineLabelFromTheSecondByte)
{
    const MemorySource source = source_of("train.bin", record({19, 87}, {1, 2, 3}));

    SA_ASSERT_OK_AND_ASSIGN(const SampleList fine,
                            read_cifar(options_of(CIFARVariant::CIFAR100Fine, "train"), source));

    EXPECT_EQ(fine.class_count, 100);
    ASSERT_EQ(fine.samples.size(), 1u);
    EXPECT_EQ(fine.samples[0].class_index, 87);
    ASSERT_TRUE(fine.samples[0].stored.has_value());
    EXPECT_EQ(fine.samples[0].stored->offset, 2u);
}

/** @brief Tests that CIFAR-100 takes the first label byte as the coarse label. */
TEST(Small_DatasetsCifar, CIFAR100TakesTheCoarseLabelFromTheFirstByte)
{
    const MemorySource source = source_of("train.bin", record({19, 87}, {1, 2, 3}));

    SA_ASSERT_OK_AND_ASSIGN(const SampleList coarse,
                            read_cifar(options_of(CIFARVariant::CIFAR100Coarse, "train"), source));

    EXPECT_EQ(coarse.class_count, 20);
    ASSERT_EQ(coarse.samples.size(), 1u);
    EXPECT_EQ(coarse.samples[0].class_index, 19);
}

/** @brief Tests that a label at or above the class count names the file and the record. */
TEST_P(Small_DatasetsCifarLabel, ALabelOutsideTheClassesNamesItsRecord)
{
    const MemorySource source = source_of(GetParam().file.c_str(), GetParam().bytes);

    SA_EXPECT_FAILURE(read_cifar(options_of(GetParam().variant, "test"), source),
                      StatusCode::InvalidFile, GetParam().file, GetParam().record,
                      GetParam().label);
}

INSTANTIATE_TEST_SUITE_P(
    Rows, Small_DatasetsCifarLabel,
    ::testing::Values(
        LabelRow{"CIFAR10", CIFARVariant::CIFAR10, "test_batch.bin",
                 joined(record({3}, {0, 0, 0}), record({10}, {0, 0, 0})), "record 2 of 2",
                 "label 10"},
        LabelRow{"CIFAR100Coarse", CIFARVariant::CIFAR100Coarse, "test.bin",
                 record({20, 5}, {0, 0, 0}), "record 1 of 1", "label 20"}),
    label_row_name);

/** @brief Tests that a file that is empty or ends inside a record is refused. */
TEST_P(Small_DatasetsCifarSize, AFileThatIsNotWholeRecordsIsRefused)
{
    const MemorySource source = source_of("test_batch.bin", GetParam().bytes);

    SA_EXPECT_FAILURE(read_cifar(options_of(CIFARVariant::CIFAR10, "test"), source),
                      StatusCode::InvalidFile, "test_batch.bin", GetParam().size,
                      GetParam().expected);
}

INSTANTIATE_TEST_SUITE_P(
    Rows, Small_DatasetsCifarSize,
    ::testing::Values(
        SizeRow{"EndsInsideARecord", cut_record(), "3072 bytes", "3073 bytes"},
        SizeRow{"Empty", {}, "0 bytes", ""}),
    size_row_name);

/** @brief Tests that a split other than train and test is refused. */
TEST(Small_DatasetsCifar, AnotherSplitIsRefused)
{
    const MemorySource source;

    SA_EXPECT_FAILURE(read_cifar(options_of(CIFARVariant::CIFAR10, "val"), source),
                      StatusCode::InvalidDescription, "'val'");
}

/** @brief Tests that an absent file of the split is reported with its path. */
TEST(Medium_DatasetsCifar, AnAbsentFileIsNotFound)
{
    const Support::ScratchDirectory scratch("sushiai_test_datasets_cifar_absent");
    scratch.write("data_batch_1.bin", record({0}, {0, 0, 0}));
    CIFAROptions options;
    options.root = scratch.path();

    SA_EXPECT_FAILURE(read_cifar(options), StatusCode::FileNotFound, "data_batch_2.bin");
}

/** @brief Tests that the file-system form lists the records of a batch written to disk. */
TEST(Medium_DatasetsCifar, ABatchOnDiskListsItsRecords)
{
    const Support::ScratchDirectory scratch("sushiai_test_datasets_cifar_disk");
    scratch.write("test_batch.bin", joined(record({3}, {10, 20, 30}), record({9}, {40, 50, 60})));
    CIFAROptions options;
    options.root = scratch.path();
    options.split = "test";

    SA_ASSERT_OK_AND_ASSIGN(const SampleList list, read_cifar(options));

    ASSERT_EQ(list.samples.size(), 2u);
    EXPECT_EQ(list.samples[1].class_index, 9);
    EXPECT_EQ(list.samples[1].stored->offset, 3074u);
}
