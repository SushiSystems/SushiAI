/**************************************************************************/
/* test_in_memory_dataset.cpp                                             */
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


/**
 * @file test_in_memory_dataset.cpp
 * @brief Tests that the datasets built in memory serve their batches, journal their reads and
 *        refuse what the Dataset contract refuses.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include <SushiAI/dtype/dtype.hpp>
#include <SushiAI/shape/shape.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/data/dataset.hpp>
#include <SushiAI/data/epoch.hpp>
#include <SushiAI/data/in_memory_dataset.hpp>
#include <SushiAI/data/input.hpp>
#include <SushiAI/data/roles.hpp>
#include <SushiAI/data/target.hpp>

using namespace SushiAI;

namespace
{
    /** @brief The samples per batch of every dataset below. */
    constexpr int64_t K_ROWS = 4;

    /** @brief Returns the input of every dataset below: three float32 values per sample. */
    Data::InputDescription three_floats()
    {
        return Data::InputDescription{Shape{3}, DType::Float32};
    }

    /** @brief Returns the one role of every dataset below: an int32 label over 2 classes. */
    std::vector<Data::TargetDescription> labels()
    {
        return {Data::TargetDescription{std::string(Data::K_ROLE_LABELS), Shape{}, DType::Int32,
                                        2}};
    }
} // namespace

/** @brief Tests that the dataset states the extents it was built with. */
TEST(Small_InMemoryDataset, StatesTheExtentsItWasBuiltWith)
{
    const Data::InMemoryDataset dataset(5, K_ROWS, three_floats(), labels());

    EXPECT_EQ(dataset.batch_count(), 5u);
    EXPECT_EQ(dataset.batch_size(), K_ROWS);
    EXPECT_EQ(dataset.sample_shape(), Shape{3});
    EXPECT_EQ(dataset.sample_dtype(), DType::Float32);
    ASSERT_EQ(dataset.targets().size(), 1u);
    EXPECT_EQ(dataset.targets()[0].role, Data::K_ROLE_LABELS);
}

/** @brief Tests that a read gives a batch of the layout the dataset states. */
TEST(Small_InMemoryDataset, AReadGivesTheLayoutTheDatasetStates)
{
    const Data::InMemoryDataset dataset(2, K_ROWS, three_floats(), labels());
    Data::Batch batch;

    ASSERT_TRUE(dataset.read(1, batch).is_ok());

    ASSERT_EQ(batch.inputs.rank, 2);
    EXPECT_EQ(batch.inputs.shape[0], K_ROWS);
    EXPECT_EQ(batch.inputs.shape[1], 3);
    ASSERT_EQ(batch.targets.size(), 1u);
    EXPECT_EQ(batch.targets[0].role, Data::K_ROLE_LABELS);
    EXPECT_EQ(batch.targets[0].values.dtype, DType::Int32);
}

/** @brief Tests that the journal holds the indices read, in the order they were read. */
TEST(Small_InMemoryDataset, TheJournalHoldsTheIndicesReadInOrder)
{
    const Data::InMemoryDataset dataset(5, K_ROWS, three_floats(), labels());
    Data::Batch batch;

    ASSERT_TRUE(dataset.read(3, batch).is_ok());
    ASSERT_TRUE(dataset.read(0, batch).is_ok());
    ASSERT_TRUE(dataset.read(3, batch).is_ok());

    EXPECT_EQ(dataset.journal(), (std::vector<std::size_t>{3, 0, 3}));
}

/** @brief Tests that a read of the index set with fail_at is refused and still journaled. */
TEST(Small_InMemoryDataset, AReadOfTheFailingIndexIsRefused)
{
    Data::InMemoryDataset dataset(5, K_ROWS, three_floats(), labels());
    dataset.fail_at(2);
    Data::Batch batch;

    SA_EXPECT_FAILURE(dataset.read(2, batch), StatusCode::InvalidFile, "batch 2", "unreadable");
    EXPECT_TRUE(dataset.read(1, batch).is_ok());
    EXPECT_EQ(dataset.journal(), (std::vector<std::size_t>{2, 1}));
}

/** @brief Tests that a range read checks its range before it reads. */
TEST(Small_InMemoryDataset, ABadRangeIsRefusedWithAStatusAndReadsNothing)
{
    const Data::InMemoryDataset dataset(2, K_ROWS, three_floats(), labels());
    Data::Batch batch;

    SA_EXPECT_FAILURE(dataset.read_rows(0, Data::RowRange{3, 2}, batch),
                      StatusCode::InvalidDescription, "Rows 3 to 5");
    EXPECT_TRUE(dataset.journal().empty());
}

/** @brief Tests that a batch maker builds the batch a read hands out. */
TEST(Small_InMemoryDataset, ABatchMakerBuildsTheBatchOfAnIndex)
{
    const Data::InMemoryDataset dataset(
        3, K_ROWS, three_floats(), labels(),
        [](std::int64_t, std::size_t index)
        {
            Data::Batch made;
            made.targets.assign(1, Data::BatchTarget{"batch-" + std::to_string(index),
                                                     SushiBLAS::Tensor()});
            return made;
        });
    Data::Batch batch;

    ASSERT_TRUE(dataset.read(2, batch).is_ok());

    ASSERT_EQ(batch.targets.size(), 1u);
    EXPECT_EQ(batch.targets[0].role, "batch-2");
}

/** @brief Tests that a read past the last batch stops the process. */
TEST(Medium_InMemoryDatasetDeathTest, AReadPastTheLastBatchStopsTheProcess)
{
    const Data::InMemoryDataset dataset(2, K_ROWS, three_floats(), labels());
    Data::Batch batch;

    SA_EXPECT_FATAL((void)dataset.read(2, batch), "Batch 2 is out of range");
}

/** @brief Tests that the plain dataset is not EpochAware and the epoch dataset is. */
TEST(Small_InMemoryEpochDataset, OnlyTheEpochDatasetIsEpochAware)
{
    const Data::InMemoryDataset plain(1, K_ROWS, three_floats(), labels());
    const Data::InMemoryEpochDataset aware(1, K_ROWS, three_floats(), labels());

    EXPECT_EQ(dynamic_cast<const Data::EpochAware*>(&plain), nullptr);
    EXPECT_NE(dynamic_cast<const Data::EpochAware*>(&aware), nullptr);
}

/** @brief Tests that the batch maker of an epoch dataset is told the epoch last begun. */
TEST(Small_InMemoryEpochDataset, TheBatchMakerIsToldTheEpochLastBegun)
{
    const Data::InMemoryEpochDataset dataset(
        2, K_ROWS, three_floats(), labels(),
        [](std::int64_t epoch, std::size_t)
        {
            Data::Batch made;
            made.targets.assign(1, Data::BatchTarget{std::to_string(epoch), SushiBLAS::Tensor()});
            return made;
        });
    Data::Batch batch;

    ASSERT_TRUE(dataset.read(0, batch).is_ok());
    EXPECT_EQ(batch.targets[0].role, "0");

    Data::begin_epoch(dataset, 6);
    ASSERT_TRUE(dataset.read(0, batch).is_ok());

    EXPECT_EQ(batch.targets[0].role, "6");
}
