/**************************************************************************/
/* test_epoch.cpp                                                         */
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
 * @file test_epoch.cpp
 * @brief Tests that begin_epoch reaches a dataset that implements EpochAware and leaves one
 *        that does not.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstdint>
#include <vector>
#include <gtest/gtest.h>
#include <SushiAI/dtype/dtype.hpp>
#include <SushiAI/shape/shape.hpp>
#include <SushiAI/data/epoch.hpp>
#include <SushiAI/data/in_memory_dataset.hpp>
#include <SushiAI/data/input.hpp>
#include <SushiAI/data/target.hpp>

using namespace SushiAI;

namespace
{
    /** @brief Returns the input of every dataset below: one float32 value per sample. */
    Data::InputDescription one_float()
    {
        return Data::InputDescription{Shape{1}, DType::Float32};
    }

    /** @brief Calls Data::begin_epoch on @p dataset for each of @p epochs, in order. */
    void begin_each(const Data::Dataset& dataset, const std::vector<std::int64_t>& epochs)
    {
        for (std::int64_t epoch : epochs)
            Data::begin_epoch(dataset, epoch);
    }

    /** @brief Runs each case with a sequence of epochs. */
    class Small_DataEpochSequence : public ::testing::TestWithParam<std::vector<std::int64_t>>
    {

    };
} // namespace

/** @brief Tests that every call reaches the dataset, so the latest epoch is the one it holds. */
TEST_P(Small_DataEpochSequence, EveryCallReachesAnEpochAwareDataset)
{
    const std::vector<std::int64_t>& epochs = GetParam();
    const Data::InMemoryEpochDataset dataset(1, 1, one_float(), {});

    begin_each(dataset, epochs);

    EXPECT_EQ(dataset.epoch(), epochs.back());
    EXPECT_EQ(dataset.begin_count(), epochs.size());
}

INSTANTIATE_TEST_SUITE_P(Epochs, Small_DataEpochSequence,
                         ::testing::Values(std::vector<std::int64_t>{4},
                                           std::vector<std::int64_t>{0, 1, 0}));

/** @brief Tests that begin_epoch on a dataset that is not EpochAware reads nothing from it. */
TEST(Small_DataEpoch, ADatasetThatIsNotEpochAwareIsLeftAlone)
{
    const Data::InMemoryDataset dataset(1, 1, one_float(), {});

    Data::begin_epoch(dataset, 4);

    EXPECT_TRUE(dataset.journal().empty());
}

/** @brief Tests that a dataset that is not EpochAware still reads after begin_epoch. */
TEST(Small_DataEpoch, ADatasetThatIsNotEpochAwareStillReads)
{
    const Data::InMemoryDataset dataset(2, 1, one_float(), {});
    Data::Batch batch;

    Data::begin_epoch(dataset, 4);

    EXPECT_TRUE(dataset.read(1, batch).is_ok());
    EXPECT_EQ(dataset.journal(), std::vector<std::size_t>{1});
}
