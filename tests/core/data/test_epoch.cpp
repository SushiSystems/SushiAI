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
/**************************************************************************/


/**
 * @file test_epoch.cpp
 * @brief Tests that begin_epoch reaches a dataset that implements EpochAware and leaves one
 *        that does not.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstdint>
#include <string_view>
#include <vector>
#include <gtest/gtest.h>
#include <SushiAI/shape/shape.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/data/dataset.hpp>
#include <SushiAI/data/epoch.hpp>
#include <SushiAI/data/in_memory_epoch_aware.hpp>
#include <SushiAI/data/target.hpp>

using namespace SushiAI;

namespace
{
    /** @brief Holds one empty batch, counts the reads it serves and implements no EpochAware. */
    class CountingDataset : public Data::Dataset
    {
        public:
            int64_t batch_size() const noexcept override
            {
                return 1;
            }

            Shape sample_shape() const override
            {
                return Shape{1};
            }

            const std::vector<Data::TargetDescription>& targets() const noexcept override
            {
                return targets_;
            }

            std::size_t batch_count() const noexcept override
            {
                return 1;
            }

            [[nodiscard]] Status read(std::size_t, Data::Batch&) const override
            {
                ++reads_;
                return Status();
            }

            [[nodiscard]] Status read_rows(std::size_t, Data::RowRange,
                                           Data::Batch&) const override
            {
                ++reads_;
                return Status();
            }

            std::string_view name() const noexcept override
            {
                return "counting";
            }

            /** @brief Returns how many reads the dataset served. */
            std::size_t reads() const noexcept
            {
                return reads_;
            }

        private:
            /** @brief Holds no role. */
            std::vector<Data::TargetDescription> targets_;

            /** @brief Counts the calls of read() and read_rows(). */
            mutable std::size_t reads_ = 0;
    };

    /** @brief A CountingDataset that keeps the epochs it is told through InMemoryEpochAware. */
    class EpochAwareDataset final : public CountingDataset, public Data::InMemoryEpochAware
    {

    };

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
    const EpochAwareDataset dataset;

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
    const CountingDataset dataset;

    Data::begin_epoch(dataset, 4);

    EXPECT_EQ(dataset.reads(), 0u);
}
