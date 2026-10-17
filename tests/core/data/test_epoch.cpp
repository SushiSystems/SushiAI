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
 * @file test_epoch.cpp
 * @brief Tests that begin_epoch reaches a dataset that implements EpochAware.
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
#include <SushiAI/data/target.hpp>

using namespace SushiAI;

namespace
{
    /** @brief Holds one empty batch and records the epoch it was last told to begin. */
    class EpochRecorder final : public Data::Dataset, public Data::EpochAware
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
                return Status();
            }

            std::string_view name() const noexcept override
            {
                return "epoch-recorder";
            }

            void begin_epoch(std::int64_t epoch) const noexcept override
            {
                epoch_ = epoch;
                ++calls_;
            }

            /** @brief Returns the epoch last begun, or -1 before the first. */
            std::int64_t epoch() const noexcept
            {
                return epoch_;
            }

            /** @brief Returns how many times begin_epoch() was called. */
            int calls() const noexcept
            {
                return calls_;
            }

        private:
            /** @brief Holds no role. */
            std::vector<Data::TargetDescription> targets_;

            /** @brief Holds the epoch last begun. */
            mutable std::int64_t epoch_ = -1;

            /** @brief Counts the calls to begin_epoch(). */
            mutable int calls_ = 0;
    };
} // namespace

/** @brief Tests that begin_epoch hands the epoch to a dataset that implements EpochAware. */
TEST(Small_DataEpoch, BeginEpochReachesAnEpochAwareDataset)
{
    const EpochRecorder dataset;

    Data::begin_epoch(static_cast<const Data::Dataset&>(dataset), 4);

    EXPECT_EQ(dataset.epoch(), 4);
    EXPECT_EQ(dataset.calls(), 1);
}

/** @brief Tests that every call reaches the dataset, so the latest epoch is the one it holds. */
TEST(Small_DataEpoch, EveryCallReachesTheDataset)
{
    const EpochRecorder dataset;

    Data::begin_epoch(dataset, 0);
    Data::begin_epoch(dataset, 1);
    Data::begin_epoch(dataset, 0);

    EXPECT_EQ(dataset.epoch(), 0);
    EXPECT_EQ(dataset.calls(), 3);
}
