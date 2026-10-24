/**************************************************************************/
/* test_materialised_dataset.cpp                                          */
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
 * @file test_materialised_dataset.cpp
 * @brief Tests the dataset that holds device batches, and how share_batch and prepare_batch
 *        treat the tensors it holds.
 * @author Mustafa Garip
 */

#include <cstdint>
#include <string>
#include <utility>
#include <vector>
#include <gtest/gtest.h>
#include <SushiBLAS/tensor/tensor.hpp>
#include <SushiAI/dtype/dtype.hpp>
#include <SushiAI/shape/shape.hpp>
#include <SushiAI/data/dataset.hpp>
#include <SushiAI/data/input.hpp>
#include <SushiAI/data/materialised_dataset.hpp>
#include <SushiAI/data/roles.hpp>
#include <SushiAI/data/target.hpp>
#include <SushiAI/engine_fixture/engine_test.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>

#include "batch_rows.hpp"

using namespace SushiAI;
namespace SaTest = SushiAI::Test;

namespace
{
    /** @brief Holds the samples per batch of every case. */
    constexpr int64_t K_ROWS = 3;

    /** @brief Returns the input of every case: four float32 values per sample. */
    Data::InputDescription four_floats()
    {
        return Data::InputDescription{Shape{4}, DType::Float32};
    }

    /** @brief Returns the one role of every case: an int32 label per sample over 2 classes. */
    std::vector<Data::TargetDescription> labels()
    {
        return {Data::TargetDescription{std::string(Data::K_ROLE_LABELS), Shape{}, DType::Int32,
                                        2}};
    }

    /** @brief Provides the engine the batches are allocated on, and builds the dataset. */
    class Medium_MaterialisedDataset : public SaTest::EngineTest
    {
        protected:
            /** @brief Returns a dataset of two allocated batches named "held". */
            Data::MaterialisedDataset two_batches()
            {
                std::vector<Data::Batch> batches;
                for (int b = 0; b < 2; ++b)
                {
                    Result<Data::Batch> batch =
                        Data::allocate_batch(*engine, K_ROWS, four_floats(), labels());
                    EXPECT_TRUE(SaTest::status_is_ok(Detail::status_of(batch)));
                    if (batch.has_value())
                        batches.push_back(std::move(batch).value());
                }
                return Data::MaterialisedDataset(std::move(batches), K_ROWS, four_floats(),
                                                 labels(), "held");
            }
    };
} // namespace

/** @brief Tests that the dataset states the extents, the input and the roles it was built with. */
TEST_F(Medium_MaterialisedDataset, StatesWhatItWasBuiltWith)
{
    const Data::MaterialisedDataset dataset = two_batches();

    EXPECT_EQ(dataset.batch_size(), K_ROWS);
    EXPECT_EQ(dataset.batch_count(), 2u);
    EXPECT_EQ(dataset.name(), "held");
    EXPECT_EQ(dataset.sample_dtype(), DType::Float32);
    ASSERT_EQ(dataset.sample_shape().rank(), 1);
    EXPECT_EQ(dataset.sample_shape()[0], 4);
    ASSERT_EQ(dataset.targets().size(), 1u);
    EXPECT_EQ(dataset.targets()[0].role, Data::K_ROLE_LABELS);
}

/** @brief Tests that a read hands out the held tensors of its index and copies no element. */
TEST_F(Medium_MaterialisedDataset, AReadSharesTheTensorsOfItsIndex)
{
    const Data::MaterialisedDataset dataset = two_batches();
    Data::Batch first;
    Data::Batch second;
    Data::Batch again;

    ASSERT_TRUE(SaTest::status_is_ok(dataset.read(0, first)));
    ASSERT_TRUE(SaTest::status_is_ok(dataset.read(1, second)));
    ASSERT_TRUE(SaTest::status_is_ok(dataset.read(1, again)));

    EXPECT_NE(first.inputs.data(), second.inputs.data());
    EXPECT_EQ(second.inputs.data(), again.inputs.data());
    ASSERT_EQ(again.targets.size(), 1u);
    EXPECT_EQ(second.targets[0].values.data(), again.targets[0].values.data());
}

/** @brief Tests that share_batch makes a batch hold the same tensors under the same roles. */
TEST_F(Medium_MaterialisedDataset, ShareBatchCopiesNoElement)
{
    SA_ASSERT_OK_AND_ASSIGN(const Data::Batch held,
                            Data::allocate_batch(*engine, K_ROWS, four_floats(), labels()));
    Data::Batch batch;

    SA_EXPECT_OK(Data::share_batch(held, batch));

    EXPECT_EQ(batch.inputs.data(), held.inputs.data());
    ASSERT_EQ(batch.targets.size(), 1u);
    EXPECT_EQ(batch.targets[0].role, Data::K_ROLE_LABELS);
    EXPECT_EQ(batch.targets[0].values.data(), held.targets[0].values.data());
}

/** @brief Tests that prepare_batch replaces tensors a dataset shares, so it never writes them. */
TEST_F(Medium_MaterialisedDataset, PrepareBatchReplacesTensorsTheDatasetShares)
{
    const Data::MaterialisedDataset dataset = two_batches();
    Data::Batch batch;
    ASSERT_TRUE(SaTest::status_is_ok(dataset.read(0, batch)));
    const void* const shared = batch.inputs.data();

    SA_EXPECT_OK(Data::prepare_batch(*engine, K_ROWS, four_floats(), labels(), batch));

    EXPECT_NE(batch.inputs.data(), shared);
    EXPECT_EQ(batch.inputs.shape[0], K_ROWS);
    EXPECT_EQ(batch.inputs.shape[1], 4);
}

/** @brief Tests that a range read views the held rows of its index and copies no element. */
TEST_F(Medium_MaterialisedDataset, ARangeReadViewsTheRowsOfItsIndex)
{
    const Data::MaterialisedDataset dataset = two_batches();
    const Data::RowRange rows{1, 2};
    Data::Batch whole;
    Data::Batch part;

    ASSERT_TRUE(SaTest::status_is_ok(dataset.read(1, whole)));
    ASSERT_TRUE(SaTest::status_is_ok(dataset.read_rows(1, rows, part)));

    EXPECT_EQ(part.inputs.shape[0], rows.count);
    EXPECT_EQ(part.inputs.storage.get(), whole.inputs.storage.get());
    EXPECT_TRUE(DataTest::holds_rows_of(whole, K_ROWS, rows, part));
}
