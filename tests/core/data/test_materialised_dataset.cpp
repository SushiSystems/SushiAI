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
 * @file test_materialised_dataset.cpp
 * @brief Tests that the dataset that holds device batches gives tensors that match what it
 *        states, and that share_batch copies no element.
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

/** @brief Tests that a read gives tensors that match the type, shape and roles the dataset states. */
TEST_F(Medium_MaterialisedDataset, AReadGivesTensorsThatMatchWhatTheDatasetStates)
{
    const Data::MaterialisedDataset dataset = two_batches();
    Data::Batch batch;

    ASSERT_TRUE(SaTest::status_is_ok(dataset.read(1, batch)));

    EXPECT_EQ(batch.inputs.dtype, dataset.sample_dtype());
    ASSERT_EQ(batch.inputs.rank, 2);
    EXPECT_EQ(batch.inputs.shape[0], dataset.batch_size());
    EXPECT_EQ(batch.inputs.shape[1], dataset.sample_shape()[0]);
    ASSERT_EQ(batch.targets.size(), dataset.targets().size());
    EXPECT_EQ(batch.targets[0].role, dataset.targets()[0].role);
    EXPECT_EQ(batch.targets[0].values.dtype, dataset.targets()[0].dtype);
    EXPECT_EQ(batch.targets[0].values.shape[0], dataset.batch_size());
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
