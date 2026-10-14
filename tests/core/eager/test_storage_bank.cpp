/**************************************************************************/
/* test_storage_bank.cpp                                                  */
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
 * @file test_storage_bank.cpp
 * @brief Tests that StorageBank holds, allocates and lets go of tensors by storage id.
 * @author Mustafa Garip
 */

#include <vector>
#include <gtest/gtest.h>
#include <SushiAI/SushiAI.h>
#include <SushiAI/core/allocation.hpp>
#include <SushiAI/core/common.hpp>
#include <SushiAI/core/shape.hpp>
#include <SushiAI/eager/storage_bank.hpp>
#include <SushiAI/eager/storage_id.hpp>
#include <SushiAI/engine_fixture/engine_test.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/tensor_values/tensor_values.hpp>

using namespace SushiAI;
using Eager::StorageBank;
using Eager::StorageID;
// SaTest stays reachable inside a TEST_F body; see tests/core/eager/README.md.
namespace SaTest = SushiAI::Test;

namespace
{
    /** @brief Provides the Engine every case allocates on. */
    class Unit_StorageBank : public SaTest::EngineTest
    {

    };
} // namespace

TEST_F(Unit_StorageBank, AllocateHoldsATensorOfTheShapeAndTypeAsked)
{
    StorageBank bank(*engine);
    const StorageID id = 7;

    SA_EXPECT_OK(bank.allocate(id, Shape{4, 3}, DType::Float64));

    ASSERT_TRUE(bank.holds(id));
    EXPECT_EQ(bank.tensor(id).num_elements, 12);
    EXPECT_EQ(bank.tensor(id).dtype, DType::Float64);
    EXPECT_EQ(bank.size(), 1u);
    EXPECT_EQ(bank.allocation_count(), 1u);
}

TEST_F(Unit_StorageBank, AdoptSharesTheStorageOfTheCallersTensor)
{
    StorageBank bank(*engine);
    SA_ASSERT_OK_AND_ASSIGN(SushiBLAS::Tensor owned, allocate_tensor(*engine, Shape{5}));
    SaTest::write_tensor(owned, std::vector<double>{1.0, 2.0, 3.0, 4.0, 5.0});

    SA_EXPECT_OK(bank.adopt(3, owned));

    EXPECT_EQ(bank.tensor(3).data(), owned.data());
    EXPECT_EQ(SaTest::read_tensor(bank.tensor(3)),
              (std::vector<double>{1.0, 2.0, 3.0, 4.0, 5.0}));
    EXPECT_EQ(bank.allocation_count(), 0u);
}

TEST_F(Unit_StorageBank, ReleaseForgetsTheTensorAndIgnoresAnUnknownId)
{
    StorageBank bank(*engine);
    SA_EXPECT_OK(bank.allocate(1, Shape{2}, DType::Float32));

    bank.release(1);
    bank.release(99);

    EXPECT_FALSE(bank.holds(1));
    EXPECT_EQ(bank.size(), 0u);
    EXPECT_EQ(bank.allocation_count(), 1u);
}

TEST_F(Unit_StorageBank, ASecondTensorUnderOneIdStopsTheProcess)
{
    StorageBank bank(*engine);
    SA_EXPECT_OK(bank.allocate(1, Shape{2}, DType::Float32));

    SA_EXPECT_FATAL((void)bank.allocate(1, Shape{2}, DType::Float32), "Storage 1",
                    "already holds");
}

TEST_F(Unit_StorageBank, AskingForAnUnknownIdStopsTheProcess)
{
    StorageBank bank(*engine);

    SA_EXPECT_FATAL((void)bank.tensor(4), "Storage 4", "holds no tensor");
}
