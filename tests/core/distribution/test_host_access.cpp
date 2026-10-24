/**************************************************************************/
/* test_host_access.cpp                                                   */
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
 * @file test_host_access.cpp
 * @brief Tests which tensors the host copies of a replica may address.
 * @author Mustafa Garip
 */

#include <gtest/gtest.h>
#include <SushiBLAS/engine/engine.hpp>
#include <SushiBLAS/tensor/tensor.hpp>
#include <SushiRuntime/api/data/buffer.hpp>
#include <SushiAI/allocation/allocation.hpp>
#include <SushiAI/distribution/host_access.hpp>
#include <SushiAI/dtype/dtype.hpp>
#include <SushiAI/engine_fixture/engine_test.hpp>
#include <SushiAI/shape/shape.hpp>

namespace
{
    using SushiAI::Distribution::is_host_visible;
    using SushiBLAS::Core::DataType;
    using SushiRuntime::Residency;

    /** @brief Gives each case an engine on the shared runtime. */
    class Medium_HostAccess : public SushiAI::Test::EngineTest
    {

    };
} // namespace

TEST_F(Medium_HostAccess, SharedAndHostMemoryAreVisible)
{
    EXPECT_TRUE(is_host_visible(engine->create_tensor({4}, DataType::Float32, Residency::Shared)));
    EXPECT_TRUE(is_host_visible(engine->create_tensor({4}, DataType::Float32, Residency::Host)));
}

TEST_F(Medium_HostAccess, DeviceMemoryAndNoStorageAreNot)
{
    EXPECT_FALSE(is_host_visible(engine->create_tensor({4}, DataType::Float32, Residency::Device)));
    EXPECT_FALSE(is_host_visible(SushiBLAS::Tensor()));
}

TEST_F(Medium_HostAccess, ThePoolAllocatesVisibleTensors)
{
    const SushiBLAS::Tensor tensor =
        SushiAI::allocate_tensor(*engine, SushiAI::Shape{4}, SushiAI::DType::Float32).value();
    EXPECT_TRUE(is_host_visible(tensor));
}
