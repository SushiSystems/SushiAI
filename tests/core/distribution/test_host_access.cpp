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
 * @file test_host_access.cpp
 * @brief Tests which tensors the host copies of a replica may address.
 * @author Mustafa Garip
 */

#include <string>
#include <vector>
#include <gtest/gtest.h>
#include <SushiBLAS/engine/engine.hpp>
#include <SushiBLAS/tensor/tensor.hpp>
#include <SushiRuntime/api/data/buffer.hpp>
#include <SushiAI/distribution/host_access.hpp>
#include <SushiAI/engine_fixture/engine_test.hpp>
#include <support/test_names.hpp>

namespace
{
    using SushiAI::Distribution::is_host_visible;
    using SushiBLAS::Core::DataType;
    using SushiRuntime::Residency;

    /** @brief Makes a tensor of four float32 elements in shared memory. */
    SushiBLAS::Tensor make_shared_tensor(SushiBLAS::Engine& engine)
    {
        return engine.create_tensor({4}, DataType::Float32, Residency::Shared);
    }

    /** @brief Makes a tensor of four float32 elements in host memory. */
    SushiBLAS::Tensor make_host_tensor(SushiBLAS::Engine& engine)
    {
        return engine.create_tensor({4}, DataType::Float32, Residency::Host);
    }

    /** @brief Makes a tensor of four float32 elements in device memory. */
    SushiBLAS::Tensor make_device_tensor(SushiBLAS::Engine& engine)
    {
        return engine.create_tensor({4}, DataType::Float32, Residency::Device);
    }

    /** @brief Makes a tensor with no storage. */
    SushiBLAS::Tensor make_empty_tensor(SushiBLAS::Engine& engine)
    {
        static_cast<void>(engine);
        return SushiBLAS::Tensor();
    }

    /** @brief Holds a tensor to make and whether the host can address its storage. */
    struct ResidencyRow
    {
        /** @brief Names the row in the test report. */
        const char* name;

        /** @brief Makes the tensor on the engine. */
        SushiBLAS::Tensor (*make)(SushiBLAS::Engine&);

        /** @brief Holds whether the host can address the tensor's storage. */
        bool visible;
    };

    /** @brief Lists the residencies of a tensor, an empty tensor included. */
    std::vector<ResidencyRow> residency_rows()
    {
        return {
            {"SharedMemory", &make_shared_tensor, true},
            {"HostMemory", &make_host_tensor, true},
            {"DeviceMemory", &make_device_tensor, false},
            {"NoStorage", &make_empty_tensor, false},
        };
    }

    /** @brief Gives each case an engine on the shared runtime and a residency to check. */
    class Medium_HostAccess : public SushiAI::Test::EngineTest,
                              public ::testing::WithParamInterface<ResidencyRow>
    {

    };
} // namespace

/** @brief Tests that a tensor is host-visible exactly when its storage is shared or host memory. */
TEST_P(Medium_HostAccess, ATensorIsHostVisibleWhenItsStorageIsSharedOrHostMemory)
{
    const ResidencyRow& row = GetParam();

    const SushiBLAS::Tensor tensor = row.make(*engine);

    EXPECT_EQ(is_host_visible(tensor), row.visible);
}

INSTANTIATE_TEST_SUITE_P(Residencies, Medium_HostAccess, ::testing::ValuesIn(residency_rows()),
                         [](const ::testing::TestParamInfo<ResidencyRow>& info)
                         {
                             return ::SushiAI::Test::parameter_name(info.param.name);
                         });
