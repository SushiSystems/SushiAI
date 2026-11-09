/**************************************************************************/
/* test_served_kernel.cpp                                                 */
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
 * @file test_served_kernel.cpp
 * @brief Tests make_served_kernel(), which builds the replica kernel one machine serves.
 * @author Mustafa Garip
 */

#include <memory>
#include <string>
#include <gtest/gtest.h>
#include <SushiAI/distribution/replica_kernel.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <support/test_names.hpp>

#include "cluster_test_files.hpp"
#include "registries.hpp"
#include "served_kernel.hpp"
#include "training_replica_model.hpp"

namespace
{
    using SushiAI::Test::require_value;
    using namespace SushiAI;
    using namespace SushiAI::Apps::ClusterTest;

    /** @brief Holds a machine name and what the served kernel of the master and the worker does. */
    struct MachineCase
    {
        /** @brief Names the case. */
        const char* name;

        /** @brief The machine the kernel is made for. */
        const char* machine;
    };

    /** @brief Runs the rows of machine names the placement matches with the section's labels. */
    class Small_ServedKernelMachine : public ::testing::TestWithParam<MachineCase> {};

    /** @brief Returns the files of @p files as a replica source. */
    std::shared_ptr<const Apps::ReplicaSource> source_of(const ClusterTestSpec& files)
    {
        return Apps::ReplicaSource::make(Apps::compose_registries().value(), files.spec,
                                         files.model_spec, "architecture")
            .value();
    }
} // namespace

/** @brief Tests that a machine no replica names is refused, naming the machine. */
TEST(Small_ServedKernel, RefusesAMachineNoReplicaNames)
{
    const ClusterTestSpec files;

    const Result<std::unique_ptr<Distribution::ReplicaKernel>> kernel =
        Apps::make_served_kernel(source_of(files), files.section(), "nobody");

    SA_EXPECT_FAILURE(kernel.status(), StatusCode::InvalidDescription);
    EXPECT_NE(std::string(kernel.status().message()).find("'nobody'"), std::string::npos);
}

/** @brief Tests that a machine is found by the name placement matches it with. */
TEST_P(Small_ServedKernelMachine, FindsTheReplicaOfAMachineNamedInAnotherForm)
{
    const ClusterTestSpec files;

    const Result<std::unique_ptr<Distribution::ReplicaKernel>> kernel =
        Apps::make_served_kernel(source_of(files), files.section(), GetParam().machine);

    EXPECT_TRUE(kernel.has_value()) << kernel.status().message();
}

INSTANTIATE_TEST_SUITE_P(Rows, Small_ServedKernelMachine,
                         ::testing::Values(MachineCase{"AnUpperCaseQualifiedName", "WORKER.lab"},
                                           MachineCase{"TheExactLabel", "worker"}),
                         [](const ::testing::TestParamInfo<MachineCase>& info)
                         {
                             return ::SushiAI::Test::parameter_name(info.param.name);
                         });
