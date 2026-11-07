/**************************************************************************/
/* test_cluster_training.cpp                                              */
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
 * @file test_cluster_training.cpp
 * @brief Tests what `sushiai train` builds for a training file with and without a cluster.
 * @author Mustafa Garip
 */

#include <gtest/gtest.h>
#include <SushiRuntime/sushiruntime.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/config/value.hpp>

#include "cluster_test_files.hpp"
#include "cluster_training.hpp"

namespace
{
    using SushiAI::Test::require_value;
    using namespace SushiAI;
    using namespace SushiAI::Apps::ClusterTest;
} // namespace

/** @brief Tests that a file without a cluster section keeps the default runtime. */
TEST(Small_ClusterTraining, AFileWithoutAClusterKeepsTheDefaultRuntime)
{
    ClusterTestSpec files;
    files.spec.cluster = Config::Value();

    const SushiRuntime::RuntimeConfig config =
        Apps::training_runtime_config(files.spec).value();

    EXPECT_EQ(config.node_name, SushiRuntime::RuntimeConfig().node_name);
}
