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
TEST(Medium_ClusterTraining, AFileWithoutAClusterKeepsTheDefaultRuntime)
{
    ClusterTestFiles files("sushiai_cluster_training_none.json");
    files.spec.cluster = Config::Value();

    const SushiRuntime::RuntimeConfig config =
        Apps::training_runtime_config(files.spec).value();

    EXPECT_EQ(config.node_name, SushiRuntime::RuntimeConfig().node_name);
}

/** @brief Tests that a file with a cluster section names the master after replica 0. */
TEST(Medium_ClusterTraining, AFileWithAClusterNamesTheMasterAfterReplicaZero)
{
    const ClusterTestFiles files("sushiai_cluster_training_master.json");

    EXPECT_EQ(Apps::training_runtime_config(files.spec).value().node_name, "master");
}
