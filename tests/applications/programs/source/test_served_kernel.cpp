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
 * @file test_served_kernel.cpp
 * @brief Tests make_served_kernel(), which builds the replica kernel one machine serves.
 * @author Mustafa Garip
 */

#include <memory>
#include <string>
#include <gtest/gtest.h>
#include <SushiAI/distribution/replica_kernel.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>

#include "cluster_test_files.hpp"
#include "registries.hpp"
#include "served_kernel.hpp"
#include "training_replica_model.hpp"

namespace
{
    using SushiAI::Test::require_value;
    using namespace SushiAI;
    using namespace SushiAI::Apps::ClusterTest;

    /** @brief Returns the files of @p files as a replica source. */
    std::shared_ptr<const Apps::ReplicaSource> source_of(const ClusterTestFiles& files)
    {
        return Apps::ReplicaSource::make(Apps::compose_registries().value(), files.spec,
                                         files.model_spec)
            .value();
    }
} // namespace

/** @brief Tests that a machine no replica names is refused, naming the machine. */
TEST(Medium_ServedKernel, RefusesAMachineNoReplicaNames)
{
    const ClusterTestFiles files("sushiai_served_kernel_unnamed.json");

    const Result<std::unique_ptr<Distribution::ReplicaKernel>> kernel =
        Apps::make_served_kernel(source_of(files), files.section(), "nobody");

    SA_EXPECT_FAILURE(kernel.status(), StatusCode::InvalidDescription);
    EXPECT_NE(std::string(kernel.status().message()).find("'nobody'"), std::string::npos);
}

/** @brief Tests that a machine is found by the name placement matches it with, not the exact text. */
TEST(Medium_ServedKernel, Regression_FindsTheReplicaOfAMachineNamedInAnotherCase)
{
    const ClusterTestFiles files("sushiai_served_kernel_case.json");

    const Result<std::unique_ptr<Distribution::ReplicaKernel>> kernel =
        Apps::make_served_kernel(source_of(files), files.section(), "WORKER.lab");

    EXPECT_TRUE(kernel.has_value()) << kernel.status().message();
}
