/**************************************************************************/
/* test_weights_import.cpp                                                */
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
 * @file test_weights_import.cpp
 * @brief Tests the containers a weights import reads and that a failed import writes no file.
 * @author Mustafa Garip
 */

#include <filesystem>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/io/weight_source.hpp>
#include <SushiAI/scratch_files/scratch_directory.hpp>
#include "registries.hpp"
#include "weights_import.hpp"

using namespace SushiAI;
// SaTest stays reachable inside a TEST body; see tests/applications/programs/README.md.
namespace SaTest = SushiAI::Test;

/** @brief Tests that the composed sources are npz, the PyTorch zip and safetensors, in that order. */
TEST(Small_WeightsImport, TheComposedSourcesAreNpzThenPytorchThenSafetensors)
{
    SA_ASSERT_OK_AND_ASSIGN(const IO::WeightSourceRegistry sources, Apps::compose_weight_sources());

    ASSERT_EQ(sources.rows().size(), 3u);
    EXPECT_EQ(sources.rows()[0].format, "npz");
    EXPECT_EQ(sources.rows()[1].format, "torch");
    EXPECT_EQ(sources.rows()[2].format, "safetensors");
}

/** @brief Tests that an import of a model file that is not there fails and writes no output. */
TEST(Small_WeightsImport, AnImportOfAMissingModelFileFailsAndWritesNoFile)
{
    SA_ASSERT_OK_AND_ASSIGN(const Apps::Registries registries, Apps::compose_registries());
    const SaTest::ScratchDirectory scratch("sushiai_weights_import");
    std::filesystem::create_directories(scratch.path());
    Apps::WeightsImportRequest request;
    request.model = scratch.path() / "absent_model.json";
    request.weights = scratch.path() / "absent_weights.safetensors";
    request.output = scratch.path() / "out.sushi";

    const Result<Apps::WeightsImportReport> report = Apps::import_weights_file(request, registries);

    EXPECT_FALSE(report.has_value());
    EXPECT_FALSE(std::filesystem::exists(request.output));
}
