/**************************************************************************/
/* test_weights_export.cpp                                                */
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
 * @file test_weights_export.cpp
 * @brief Tests the formats a weights export writes and that a refused export writes no file.
 * @author Mustafa Garip
 */

#include <filesystem>
#include <vector>
#include <gtest/gtest.h>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/scratch_files/scratch_directory.hpp>
#include "registries.hpp"
#include "weights_export.hpp"

using namespace SushiAI;
// SaTest stays reachable inside a TEST body; see tests/applications/programs/README.md.
namespace SaTest = SushiAI::Test;

/** @brief Tests that the composed writers are safetensors, then the PyTorch state dictionary. */
TEST(Small_WeightsExport, TheComposedWritersAreSafetensorsThenTorch)
{
    const std::vector<Apps::WeightWriterEntry> writers = Apps::compose_weight_writers();

    ASSERT_EQ(writers.size(), 2u);
    EXPECT_EQ(writers[0].format, "safetensors");
    EXPECT_EQ(writers[1].format, "torch");
    EXPECT_NE(writers[0].write, nullptr);
    EXPECT_NE(writers[1].write, nullptr);
}

/** @brief Tests that an unknown weight format is refused with both formats named, and no file. */
TEST(Small_WeightsExport, AnUnknownFormatNamesTheFormatsAndWritesNoFile)
{
    SA_ASSERT_OK_AND_ASSIGN(const Apps::Registries registries, Apps::compose_registries());
    const SaTest::ScratchDirectory scratch("sushiai_weights_export");
    Apps::WeightsExportRequest request;
    request.format = "npz";
    request.model = scratch.path() / "absent_model.json";
    request.weights = scratch.path() / "absent_weights.sushi";
    request.output = scratch.path() / "out.safetensors";

    SA_EXPECT_FAILURE(Apps::export_weights_file(request, registries),
                      StatusCode::InvalidDescription, "'npz'", "'safetensors'", "'torch'");
    EXPECT_FALSE(std::filesystem::exists(request.output));
}

/** @brief Tests that an export of a model file that is not there fails and writes no output. */
TEST(Small_WeightsExport, AnExportOfAMissingModelFileFailsAndWritesNoFile)
{
    SA_ASSERT_OK_AND_ASSIGN(const Apps::Registries registries, Apps::compose_registries());
    const SaTest::ScratchDirectory scratch("sushiai_weights_export_missing");
    Apps::WeightsExportRequest request;
    request.format = "safetensors";
    request.model = scratch.path() / "absent_model.json";
    request.weights = scratch.path() / "absent_weights.sushi";
    request.output = scratch.path() / "out.safetensors";

    const Result<Apps::WeightsExportReport> report = Apps::export_weights_file(request, registries);

    EXPECT_FALSE(report.has_value());
    EXPECT_FALSE(std::filesystem::exists(request.output));
}
