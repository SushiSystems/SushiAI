/**************************************************************************/
/* test_ultralytics_checkpoint.cpp                                        */
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
 * @file test_ultralytics_checkpoint.cpp
 * @brief Tests the reader of Ultralytics checkpoints on the exported fixture and on refused bytes.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/config/value.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/io/checkpoint_format.hpp>
#include <SushiAI/vision/importers/ultralytics_checkpoint.hpp>

using namespace SushiAI;

namespace
{
    /** @brief Holds the normalisations of YOLO26n, each with one step counter. */
    constexpr std::size_t K_STEP_COUNTERS = 114;

    /** @brief Returns the checkpoint the reference exporter writes. */
    std::filesystem::path fixture_checkpoint()
    {
        return std::filesystem::path(SA_REPOSITORY_ROOT) / "tests" / "vision" / "registration" /
               "fixtures" / "yolo26" / "model.pt";
    }

    /** @brief Returns the weight file the reference exporter writes beside the checkpoint. */
    std::filesystem::path fixture_weights()
    {
        return fixture_checkpoint().parent_path() / "weights.sushi";
    }

    /** @brief Reads the fixture checkpoint, or skips when the exporter has not run. */
    class Medium_UltralyticsCheckpoint : public ::testing::Test
    {
        protected:
            /** @brief Skips the test when model.pt is absent. */
            void SetUp() override
            {
                if (!std::filesystem::exists(fixture_checkpoint()))
                    GTEST_SKIP() << "run tools/reference/export_yolo26_reference.py tests first";
            }
    };

    /** @brief Reads the fixture weight file, or skips when it is absent or kept elsewhere. */
    class Medium_UltralyticsWeightFile : public ::testing::Test
    {
        protected:
            /** @brief Skips the test when weights.sushi is not in the fixture folder. */
            void SetUp() override
            {
                if (!std::filesystem::exists(fixture_weights()))
                    GTEST_SKIP() << "run tools/reference/export_yolo26_reference.py tests; "
                                    "--external-weights keeps weights.sushi out of this folder";
            }
    };
} // namespace

/** @brief Tests that the checkpoint's description is the yaml of the scale n it was built at. */
TEST_F(Medium_UltralyticsCheckpoint, ItReadsTheDescriptionOfScaleN)
{
    SA_ASSERT_OK_AND_ASSIGN(const Vision::Importers::UltralyticsCheckpoint checkpoint,
                            Vision::Importers::load_ultralytics_checkpoint(fixture_checkpoint()));

    SA_ASSERT_OK_AND_ASSIGN(const Config::Value& scale,
                            checkpoint.architecture.at("scale", "yaml"));
    EXPECT_EQ(scale.as_string("scale").value(), "n");
}

/** @brief Tests that tensors are named by their path below the layer list, first one first. */
TEST_F(Medium_UltralyticsCheckpoint, ItNamesTensorsByTheirPathBelowTheLayerList)
{
    SA_ASSERT_OK_AND_ASSIGN(const Vision::Importers::UltralyticsCheckpoint checkpoint,
                            Vision::Importers::load_ultralytics_checkpoint(fixture_checkpoint()));

    ASSERT_FALSE(checkpoint.tensors.empty());
    EXPECT_EQ(checkpoint.tensors.front().name, "0.conv.weight");
}

/** @brief Tests that the step counter of every normalisation is left out and listed. */
TEST_F(Medium_UltralyticsCheckpoint, ItLeavesOutEveryStepCounter)
{
    SA_ASSERT_OK_AND_ASSIGN(const Vision::Importers::UltralyticsCheckpoint checkpoint,
                            Vision::Importers::load_ultralytics_checkpoint(fixture_checkpoint()));

    EXPECT_EQ(checkpoint.skipped.size(), K_STEP_COUNTERS);
}

/** @brief Tests that the exporter's weight file is a version 2 file of model state alone. */
TEST_F(Medium_UltralyticsWeightFile, TheExportedWeightFileIsVersionTwoWithNoRunState)
{
    std::ifstream stream(fixture_weights(), std::ios::binary);
    SA_ASSERT_OK_AND_ASSIGN(const IO::CheckpointReader reader, IO::CheckpointReader::open(stream));

    EXPECT_EQ(reader.version(), IO::K_CHECKPOINT_VERSION);
    EXPECT_EQ(reader.run_state_count(), 0u);
    EXPECT_GT(reader.count(), 0u);
}

/** @brief Tests that bytes that are no zip archive are refused as no checkpoint. */
TEST(Small_UltralyticsCheckpoint, BytesThatAreNoArchiveAreRefused)
{
    const std::vector<std::byte> bytes(64, std::byte{0x41});

    SA_EXPECT_FAILURE(Vision::Importers::read_ultralytics_checkpoint(
                          SushiRuntime::span<const std::byte>(bytes.data(), bytes.size()),
                          "bad.pt"),
                      StatusCode::InvalidFile, "'bad.pt'");
}
