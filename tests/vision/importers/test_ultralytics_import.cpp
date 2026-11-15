/**************************************************************************/
/* test_ultralytics_import.cpp                                            */
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
 * @file test_ultralytics_import.cpp
 * @brief Tests the Ultralytics importer on the YOLO26n checkpoint the reference exporter writes.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/config/model_config.hpp>
#include <SushiAI/config/model_writer.hpp>
#include <SushiAI/dtype/dtype.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/io/imported_model.hpp>
#include <SushiAI/reference_files/reference_file.hpp>
#include <SushiAI/vision/importers/ultralytics_import.hpp>

using namespace SushiAI;

namespace
{
    /** @brief Holds the floating tensors MANIFEST.json lists for weights.sushi. */
    constexpr std::size_t K_REFERENCE_TENSORS = 594;

    /** @brief Returns the fixture folder of the YOLO26n reference set. */
    std::filesystem::path fixture_folder()
    {
        return std::filesystem::path(SA_REPOSITORY_ROOT) / "tests" / "vision" / "registration" /
               "fixtures" / "yolo26";
    }

    /** @brief Returns the hand-written YOLO26 model file. */
    std::filesystem::path yolo26_model_path()
    {
        return std::filesystem::path(SA_REPOSITORY_ROOT) / "applications" / "examples" / "yolo26" /
               "model.json";
    }

    /**
     * @brief Expects every tensor of @p imported to equal, byte for byte and in shape, the
     *        record of @p reference named "model." and its name.
     */
    void expect_same_tensors(const std::vector<IO::ImportedTensor>& imported,
                             SushiAI::Test::ReferenceFile& reference)
    {
        for (const IO::ImportedTensor& tensor : imported)
        {
            const SushiAI::Test::ReferenceTensor wanted = reference.read("model." + tensor.record.name);
            ASSERT_TRUE(wanted.found) << tensor.record.name << ": " << wanted.reason;
            EXPECT_EQ(wanted.dtype, DType::Float32) << tensor.record.name;
            EXPECT_EQ(tensor.record.shape, wanted.shape) << tensor.record.name;
            ASSERT_EQ(tensor.payload.size(), wanted.payload.size()) << tensor.record.name;
            EXPECT_EQ(std::memcmp(tensor.payload.data(), wanted.payload.data(),
                                  wanted.payload.size()),
                      0)
                << tensor.record.name;
        }
    }

    /** @brief Imports model.pt of the fixture folder, or skips when the exporter has not run. */
    class Medium_UltralyticsImport : public ::testing::Test
    {
        protected:
            /** @brief Skips the test when the fixture folder lacks model.pt or weights.sushi. */
            void SetUp() override
            {
                if (!std::filesystem::exists(fixture_folder() / "model.pt") ||
                    !std::filesystem::exists(fixture_folder() / "weights.sushi"))
                    GTEST_SKIP() << "run tools/reference/export_yolo26_reference.py tests first";
            }

            /** @brief Returns the import of the fixture's model.pt. */
            static Result<IO::ImportedModel> import_fixture()
            {
                const std::filesystem::path files[] = {fixture_folder() / "model.pt"};
                return Vision::Importers::import_ultralytics_files(
                    SushiRuntime::span<const std::filesystem::path>(files, 1));
            }
    };
} // namespace

/** @brief Tests that the import writes every floating tensor Ultralytics' exporter wrote. */
TEST_F(Medium_UltralyticsImport, ItImportsEveryFloatingTensorOfTheCheckpoint)
{
    SA_ASSERT_OK_AND_ASSIGN(const IO::ImportedModel imported, import_fixture());

    EXPECT_EQ(imported.weights.size(), K_REFERENCE_TENSORS);
}

/** @brief Tests that each imported tensor is the exported one, bit for bit, in its layout. */
TEST_F(Medium_UltralyticsImport, EachTensorIsTheExportedOneBitForBit)
{
    SA_ASSERT_OK_AND_ASSIGN(const IO::ImportedModel imported, import_fixture());
    SushiAI::Test::ReferenceFile reference(fixture_folder().string(), "weights.sushi");
    ASSERT_TRUE(reference.is_present()) << reference.reason();

    expect_same_tensors(imported.weights, reference);
}

/** @brief Tests that the imported model is the YOLO26 model file written by hand. */
TEST_F(Medium_UltralyticsImport, TheModelIsTheHandWrittenYolo26ModelFile)
{
    SA_ASSERT_OK_AND_ASSIGN(const IO::ImportedModel imported, import_fixture());
    SA_ASSERT_OK_AND_ASSIGN(const Config::ModelSpec wanted, Config::load_model(yolo26_model_path()));

    SA_ASSERT_OK_AND_ASSIGN(const std::string found_text, Config::write_model(imported.model));
    SA_ASSERT_OK_AND_ASSIGN(const std::string wanted_text, Config::write_model(wanted));
    EXPECT_EQ(found_text, wanted_text);
}

/** @brief Tests that a call with two files is refused before any file is opened. */
TEST(Small_UltralyticsImport, TwoFilesAreRefused)
{
    const std::filesystem::path files[] = {"a.pt", "b.pt"};

    SA_EXPECT_FAILURE(Vision::Importers::import_ultralytics_files(
                          SushiRuntime::span<const std::filesystem::path>(files, 2)),
                      StatusCode::InvalidDescription, "'ultralytics'", "2 files");
}
