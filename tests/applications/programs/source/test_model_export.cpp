/**************************************************************************/
/* test_model_export.cpp                                                  */
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
 * @file test_model_export.cpp
 * @brief Tests `sushiai export` through the registries: each format's fixture comes back.
 * @author Mustafa Garip
 */

#include <filesystem>
#include <fstream>
#include <string>
#include <string_view>
#include <vector>
#include <gtest/gtest.h>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/config/model_writer.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/io/imported_model.hpp>
#include <SushiAI/io/imported_model_writer.hpp>
#include <SushiAI/io/importer_registry.hpp>
#include <SushiAI/io/model_files.hpp>
#include <SushiAI/scratch_files/scratch_directory.hpp>
#include "model_export.hpp"
#include "registries.hpp"
#include "weights_export.hpp"
#include "weights_import.hpp"

using namespace SushiAI;
// SaTest stays reachable inside a TEST body; see tests/applications/programs/README.md.
namespace SaTest = SushiAI::Test;

namespace
{
    /** @brief Holds a 32 by 32 Darknet network: a normalised convolution, a head and a yolo. */
    constexpr std::string_view K_SMALL_CFG =
        "[net]\nwidth=32\nheight=32\nchannels=3\n"
        "[convolutional]\nbatch_normalize=1\nfilters=4\nsize=3\nstride=1\npad=1\n"
        "activation=leaky\n"
        "[convolutional]\nfilters=18\nsize=1\nstride=1\npad=1\nactivation=linear\n"
        "[yolo]\nmask=0,1,2\nanchors=4,8, 8,16, 16,24\nclasses=1\nnum=3\n";

    /** @brief Returns the folder of the importers' fixtures for @p format. */
    std::filesystem::path fixture_folder(std::string_view format)
    {
        return std::filesystem::path(SA_REPOSITORY_ROOT) / "tests" / "vision" / "importers" /
               "fixtures" / std::string(format);
    }

    /** @brief Writes @p text to @p path and returns the path. */
    std::filesystem::path write_text(const std::filesystem::path& path, std::string_view text)
    {
        std::filesystem::create_directories(path.parent_path());
        std::ofstream stream(path, std::ios::binary);
        stream.write(text.data(), static_cast<std::streamsize>(text.size()));
        return path;
    }

    /** @brief Returns the path in @p written whose file name is @p name, or an empty path. */
    std::filesystem::path find_written(const std::vector<std::filesystem::path>& written,
                                       std::string_view name)
    {
        for (const std::filesystem::path& path : written)
        {
            if (path.filename() == name)
                return path;
        }

        return {};
    }

    /** @brief Expects the same model text and every tensor equal bit for bit. */
    void expect_same_model(const IO::ImportedModel& expected, const IO::ImportedModel& actual)
    {
        SA_ASSERT_OK_AND_ASSIGN(const std::string expected_text,
                                Config::write_model(expected.model));
        SA_ASSERT_OK_AND_ASSIGN(const std::string actual_text, Config::write_model(actual.model));
        EXPECT_EQ(expected_text, actual_text);
        ASSERT_EQ(expected.weights.size(), actual.weights.size());
        for (const IO::ImportedTensor& tensor : expected.weights)
        {
            const IO::ImportedTensor* found = nullptr;
            for (const IO::ImportedTensor& candidate : actual.weights)
            {
                if (candidate.record.name == tensor.record.name)
                    found = &candidate;
            }
            ASSERT_NE(found, nullptr) << tensor.record.name;
            EXPECT_EQ(found->record.dtype, tensor.record.dtype) << tensor.record.name;
            EXPECT_EQ(found->record.shape.to_string(), tensor.record.shape.to_string())
                << tensor.record.name;
            EXPECT_EQ(found->payload, tensor.payload) << tensor.record.name;
        }
    }

    /**
     * @brief Imports @p files as @p format, writes SushiAI's files, exports them as @p format
     *        and imports the exported files back, all through the composed registries.
     */
    void expect_format_round_trip(std::string_view format,
                                  const std::vector<std::filesystem::path>& files,
                                  const std::filesystem::path& scratch)
    {
        SA_ASSERT_OK_AND_ASSIGN(const Apps::Registries registries, Apps::compose_registries());
        SA_ASSERT_OK_AND_ASSIGN(
            const IO::ImportedModel imported,
            IO::import_model(registries.importers, format,
                             SushiRuntime::span<const std::filesystem::path>(files.data(),
                                                                             files.size())));
        SA_ASSERT_OK_AND_ASSIGN(const IO::WrittenImport written,
                                IO::write_imported_model(imported, scratch / "sushi",
                                                         IO::ExistingFiles::Refuse));

        Apps::ModelExportRequest request;
        request.format = std::string(format);
        request.model = find_written(written.files, IO::K_IMPORTED_MODEL_FILE);
        request.weights = find_written(written.files, IO::K_IMPORTED_WEIGHTS_FILE);
        request.training = find_written(written.files, IO::K_IMPORTED_TRAINING_FILE);
        request.output = scratch / "exported";
        SA_ASSERT_OK_AND_ASSIGN(const Apps::ModelExportReport report,
                                Apps::export_model_files(request, registries));
        ASSERT_FALSE(report.written.empty());

        SA_ASSERT_OK_AND_ASSIGN(
            const IO::ImportedModel back,
            IO::import_model(registries.importers, format,
                             SushiRuntime::span<const std::filesystem::path>(
                                 report.written.data(), report.written.size())));
        expect_same_model(imported, back);
    }
} // namespace

/** @brief Tests that the Caffe fixture exports through the registry and imports back unchanged. */
TEST(Small_ModelExport, TheCaffeFixtureComesBackThroughTheRegistry)
{
    const SaTest::ScratchDirectory scratch("sushiai_model_export_caffe");
    expect_format_round_trip("caffe",
                             {fixture_folder("caffe") / "lenet.prototxt",
                              fixture_folder("caffe") / "lenet.caffemodel"},
                             scratch.path());
}

/** @brief Tests that a Darknet .cfg exports through the registry and imports back unchanged. */
TEST(Small_ModelExport, ADarknetConfigComesBackThroughTheRegistry)
{
    const SaTest::ScratchDirectory scratch("sushiai_model_export_darknet");
    const std::filesystem::path cfg = write_text(scratch.path() / "small.cfg", K_SMALL_CFG);
    expect_format_round_trip("darknet", {cfg}, scratch.path());
}

/** @brief Tests that a format with no exporter row is refused with the rows named, and no file. */
TEST(Small_ModelExport, AnUnknownFormatNamesTheExportersAndWritesNoFile)
{
    SA_ASSERT_OK_AND_ASSIGN(const Apps::Registries registries, Apps::compose_registries());
    const SaTest::ScratchDirectory scratch("sushiai_model_export_unknown");
    const std::filesystem::path cfg = write_text(scratch.path() / "small.cfg", K_SMALL_CFG);
    const std::filesystem::path files[] = {cfg};
    SA_ASSERT_OK_AND_ASSIGN(const IO::ImportedModel imported,
                            IO::import_model(registries.importers, "darknet",
                                             SushiRuntime::span<const std::filesystem::path>(
                                                 files, 1)));
    SA_ASSERT_OK_AND_ASSIGN(const IO::WrittenImport written,
                            IO::write_imported_model(imported, scratch.path() / "sushi",
                                                     IO::ExistingFiles::Refuse));

    Apps::ModelExportRequest request;
    request.format = "tflite";
    request.model = find_written(written.files, IO::K_IMPORTED_MODEL_FILE);
    request.output = scratch.path() / "exported";

    SA_EXPECT_FAILURE(Apps::export_model_files(request, registries),
                      StatusCode::InvalidDescription, "'tflite'", "onnx", "caffe", "darknet");
    EXPECT_FALSE(std::filesystem::exists(request.output / "model.cfg"));
}

/** @brief Tests that a second export refuses the files of the first, and --force replaces them. */
TEST(Small_ModelExport, ASecondExportIsRefusedUnlessForced)
{
    SA_ASSERT_OK_AND_ASSIGN(const Apps::Registries registries, Apps::compose_registries());
    const SaTest::ScratchDirectory scratch("sushiai_model_export_force");
    const std::filesystem::path cfg = write_text(scratch.path() / "small.cfg", K_SMALL_CFG);
    const std::filesystem::path files[] = {cfg};
    SA_ASSERT_OK_AND_ASSIGN(const IO::ImportedModel imported,
                            IO::import_model(registries.importers, "darknet",
                                             SushiRuntime::span<const std::filesystem::path>(
                                                 files, 1)));
    SA_ASSERT_OK_AND_ASSIGN(const IO::WrittenImport written,
                            IO::write_imported_model(imported, scratch.path() / "sushi",
                                                     IO::ExistingFiles::Refuse));

    Apps::ModelExportRequest request;
    request.format = "darknet";
    request.model = find_written(written.files, IO::K_IMPORTED_MODEL_FILE);
    request.output = scratch.path() / "exported";
    request.name = "small";
    SA_ASSERT_OK_AND_ASSIGN(const Apps::ModelExportReport first,
                            Apps::export_model_files(request, registries));
    EXPECT_EQ(first.written.size(), 1u);
    EXPECT_TRUE(std::filesystem::exists(request.output / "small.cfg"));

    SA_EXPECT_FAILURE(Apps::export_model_files(request, registries), StatusCode::Incompatible,
                      "small.cfg");
    request.existing = IO::ExistingFiles::Replace;
    SA_EXPECT_OK(Apps::export_model_files(request, registries));
}

/**
 * @brief Tests that the ONNX fixture's weights go to each weight format and come back bit for
 *        bit through `import weights`.
 */
TEST(Small_ModelExport, TheFixtureWeightsComeBackFromEachWeightFormat)
{
    SA_ASSERT_OK_AND_ASSIGN(const Apps::Registries registries, Apps::compose_registries());
    const SaTest::ScratchDirectory scratch("sushiai_model_export_weights");
    const std::filesystem::path files[] = {fixture_folder("onnx") / "silu_resize.onnx"};
    SA_ASSERT_OK_AND_ASSIGN(const IO::ImportedModel imported,
                            IO::import_model(registries.importers, "onnx",
                                             SushiRuntime::span<const std::filesystem::path>(
                                                 files, 1)));
    ASSERT_FALSE(imported.weights.empty());
    SA_ASSERT_OK_AND_ASSIGN(const IO::WrittenImport written,
                            IO::write_imported_model(imported, scratch.path() / "sushi",
                                                     IO::ExistingFiles::Refuse));
    const std::filesystem::path model = find_written(written.files, IO::K_IMPORTED_MODEL_FILE);
    const std::filesystem::path map = write_text(scratch.path() / "map.json", "{}");

    for (const char* const format : {"safetensors", "torch"})
    {
        Apps::WeightsExportRequest exported;
        exported.format = format;
        exported.model = model;
        exported.weights = find_written(written.files, IO::K_IMPORTED_WEIGHTS_FILE);
        exported.output = scratch.path() / (std::string(format) + ".bin");
        SA_ASSERT_OK_AND_ASSIGN(const Apps::WeightsExportReport report,
                                Apps::export_weights_file(exported, registries));
        EXPECT_EQ(report.written, imported.weights.size()) << format;

        Apps::WeightsImportRequest back;
        back.model = model;
        back.weights = exported.output;
        back.map = map;
        back.output = scratch.path() / (std::string(format) + ".sushi");
        SA_ASSERT_OK_AND_ASSIGN(const Apps::WeightsImportReport imported_back,
                                Apps::import_weights_file(back, registries));
        EXPECT_EQ(imported_back.written, imported.weights.size()) << format;

        IO::ModelFilePaths paths;
        paths.model = model;
        paths.weights = back.output;
        SA_ASSERT_OK_AND_ASSIGN(const IO::ImportedModel returned, IO::read_model_files(paths));
        expect_same_model(imported, returned);
    }
}
