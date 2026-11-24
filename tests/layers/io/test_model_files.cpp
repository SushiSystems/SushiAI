/**************************************************************************/
/* test_model_files.cpp                                                   */
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
 * @file test_model_files.cpp
 * @brief Tests the reader of the three files of a SushiAI model against the files the import
 *        writer wrote.
 * @author Mustafa Garip
 */

#include <cstring>
#include <filesystem>
#include <string>
#include <utility>
#include <vector>
#include <gtest/gtest.h>
#include <SushiAI/config/model_writer.hpp>
#include <SushiAI/dtype/dtype.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/io/imported_model.hpp>
#include <SushiAI/io/imported_model_writer.hpp>
#include <SushiAI/io/model_files.hpp>
#include <SushiAI/model_text/model_text.hpp>
#include <SushiAI/scratch_files/scratch_directory.hpp>
#include <SushiAI/shape/shape.hpp>

using namespace SushiAI;

namespace
{
    using SushiAI::Test::require_value;
    using SushiAI::Test::ScratchDirectory;

    /** @brief Returns a float32 tensor named @p name holding @p values under @p shape. */
    IO::ImportedTensor make_tensor(std::string name, const Shape& shape,
                                   const std::vector<float>& values)
    {
        IO::ImportedTensor tensor;
        tensor.record.name = std::move(name);
        tensor.record.dtype = DType::Float32;
        tensor.record.shape = shape;
        tensor.payload.resize(values.size() * sizeof(float));
        std::memcpy(tensor.payload.data(), values.data(), tensor.payload.size());
        return tensor;
    }
} // namespace

/** @brief Tests that the model, the tensors, the recipe and the data block come back. */
TEST(Small_ModelFiles, ReadsTheFilesTheImportWriterWrote)
{
    IO::ImportedModel written;
    written.model = SushiAI::Test::model_from(R"({
        "format_version": 2,
        "input": { "shape": [4] },
        "layers": [ { "type": "linear", "units": 2 } ]
    })");
    written.weights.push_back(make_tensor("0.weight", Shape{2, 4}, {1, 2, 3, 4, 5, 6, 7, 8}));
    written.weights.push_back(make_tensor("0.bias", Shape{2}, {9, 10}));
    written.training = SushiAI::Test::attributes_from(R"({ "epochs": 3 })");
    written.data = SushiAI::Test::attributes_from(R"({ "pipeline": "darknet" })");

    const ScratchDirectory scratch("sushiai_model_files");
    SA_EXPECT_OK(IO::write_imported_model(written, scratch.path(), IO::ExistingFiles::Refuse));

    IO::ModelFilePaths paths;
    paths.model = scratch.path() / "model.json";
    paths.weights = scratch.path() / "weights.sushi";
    paths.training = scratch.path() / "train.json";
    SA_ASSERT_OK_AND_ASSIGN(const IO::ImportedModel read, IO::read_model_files(paths));

    EXPECT_EQ(require_value(Config::write_model(read.model)),
              require_value(Config::write_model(written.model)));
    ASSERT_EQ(read.weights.size(), 2u);
    for (std::size_t i = 0; i < 2; ++i)
    {
        EXPECT_EQ(read.weights[i].record.name, written.weights[i].record.name);
        EXPECT_EQ(read.weights[i].record.shape, written.weights[i].record.shape);
        EXPECT_EQ(read.weights[i].payload, written.weights[i].payload);
    }
    EXPECT_EQ(read.training.keys(), (std::vector<std::string>{"epochs"}));
    EXPECT_TRUE(read.data.has("pipeline"));
}

/** @brief Tests that a model file alone gives no weights, no recipe and no data. */
TEST(Small_ModelFiles, AModelFileAloneStatesNothingElse)
{
    IO::ImportedModel written;
    written.model = SushiAI::Test::model_from(R"({
        "format_version": 2,
        "input": { "shape": [4] },
        "layers": [ { "type": "relu" } ]
    })");
    const ScratchDirectory scratch("sushiai_model_file_alone");
    SA_EXPECT_OK(IO::write_imported_model(written, scratch.path(), IO::ExistingFiles::Refuse));

    IO::ModelFilePaths paths;
    paths.model = scratch.path() / "model.json";
    SA_ASSERT_OK_AND_ASSIGN(const IO::ImportedModel read, IO::read_model_files(paths));
    EXPECT_TRUE(read.weights.empty());
    EXPECT_TRUE(read.training.is_null());
    EXPECT_TRUE(read.data.is_null());
}

/** @brief Tests that a missing weight file is refused by name. */
TEST(Small_ModelFiles, AMissingWeightFileIsRefused)
{
    IO::ImportedModel written;
    written.model = SushiAI::Test::model_from(R"({
        "format_version": 2,
        "input": { "shape": [4] },
        "layers": [ { "type": "relu" } ]
    })");
    const ScratchDirectory scratch("sushiai_model_file_missing");
    SA_EXPECT_OK(IO::write_imported_model(written, scratch.path(), IO::ExistingFiles::Refuse));

    IO::ModelFilePaths paths;
    paths.model = scratch.path() / "model.json";
    paths.weights = scratch.path() / "absent.sushi";
    SA_EXPECT_FAILURE(IO::read_model_files(paths), StatusCode::FileNotFound, "absent.sushi");
}
