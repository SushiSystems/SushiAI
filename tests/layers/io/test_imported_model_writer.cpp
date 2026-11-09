/**************************************************************************/
/* test_imported_model_writer.cpp                                         */
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
 * @file test_imported_model_writer.cpp
 * @brief Tests the writer of an imported model: the training file it composes, the three files it
 *        writes and what it refuses before it writes any.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <ios>
#include <iterator>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>
#include <gtest/gtest.h>
#include <SushiAI/config/model_config.hpp>
#include <SushiAI/config/train_config.hpp>
#include <SushiAI/config/value.hpp>
#include <SushiAI/dtype/dtype.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/shape/shape.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/io/checkpoint_format.hpp>
#include <SushiAI/io/imported_model.hpp>
#include <SushiAI/io/imported_model_writer.hpp>
#include <SushiAI/model_text/model_text.hpp>
#include <SushiAI/scratch_files/scratch_directory.hpp>

using namespace SushiAI;
using IO::ExistingFiles;
using IO::ImportedModel;
using IO::ImportedTensor;

namespace
{
    using SushiAI::Test::require_value;
    using SushiAI::Test::ScratchDirectory;

    /** @brief Returns a float32 tensor named @p name holding @p values under @p shape. */
    ImportedTensor make_tensor(std::string name, const Shape& shape,
                               const std::vector<float>& values)
    {
        ImportedTensor tensor;
        tensor.record.name = std::move(name);
        tensor.record.dtype = DType::Float32;
        tensor.record.shape = shape;
        tensor.payload.resize(values.size() * sizeof(float));
        std::memcpy(tensor.payload.data(), values.data(), tensor.payload.size());
        return tensor;
    }

    /** @brief Returns a one-row model with a task, two tensors, a recipe and one note. */
    ImportedModel make_imported()
    {
        ImportedModel imported;
        imported.model = SushiAI::Test::model_from(R"({
            "format_version": 2,
            "input": { "shape": [4] },
            "layers": [ { "type": "linear", "units": 2 } ],
            "task": { "type": "stub_task", "classes": 2 }
        })");
        imported.weights.push_back(
            make_tensor("0.weight", Shape{2, 4}, {1, 2, 3, 4, 5, 6, 7, 8}));
        imported.weights.push_back(make_tensor("0.bias", Shape{2}, {9, 10}));
        imported.training = SushiAI::Test::attributes_from(R"({
            "objective": { "type": "stub_objective" },
            "optimizer": { "type": "sgd", "momentum": 0.9 }
        })");
        imported.notes.push_back("a note");
        return imported;
    }

    /** @brief Returns the file name of each path @p written holds, in order. */
    std::vector<std::string> file_names(const IO::WrittenImport& written)
    {
        std::vector<std::string> names;
        for (const std::filesystem::path& file : written.files)
            names.push_back(file.filename().string());
        return names;
    }

    /** @brief Holds a record of a weight file and the payload that follows it. */
    struct ReadTensor
    {
        IO::TensorRecord record;            /**< The record header. */
        std::vector<std::byte> payload;     /**< The bytes after the header. */
    };

    /** @brief Reads every record of the weight file at @p path, in order. */
    std::vector<ReadTensor> read_weight_file(const std::filesystem::path& path)
    {
        std::vector<ReadTensor> tensors;
        std::ifstream in(path, std::ios::binary);
        EXPECT_TRUE(in.is_open());
        IO::CheckpointReader reader = require_value(IO::CheckpointReader::open(in));
        while (reader.read_count() < reader.count())
        {
            ReadTensor tensor;
            tensor.record = require_value(reader.next());
            tensor.payload.resize(tensor.record.payload_bytes());
            SA_EXPECT_OK(reader.read_payload(tensor.record, tensor.payload.data()));
            tensors.push_back(std::move(tensor));
        }
        return tensors;
    }

    /** @brief Returns the bytes of the file at @p path; empty when it cannot be opened. */
    std::string read_bytes(const std::filesystem::path& path)
    {
        std::ifstream in(path, std::ios::binary);
        return std::string(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
    }
} // namespace

/** @brief Tests the keys of the composed training file, in the order the writer states them. */
TEST(Small_ImportedModelWriter, ComposesTheTrainingFileAroundTheRecipe)
{
    SA_ASSERT_OK_AND_ASSIGN(const Config::Value training,
                            IO::compose_training_file(make_imported()));

    EXPECT_EQ(training.keys(),
              (std::vector<std::string>{"sushiai_train", "model", "weights", "data",
                                        "checkpoint", "objective", "optimizer"}));
    EXPECT_EQ(require_value(require_value(training.at("sushiai_train", "file")).as_int("version")),
              Config::K_TRAIN_VERSION);
    EXPECT_EQ(require_value(require_value(training.at("model", "file")).as_string("model")),
              "model.json");
    EXPECT_EQ(require_value(require_value(training.at("weights", "file")).as_string("weights")),
              "weights.sushi");
    EXPECT_EQ(require_value(require_value(training.at("checkpoint",
                                                      "file")).as_string("checkpoint")),
              "trained.sushi");
    const Config::Value& optimizer = require_value(training.at("optimizer", "file"));
    EXPECT_EQ(require_value(require_value(optimizer.at("type", "optimizer")).as_string("type")),
              "sgd");
}

/** @brief Tests that the data block is null, since no foreign file names a dataset. */
TEST(Small_ImportedModelWriter, LeavesTheDataBlockNull)
{
    SA_ASSERT_OK_AND_ASSIGN(const Config::Value training,
                            IO::compose_training_file(make_imported()));

    ASSERT_TRUE(training.has("data"));
    EXPECT_TRUE(require_value(training.at("data", "file")).is_null());
}

/** @brief Tests that a model without tensors states no "weights" key. */
TEST(Small_ImportedModelWriter, StatesNoWeightsKeyWithoutTensors)
{
    ImportedModel imported = make_imported();
    imported.weights.clear();

    SA_ASSERT_OK_AND_ASSIGN(const Config::Value training, IO::compose_training_file(imported));
    EXPECT_FALSE(training.has("weights"));
}

/** @brief Tests that a recipe stating a key of the writer is refused by that key. */
TEST(Small_ImportedModelWriter, ARecipeThatStatesAWriterKeyIsRefused)
{
    ImportedModel imported = make_imported();
    imported.training = SushiAI::Test::attributes_from(R"({ "model": "other.json" })");

    SA_EXPECT_FAILURE(IO::compose_training_file(imported), StatusCode::InvalidDescription,
                      "'model'", "training recipe");
}

/** @brief Tests that a recipe that is no object is refused by its kind. */
TEST(Small_ImportedModelWriter, ARecipeThatIsNoObjectIsRefused)
{
    ImportedModel imported = make_imported();
    imported.training = Config::Value::integer(3);

    SA_EXPECT_FAILURE(IO::compose_training_file(imported), StatusCode::InvalidDescription,
                      "must be an object");
}

namespace
{
    /** @brief Holds whether the model has tensors and the files an import then writes. */
    struct FileSetRow
    {
        std::string name;                 /**< The row's name. */
        bool weighted;                    /**< Whether the model holds tensors. */
        std::vector<std::string> files;   /**< The file names written, in order. */
    };

    /** @brief Runs the file set of an import over a model with and without tensors. */
    class Medium_ImportedModelFiles : public ::testing::TestWithParam<FileSetRow>
    {
    };
} // namespace

/** @brief Tests that the files are written and named in the order model, weights, train. */
TEST_P(Medium_ImportedModelFiles, TheFilesWrittenAreNamedInOrder)
{
    const FileSetRow& row = GetParam();
    const ScratchDirectory scratch("sushiai_import_file_set");
    ImportedModel imported = make_imported();
    if (!row.weighted)
        imported.weights.clear();

    SA_ASSERT_OK_AND_ASSIGN(
        const IO::WrittenImport written,
        IO::write_imported_model(imported, scratch.path(), ExistingFiles::Refuse));

    EXPECT_EQ(file_names(written), row.files);
    EXPECT_EQ(std::filesystem::exists(scratch.path() / "weights.sushi"), row.weighted);
}

INSTANTIATE_TEST_SUITE_P(
    Rows, Medium_ImportedModelFiles,
    ::testing::Values(
        FileSetRow{"WithTensors", true, {"model.json", "weights.sushi", "train.json"}},
        FileSetRow{"WithoutTensors", false, {"model.json", "train.json"}}),
    [](const ::testing::TestParamInfo<FileSetRow>& info) { return info.param.name; });

/** @brief Tests that the model file reads back to the rows and the task it was given. */
TEST(Medium_ImportedModelWriter, TheModelFileReadsBack)
{
    const ScratchDirectory scratch("sushiai_import_model_file");
    const ImportedModel imported = make_imported();
    SA_ASSERT_OK_AND_ASSIGN(
        const IO::WrittenImport written,
        IO::write_imported_model(imported, scratch.path(), ExistingFiles::Refuse));

    const Config::ModelSpec read = require_value(Config::load_model(written.files.front()));
    EXPECT_EQ(read.input_shape, imported.model.input_shape);
    ASSERT_EQ(read.layers.size(), 1u);
    EXPECT_EQ(read.layers[0].type, "linear");
    EXPECT_EQ(require_value(require_value(read.task.at("type", "task")).as_string("type")),
              "stub_task");
}

namespace
{
    /** @brief Names the position of a tensor in the weight file. */
    struct TensorRow
    {
        std::string name;       /**< The row's name. */
        std::size_t index;      /**< The tensor's position in the model and in the file. */
    };

    /** @brief Runs the weight file check over each tensor of the model. */
    class Medium_ImportedModelWeightFile : public ::testing::TestWithParam<TensorRow>
    {
    };
} // namespace

/** @brief Tests that the weight file holds a tensor's record and bytes at its position. */
TEST_P(Medium_ImportedModelWeightFile, TheWeightFileHoldsTheTensor)
{
    const ScratchDirectory scratch("sushiai_import_weight_file");
    const ImportedModel imported = make_imported();
    SA_ASSERT_OK_AND_ASSIGN(
        const IO::WrittenImport written,
        IO::write_imported_model(imported, scratch.path(), ExistingFiles::Refuse));

    const std::vector<ReadTensor> read = read_weight_file(written.files[1]);

    ASSERT_EQ(read.size(), imported.weights.size());
    const ReadTensor& got = read[GetParam().index];
    const ImportedTensor& wanted = imported.weights[GetParam().index];
    EXPECT_EQ(got.record.name, wanted.record.name);
    EXPECT_EQ(got.record.dtype, wanted.record.dtype);
    EXPECT_EQ(got.record.shape, wanted.record.shape);
    EXPECT_EQ(got.payload, wanted.payload);
}

INSTANTIATE_TEST_SUITE_P(Rows, Medium_ImportedModelWeightFile,
                         ::testing::Values(TensorRow{"TheWeight", 0}, TensorRow{"TheBias", 1}),
                         [](const ::testing::TestParamInfo<TensorRow>& info)
                         { return info.param.name; });

/** @brief Tests that the composed training file is refused by its "data" key. */
TEST(Small_ImportedModelWriter, TheTrainingFileIsRefusedUntilItsDataIsFilled)
{
    SA_ASSERT_OK_AND_ASSIGN(const Config::Value training,
                            IO::compose_training_file(make_imported()));

    SA_EXPECT_FAILURE(Config::parse_train(training, "<test>", std::filesystem::path()),
                      StatusCode::InvalidDescription, "'data'", "must be an object", "null");
}

/** @brief Tests that the training file on disk is refused by its "data" key when it is read. */
TEST(Medium_ImportedModelWriter, TheTrainingFileOnDiskIsRefusedUntilItsDataIsFilled)
{
    const ScratchDirectory scratch("sushiai_import_training_file");
    SA_ASSERT_OK_AND_ASSIGN(
        const IO::WrittenImport written,
        IO::write_imported_model(make_imported(), scratch.path(), ExistingFiles::Refuse));

    SA_EXPECT_FAILURE(Config::load_train(written.files.back()), StatusCode::InvalidDescription,
                      "'data'", "must be an object", "null");
}

/** @brief Tests that an existing file is refused by name and left as it was. */
TEST(Medium_ImportedModelWriter, AnExistingFileIsRefusedAndLeftAlone)
{
    const ScratchDirectory scratch("sushiai_import_existing");
    std::filesystem::create_directories(scratch.path());
    {
        std::ofstream kept(scratch.path() / "train.json", std::ios::binary);
        kept << "kept";
    }

    SA_EXPECT_FAILURE(
        IO::write_imported_model(make_imported(), scratch.path(), ExistingFiles::Refuse),
        StatusCode::Incompatible, "train.json", "exists already");

    EXPECT_EQ(read_bytes(scratch.path() / "train.json"), "kept");
    EXPECT_FALSE(std::filesystem::exists(scratch.path() / "model.json"));
}

/** @brief Tests that Replace overwrites a file that is already in the directory. */
TEST(Medium_ImportedModelWriter, ReplaceOverwritesAnExistingFile)
{
    const ScratchDirectory scratch("sushiai_import_replace");
    std::filesystem::create_directories(scratch.path());
    {
        std::ofstream stale(scratch.path() / "train.json", std::ios::binary);
        stale << "stale";
    }

    SA_EXPECT_OK(
        IO::write_imported_model(make_imported(), scratch.path(), ExistingFiles::Replace));

    EXPECT_NE(read_bytes(scratch.path() / "train.json").find("sushiai_train"),
              std::string::npos);
}

/** @brief Tests that a tensor whose bytes differ from its record is refused by its name. */
TEST(Small_ImportedModelWriter, ATensorOfTheWrongSizeIsRefused)
{
    ImportedModel imported = make_imported();
    imported.weights[1].payload.pop_back();

    SA_EXPECT_FAILURE(IO::check_imported_tensors(imported.weights),
                      StatusCode::InvalidDescription, "'0.bias'", "7 bytes", "needs 8");
}

/** @brief Tests that a tensor name stated twice is refused by that name. */
TEST(Small_ImportedModelWriter, ATensorNamedTwiceIsRefused)
{
    ImportedModel imported = make_imported();
    imported.weights[1].record.name = "0.weight";

    SA_EXPECT_FAILURE(IO::check_imported_tensors(imported.weights),
                      StatusCode::InvalidDescription, "'0.weight'", "twice");
}

/** @brief Tests that a refused tensor stops the write before any file exists. */
TEST(Medium_ImportedModelWriter, ARefusedImportWritesNoFile)
{
    const ScratchDirectory scratch("sushiai_import_wrong_size");
    ImportedModel imported = make_imported();
    imported.weights[1].payload.pop_back();

    EXPECT_FALSE(
        IO::write_imported_model(imported, scratch.path(), ExistingFiles::Refuse).has_value());

    EXPECT_FALSE(std::filesystem::exists(scratch.path() / "model.json"));
}
