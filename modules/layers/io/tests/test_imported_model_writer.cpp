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
 * @file test_imported_model_writer.cpp
 * @brief Tests the writer of an imported model: the training file it composes, the three
 *        files it writes and what it refuses before it writes any.
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
#include <SushiAI/core/common.hpp>
#include <SushiAI/core/result.hpp>
#include <SushiAI/core/shape.hpp>
#include <SushiAI/core/status.hpp>
#include <SushiAI/io/checkpoint_format.hpp>
#include <SushiAI/io/imported_model.hpp>
#include <SushiAI/io/imported_model_writer.hpp>
#include <common/model_text.hpp>
#include <common/test_status.hpp>
#include <common/test_throws.hpp>

using namespace SushiAI;
using IO::ExistingFiles;
using IO::ImportedModel;
using IO::ImportedTensor;

namespace
{
    /** @brief Removes a directory under the system's temporary folder, before and after. */
    class ScratchDirectory
    {
        public:
            /** @brief Removes whatever a previous run left under @p name. */
            explicit ScratchDirectory(const char* name)
                : path_(std::filesystem::temp_directory_path() / name)
            {
                std::error_code ignored;
                std::filesystem::remove_all(path_, ignored);
            }

            /** @brief Removes the directory and what it holds. */
            ~ScratchDirectory()
            {
                std::error_code ignored;
                std::filesystem::remove_all(path_, ignored);
            }

            ScratchDirectory(const ScratchDirectory&) = delete;
            ScratchDirectory& operator=(const ScratchDirectory&) = delete;
            ScratchDirectory(ScratchDirectory&&) = delete;
            ScratchDirectory& operator=(ScratchDirectory&&) = delete;

            /** @brief Returns the directory, which need not exist. */
            const std::filesystem::path& path() const noexcept { return path_; }

        private:
            /** @brief Holds the path of the directory. */
            std::filesystem::path path_;
    };

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

    /** @brief Returns the bytes of the file at @p path; empty when it cannot be opened. */
    std::string read_bytes(const std::filesystem::path& path)
    {
        std::ifstream in(path, std::ios::binary);
        return std::string(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
    }
} // namespace

/** @brief Tests the keys of the composed training file, in the order the writer states them. */
TEST(Unit_ImportedModelWriter, ComposesTheTrainingFileAroundTheRecipe)
{
    SA_ASSERT_OK_AND_ASSIGN(const Config::Value training,
                            IO::compose_training_file(make_imported()));

    EXPECT_EQ(training.keys(),
              (std::vector<std::string>{"sushiai_train", "model", "weights", "data",
                                        "checkpoint", "objective", "optimizer"}));
    EXPECT_EQ(training.at("sushiai_train", "file").as_int("version"), Config::K_TRAIN_VERSION);
    EXPECT_EQ(training.at("model", "file").as_string("model"), "model.json");
    EXPECT_EQ(training.at("weights", "file").as_string("weights"), "weights.sushi");
    EXPECT_EQ(training.at("checkpoint", "file").as_string("checkpoint"), "trained.sushi");
    EXPECT_EQ(training.at("optimizer", "file").at("type", "optimizer").as_string("type"), "sgd");
}

/** @brief Tests that the data block is null, since no foreign file names a dataset. */
TEST(Unit_ImportedModelWriter, LeavesTheDataBlockNull)
{
    SA_ASSERT_OK_AND_ASSIGN(const Config::Value training,
                            IO::compose_training_file(make_imported()));

    ASSERT_TRUE(training.has("data"));
    EXPECT_TRUE(training.at("data", "file").is_null());
}

/** @brief Tests that a model without tensors states no "weights" key. */
TEST(Unit_ImportedModelWriter, StatesNoWeightsKeyWithoutTensors)
{
    ImportedModel imported = make_imported();
    imported.weights.clear();

    SA_ASSERT_OK_AND_ASSIGN(const Config::Value training, IO::compose_training_file(imported));
    EXPECT_FALSE(training.has("weights"));
}

/** @brief Tests that a recipe stating a key of the writer is refused by that key. */
TEST(Unit_ImportedModelWriter, ARecipeThatStatesAWriterKeyIsRefused)
{
    ImportedModel imported = make_imported();
    imported.training = SushiAI::Test::attributes_from(R"({ "model": "other.json" })");

    SA_EXPECT_FAILURE(IO::compose_training_file(imported), StatusCode::InvalidDescription,
                      "'model'", "training recipe");
}

/** @brief Tests that a recipe that is no object is refused by its kind. */
TEST(Unit_ImportedModelWriter, ARecipeThatIsNoObjectIsRefused)
{
    ImportedModel imported = make_imported();
    imported.training = Config::Value::integer(3);

    SA_EXPECT_FAILURE(IO::compose_training_file(imported), StatusCode::InvalidDescription,
                      "must be an object");
}

/** @brief Tests that the three files are written and named in the order model, weights, train. */
TEST(Unit_ImportedModelWriter, WritesTheThreeFiles)
{
    const ScratchDirectory scratch("sushiai_import_three_files");
    SA_ASSERT_OK_AND_ASSIGN(
        const IO::WrittenImport written,
        IO::write_imported_model(make_imported(), scratch.path(), ExistingFiles::Refuse));

    ASSERT_EQ(written.files.size(), 3u);
    EXPECT_EQ(written.files[0], scratch.path() / "model.json");
    EXPECT_EQ(written.files[1], scratch.path() / "weights.sushi");
    EXPECT_EQ(written.files[2], scratch.path() / "train.json");
    for (const std::filesystem::path& file : written.files)
        EXPECT_TRUE(std::filesystem::is_regular_file(file)) << file.string();
}

/** @brief Tests that the model file reads back to the rows and the task it was given. */
TEST(Unit_ImportedModelWriter, TheModelFileReadsBack)
{
    const ScratchDirectory scratch("sushiai_import_model_file");
    const ImportedModel imported = make_imported();
    SA_ASSERT_OK_AND_ASSIGN(
        const IO::WrittenImport written,
        IO::write_imported_model(imported, scratch.path(), ExistingFiles::Refuse));

    const Config::ModelSpec read = Config::load_model(written.files.front());
    EXPECT_EQ(read.input_shape, imported.model.input_shape);
    ASSERT_EQ(read.layers.size(), 1u);
    EXPECT_EQ(read.layers[0].type, "linear");
    EXPECT_EQ(read.task.at("type", "task").as_string("type"), "stub_task");
}

/** @brief Tests that the weight file holds each tensor's record and bytes, in order. */
TEST(Unit_ImportedModelWriter, TheWeightFileHoldsEveryTensor)
{
    const ScratchDirectory scratch("sushiai_import_weight_file");
    const ImportedModel imported = make_imported();
    SA_ASSERT_OK_AND_ASSIGN(
        const IO::WrittenImport written,
        IO::write_imported_model(imported, scratch.path(), ExistingFiles::Refuse));

    std::ifstream in(written.files[1], std::ios::binary);
    ASSERT_TRUE(in.is_open());
    SA_ASSERT_OK_AND_ASSIGN(IO::CheckpointReader reader, IO::CheckpointReader::open(in));
    ASSERT_EQ(reader.count(), 2u);

    for (const ImportedTensor& wanted : imported.weights)
    {
        SA_ASSERT_OK_AND_ASSIGN(const IO::TensorRecord record, reader.next());
        EXPECT_EQ(record.name, wanted.record.name);
        EXPECT_EQ(record.dtype, wanted.record.dtype);
        EXPECT_EQ(record.shape, wanted.record.shape);

        std::vector<std::byte> payload(record.payload_bytes());
        SA_EXPECT_OK(reader.read_payload(record, payload.data()));
        EXPECT_EQ(payload, wanted.payload) << wanted.record.name;
    }
}

/** @brief Tests that `train` refuses the written training file by its "data" key. */
TEST(Unit_ImportedModelWriter, TheTrainingFileIsRefusedUntilItsDataIsFilled)
{
    const ScratchDirectory scratch("sushiai_import_training_file");
    SA_ASSERT_OK_AND_ASSIGN(
        const IO::WrittenImport written,
        IO::write_imported_model(make_imported(), scratch.path(), ExistingFiles::Refuse));

    const std::filesystem::path training = written.files.back();
    SA_EXPECT_ERROR(Config::load_train(training), "'data'", "must be an object", "null");
}

/** @brief Tests that a model without tensors writes two files and no weight file. */
TEST(Unit_ImportedModelWriter, WritesNoWeightFileWithoutTensors)
{
    const ScratchDirectory scratch("sushiai_import_no_weights");
    ImportedModel imported = make_imported();
    imported.weights.clear();

    SA_ASSERT_OK_AND_ASSIGN(
        const IO::WrittenImport written,
        IO::write_imported_model(imported, scratch.path(), ExistingFiles::Refuse));

    ASSERT_EQ(written.files.size(), 2u);
    EXPECT_FALSE(std::filesystem::exists(scratch.path() / "weights.sushi"));
}

/** @brief Tests that an existing file is refused by name and left as it was. */
TEST(Unit_ImportedModelWriter, AnExistingFileIsRefusedAndLeftAlone)
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
TEST(Unit_ImportedModelWriter, ReplaceOverwritesAnExistingFile)
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

/** @brief Tests that a tensor whose bytes differ from its record stops the write of any file. */
TEST(Unit_ImportedModelWriter, ATensorOfTheWrongSizeWritesNothing)
{
    const ScratchDirectory scratch("sushiai_import_wrong_size");
    ImportedModel imported = make_imported();
    imported.weights[1].payload.pop_back();

    SA_EXPECT_FAILURE(
        IO::write_imported_model(imported, scratch.path(), ExistingFiles::Refuse),
        StatusCode::InvalidDescription, "'0.bias'", "7 bytes", "needs 8");

    EXPECT_FALSE(std::filesystem::exists(scratch.path() / "model.json"));
}

/** @brief Tests that a tensor name stated twice is refused by that name. */
TEST(Unit_ImportedModelWriter, ATensorNamedTwiceIsRefused)
{
    const ScratchDirectory scratch("sushiai_import_named_twice");
    ImportedModel imported = make_imported();
    imported.weights[1].record.name = "0.weight";

    SA_EXPECT_FAILURE(
        IO::write_imported_model(imported, scratch.path(), ExistingFiles::Refuse),
        StatusCode::InvalidDescription, "'0.weight'", "twice");
}
