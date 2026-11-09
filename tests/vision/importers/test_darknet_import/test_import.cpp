/**************************************************************************/
/* test_import.cpp                                                        */
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
 * @file test_import.cpp
 * @brief Tests that each Darknet parity model imports with the rows, the task and the weights it
 *        declares.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <set>
#include <string>
#include <string_view>
#include <vector>
#include <gtest/gtest.h>
#include <SushiAI/config/value.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/io/imported_model.hpp>
#include <SushiAI/scratch_files/scratch_directory.hpp>
#include <SushiAI/vision/darknet_forward_cases/darknet_forward.hpp>
#include <SushiAI/vision/darknet_import_cases/darknet_import.hpp>
#include <SushiAI/vision/detect/anchor/anchor_task.hpp>
#include <SushiAI/vision/detect/anchor/head_options.hpp>
#include <SushiAI/vision/importers/darknet_config.hpp>
#include <SushiAI/vision/importers/darknet_import.hpp>
#include <SushiAI/vision/importers/darknet_model.hpp>
#include <SushiAI/vision/importers/darknet_weights.hpp>

using namespace SushiAI;
using IO::ImportedModel;
using Vision::Detect::Anchor::AnchorTask;
using Vision::Importers::DarknetConfig;
using Vision::Importers::DarknetConvolution;
using Vision::Importers::DarknetConvolutionShape;
using Vision::Importers::DarknetModel;
using Vision::Importers::DarknetWeights;
namespace SaTest = SushiAI::Test;

namespace
{
    /** @brief Returns the string under @p key of @p value; a refusal fails the test run. */
    const std::string& string_of(const Config::Value& value, std::string_view key)
    {
        return SaTest::require_value(SaTest::require_value(value.at(key, key)).as_string(key));
    }

    /** @brief Holds a 32 by 32 network: a normalised convolution, a plain one and one head. */
    constexpr std::string_view K_SMALL_CFG =
        "[net]\nbatch=8\nsubdivisions=2\nwidth=32\nheight=32\nchannels=3\nmomentum=0.9\n"
        "decay=0.0005\nlearning_rate=0.001\nburn_in=100\nmax_batches=2000\npolicy=steps\n"
        "steps=1600,1800\nscales=.1,.1\n"
        "[convolutional]\nbatch_normalize=1\nfilters=4\nsize=3\nstride=1\npad=1\n"
        "activation=leaky\n"
        "[convolutional]\nfilters=18\nsize=1\nstride=1\npad=1\nactivation=linear\n"
        "[yolo]\nmask=0,1,2\nanchors=4,8, 8,16, 16,24\nclasses=1\nnum=3\njitter=.3\n"
        "ignore_thresh=.7\ntruth_thresh=1\nrandom=1\n";

    /** @brief Holds a parsed .cfg and its translation, which an import reads together. */
    struct Translated
    {
        DarknetConfig config; /**< The sections of the text. */
        DarknetModel model;   /**< The translation, every batch norm in its own row. */
    };

    /** @brief Parses and translates K_SMALL_CFG. */
    Result<Translated> translate_small()
    {
        Translated translated;
        SA_TRY_ASSIGN(translated.config,
                      Vision::Importers::parse_darknet_config(K_SMALL_CFG, "small.cfg"));
        SA_TRY_ASSIGN(translated.model,
                      Vision::Importers::translate_darknet_model(
                          translated.config, Vision::Importers::DarknetTranslation{}));
        return translated;
    }

    /** @brief Returns weights of the lengths @p shapes state, every float 1. */
    DarknetWeights make_weights(const std::vector<DarknetConvolutionShape>& shapes)
    {
        DarknetWeights weights;
        for (const DarknetConvolutionShape& shape : shapes)
        {
            const auto filters = static_cast<std::size_t>(shape.filters);
            DarknetConvolution convolution;
            convolution.biases.assign(filters, 1.0f);
            convolution.weights.assign(
                filters * static_cast<std::size_t>(shape.channels * shape.size * shape.size),
                1.0f);
            if (shape.normalised)
            {
                convolution.scales.assign(filters, 1.0f);
                convolution.rolling_mean.assign(filters, 1.0f);
                convolution.rolling_variance.assign(filters, 1.0f);
            }
            weights.convolutions.push_back(convolution);
        }

        return weights;
    }

    /** @brief Returns the type of each row of a model, in order. */
    std::vector<decltype(Config::LayerSpec::type)> types_of(const Config::ModelSpec& model)
    {
        std::vector<decltype(Config::LayerSpec::type)> types;
        for (const Config::LayerSpec& layer : model.layers)
            types.push_back(layer.type);

        return types;
    }

    /** @brief Returns the non-empty tensor names the targets of a translation state. */
    std::set<std::string> target_names_of(const DarknetModel& model)
    {
        std::set<std::string> names;
        for (const Vision::Importers::DarknetWeightTarget& target : model.targets)
        {
            for (const std::string* name :
                 {&target.weight, &target.bias, &target.scale, &target.mean, &target.variance})
            {
                if (!name->empty())
                    names.insert(*name);
            }
        }

        return names;
    }

    /** @brief Returns the names of the tensors of an import. */
    std::set<std::string> tensor_names_of(const ImportedModel& imported)
    {
        std::set<std::string> names;
        for (const IO::ImportedTensor& tensor : imported.weights)
            names.insert(tensor.record.name);

        return names;
    }

    /** @brief Imports the translation of K_SMALL_CFG without weights. */
    Result<ImportedModel> import_small_without_weights(const Translated& translated)
    {
        return Vision::Importers::import_darknet_model(translated.config, translated.model,
                                                       nullptr);
    }

    /** @brief Holds the files of one call of import_darknet_files the importer refuses. */
    struct FileCountRefusal
    {
        /** @brief Names the row. */
        const char* name;

        /** @brief Holds the paths of the call. */
        std::vector<std::filesystem::path> files;

        /** @brief Holds the text the refusal opens its subject with. */
        const char* subject;

        /** @brief Holds the count the refusal states. */
        const char* count;
    };

    /** @brief Holds one refused call of import_darknet_files per row. */
    class Small_DarknetImportFileCount : public ::testing::TestWithParam<FileCountRefusal>
    {
    };
} // namespace

/** @brief Tests that the imported rows are those of the translation. */
TEST(Small_DarknetImport, KeepsTheRowsOfTheTranslation)
{
    SA_ASSERT_OK_AND_ASSIGN(const Translated translated, translate_small());
    SA_ASSERT_OK_AND_ASSIGN(const ImportedModel imported, import_small_without_weights(translated));

    EXPECT_EQ(imported.model.input_shape, translated.model.spec.input_shape);
    EXPECT_EQ(types_of(imported.model), types_of(translated.model.spec));
}

/** @brief Tests that the imported outputs are those of the translation. */
TEST(Small_DarknetImport, KeepsTheOutputsOfTheTranslation)
{
    SA_ASSERT_OK_AND_ASSIGN(const Translated translated, translate_small());
    SA_ASSERT_OK_AND_ASSIGN(const ImportedModel imported, import_small_without_weights(translated));

    const Config::ModelSpec& wanted = translated.model.spec;
    ASSERT_EQ(imported.model.outputs.size(), 1u);
    EXPECT_EQ(imported.model.outputs[0].name, "head.0");
    EXPECT_EQ(imported.model.outputs[0].node, wanted.outputs[0].node);
}

/** @brief Tests that the task section reads back as the classes, anchors and head of the .cfg. */
TEST(Small_DarknetImport, StatesTheHeadsAsAnAnchorDetectionTask)
{
    SA_ASSERT_OK_AND_ASSIGN(const Translated translated, translate_small());
    SA_ASSERT_OK_AND_ASSIGN(const ImportedModel imported, import_small_without_weights(translated));

    EXPECT_EQ(string_of(imported.model.task, "type"), "anchor_detection");
    SA_ASSERT_OK_AND_ASSIGN(const AnchorTask task,
                            Vision::Detect::Anchor::read_anchor_task(imported.model.task));

    EXPECT_EQ(task.num_classes, 1);
    ASSERT_EQ(task.heads.size(), 1u);
    EXPECT_EQ(task.heads[0].output, "head.0");
    EXPECT_EQ(task.heads[0].options.coding, Vision::Detect::Anchor::HeadCoding::Logit);
    EXPECT_EQ(task.layout.head_anchors,
              (std::vector<std::vector<std::int64_t>>{{0, 1, 2}}));

    // 4 / 32 and 24 / 32 are exact in float.
    ASSERT_EQ(task.layout.anchors.size(), 3u);
    EXPECT_EQ(task.layout.anchors[0].width, 0.125f);
    EXPECT_EQ(task.layout.anchors[2].height, 0.75f);
}

/** @brief Tests that an import without weights holds no tensor. */
TEST(Small_DarknetImport, HoldsNoTensorWithoutWeights)
{
    SA_ASSERT_OK_AND_ASSIGN(const Translated translated, translate_small());
    SA_ASSERT_OK_AND_ASSIGN(const ImportedModel imported, import_small_without_weights(translated));

    EXPECT_TRUE(imported.weights.empty());
}

/** @brief Tests that the tensors carry exactly the names the translation's targets state. */
TEST(Small_DarknetImport, NamesEveryTensorAsItsTargetDoes)
{
    SA_ASSERT_OK_AND_ASSIGN(const Translated translated, translate_small());
    const DarknetWeights weights = make_weights(translated.model.convolutions);
    SA_ASSERT_OK_AND_ASSIGN(const ImportedModel imported,
                            Vision::Importers::import_darknet_model(
                                translated.config, translated.model, &weights));

    EXPECT_EQ(tensor_names_of(imported), target_names_of(translated.model));
    EXPECT_EQ(imported.weights.size(), 7u);
}

/** @brief Tests that the recipe is the one the .cfg's training settings state. */
TEST(Small_DarknetImport, CarriesTheRecipe)
{
    SA_ASSERT_OK_AND_ASSIGN(const Translated translated, translate_small());
    SA_ASSERT_OK_AND_ASSIGN(const ImportedModel imported, import_small_without_weights(translated));

    EXPECT_EQ(imported.training.keys(),
              (std::vector<std::string>{"objective", "optimizer", "schedule"}));
}

/** @brief Tests that the import carries the notes of the .cfg's training settings. */
TEST(Small_DarknetImport, CarriesTheNotes)
{
    SA_ASSERT_OK_AND_ASSIGN(const Translated translated, translate_small());
    SA_ASSERT_OK_AND_ASSIGN(const ImportedModel imported, import_small_without_weights(translated));

    EXPECT_EQ(imported.notes.size(), 5u);
}

/** @brief Tests that the file importer refuses a call with no file and one with three. */
TEST_P(Small_DarknetImportFileCount, IsRefused)
{
    const FileCountRefusal& row = GetParam();

    SA_EXPECT_FAILURE(Vision::Importers::import_darknet_files(row.files),
                      StatusCode::InvalidDescription, row.subject, row.count);
}

INSTANTIATE_TEST_SUITE_P(
    Rows, Small_DarknetImportFileCount,
    ::testing::Values(FileCountRefusal{"NoFile", {}, "'darknet'", "0 files"},
                      FileCountRefusal{"ThreeFiles", {"a.cfg", "a.weights", "b.weights"}, ".cfg",
                                       "3 files"}),
    [](const ::testing::TestParamInfo<FileCountRefusal>& info)
    {
        return std::string(info.param.name);
    });

/** @brief Tests that a .cfg that is not on disk is refused by its path. */
TEST(Medium_DarknetImport, AMissingCfgIsRefusedByName)
{
    const std::vector<std::filesystem::path> files = {"sushiai_absent_import.cfg"};

    SA_EXPECT_FAILURE(Vision::Importers::import_darknet_files(files),
                      StatusCode::FileNotFound, "sushiai_absent_import.cfg");
}

namespace
{
    /** @brief Runs once per parity model; a case skips when the model's files are missing. */
    class Medium_DarknetImportEndToEnd : public SaTest::DarknetImportTest {};

    /** @brief Lists the stems of the parity table, one case per forward suite. */
    std::vector<std::string_view> list_stems()
    {
        std::vector<std::string_view> stems;
        for (const SaTest::DarknetParityModel& model : SaTest::list_darknet_parity_models())
            stems.push_back(model.stem);

        return stems;
    }
} // namespace

/** @brief Tests that the imported model file, without its task, is the text of the translation. */
TEST_P(Medium_DarknetImportEndToEnd, TheImportedRowsAreThoseOfTheTranslation)
{
    SaTest::check_imported_rows(run);
}

/** @brief Tests that the task reads back as the anchors, the heads and the criterion. */
TEST_P(Medium_DarknetImportEndToEnd, TheImportedTaskIsThatOfTheTranslation)
{
    SaTest::check_imported_task(run);
}

/**
 * @brief Tests that a state filled from the imported tensors equals, byte for byte, the state
 *        store_darknet_weights fills.
 */
TEST_P(Medium_DarknetImportEndToEnd, TheImportedWeightsAreThoseOfTheDirectStore)
{
    SaTest::check_imported_weights(*engine, run);
}

/** @brief Tests that the imported tensors supply every entry of the state exactly once. */
TEST_P(Medium_DarknetImportEndToEnd, TheImportedTensorsSupplyEveryEntryOnce)
{
    SaTest::check_imported_entries(*engine, run);
}

/** @brief Tests that the files the import writes read back as the import. */
TEST_P(Medium_DarknetImportEndToEnd, TheWrittenFilesReadBackAsTheImport)
{
    const SaTest::ScratchDirectory directory("sushiai_test_darknet_written_files");
    SaTest::check_written_files(run, directory.path());
}

/** @brief Tests that the written weights load into a state equal to the direct store's. */
TEST_P(Medium_DarknetImportEndToEnd, TheWrittenWeightsLoadAsTheDirectStore)
{
    const SaTest::ScratchDirectory directory("sushiai_test_darknet_written_weights");
    SaTest::check_written_weights(*engine, run, directory.path());
}

INSTANTIATE_TEST_SUITE_P(ParityModels, Medium_DarknetImportEndToEnd,
                         ::testing::ValuesIn(list_stems()));
