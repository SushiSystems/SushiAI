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

    /** @brief Holds the tolerance of an anchor read back from the task section, absolute. */
    constexpr float K_FLOAT_TOLERANCE = 1e-6f;

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
} // namespace

/** @brief Tests that the imported rows and outputs are those of the translation. */
TEST(Medium_DarknetImport, KeepsTheRowsAndTheOutputsOfTheTranslation)
{
    SA_ASSERT_OK_AND_ASSIGN(const Translated translated, translate_small());
    SA_ASSERT_OK_AND_ASSIGN(const ImportedModel imported,
                            Vision::Importers::import_darknet_model(
                                translated.config, translated.model, nullptr));

    const Config::ModelSpec& wanted = translated.model.spec;
    EXPECT_EQ(imported.model.input_shape, wanted.input_shape);
    ASSERT_EQ(imported.model.layers.size(), wanted.layers.size());
    for (std::size_t row = 0; row < wanted.layers.size(); ++row)
        EXPECT_EQ(imported.model.layers[row].type, wanted.layers[row].type) << "row " << row;

    ASSERT_EQ(imported.model.outputs.size(), 1u);
    EXPECT_EQ(imported.model.outputs[0].name, "head.0");
    EXPECT_EQ(imported.model.outputs[0].node, wanted.outputs[0].node);
}

/** @brief Tests that the task section reads back as the classes, anchors and head of the .cfg. */
TEST(Medium_DarknetImport, StatesTheHeadsAsAnAnchorDetectionTask)
{
    SA_ASSERT_OK_AND_ASSIGN(const Translated translated, translate_small());
    SA_ASSERT_OK_AND_ASSIGN(const ImportedModel imported,
                            Vision::Importers::import_darknet_model(
                                translated.config, translated.model, nullptr));

    EXPECT_EQ(string_of(imported.model.task, "type"),
              "anchor_detection");
    SA_ASSERT_OK_AND_ASSIGN(const AnchorTask task,
                            Vision::Detect::Anchor::read_anchor_task(imported.model.task));

    EXPECT_EQ(task.num_classes, 1);
    ASSERT_EQ(task.heads.size(), 1u);
    EXPECT_EQ(task.heads[0].output, "head.0");
    EXPECT_EQ(task.heads[0].options.coding, Vision::Detect::Anchor::HeadCoding::Logit);
    EXPECT_EQ(task.layout.head_anchors,
              (std::vector<std::vector<std::int64_t>>{{0, 1, 2}}));
    ASSERT_EQ(task.layout.anchors.size(), 3u);
    EXPECT_NEAR(task.layout.anchors[0].width, 4.0f / 32.0f, K_FLOAT_TOLERANCE);
    EXPECT_NEAR(task.layout.anchors[2].height, 24.0f / 32.0f, K_FLOAT_TOLERANCE);
}

/** @brief Tests that an import without weights holds no tensor. */
TEST(Medium_DarknetImport, HoldsNoTensorWithoutWeights)
{
    SA_ASSERT_OK_AND_ASSIGN(const Translated translated, translate_small());
    SA_ASSERT_OK_AND_ASSIGN(const ImportedModel imported,
                            Vision::Importers::import_darknet_model(
                                translated.config, translated.model, nullptr));

    EXPECT_TRUE(imported.weights.empty());
}

/** @brief Tests that the tensors carry exactly the names the translation's targets state. */
TEST(Medium_DarknetImport, NamesEveryTensorAsItsTargetDoes)
{
    SA_ASSERT_OK_AND_ASSIGN(const Translated translated, translate_small());
    const DarknetWeights weights = make_weights(translated.model.convolutions);
    SA_ASSERT_OK_AND_ASSIGN(const ImportedModel imported,
                            Vision::Importers::import_darknet_model(
                                translated.config, translated.model, &weights));

    std::set<std::string> wanted;
    for (const Vision::Importers::DarknetWeightTarget& target : translated.model.targets)
    {
        for (const std::string* name :
             {&target.weight, &target.bias, &target.scale, &target.mean, &target.variance})
        {
            if (!name->empty())
                wanted.insert(*name);
        }
    }

    std::set<std::string> found;
    for (const IO::ImportedTensor& tensor : imported.weights)
        found.insert(tensor.record.name);

    EXPECT_EQ(found, wanted);
    EXPECT_EQ(imported.weights.size(), 7u);
}

/** @brief Tests that the recipe and its notes are those of the .cfg's training settings. */
TEST(Medium_DarknetImport, CarriesTheRecipeAndItsNotes)
{
    SA_ASSERT_OK_AND_ASSIGN(const Translated translated, translate_small());
    SA_ASSERT_OK_AND_ASSIGN(const ImportedModel imported,
                            Vision::Importers::import_darknet_model(
                                translated.config, translated.model, nullptr));

    EXPECT_EQ(imported.training.keys(),
              (std::vector<std::string>{"objective", "optimizer", "schedule"}));
    EXPECT_EQ(imported.notes.size(), 5u);
}

/** @brief Tests that the file importer refuses a call with no file and one with three. */
TEST(Medium_DarknetImport, AnotherCountOfFilesIsRefused)
{
    const std::vector<std::filesystem::path> three = {"a.cfg", "a.weights", "b.weights"};

    SA_EXPECT_FAILURE(Vision::Importers::import_darknet_files({}),
                      StatusCode::InvalidDescription, "'darknet'", "0 files");
    SA_EXPECT_FAILURE(Vision::Importers::import_darknet_files(three),
                      StatusCode::InvalidDescription, ".cfg", "3 files");
}

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

/**
 * @brief Tests that the imported model file, without its task, is the text of the translated
 *        model, and that the task reads back as the anchors, the heads and the criterion.
 */
TEST_P(Medium_DarknetImportEndToEnd, TheImportedRowsAndTaskAreThoseOfTheTranslation)
{
    SaTest::check_imported_rows(run);
}

/**
 * @brief Tests that a state filled from the imported tensors equals, byte for byte, the state
 *        store_darknet_weights fills, and that the tensors supply every entry exactly once.
 */
TEST_P(Medium_DarknetImportEndToEnd, TheImportedWeightsAreThoseOfTheDirectStore)
{
    SaTest::check_imported_weights(*engine, run);
}

/**
 * @brief Tests that the three files the import writes read back as the import, and that the
 *        written weights load into a state equal to the direct store's.
 */
TEST_P(Medium_DarknetImportEndToEnd, TheWrittenFilesLoadAsTheImport)
{
    const SaTest::ScratchDirectory directory("sushiai_test_darknet_written_import");
    SaTest::check_written_import(*engine, run, directory.path());
}

INSTANTIATE_TEST_SUITE_P(ParityModels, Medium_DarknetImportEndToEnd,
                         ::testing::ValuesIn(list_stems()));
