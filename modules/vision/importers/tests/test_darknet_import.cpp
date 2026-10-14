/**************************************************************************/
/* test_darknet_import.cpp                                                */
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
 * @file test_darknet_import.cpp
 * @brief Tests the Darknet importer: the task, the rows, the tensors and the recipe of a .cfg.
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
#include <SushiAI/core/result.hpp>
#include <SushiAI/core/status.hpp>
#include <SushiAI/io/imported_model.hpp>
#include <SushiAI/vision/detect/anchor/anchor_task.hpp>
#include <SushiAI/vision/detect/anchor/head_options.hpp>
#include <SushiAI/vision/importers/darknet_config.hpp>
#include <SushiAI/vision/importers/darknet_import.hpp>
#include <SushiAI/vision/importers/darknet_model.hpp>
#include <SushiAI/vision/importers/darknet_weights.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>

using namespace SushiAI;
using IO::ImportedModel;
using Vision::Detect::Anchor::AnchorTask;
using Vision::Importers::DarknetConfig;
using Vision::Importers::DarknetConvolution;
using Vision::Importers::DarknetConvolutionShape;
using Vision::Importers::DarknetModel;
using Vision::Importers::DarknetWeights;

namespace
{
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
TEST(Unit_DarknetImport, KeepsTheRowsAndTheOutputsOfTheTranslation)
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
TEST(Unit_DarknetImport, StatesTheHeadsAsAnAnchorDetectionTask)
{
    SA_ASSERT_OK_AND_ASSIGN(const Translated translated, translate_small());
    SA_ASSERT_OK_AND_ASSIGN(const ImportedModel imported,
                            Vision::Importers::import_darknet_model(
                                translated.config, translated.model, nullptr));

    EXPECT_EQ(imported.model.task.at("type", "task").as_string("type"), "anchor_detection");
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
TEST(Unit_DarknetImport, HoldsNoTensorWithoutWeights)
{
    SA_ASSERT_OK_AND_ASSIGN(const Translated translated, translate_small());
    SA_ASSERT_OK_AND_ASSIGN(const ImportedModel imported,
                            Vision::Importers::import_darknet_model(
                                translated.config, translated.model, nullptr));

    EXPECT_TRUE(imported.weights.empty());
}

/** @brief Tests that the tensors carry exactly the names the translation's targets state. */
TEST(Unit_DarknetImport, NamesEveryTensorAsItsTargetDoes)
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
TEST(Unit_DarknetImport, CarriesTheRecipeAndItsNotes)
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
TEST(Unit_DarknetImport, AnotherCountOfFilesIsRefused)
{
    const std::vector<std::filesystem::path> three = {"a.cfg", "a.weights", "b.weights"};

    SA_EXPECT_FAILURE(Vision::Importers::import_darknet_files({}),
                      StatusCode::InvalidDescription, "'darknet'", "0 files");
    SA_EXPECT_FAILURE(Vision::Importers::import_darknet_files(three),
                      StatusCode::InvalidDescription, ".cfg", "3 files");
}

/** @brief Tests that a .cfg that is not on disk is refused by its path. */
TEST(Unit_DarknetImport, AMissingCfgIsRefusedByName)
{
    const std::vector<std::filesystem::path> files = {"sushiai_absent_import.cfg"};

    SA_EXPECT_FAILURE(Vision::Importers::import_darknet_files(files),
                      StatusCode::FileNotFound, "sushiai_absent_import.cfg");
}
