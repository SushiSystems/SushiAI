/**************************************************************************/
/* test_ultralytics_files.cpp                                             */
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
 * @file test_ultralytics_files.cpp
 * @brief Tests that the yolov8n and yolo11n model files build, and what their task sections state.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <utility>
#include <vector>
#include <gtest/gtest.h>
#include <SushiAI/config/model_config.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/graph/builder.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/nn/buffer.hpp>
#include <SushiAI/nn/model_graph.hpp>
#include <SushiAI/nn/model_outputs.hpp>
#include <SushiAI/nn/module.hpp>
#include <SushiAI/nn/node_registry.hpp>
#include <SushiAI/nn/layer_registry.hpp>
#include <SushiAI/nn/trace.hpp>
#include <SushiAI/shape/shape.hpp>
#include <SushiAI/train/objective_context.hpp>
#include <SushiAI/vision/detect/free/free_task.hpp>
#include <SushiAI/vision/registration/registration.hpp>
#include <SushiAI/vision/training/free_model_facts.hpp>
#include "registration_test_support.hpp"

using namespace SushiAI;
using NN::ModelGraph;
using Vision::Detect::Free::FreeTask;
using Vision::Training::FreeModelFacts;

namespace
{
    using Test::composed;
    using Test::count_parameters;
    using Test::trace_outputs;

    /** @brief Names one model file of the examples and what it is checked against. */
    struct UltralyticsFile
    {
        /** @brief Holds the folder of the file under applications/examples. */
        const char* folder;

        /** @brief Holds the row count of the file. */
        std::size_t rows;

        /**
         * @brief Holds the parameter count of the Ultralytics summary line at 80 classes; a
         *        figure from memory, which this file's own count of the rows matches once the
         *        ramp of the distribution expectation is taken out.
         */
        std::int64_t reference_parameters;
    };

    /** @brief Lists the two files whose parameter counts and levels the tests pin. */
    const UltralyticsFile K_FILES[] = {
        {"yolov8n", 23, 3157200},
        {"yolo11n", 24, 2624080},
    };

    /** @brief Lists the folders of every anchor-free file that states a task section. */
    const char* const K_TASK_FOLDERS[] = {"yolov8n", "yolo11n", "yolo26"};

    /** @brief Holds the distribution bins per box side the files state. */
    constexpr std::int64_t K_REG_MAX = 16;

    /**
     * @brief Holds the elements of the fixed ramp 0..reg_max-1 that Ultralytics counts as a
     *        parameter and this framework keeps as no state record.
     */
    constexpr std::int64_t K_RAMP_ELEMENTS = K_REG_MAX;

    /** @brief Holds the class count of the files. */
    constexpr std::int64_t K_CLASSES = 80;

    /** @brief Holds the side of the network input in pixels. */
    constexpr std::int64_t K_INPUT_SIDE = 640;

    /** @brief Holds the strides of the three levels, finest first. */
    constexpr std::int64_t K_STRIDES[] = {8, 16, 32};

    /** @brief Holds the number of levels the head reads. */
    constexpr std::size_t K_LEVELS = 3;

    /** @brief Holds the bound of a threshold read back; each is the same float32 constant. */
    constexpr float K_THRESHOLD_TOLERANCE = 0.0f;

    /** @brief Returns the path of the model file in @p folder. */
    std::filesystem::path model_path(const char* folder)
    {
        return std::filesystem::path(SA_REPOSITORY_ROOT) / "applications" / "examples" / folder
               / "model.json";
    }

    /** @brief Returns the shape @p outputs publishes under @p name, or an empty shape. */
    Shape shape_of(const std::vector<Train::OutputShape>& outputs, const std::string& name)
    {
        for (const Train::OutputShape& output : outputs)
            if (output.name == name)
                return output.shape;
        return Shape{};
    }
} // namespace

/** @brief Tests that each file holds its rows, the input of 640 pixels and one output node. */
TEST(Medium_UltralyticsFiles, HoldTheirRowsAndTheirInput)
{
    for (const UltralyticsFile& file : K_FILES)
    {
        SCOPED_TRACE(file.folder);
        const Config::ModelSpec spec = Config::load_model(model_path(file.folder));

        EXPECT_EQ(spec.layers.size(), file.rows);
        EXPECT_EQ(spec.input_shape, (Shape{K_INPUT_SIDE, K_INPUT_SIDE, 3}));
        EXPECT_EQ(spec.outputs.size(), 1u);
    }
}

/**
 * @brief Tests that the parameter count of each built model is the Ultralytics summary figure
 *        less the elements of the ramp, which the framework holds as no record.
 */
TEST(Medium_UltralyticsFiles, BuildToTheParameterCountOfTheReference)
{
    SA_ASSERT_OK_AND_ASSIGN(const NN::NodeRegistry registry, composed());

    for (const UltralyticsFile& file : K_FILES)
    {
        SCOPED_TRACE(file.folder);
        Config::ModelSpec spec = Config::load_model(model_path(file.folder));
        SA_ASSERT_OK_AND_ASSIGN(ModelGraph model, NN::build_model(registry, spec));
        (void)trace_outputs(model, spec.input_shape);

        EXPECT_EQ(count_parameters(model), file.reference_parameters - K_RAMP_ELEMENTS);
    }
}

/**
 * @brief Tests that each model publishes the boxes and the scores of three levels at strides 8,
 *        16 and 32, 4 * reg_max and 80 channels wide, and publishes no other port.
 */
TEST(Medium_UltralyticsFiles, PublishTheOneToManyPortsAtTheLevelStrides)
{
    SA_ASSERT_OK_AND_ASSIGN(const NN::NodeRegistry registry, composed());

    for (const UltralyticsFile& file : K_FILES)
    {
        SCOPED_TRACE(file.folder);
        Config::ModelSpec spec = Config::load_model(model_path(file.folder));
        SA_ASSERT_OK_AND_ASSIGN(ModelGraph model, NN::build_model(registry, spec));

        const std::vector<Train::OutputShape> outputs = trace_outputs(model, spec.input_shape);

        EXPECT_EQ(outputs.size(), 2 * K_LEVELS);
        for (std::size_t level = 0; level < K_LEVELS; ++level)
        {
            SCOPED_TRACE(level);
            const std::int64_t side = K_INPUT_SIDE / K_STRIDES[level];
            const std::string prefix = "one2many.level" + std::to_string(level);

            EXPECT_EQ(shape_of(outputs, prefix + ".boxes"), (Shape{1, side, side, 4 * K_REG_MAX}));
            EXPECT_EQ(shape_of(outputs, prefix + ".scores"), (Shape{1, side, side, K_CLASSES}));
        }
    }
}

/** @brief Tests that the traced outputs of each model give strides 8, 16, 32 and reg_max 16. */
TEST(Medium_UltralyticsFiles, GiveTheFactsOfAnAnchorFreeModel)
{
    SA_ASSERT_OK_AND_ASSIGN(const NN::NodeRegistry registry, composed());

    for (const UltralyticsFile& file : K_FILES)
    {
        SCOPED_TRACE(file.folder);
        Config::ModelSpec spec = Config::load_model(model_path(file.folder));
        SA_ASSERT_OK_AND_ASSIGN(ModelGraph model, NN::build_model(registry, spec));
        const std::vector<Train::OutputShape> outputs = trace_outputs(model, spec.input_shape);

        SA_ASSERT_OK_AND_ASSIGN(
            const FreeModelFacts facts,
            Vision::Training::derive_free_model_facts(spec.input_shape, outputs));

        EXPECT_EQ(facts.reg_max, K_REG_MAX);
        EXPECT_EQ(facts.classes, K_CLASSES);
        EXPECT_FALSE(facts.has_one_to_one);
        ASSERT_EQ(facts.levels.size(), K_LEVELS);
        for (std::size_t level = 0; level < K_LEVELS; ++level)
        {
            SCOPED_TRACE(level);
            EXPECT_EQ(facts.levels[level].stride, K_STRIDES[level]);
            EXPECT_EQ(facts.levels[level].height, K_INPUT_SIDE / K_STRIDES[level]);
            EXPECT_EQ(facts.levels[level].width, K_INPUT_SIDE / K_STRIDES[level]);
        }
    }
}

/**
 * @brief Tests that the task section of each anchor-free file reads back as the FreeTask
 *        defaults: score 0.25, intersection over union suppression at 0.7.
 */
TEST(Medium_UltralyticsFiles, TheTaskSectionReadsBackAsTheDefaults)
{
    const FreeTask defaults;

    for (const char* const folder : K_TASK_FOLDERS)
    {
        SCOPED_TRACE(folder);
        const Config::ModelSpec spec = Config::load_model(model_path(folder));

        SA_ASSERT_OK_AND_ASSIGN(const FreeTask task,
                                Vision::Detect::Free::read_free_task(spec.task));

        EXPECT_NEAR(task.score_threshold, defaults.score_threshold, K_THRESHOLD_TOLERANCE);
        EXPECT_NEAR(task.suppression_threshold, defaults.suppression_threshold,
                    K_THRESHOLD_TOLERANCE);
        EXPECT_EQ(task.criterion, defaults.criterion);
    }
}
