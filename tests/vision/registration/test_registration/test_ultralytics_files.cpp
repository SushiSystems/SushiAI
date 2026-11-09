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
#include <support/test_names.hpp>

using namespace SushiAI;
using NN::ModelGraph;
using Vision::Detect::Free::FreeTask;
using Vision::Training::FreeModelFacts;

namespace
{
    using SushiAI::Test::require_value;
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
         * @brief Holds the parameter count in the summary line Ultralytics prints when it loads
         *        yolov8n.yaml (3157200 parameters) and yolo11n.yaml (2624080 parameters) at 80
         *        classes. The figure is the published one, not a count made here; the ramp of
         *        the distribution expectation is taken out of it before the comparison.
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

    /** @brief Holds the score threshold the task section of each file reads back as. */
    constexpr float K_TASK_SCORE_THRESHOLD = 0.25f;

    /** @brief Holds the suppression threshold the task section of each file reads back as. */
    constexpr float K_TASK_SUPPRESSION_THRESHOLD = 0.7f;

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

namespace
{
    /** @brief Gives each file of K_FILES its row, so a test runs once per file. */
    class Medium_UltralyticsFiles : public ::testing::TestWithParam<UltralyticsFile> {};

    /** @brief Gives each folder of K_TASK_FOLDERS to a test of the task section. */
    class Medium_UltralyticsTaskSection : public ::testing::TestWithParam<const char*> {};

    /** @brief Names a file in the name of its test. */
    std::string file_name(const ::testing::TestParamInfo<UltralyticsFile>& info)
    {
        return SushiAI::Test::parameter_name(info.param.folder);
    }

    /** @brief Names a folder in the name of its test. */
    std::string folder_name(const ::testing::TestParamInfo<const char*>& info)
    {
        return SushiAI::Test::parameter_name(info.param);
    }

    /** @brief Loads the model file of @p file. */
    Config::ModelSpec load_spec(const UltralyticsFile& file)
    {
        return require_value(Config::load_model(model_path(file.folder)));
    }

    /**
     * @brief Builds @p spec over the composed registry and lists the outputs it traces to.
     * @param spec The model file, which must build.
     */
    std::vector<Train::OutputShape> traced_outputs(const Config::ModelSpec& spec)
    {
        const NN::NodeRegistry registry = require_value(composed());
        ModelGraph model = require_value(NN::build_model(registry, spec));

        return trace_outputs(model, spec.input_shape);
    }

    /** @brief Derives the facts of the head of @p spec from the outputs it traces to. */
    FreeModelFacts head_facts(const Config::ModelSpec& spec)
    {
        return require_value(
            Vision::Training::derive_free_model_facts(spec.input_shape, traced_outputs(spec)));
    }
} // namespace

/** @brief Tests that the file holds its rows. */
TEST_P(Medium_UltralyticsFiles, HoldsItsRows)
{
    const Config::ModelSpec spec = load_spec(GetParam());

    EXPECT_EQ(spec.layers.size(), GetParam().rows);
}

/** @brief Tests that the file takes a 640 by 640 by 3 input. */
TEST_P(Medium_UltralyticsFiles, TakesA640By640By3Input)
{
    const Config::ModelSpec spec = load_spec(GetParam());

    EXPECT_EQ(spec.input_shape, (Shape{K_INPUT_SIDE, K_INPUT_SIDE, 3}));
}

/** @brief Tests that the file publishes one output. */
TEST_P(Medium_UltralyticsFiles, PublishesOneOutput)
{
    const Config::ModelSpec spec = load_spec(GetParam());

    EXPECT_EQ(spec.outputs.size(), 1u);
}

/**
 * @brief Tests that the parameter count of the built model is the Ultralytics summary figure
 *        less the elements of the ramp, which the framework holds as no record.
 */
TEST_P(Medium_UltralyticsFiles, BuildToTheParameterCountOfTheReference)
{
    SA_ASSERT_OK_AND_ASSIGN(const NN::NodeRegistry registry, composed());
    const Config::ModelSpec spec = load_spec(GetParam());
    SA_ASSERT_OK_AND_ASSIGN(ModelGraph model, NN::build_model(registry, spec));
    (void)trace_outputs(model, spec.input_shape);

    EXPECT_EQ(count_parameters(model), GetParam().reference_parameters - K_RAMP_ELEMENTS);
}

/** @brief Tests that the model publishes the three boxes and the three scores and no other port. */
TEST_P(Medium_UltralyticsFiles, PublishOnlyTheOneToManyPorts)
{
    const std::vector<Train::OutputShape> outputs = traced_outputs(load_spec(GetParam()));

    EXPECT_EQ(outputs.size(), 2 * K_LEVELS);
}

/** @brief Tests that the boxes of level 0 are 4 * reg_max channels wide at stride 8. */
TEST_P(Medium_UltralyticsFiles, PublishTheBoxesOfLevelZeroAtStrideEight)
{
    const std::vector<Train::OutputShape> outputs = traced_outputs(load_spec(GetParam()));

    EXPECT_EQ(shape_of(outputs, "one2many.level0.boxes"),
              (Shape{1, K_INPUT_SIDE / 8, K_INPUT_SIDE / 8, 4 * K_REG_MAX}));
}

/** @brief Tests that the boxes of level 1 are 4 * reg_max channels wide at stride 16. */
TEST_P(Medium_UltralyticsFiles, PublishTheBoxesOfLevelOneAtStrideSixteen)
{
    const std::vector<Train::OutputShape> outputs = traced_outputs(load_spec(GetParam()));

    EXPECT_EQ(shape_of(outputs, "one2many.level1.boxes"),
              (Shape{1, K_INPUT_SIDE / 16, K_INPUT_SIDE / 16, 4 * K_REG_MAX}));
}

/** @brief Tests that the boxes of level 2 are 4 * reg_max channels wide at stride 32. */
TEST_P(Medium_UltralyticsFiles, PublishTheBoxesOfLevelTwoAtStrideThirtyTwo)
{
    const std::vector<Train::OutputShape> outputs = traced_outputs(load_spec(GetParam()));

    EXPECT_EQ(shape_of(outputs, "one2many.level2.boxes"),
              (Shape{1, K_INPUT_SIDE / 32, K_INPUT_SIDE / 32, 4 * K_REG_MAX}));
}

/** @brief Tests that the scores of level 0 are 80 channels wide at stride 8. */
TEST_P(Medium_UltralyticsFiles, PublishTheScoresOfLevelZeroAtStrideEight)
{
    const std::vector<Train::OutputShape> outputs = traced_outputs(load_spec(GetParam()));

    EXPECT_EQ(shape_of(outputs, "one2many.level0.scores"),
              (Shape{1, K_INPUT_SIDE / 8, K_INPUT_SIDE / 8, K_CLASSES}));
}

/** @brief Tests that the scores of level 1 are 80 channels wide at stride 16. */
TEST_P(Medium_UltralyticsFiles, PublishTheScoresOfLevelOneAtStrideSixteen)
{
    const std::vector<Train::OutputShape> outputs = traced_outputs(load_spec(GetParam()));

    EXPECT_EQ(shape_of(outputs, "one2many.level1.scores"),
              (Shape{1, K_INPUT_SIDE / 16, K_INPUT_SIDE / 16, K_CLASSES}));
}

/** @brief Tests that the scores of level 2 are 80 channels wide at stride 32. */
TEST_P(Medium_UltralyticsFiles, PublishTheScoresOfLevelTwoAtStrideThirtyTwo)
{
    const std::vector<Train::OutputShape> outputs = traced_outputs(load_spec(GetParam()));

    EXPECT_EQ(shape_of(outputs, "one2many.level2.scores"),
              (Shape{1, K_INPUT_SIDE / 32, K_INPUT_SIDE / 32, K_CLASSES}));
}

/** @brief Tests that the traced outputs of the model give reg_max 16. */
TEST_P(Medium_UltralyticsFiles, GiveARegMaxOfSixteen)
{
    EXPECT_EQ(head_facts(load_spec(GetParam())).reg_max, K_REG_MAX);
}

/** @brief Tests that the traced outputs of the model give 80 classes. */
TEST_P(Medium_UltralyticsFiles, GiveEightyClasses)
{
    EXPECT_EQ(head_facts(load_spec(GetParam())).classes, K_CLASSES);
}

/** @brief Tests that the traced outputs of the model give no one-to-one copy. */
TEST_P(Medium_UltralyticsFiles, GiveNoOneToOneCopy)
{
    EXPECT_FALSE(head_facts(load_spec(GetParam())).has_one_to_one);
}

/** @brief Tests that the traced outputs of the model give three levels at strides 8, 16 and 32. */
TEST_P(Medium_UltralyticsFiles, GiveThreeLevelsAtTheLevelStrides)
{
    const FreeModelFacts facts = head_facts(load_spec(GetParam()));

    ASSERT_EQ(facts.levels.size(), K_LEVELS);
    EXPECT_EQ(facts.levels[0].stride, K_STRIDES[0]);
    EXPECT_EQ(facts.levels[1].stride, K_STRIDES[1]);
    EXPECT_EQ(facts.levels[2].stride, K_STRIDES[2]);
}

/** @brief Tests that the traced outputs of the model give square levels of the input over the stride. */
TEST_P(Medium_UltralyticsFiles, GiveSquareLevelsOfTheInputOverTheStride)
{
    const FreeModelFacts facts = head_facts(load_spec(GetParam()));

    ASSERT_EQ(facts.levels.size(), K_LEVELS);
    EXPECT_EQ(facts.levels[0].height, K_INPUT_SIDE / K_STRIDES[0]);
    EXPECT_EQ(facts.levels[0].width, K_INPUT_SIDE / K_STRIDES[0]);
    EXPECT_EQ(facts.levels[1].height, K_INPUT_SIDE / K_STRIDES[1]);
    EXPECT_EQ(facts.levels[1].width, K_INPUT_SIDE / K_STRIDES[1]);
    EXPECT_EQ(facts.levels[2].height, K_INPUT_SIDE / K_STRIDES[2]);
    EXPECT_EQ(facts.levels[2].width, K_INPUT_SIDE / K_STRIDES[2]);
}

INSTANTIATE_TEST_SUITE_P(Files, Medium_UltralyticsFiles, ::testing::ValuesIn(K_FILES),
                         file_name);

/**
 * @brief Tests that the task section of an anchor-free file reads back as score 0.25, with
 *        the values written out here and not read from FreeTask{}, the code under test.
 */
TEST_P(Medium_UltralyticsTaskSection, ReadsBackTheDefaultScoreThreshold)
{
    const Config::ModelSpec spec = require_value(Config::load_model(model_path(GetParam())));

    SA_ASSERT_OK_AND_ASSIGN(const FreeTask task, Vision::Detect::Free::read_free_task(spec.task));

    EXPECT_EQ(task.score_threshold, K_TASK_SCORE_THRESHOLD);
}

/** @brief Tests that the task section reads back as suppression at 0.7. */
TEST_P(Medium_UltralyticsTaskSection, ReadsBackTheDefaultSuppressionThreshold)
{
    const Config::ModelSpec spec = require_value(Config::load_model(model_path(GetParam())));

    SA_ASSERT_OK_AND_ASSIGN(const FreeTask task, Vision::Detect::Free::read_free_task(spec.task));

    EXPECT_EQ(task.suppression_threshold, K_TASK_SUPPRESSION_THRESHOLD);
}

/** @brief Tests that the task section reads back as suppression by intersection over union. */
TEST_P(Medium_UltralyticsTaskSection, ReadsBackIntersectionOverUnionSuppression)
{
    const Config::ModelSpec spec = require_value(Config::load_model(model_path(GetParam())));

    SA_ASSERT_OK_AND_ASSIGN(const FreeTask task, Vision::Detect::Free::read_free_task(spec.task));

    EXPECT_EQ(task.criterion, Vision::Detect::SuppressionCriterion::IoU);
}

INSTANTIATE_TEST_SUITE_P(Folders, Medium_UltralyticsTaskSection,
                         ::testing::ValuesIn(K_TASK_FOLDERS), folder_name);
