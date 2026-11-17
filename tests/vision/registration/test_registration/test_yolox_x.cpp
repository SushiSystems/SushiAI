/**************************************************************************/
/* test_yolox_x.cpp                                                       */
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
 * @file test_yolox_x.cpp
 * @brief Tests that the yolox_x model file builds, what it publishes, and that its name map fills
 *        every record of the model.
 * @author Mustafa Garip
 */

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <map>
#include <string>
#include <string_view>
#include <utility>
#include <vector>
#include <gtest/gtest.h>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/config/model_config.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/io/import_weights.hpp>
#include <SushiAI/io/name_map.hpp>
#include <SushiAI/nn/buffer.hpp>
#include <SushiAI/nn/model_graph.hpp>
#include <SushiAI/nn/module.hpp>
#include <SushiAI/nn/node_registry.hpp>
#include <SushiAI/shape/shape.hpp>
#include <SushiAI/tensor_values/tensor_values.hpp>
#include <SushiAI/train/objective_context.hpp>
#include <SushiAI/vision/detect/suppression_criterion.hpp>
#include <SushiAI/vision/detect/yolox/yolox_head_contract.hpp>
#include <SushiAI/vision/detect/yolox/yolox_task.hpp>
#include "registration_test_support.hpp"

using namespace SushiAI;
using NN::ModelGraph;
namespace SaTest = SushiAI::Test;

namespace
{
    using Test::composed;
    using Test::trace_outputs;

    /** @brief Holds the row count of the yolox_x model file. */
    constexpr std::size_t K_ROWS = 25;

    /** @brief Holds the height of the network input in pixels, the MOT17 frame fitted. */
    constexpr std::int64_t K_INPUT_HEIGHT = 800;

    /** @brief Holds the width of the network input in pixels. */
    constexpr std::int64_t K_INPUT_WIDTH = 1440;

    /** @brief Holds the strides of the three levels, finest first. */
    constexpr std::int64_t K_STRIDES[] = {8, 16, 32};

    /** @brief Holds the class count of the MOT17 model: pedestrians. */
    constexpr std::int64_t K_CLASSES = 1;

    /**
     * @brief Holds the unit width of YOLOX-X: the 64 channels of the stem times the width
     *        multiplier 1.25.
     */
    constexpr std::int64_t K_UNIT = 80;

    /** @brief Holds the repeats of a one-stage CSP layer of YOLOX-X: round(3 * 1.33). */
    constexpr std::int64_t K_REPEATS = 4;

    /** @brief Holds the width of the head's stems and towers: 256 times the width multiplier. */
    constexpr std::int64_t K_HEAD_WIDTH = 4 * K_UNIT;

    /** @brief Holds the score threshold of the task section. */
    constexpr float K_SCORE_THRESHOLD = 0.1f;

    /** @brief Holds the suppression threshold of the task section. */
    constexpr float K_SUPPRESSION_THRESHOLD = 0.7f;

    /** @brief Returns the path of the yolox_x model file. */
    std::filesystem::path model_path()
    {
        return std::filesystem::path(SA_REPOSITORY_ROOT) / "applications" / "examples" / "yolox_x"
               / "model.json";
    }

    /** @brief Returns the path of the yolox_x name map. */
    std::filesystem::path map_path()
    {
        return std::filesystem::path(SA_REPOSITORY_ROOT) / "applications" / "examples" / "yolox_x"
               / "map.json";
    }

    /** @brief Loads the yolox_x model file. */
    Config::ModelSpec load_spec()
    {
        return SaTest::require_value(Config::load_model(model_path()));
    }

    /** @brief Builds the model of @p spec over the composed registry. */
    ModelGraph build(const Config::ModelSpec& spec)
    {
        const NN::NodeRegistry registry = SaTest::require_value(composed());

        return SaTest::require_value(NN::build_model(registry, spec));
    }

    /** @brief Gives each level of the head, finest first, to a test. */
    class Medium_YoloxXLevel : public ::testing::TestWithParam<std::size_t> {};

    /** @brief Names a level in the name of its test. */
    std::string level_name(const ::testing::TestParamInfo<std::size_t>& info)
    {
        return "level" + std::to_string(info.param);
    }

    /** @brief Holds one tensor of the PyTorch state dictionary, in the file's layout. */
    struct FileTensor
    {
        std::string name; /**< The name in the state dictionary below "model". */
        Shape shape;      /**< The extents as PyTorch stores them, [out, in, kh, kw] for a filter. */
    };

    /** @brief Holds the tensors a YOLOX state dictionary lists, and the counters it adds. */
    struct FileTensors
    {
        std::vector<FileTensor> tensors;       /**< Weights, biases and running statistics. */
        std::vector<std::string> counters;     /**< The num_batches_tracked of every BatchNorm. */
    };

    /** @brief Appends the conv and bn of a YOLOX BaseConv named @p prefix. */
    void add_base_conv(FileTensors& file, const std::string& prefix, std::int64_t in,
                       std::int64_t out, std::int64_t kernel)
    {
        file.tensors.push_back({prefix + ".conv.weight", Shape{out, in, kernel, kernel}});
        for (const char* const leaf : {"weight", "bias", "running_mean", "running_var"})
            file.tensors.push_back({prefix + ".bn." + leaf, Shape{out}});
        file.counters.push_back(prefix + ".bn.num_batches_tracked");
    }

    /**
     * @brief Appends a CSPLayer: conv1 and conv2 narrow to half the output, conv3 joins both
     *        halves, and each of @p repeats bottlenecks is a 1x1 then a 3x3 at the half width.
     */
    void add_csp_layer(FileTensors& file, const std::string& prefix, std::int64_t in,
                       std::int64_t out, std::int64_t repeats)
    {
        const std::int64_t hidden = out / 2;
        add_base_conv(file, prefix + ".conv1", in, hidden, 1);
        add_base_conv(file, prefix + ".conv2", in, hidden, 1);
        add_base_conv(file, prefix + ".conv3", 2 * hidden, out, 1);
        for (std::int64_t k = 0; k < repeats; ++k)
        {
            add_base_conv(file, prefix + ".m." + std::to_string(k) + ".conv1", hidden, hidden, 1);
            add_base_conv(file, prefix + ".m." + std::to_string(k) + ".conv2", hidden, hidden, 3);
        }
    }

    /** @brief Appends an SPPBottleneck: conv1 to half the input, conv2 over four stacked maps. */
    void add_spp(FileTensors& file, const std::string& prefix, std::int64_t in, std::int64_t out)
    {
        add_base_conv(file, prefix + ".conv1", in, in / 2, 1);
        add_base_conv(file, prefix + ".conv2", 4 * (in / 2), out, 1);
    }

    /** @brief Appends the head of one level: its stem, two towers and three 1x1 predictors. */
    void add_head_level(FileTensors& file, std::int64_t level, std::int64_t in)
    {
        const std::string at = std::to_string(level);
        add_base_conv(file, "head.stems." + at, in, K_HEAD_WIDTH, 1);
        for (const char* const tower : {"cls_convs", "reg_convs"})
        {
            for (std::int64_t k = 0; k < 2; ++k)
                add_base_conv(file, std::string("head.") + tower + "." + at + "." + std::to_string(k),
                              K_HEAD_WIDTH, K_HEAD_WIDTH, 3);
        }
        const std::pair<const char*, std::int64_t> predictors[] = {
            {"cls_preds", K_CLASSES}, {"reg_preds", 4}, {"obj_preds", 1}};
        for (const auto& [name, channels] : predictors)
        {
            file.tensors.push_back({std::string("head.") + name + "." + at + ".weight",
                                    Shape{channels, K_HEAD_WIDTH, 1, 1}});
            file.tensors.push_back({std::string("head.") + name + "." + at + ".bias",
                                    Shape{channels}});
        }
    }

    /**
     * @brief Lists the state dictionary of YOLOX-X with one class, written from the structure of
     *        the YOLOX source: CSPDarknet, YOLOPAFPN and YOLOXHead at depth 1.33 and width 1.25.
     *        It is derived here from the architecture, not read from the model file.
     */
    FileTensors yolox_x_state_dictionary()
    {
        FileTensors file;
        const std::string backbone = "backbone.backbone.";

        // The stem is a Focus whose convolution reads four stacked phases of three channels.
        add_base_conv(file, backbone + "stem.conv", 12, K_UNIT, 3);
        add_base_conv(file, backbone + "dark2.0", K_UNIT, 2 * K_UNIT, 3);
        add_csp_layer(file, backbone + "dark2.1", 2 * K_UNIT, 2 * K_UNIT, K_REPEATS);
        add_base_conv(file, backbone + "dark3.0", 2 * K_UNIT, 4 * K_UNIT, 3);
        add_csp_layer(file, backbone + "dark3.1", 4 * K_UNIT, 4 * K_UNIT, 3 * K_REPEATS);
        add_base_conv(file, backbone + "dark4.0", 4 * K_UNIT, 8 * K_UNIT, 3);
        add_csp_layer(file, backbone + "dark4.1", 8 * K_UNIT, 8 * K_UNIT, 3 * K_REPEATS);
        add_base_conv(file, backbone + "dark5.0", 8 * K_UNIT, 16 * K_UNIT, 3);
        add_spp(file, backbone + "dark5.1", 16 * K_UNIT, 16 * K_UNIT);
        add_csp_layer(file, backbone + "dark5.2", 16 * K_UNIT, 16 * K_UNIT, K_REPEATS);

        add_base_conv(file, "backbone.lateral_conv0", 16 * K_UNIT, 8 * K_UNIT, 1);
        add_csp_layer(file, "backbone.C3_p4", 16 * K_UNIT, 8 * K_UNIT, K_REPEATS);
        add_base_conv(file, "backbone.reduce_conv1", 8 * K_UNIT, 4 * K_UNIT, 1);
        add_csp_layer(file, "backbone.C3_p3", 8 * K_UNIT, 4 * K_UNIT, K_REPEATS);
        add_base_conv(file, "backbone.bu_conv2", 4 * K_UNIT, 4 * K_UNIT, 3);
        add_csp_layer(file, "backbone.C3_n3", 8 * K_UNIT, 8 * K_UNIT, K_REPEATS);
        add_base_conv(file, "backbone.bu_conv1", 8 * K_UNIT, 8 * K_UNIT, 3);
        add_csp_layer(file, "backbone.C3_n4", 16 * K_UNIT, 16 * K_UNIT, K_REPEATS);

        add_head_level(file, 0, 4 * K_UNIT);
        add_head_level(file, 1, 8 * K_UNIT);
        add_head_level(file, 2, 16 * K_UNIT);
        return file;
    }

    /** @brief Returns the shape a record states for a file tensor of @p shape. */
    Shape record_shape(const Shape& shape)
    {
        if (shape.rank() != 4)
            return shape;

        return Shape{shape[0], shape[2], shape[3], shape[1]};
    }

    /** @brief Returns every record of @p model, parameters and buffers, by name. */
    std::map<std::string, Shape> records_of(ModelGraph& model)
    {
        std::map<std::string, Shape> records;
        auto on_parameter = [&records](const std::string& name, NN::Parameter& parameter)
        {
            records[name] = parameter.shape();
        };
        auto on_buffer = [&records](const std::string& name, NN::Buffer& buffer)
        {
            records[name] = buffer.shape();
        };
        const NN::StateVisitor visitor(on_parameter, on_buffer);
        model.walk_state("", visitor);
        return records;
    }
} // namespace

/** @brief Tests that the file holds its 25 rows. */
TEST(Medium_YoloxXFile, HoldsItsRows)
{
    EXPECT_EQ(load_spec().layers.size(), K_ROWS);
}

/** @brief Tests that the file takes an 800 by 1440 by 3 input. */
TEST(Medium_YoloxXFile, TakesAnInputOf800By1440By3)
{
    EXPECT_EQ(load_spec().input_shape, (Shape{K_INPUT_HEIGHT, K_INPUT_WIDTH, 3}));
}

/** @brief Tests that the file publishes nine outputs named by level and role, level by level. */
TEST(Medium_YoloxXFile, PublishesNineOutputsNamedByLevelAndRole)
{
    const Config::ModelSpec spec = load_spec();
    const std::string_view roles[] = {Vision::Detect::Yolox::K_YOLOX_CLASSES,
                                      Vision::Detect::Yolox::K_YOLOX_BOXES,
                                      Vision::Detect::Yolox::K_YOLOX_OBJECTNESS};

    ASSERT_EQ(spec.outputs.size(), 9u);
    for (std::size_t level = 0; level < 3; ++level)
    {
        for (std::size_t role = 0; role < 3; ++role)
        {
            EXPECT_EQ(spec.outputs[level * 3 + role].name,
                      Vision::Detect::Yolox::yolox_port_name(level, roles[role]));
        }
    }
}

/** @brief Tests that the three ports of a level sit on the grid of the level's stride. */
TEST_P(Medium_YoloxXLevel, PublishesItsThreePortsOnTheGridOfItsStride)
{
    const Config::ModelSpec spec = load_spec();
    ModelGraph model = build(spec);
    const std::vector<Train::OutputShape> outputs = trace_outputs(model, spec.input_shape);
    const std::int64_t stride = K_STRIDES[GetParam()];
    const std::int64_t rows = K_INPUT_HEIGHT / stride;
    const std::int64_t columns = K_INPUT_WIDTH / stride;

    ASSERT_EQ(outputs.size(), 9u);
    EXPECT_EQ(outputs[GetParam() * 3].shape, (Shape{1, rows, columns, K_CLASSES}));
    EXPECT_EQ(outputs[GetParam() * 3 + 1].shape, (Shape{1, rows, columns, 4}));
    EXPECT_EQ(outputs[GetParam() * 3 + 2].shape, (Shape{1, rows, columns, 1}));
}

INSTANTIATE_TEST_SUITE_P(Levels, Medium_YoloxXLevel, ::testing::Values(0, 1, 2), level_name);

/** @brief Tests that the task section reads back the score and the suppression it states. */
TEST(Medium_YoloxXFile, TheTaskSectionReadsBackItsThresholds)
{
    const Vision::Detect::Yolox::YoloxTask task =
        SaTest::require_value(Vision::Detect::Yolox::read_yolox_task(load_spec().task));

    EXPECT_EQ(task.score_threshold, K_SCORE_THRESHOLD);
    EXPECT_EQ(task.suppression.criterion, Vision::Detect::SuppressionCriterion::IoU);
    EXPECT_EQ(task.suppression.threshold, K_SUPPRESSION_THRESHOLD);
}

/** @brief Tests that every rule of the map ends in a record the model has. */
TEST(Medium_YoloxXMap, EveryRuleNamesARecordTheModelHas)
{
    const Config::ModelSpec spec = load_spec();
    ModelGraph model = build(spec);
    (void)trace_outputs(model, spec.input_shape);
    const std::map<std::string, Shape> records = records_of(model);
    const IO::NameMap map = SaTest::require_value(IO::NameMap::read(map_path()));

    ASSERT_FALSE(map.rename_rules().empty());
    for (const IO::RenameRule& rule : map.rename_rules())
    {
        const std::string to(rule.to_prefix);
        const bool found = std::any_of(records.begin(), records.end(),
                                       [&to](const auto& record)
                                       {
                                           return record.first.rfind(to, 0) == 0;
                                       });
        EXPECT_TRUE(found) << "'" << rule.from_prefix << "' maps to '" << to
                           << "', which no record starts with";
    }
}

/**
 * @brief Tests that the YOLOX-X state dictionary, written from the architecture, renames to the
 *        records of the model exactly, shape for shape, and that nothing is left over.
 */
TEST(Medium_YoloxXMap, TheStateDictionaryFillsTheRecordsExactly)
{
    const Config::ModelSpec spec = load_spec();
    ModelGraph model = build(spec);
    (void)trace_outputs(model, spec.input_shape);
    const std::map<std::string, Shape> records = records_of(model);
    const IO::NameMap map = SaTest::require_value(IO::NameMap::read(map_path()));
    const std::vector<IO::RenameRule> rules = map.rename_rules();
    const FileTensors file = yolox_x_state_dictionary();

    std::map<std::string, Shape> renamed_shapes;
    for (const FileTensor& tensor : file.tensors)
    {
        ASSERT_FALSE(map.is_dropped(tensor.name)) << tensor.name;
        const std::string target = IO::renamed(
            tensor.name, SushiRuntime::span<const IO::RenameRule>(rules.data(), rules.size()));
        EXPECT_TRUE(renamed_shapes.emplace(target, record_shape(tensor.shape)).second)
            << "two tensors rename to '" << target << "'";
    }

    for (const auto& [name, shape] : renamed_shapes)
    {
        const auto at = records.find(name);
        ASSERT_NE(at, records.end()) << "no record '" << name << "'";
        EXPECT_EQ(at->second, shape) << name;
    }
    for (const auto& [name, shape] : records)
        EXPECT_TRUE(renamed_shapes.count(name) == 1) << "the file supplies no '" << name << "'";
    EXPECT_EQ(renamed_shapes.size(), records.size());
}

/** @brief Tests that the map drops the batch counter of every normalisation. */
TEST(Medium_YoloxXMap, TheBatchCountersAreDropped)
{
    const IO::NameMap map = SaTest::require_value(IO::NameMap::read(map_path()));
    const FileTensors file = yolox_x_state_dictionary();

    ASSERT_FALSE(file.counters.empty());
    for (const std::string& counter : file.counters)
        EXPECT_TRUE(map.is_dropped(counter)) << counter;
}

/** @brief Tests that the map names the key of the state dictionary of a ByteTrack checkpoint. */
TEST(Medium_YoloxXMap, NamesTheModelKeyOfTheCheckpoint)
{
    const IO::NameMap map = SaTest::require_value(IO::NameMap::read(map_path()));

    EXPECT_EQ(map.state_key(), "model");
}
