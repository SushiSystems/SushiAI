/**************************************************************************/
/* test_detect_head.cpp                                                   */
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
 * @file test_detect_head.cpp
 * @brief Tests DetectHead's ports, state names and gradient barriers against YOLO26n row 23.
 * @author Mustafa Garip
 */

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>
#include <gtest/gtest.h>
#include <SushiAI/config/value.hpp>
#include <SushiAI/ops/ops.hpp>
#include <SushiAI/shape/shape.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/graph/builder.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/nn/batch_norm2d.hpp>
#include <SushiAI/nn/buffer.hpp>
#include <SushiAI/nn/module.hpp>
#include <SushiAI/nn/node.hpp>
#include <SushiAI/nn/trace.hpp>
#include <SushiAI/vision/blocks/detect_head.hpp>
#include <support/tolerance.hpp>

using namespace SushiAI;
using NN::BufferUpdateList;
using NN::NormOptions;
using NN::Trace;
using Vision::Blocks::ClassBranchStyle;
using Vision::Blocks::DetectHead;
using Vision::Blocks::DetectOptions;

namespace
{
    using SushiAI::Test::require_value;
    /** @brief Holds the epsilon and the momentum MANIFEST.json records for YOLO26n. */
    constexpr NormOptions K_YOLO_NORM{1e-3, 0.03};

    /** @brief Names the row a factory is asked to build, as a diagnostic would. */
    constexpr std::string_view K_CONTEXT = "layer 23 'detect'";

    /** @brief Holds the class count of the fixture. */
    constexpr int64_t K_CLASSES = 80;

    /** @brief Holds the number of levels row 23 reads. */
    constexpr std::size_t K_LEVELS = 3;

    /** @brief Holds the widths of rows 16, 19 and 22, which row 23 reads. */
    constexpr std::array<int64_t, K_LEVELS> K_WIDTHS = {64, 128, 256};

    /** @brief Holds the H and the W of each level for a 640 by 640 image. */
    constexpr std::array<int64_t, K_LEVELS> K_SIDES = {80, 40, 20};

    /** @brief Holds the H and the W of each level of a traced graph. */
    constexpr std::array<int64_t, K_LEVELS> K_TRACED_SIDES = {8, 4, 2};

    /** @brief Holds the batch extent of a traced graph. */
    constexpr int64_t K_BATCH = 2;

    /** @brief Holds the hidden width of the fixture's box branches. */
    constexpr int64_t K_BOX_WIDTH = 16;

    /** @brief Holds the hidden width of the fixture's class branches. */
    constexpr int64_t K_CLASS_WIDTH = 80;

    /** @brief Names which of the two kinds of state a record is. */
    enum class StateKind : std::uint8_t
    {
        Parameter, /**< A learnable tensor. */
        Buffer,    /**< A stored tensor no gradient reaches. */
    };

    /** @brief Writes one piece of state as "name [extents] kind", for a readable mismatch. */
    std::string record(const std::string& name, const Shape& shape, StateKind kind)
    {
        return name + " " + shape.to_string() +
               (kind == StateKind::Buffer ? " buffer" : " parameter");
    }

    /** @brief Lists every piece of state under @p module, in the order the walk yields. */
    template<typename M>
    std::vector<std::string> walk(M& module, std::string_view prefix)
    {
        std::vector<std::string> records;
        const auto on_parameter = [&records](const std::string& name, NN::Parameter& state)
        {
            records.push_back(record(name, state.shape(), StateKind::Parameter));
        };
        const auto on_buffer = [&records](const std::string& name, NN::Buffer& state)
        {
            records.push_back(record(name, state.shape(), StateKind::Buffer));
        };
        const NN::StateVisitor visitor(on_parameter, on_buffer);
        NN::walk_state(module, prefix, visitor);

        return records;
    }

    /** @brief Appends the five records of a convolution and its norm under @p prefix. */
    void append_conv_norm(std::vector<std::string>& records, const std::string& prefix,
                          const Shape& weight, int64_t width)
    {
        records.push_back(record(prefix + ".conv.weight", weight, StateKind::Parameter));
        records.push_back(record(prefix + ".bn.weight", Shape{width}, StateKind::Parameter));
        records.push_back(record(prefix + ".bn.bias", Shape{width}, StateKind::Parameter));
        records.push_back(record(prefix + ".bn.running_mean", Shape{width}, StateKind::Buffer));
        records.push_back(record(prefix + ".bn.running_var", Shape{width}, StateKind::Buffer));
    }

    /** @brief Appends the two records of a biased 1x1 convolution under @p prefix. */
    void append_biased_pointwise(std::vector<std::string>& records, const std::string& prefix,
                                 int64_t out_channels, int64_t in_channels)
    {
        records.push_back(record(prefix + ".weight", Shape{out_channels, 1, 1, in_channels},
                                 StateKind::Parameter));
        records.push_back(record(prefix + ".bias", Shape{out_channels}, StateKind::Parameter));
    }

    /** @brief Appends the twelve records of each level's box branch under @p prefix. */
    void append_box_branches(std::vector<std::string>& records, const std::string& prefix)
    {
        for (std::size_t level = 0; level < K_LEVELS; ++level)
        {
            const std::string branch = prefix + "." + std::to_string(level);
            append_conv_norm(records, branch + ".0", Shape{K_BOX_WIDTH, 3, 3, K_WIDTHS[level]},
                             K_BOX_WIDTH);
            append_conv_norm(records, branch + ".1", Shape{K_BOX_WIDTH, 3, 3, K_BOX_WIDTH},
                             K_BOX_WIDTH);
            append_biased_pointwise(records, branch + ".2", 4, K_BOX_WIDTH);
        }
    }

    /** @brief Appends the twenty-two records of each level's class branch under @p prefix. */
    void append_class_branches(std::vector<std::string>& records, const std::string& prefix)
    {
        for (std::size_t level = 0; level < K_LEVELS; ++level)
        {
            const std::string branch = prefix + "." + std::to_string(level);
            const int64_t width = K_WIDTHS[level];
            append_conv_norm(records, branch + ".0.0", Shape{3, 3, width}, width);
            append_conv_norm(records, branch + ".0.1", Shape{K_CLASS_WIDTH, 1, 1, width},
                             K_CLASS_WIDTH);
            append_conv_norm(records, branch + ".1.0", Shape{3, 3, K_CLASS_WIDTH},
                             K_CLASS_WIDTH);
            append_conv_norm(records, branch + ".1.1",
                             Shape{K_CLASS_WIDTH, 1, 1, K_CLASS_WIDTH}, K_CLASS_WIDTH);
            append_biased_pointwise(records, branch + ".2", K_CLASSES, K_CLASS_WIDTH);
        }
    }

    /** @brief Counts the nodes of @p graph that carry the operation @p op. */
    std::size_t count_nodes(const Graph::Graph& graph, OpID op)
    {
        std::size_t count = 0;
        for (const Graph::Node& node : graph.nodes())
        {
            if (node.op == op)
                ++count;
        }

        return count;
    }

    /** @brief Parses @p text as the attributes of one row. */
    Config::Value attributes_of(std::string_view text)
    {
        return require_value(Config::parse_json(text, "<test>"));
    }

    /** @brief Lists the per-sample [side, side, width] shape of each level. */
    std::vector<Shape> level_shapes(const std::array<int64_t, K_LEVELS>& sides,
                                    const std::array<int64_t, K_LEVELS>& widths)
    {
        std::vector<Shape> shapes;
        for (std::size_t level = 0; level < K_LEVELS; ++level)
            shapes.push_back(Shape{sides[level], sides[level], widths[level]});

        return shapes;
    }

    /** @brief Views @p shapes as the inputs of a row. */
    NN::ShapeSpan span_of(const std::vector<Shape>& shapes)
    {
        return NN::ShapeSpan(shapes.data(), shapes.size());
    }

    /** @brief Holds a graph with the three features of a batch of K_BATCH declared in it. */
    struct FeatureGraph
    {
        /** @brief Holds the graph a head is traced into. */
        Graph::Graph graph;

        /** @brief Holds the builder over the graph. */
        Graph::GraphBuilder builder{graph};

        /** @brief Holds the id of each level's [K_BATCH, side, side, width] input. */
        std::vector<Graph::ValueID> features;

        /** @brief Declares one input per level of K_TRACED_SIDES and K_WIDTHS. */
        FeatureGraph()
        {
            for (std::size_t level = 0; level < K_LEVELS; ++level)
            {
                const int64_t side = K_TRACED_SIDES[level];
                features.push_back(
                    builder.input(Shape{K_BATCH, side, side, K_WIDTHS[level]}, "feature"));
            }
        }

        /** @brief Views the three inputs as the values a head reads. */
        NN::ValueSpan inputs() const
        {
            return NN::ValueSpan(features.data(), features.size());
        }
    };

    /** @brief Builds the head of row 23 over K_SIDES and K_WIDTHS, or fails the test. */
    std::unique_ptr<DetectHead> make_row23()
    {
        const std::vector<Shape> levels = level_shapes(K_SIDES, K_WIDTHS);
        Result<std::unique_ptr<DetectHead>> made =
            DetectHead::create(span_of(levels), K_CLASSES, K_YOLO_NORM);
        EXPECT_TRUE(Test::status_is_ok(made.status()));
        if (!made.status().is_ok())
            return nullptr;

        return std::move(made).value();
    }

    /** @brief Holds the distribution bins per box side of the fixture's wide-box heads. */
    constexpr int64_t K_REG_MAX = 16;

    /** @brief Holds the box branch width of the wide-box fixture: max(16, 64 / 4, 4 * 16). */
    constexpr int64_t K_WIDE_BOX_WIDTH = 64;

    /** @brief Holds the strides of the three fixture levels. */
    const std::vector<double> K_STRIDES = {8.0, 16.0, 32.0};

    /** @brief Builds a head over K_SIDES and K_WIDTHS with @p options, or fails the test. */
    std::unique_ptr<DetectHead> make_head(const DetectOptions& options)
    {
        const std::vector<Shape> levels = level_shapes(K_SIDES, K_WIDTHS);
        Result<std::unique_ptr<DetectHead>> made =
            DetectHead::create(span_of(levels), K_CLASSES, K_YOLO_NORM, options);
        EXPECT_TRUE(Test::status_is_ok(made.status()));
        if (!made.status().is_ok())
            return nullptr;

        return std::move(made).value();
    }

    /** @brief Returns the options of a head with wide boxes, one copy and plain class branches. */
    DetectOptions plain_single_copy_options()
    {
        DetectOptions options;
        options.reg_max = K_REG_MAX;
        options.end2end = false;
        options.class_branch = ClassBranchStyle::Plain;

        return options;
    }

    /** @brief Appends the twelve records of each level's wide box branch under @p prefix. */
    void append_wide_box_branches(std::vector<std::string>& records, const std::string& prefix)
    {
        for (std::size_t level = 0; level < K_LEVELS; ++level)
        {
            const std::string branch = prefix + "." + std::to_string(level);
            append_conv_norm(records, branch + ".0",
                             Shape{K_WIDE_BOX_WIDTH, 3, 3, K_WIDTHS[level]}, K_WIDE_BOX_WIDTH);
            append_conv_norm(records, branch + ".1",
                             Shape{K_WIDE_BOX_WIDTH, 3, 3, K_WIDE_BOX_WIDTH}, K_WIDE_BOX_WIDTH);
            append_biased_pointwise(records, branch + ".2", 4 * K_REG_MAX, K_WIDE_BOX_WIDTH);
        }
    }

    /** @brief Appends the twelve records of each level's plain class branch under @p prefix. */
    void append_plain_class_branches(std::vector<std::string>& records,
                                     const std::string& prefix)
    {
        for (std::size_t level = 0; level < K_LEVELS; ++level)
        {
            const std::string branch = prefix + "." + std::to_string(level);
            append_conv_norm(records, branch + ".0", Shape{K_CLASS_WIDTH, 3, 3, K_WIDTHS[level]},
                             K_CLASS_WIDTH);
            append_conv_norm(records, branch + ".1", Shape{K_CLASS_WIDTH, 3, 3, K_CLASS_WIDTH},
                             K_CLASS_WIDTH);
            append_biased_pointwise(records, branch + ".2", K_CLASSES, K_CLASS_WIDTH);
        }
    }

    /**
     * @brief Traces @p head once in evaluation mode and returns the constant every bias named
     *        "<...>.2.bias" starts from, keyed by its state name.
     */
    std::map<std::string, double> last_bias_constants(DetectHead& head)
    {
        FeatureGraph traced;
        std::vector<Graph::ValueID> outputs;
        SA_EXPECT_OK(head.forward(Trace::evaluation(traced.builder), traced.inputs(), outputs));

        std::map<std::string, double> constants;
        const auto on_parameter = [&constants](const std::string& name, NN::Parameter& state)
        {
            const std::string suffix = ".2.bias";
            const bool is_last = name.size() > suffix.size() &&
                                 name.compare(name.size() - suffix.size(), suffix.size(),
                                              suffix) == 0;
            if (!is_last)
                return;

            EXPECT_EQ(state.init().kind, NN::ParameterInit::Kind::Constant) << name;
            constants[name] = state.init().constant;
        };
        const auto on_buffer = [](const std::string&, NN::Buffer&) {};
        const NN::StateVisitor visitor(on_parameter, on_buffer);
        NN::walk_state(head, "", visitor);

        return constants;
    }

    /** @brief Returns log(5 / classes / (640 / stride)^2) for the fixture's class count. */
    double expected_class_bias(double stride)
    {
        const double cells = 640.0 / stride;

        return std::log(5.0 / static_cast<double>(K_CLASSES) / (cells * cells));
    }

    /** @brief Builds row 23 over K_SIDES and K_WIDTHS, or returns null when create refuses it. */
    std::unique_ptr<DetectHead> build_row23()
    {
        const std::vector<Shape> levels = level_shapes(K_SIDES, K_WIDTHS);
        Result<std::unique_ptr<DetectHead>> made =
            DetectHead::create(span_of(levels), K_CLASSES, K_YOLO_NORM);
        if (!made.status().is_ok())
            return nullptr;

        return std::move(made).value();
    }

    /** @brief Expects @p options on every BATCHNORM2D node of @p graph and every update. */
    void expect_norm_options(const Graph::Graph& graph, const BufferUpdateList& list,
                             const NormOptions& options)
    {
        // Both constants are stored as given, so they compare exactly.
        for (const Graph::Node& node : graph.nodes())
        {
            if (node.op == Ops::K_BATCHNORM2D.id)
                EXPECT_EQ(node.attributes.values[0], options.epsilon);
        }

        for (const auto& update : list.updates())
            EXPECT_EQ(update.momentum, options.momentum);
    }

    /** @brief Lists the input of each STOP_GRADIENT node of @p graph, in emission order. */
    std::vector<Graph::ValueID> detached_inputs(const Graph::Graph& graph)
    {
        std::vector<Graph::ValueID> detached;
        for (const Graph::Node& node : graph.nodes())
        {
            if (node.op == Ops::K_STOP_GRADIENT.id)
                detached.push_back(node.inputs[0]);
        }

        return detached;
    }

    /** @brief Lists the shape @p graph holds for each of @p values. */
    std::vector<Shape> shapes_of(const Graph::Graph& graph,
                                 const std::vector<Graph::ValueID>& values)
    {
        std::vector<Shape> shapes;
        for (const Graph::ValueID value : values)
            shapes.push_back(graph.value(value).shape);

        return shapes;
    }

    /**
     * @brief Lists the batched shape each port of @p head yields in a graph of K_BATCH samples
     *        and K_TRACED_SIDES: port p reads level p % K_LEVELS and keeps its channel count.
     */
    std::vector<Shape> traced_port_shapes(const DetectHead& head)
    {
        std::vector<Shape> shapes;
        for (std::size_t port = 0; port < head.ports().size(); ++port)
        {
            const std::size_t level = port % K_LEVELS;
            shapes.push_back(Shape{K_BATCH, K_TRACED_SIDES[level], K_TRACED_SIDES[level],
                                   head.ports()[port].shape[2]});
        }

        return shapes;
    }

    /** @brief Lists the shape of each port of @p head at the given @p indices. */
    std::vector<Shape> port_shapes_at(const DetectHead& head,
                                      const std::vector<std::size_t>& indices)
    {
        std::vector<Shape> shapes;
        for (const std::size_t index : indices)
            shapes.push_back(head.ports()[index].shape);

        return shapes;
    }

    /** @brief Lists the name of each port of @p head at the given @p indices. */
    std::vector<std::string> port_names_at(const DetectHead& head,
                                           const std::vector<std::size_t>& indices)
    {
        std::vector<std::string> names;
        for (const std::size_t index : indices)
            names.push_back(head.ports()[index].name);

        return names;
    }

    /** @brief Reports whether @p records holds @p wanted. */
    bool contains(const std::vector<std::string>& records, const std::string& wanted)
    {
        return std::find(records.begin(), records.end(), wanted) != records.end();
    }

    /** @brief Lists the state name of every last bias: "<copy>cv2.<level>.2.bias" and "cv3". */
    std::vector<std::string> last_bias_names()
    {
        std::vector<std::string> names;
        for (const std::string_view copy : {std::string_view(), std::string_view("one2one_")})
        {
            for (const std::string_view branch : {std::string_view("cv2"), std::string_view("cv3")})
            {
                for (std::size_t level = 0; level < K_LEVELS; ++level)
                    names.push_back(std::string(copy) + std::string(branch) + "." +
                                    std::to_string(level) + ".2.bias");
            }
        }

        return names;
    }

    /** @brief Maps every last-bias name to @p value. */
    std::map<std::string, double> all_last_biases(double value)
    {
        std::map<std::string, double> constants;
        for (const std::string& name : last_bias_names())
            constants[name] = value;

        return constants;
    }

    /**
     * @brief Compares each constant of @p got with its closed form: 1 for a box bias and
     *        expected_class_bias() of its level's stride for a class bias.
     * @return A success, or a failure naming the first constant outside its bound.
     * @details The bound is (4 + |reference|) times the double unit roundoff: four roundings in
     *          the argument of the logarithm move its result by about 4u, and the logarithm
     *          itself rounds once more in proportion to its result.
     */
    ::testing::AssertionResult strided_biases_within_bound(
        const std::map<std::string, double>& got)
    {
        for (const auto& entry : got)
        {
            const std::size_t level =
                static_cast<std::size_t>(entry.first[entry.first.find('.') + 1] - '0');
            const bool is_box = entry.first.find("cv2.") != std::string::npos;
            const double reference = is_box ? 1.0 : expected_class_bias(K_STRIDES[level]);
            const double bound =
                (4.0 + std::fabs(reference)) * TestSupport::unit_roundoff<double>();
            const ::testing::AssertionResult within = TestSupport::is_within(
                TestSupport::measure_error(entry.second, reference, bound));
            if (!within)
                return ::testing::AssertionFailure() << entry.first << ": " << within.message();
        }

        return ::testing::AssertionSuccess();
    }

    /** @brief Holds one port of row 23 and the name and shape the fixture records for it. */
    struct PortRow
    {
        std::string name;      /**< The row's name. */
        std::size_t index;     /**< The port's place in ports(). */
        std::string port_name; /**< The name forward.sacp records. */
        Shape shape;           /**< The per-sample shape forward.sacp records. */
    };

    /** @brief Lists the twelve ports of row 23: per copy the boxes of each level, then the scores. */
    std::vector<PortRow> port_rows()
    {
        return {
            {"OneToManyLevel0Boxes", 0, "one2many.level0.boxes", Shape{80, 80, 4}},
            {"OneToManyLevel1Boxes", 1, "one2many.level1.boxes", Shape{40, 40, 4}},
            {"OneToManyLevel2Boxes", 2, "one2many.level2.boxes", Shape{20, 20, 4}},
            {"OneToManyLevel0Scores", 3, "one2many.level0.scores", Shape{80, 80, 80}},
            {"OneToManyLevel1Scores", 4, "one2many.level1.scores", Shape{40, 40, 80}},
            {"OneToManyLevel2Scores", 5, "one2many.level2.scores", Shape{20, 20, 80}},
            {"OneToOneLevel0Boxes", 6, "one2one.level0.boxes", Shape{80, 80, 4}},
            {"OneToOneLevel1Boxes", 7, "one2one.level1.boxes", Shape{40, 40, 4}},
            {"OneToOneLevel2Boxes", 8, "one2one.level2.boxes", Shape{20, 20, 4}},
            {"OneToOneLevel0Scores", 9, "one2one.level0.scores", Shape{80, 80, 80}},
            {"OneToOneLevel1Scores", 10, "one2one.level1.scores", Shape{40, 40, 80}},
            {"OneToOneLevel2Scores", 11, "one2one.level2.scores", Shape{20, 20, 80}},
        };
    }

    /** @brief Runs one port of row 23 against the one head the whole suite shares. */
    class Small_BlockDetectHeadPorts : public ::testing::TestWithParam<PortRow>
    {
        protected:
            /** @brief Builds the head once for every row of the suite. */
            static void SetUpTestSuite()
            {
                head_ = build_row23();
            }

            /** @brief Frees the head after the last row. */
            static void TearDownTestSuite()
            {
                head_.reset();
            }

            /** @brief Holds the head every row reads; null when create refused it. */
            static std::unique_ptr<DetectHead> head_;
    };

    std::unique_ptr<DetectHead> Small_BlockDetectHeadPorts::head_;

    /** @brief Holds one option set and the ports a head built with it publishes. */
    struct OptionRow
    {
        std::string name;                      /**< The row's name. */
        DetectOptions options;                 /**< The options the head is built with. */
        std::size_t port_count;                /**< The number of ports. */
        std::vector<std::size_t> shape_at;     /**< The ports whose shape is checked. */
        std::vector<Shape> shapes;             /**< Their shapes. */
        std::vector<std::size_t> name_at;      /**< The ports whose name is checked. */
        std::vector<std::string> names;        /**< Their names. */
    };

    /** @brief Runs one option set. */
    class Small_BlockDetectHeadOptions : public ::testing::TestWithParam<OptionRow>
    {

    };

    /** @brief Returns the options of a head with wide boxes and both copies. */
    DetectOptions wide_options()
    {
        DetectOptions options;
        options.reg_max = K_REG_MAX;

        return options;
    }

    /** @brief Returns the options of a head with one copy. */
    DetectOptions single_copy_options()
    {
        DetectOptions options;
        options.end2end = false;

        return options;
    }

    /** @brief Lists the four option sets and the ports each publishes. */
    std::vector<OptionRow> option_rows()
    {
        return {
            {"Defaults", DetectOptions(), 4 * K_LEVELS, {0}, {Shape{80, 80, 4}}, {}, {}},
            {"WideBoxes", wide_options(), 4 * K_LEVELS, {0, 2 * K_LEVELS, K_LEVELS},
             {Shape{80, 80, 4 * K_REG_MAX}, Shape{80, 80, 4 * K_REG_MAX}, Shape{80, 80, K_CLASSES}},
             {}, {}},
            {"SingleCopy", single_copy_options(), 2 * K_LEVELS, {}, {}, {0, 2 * K_LEVELS - 1},
             {"one2many.level0.boxes", "one2many.level2.scores"}},
            {"PlainSingleCopy", plain_single_copy_options(), 2 * K_LEVELS,
             {0, 2 * K_LEVELS - 1}, {Shape{80, 80, 4 * K_REG_MAX}, Shape{20, 20, K_CLASSES}}, {},
             {}},
        };
    }

    /** @brief Holds one create refusal of the levels or the classes. */
    struct LevelRow
    {
        std::string name; /**< The row's name. */
    };

    /** @brief Holds one value the factory refuses for a key, and the words its message names. */
    struct BadValueRow
    {
        std::string name;       /**< The row's name. */
        std::string attributes; /**< The row's attributes. */
        std::string key;        /**< The key the message names. */
        std::string reason;     /**< The reason the message gives. */
        std::string extra;      /**< A further word of the message, or empty. */
    };

    /** @brief Runs one refused "reg_max". */
    class Small_BlockDetectHeadRegMax : public ::testing::TestWithParam<BadValueRow>
    {

    };

    /** @brief Runs one refused "class_branch". */
    class Small_BlockDetectHeadClassBranch : public ::testing::TestWithParam<BadValueRow>
    {

    };

    /** @brief Runs one class-branch style over the bias initialisation. */
    class Small_BlockDetectHeadBiases : public ::testing::TestWithParam<ClassBranchStyle>
    {

    };

    /** @brief Names a bad-value row by its name field. */
    std::string bad_value_name(const ::testing::TestParamInfo<BadValueRow>& info)
    {
        return info.param.name;
    }

    /** @brief Names an option row by its name field. */
    std::string option_name(const ::testing::TestParamInfo<OptionRow>& info)
    {
        return info.param.name;
    }

    /** @brief Names a port row by its name field. */
    std::string port_name_of(const ::testing::TestParamInfo<PortRow>& info)
    {
        return info.param.name;
    }

    /** @brief Names a class-branch style. */
    std::string style_name(const ::testing::TestParamInfo<ClassBranchStyle>& info)
    {
        return info.param == ClassBranchStyle::Depthwise ? "Depthwise" : "Plain";
    }
} // namespace

/** @brief Tests that row 23 reads three levels and publishes twelve ports. */
TEST(Small_BlockDetectHead, Row23ReadsThreeLevelsAndPublishesTwelvePorts)
{
    const std::unique_ptr<DetectHead> head = make_row23();
    ASSERT_NE(head, nullptr);

    EXPECT_EQ(head->input_count(), K_LEVELS);
    EXPECT_EQ(head->ports().size(), 4 * K_LEVELS);
}

/**
 * @brief Tests that a port of row 23 carries the name and the shape forward.sacp records for
 *        it; MANIFEST.json of the registration fixtures, from line 9856.
 */
TEST_P(Small_BlockDetectHeadPorts, APortCarriesItsFixtureNameAndShape)
{
    const PortRow& row = GetParam();
    ASSERT_NE(head_, nullptr);
    ASSERT_LT(row.index, head_->ports().size());

    EXPECT_EQ(head_->ports()[row.index].name, row.port_name);
    EXPECT_EQ(head_->ports()[row.index].shape, row.shape);
}

INSTANTIATE_TEST_SUITE_P(Rows, Small_BlockDetectHeadPorts, ::testing::ValuesIn(port_rows()),
                         port_name_of);

/**
 * @brief Tests that a traced head yields the 204 records MANIFEST.json lists under model.23,
 *        records 390 to 593: names, NHWC shapes and order.
 */
TEST(Small_BlockDetectHead, StateMatchesManifestRecords390To593)
{
    const std::unique_ptr<DetectHead> head = make_row23();
    ASSERT_NE(head, nullptr);
    FeatureGraph traced;
    BufferUpdateList list;
    std::vector<Graph::ValueID> outputs;
    SA_EXPECT_OK(head->forward(Trace::training(traced.builder, list), traced.inputs(), outputs));

    // MANIFEST.json of the registration fixtures, weights.sushi tensors, from line 6485.
    std::vector<std::string> expected;
    append_box_branches(expected, "model.23.cv2");
    append_class_branches(expected, "model.23.cv3");
    append_box_branches(expected, "model.23.one2one_cv2");
    append_class_branches(expected, "model.23.one2one_cv3");
    ASSERT_EQ(expected.size(), 204u);

    EXPECT_EQ(walk(*head, "model.23"), expected);
}

/** @brief Tests that a training trace detaches each feature once, in level order. */
TEST(Small_BlockDetectHead, ATrainingTraceDetachesEachFeatureOnce)
{
    const std::unique_ptr<DetectHead> head = make_row23();
    ASSERT_NE(head, nullptr);
    FeatureGraph traced;
    BufferUpdateList list;
    std::vector<Graph::ValueID> outputs;
    SA_EXPECT_OK(head->forward(Trace::training(traced.builder, list), traced.inputs(), outputs));

    EXPECT_EQ(detached_inputs(traced.graph), traced.features);
}

/** @brief Tests that a training trace appends twelve values of the ports' batched shapes. */
TEST(Small_BlockDetectHead, ATrainingTraceAppendsOneValueOfEachPortsShape)
{
    const std::unique_ptr<DetectHead> head = make_row23();
    ASSERT_NE(head, nullptr);
    FeatureGraph traced;
    BufferUpdateList list;
    std::vector<Graph::ValueID> outputs;
    SA_EXPECT_OK(head->forward(Trace::training(traced.builder, list), traced.inputs(), outputs));

    EXPECT_EQ(shapes_of(traced.graph, outputs), traced_port_shapes(*head));
}

/** @brief Tests that an evaluation trace emits no STOP_GRADIENT and no running-statistics norm. */
TEST(Small_BlockDetectHead, AnEvaluationTraceEmitsNoStopGradient)
{
    const std::unique_ptr<DetectHead> head = make_row23();
    ASSERT_NE(head, nullptr);
    FeatureGraph traced;
    std::vector<Graph::ValueID> outputs;
    SA_EXPECT_OK(head->forward(Trace::evaluation(traced.builder), traced.inputs(), outputs));

    EXPECT_EQ(count_nodes(traced.graph, Ops::K_STOP_GRADIENT.id), 0u);
    EXPECT_EQ(count_nodes(traced.graph, Ops::K_BATCHNORM2D.id), 0u);
    EXPECT_EQ(outputs.size(), 4 * K_LEVELS);
}

/**
 * @brief Tests the widths over a first level of 128 channels and 3 classes: a box branch of
 *        128 / 4 = 32 and a class branch of max(128, 3) = 128, read by state name.
 */
TEST(Small_BlockDetectHead, TheFirstLevelSetsTheBranchWidths)
{
    const std::vector<Shape> levels = {Shape{4, 4, 128}};
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<DetectHead> head,
                            DetectHead::create(span_of(levels), 3, K_YOLO_NORM));
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueID x = builder.input(Shape{K_BATCH, 4, 4, 128}, "x");
    std::vector<Graph::ValueID> outputs;
    SA_EXPECT_OK(head->forward(Trace::evaluation(builder), NN::ValueSpan(&x, 1), outputs));

    const std::vector<std::string> state = walk(*head, "");
    EXPECT_EQ(state.size(), 68u);
    EXPECT_TRUE(contains(state, record("cv2.0.0.conv.weight", Shape{32, 3, 3, 128},
                                       StateKind::Parameter)));
    EXPECT_TRUE(contains(state, record("cv2.0.2.weight", Shape{4, 1, 1, 32},
                                       StateKind::Parameter)));
    EXPECT_TRUE(contains(state, record("cv3.0.0.1.conv.weight", Shape{128, 1, 1, 128},
                                       StateKind::Parameter)));
    EXPECT_TRUE(contains(state, record("cv3.0.2.weight", Shape{3, 1, 1, 128},
                                       StateKind::Parameter)));
    EXPECT_EQ(head->ports()[1].shape, (Shape{4, 4, 3}));
}

/** @brief Tests that forward refuses two values for three levels and appends nothing. */
TEST(Small_BlockDetectHead, ForwardRefusesAWrongInputCount)
{
    const std::unique_ptr<DetectHead> head = make_row23();
    ASSERT_NE(head, nullptr);
    FeatureGraph traced;
    std::vector<Graph::ValueID> outputs;

    SA_EXPECT_FAILURE(head->forward(Trace::evaluation(traced.builder),
                                    NN::ValueSpan(traced.features.data(), 2), outputs),
                      StatusCode::InvalidDescription, "input count is 3", "given is 2");
    EXPECT_TRUE(outputs.empty());
}

/** @brief Tests that create refuses a head with no level. */
TEST(Small_BlockDetectHead, CreateRefusesNoLevel)
{
    const std::vector<Shape> none;

    SA_EXPECT_FAILURE(DetectHead::create(span_of(none), K_CLASSES, K_YOLO_NORM),
                      StatusCode::InvalidDescription, "at least one feature level");
}

/** @brief Tests that create refuses a level that is not rank 3, naming it. */
TEST(Small_BlockDetectHead, CreateRefusesALevelOfRankTwo)
{
    const std::vector<Shape> flat = {Shape{80, 80, 64}, Shape{40, 128}};

    SA_EXPECT_FAILURE(DetectHead::create(span_of(flat), K_CLASSES, K_YOLO_NORM),
                      StatusCode::InvalidDescription, "level 1", "[40, 128]", "rank-3");
}

/** @brief Tests that create refuses a class count of zero. */
TEST(Small_BlockDetectHead, CreateRefusesZeroClasses)
{
    const std::vector<Shape> levels = level_shapes(K_SIDES, K_WIDTHS);

    SA_EXPECT_FAILURE(DetectHead::create(span_of(levels), 0, K_YOLO_NORM),
                      StatusCode::InvalidDescription, "at least one class", "0");
}

/**
 * @brief Tests that make_detect_node builds row 23 from its file row: twelve ports, the first
 *        and the last of them, the norm constants on every traced norm, and the walk size.
 */
TEST(Small_BlockDetectHead, TheFactoryBuildsRow23)
{
    const std::vector<Shape> levels = level_shapes(K_SIDES, K_WIDTHS);
    const Config::Value attributes =
        attributes_of(R"({"classes": 80, "bn_epsilon": 0.001, "bn_momentum": 0.03})");
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<NN::Node> node,
                            Vision::Blocks::make_detect_node(span_of(levels), attributes,
                                                             K_CONTEXT));
    EXPECT_EQ(node->input_count(), K_LEVELS);
    ASSERT_EQ(node->ports().size(), 4 * K_LEVELS);
    EXPECT_EQ(node->ports()[0].name, "one2many.level0.boxes");
    EXPECT_EQ(node->ports()[0].shape, (Shape{80, 80, 4}));
    EXPECT_EQ(node->ports()[11].name, "one2one.level2.scores");
    EXPECT_EQ(node->ports()[11].shape, (Shape{20, 20, K_CLASSES}));

    FeatureGraph traced;
    BufferUpdateList list;
    std::vector<Graph::ValueID> outputs;
    SA_EXPECT_OK(node->forward(Trace::training(traced.builder, list), traced.inputs(), outputs));

    expect_norm_options(traced.graph, list, K_YOLO_NORM);
    EXPECT_EQ(walk(*node, "model.23").size(), 204u);
}

/** @brief Tests that the factory refuses a key the row does not take. */
TEST(Small_BlockDetectHead, TheFactoryRefusesAnUnknownKey)
{
    const std::vector<Shape> levels = level_shapes(K_SIDES, K_WIDTHS);
    const Config::Value unknown = attributes_of(R"({"classes": 80, "filters": 64})");

    SA_EXPECT_FAILURE(Vision::Blocks::make_detect_node(span_of(levels), unknown, K_CONTEXT),
                      StatusCode::InvalidDescription, "layer 23 'detect'", "'filters'");
}

/** @brief Tests that the factory refuses a row without its class count. */
TEST(Small_BlockDetectHead, TheFactoryRefusesAnAbsentClassCount)
{
    const std::vector<Shape> levels = level_shapes(K_SIDES, K_WIDTHS);
    const Config::Value no_classes = attributes_of(R"({"bn_epsilon": 0.001})");

    SA_EXPECT_FAILURE(Vision::Blocks::make_detect_node(span_of(levels), no_classes, K_CONTEXT),
                      StatusCode::InvalidDescription, "'classes'", "absent");
}

/** @brief Tests that the factory refuses a row with no feature level. */
TEST(Small_BlockDetectHead, TheFactoryRefusesNoLevel)
{
    const std::vector<Shape> none;
    const Config::Value row = attributes_of(R"({"classes": 80})");

    SA_EXPECT_FAILURE(Vision::Blocks::make_detect_node(span_of(none), row, K_CONTEXT),
                      StatusCode::InvalidDescription, "layer 23 'detect'",
                      "at least one feature level");
}

/**
 * @brief Tests that a wide-box, one-copy, plain-class head over 64, 128 and 256 channels and 80
 *        classes carries the 72 records of a 64-wide box branch emitting 4 * 16 channels.
 */
TEST(Small_BlockDetectHead, AWideBoxPlainSingleCopyHeadCarriesItsRecords)
{
    const std::unique_ptr<DetectHead> head = make_head(plain_single_copy_options());
    ASSERT_NE(head, nullptr);
    FeatureGraph traced;
    BufferUpdateList list;
    std::vector<Graph::ValueID> outputs;
    SA_EXPECT_OK(head->forward(Trace::training(traced.builder, list), traced.inputs(), outputs));

    std::vector<std::string> expected;
    append_wide_box_branches(expected, "model.22.cv2");
    append_plain_class_branches(expected, "model.22.cv3");
    ASSERT_EQ(expected.size(), 72u);

    EXPECT_EQ(walk(*head, "model.22"), expected);
}

/** @brief Tests the port count and the channel widths of one combination of the options. */
TEST_P(Small_BlockDetectHeadOptions, ThePortCountFollowsTheCopiesAndTheBoxWidthFollowsRegMax)
{
    const OptionRow& row = GetParam();
    const std::unique_ptr<DetectHead> head = make_head(row.options);
    ASSERT_NE(head, nullptr);

    EXPECT_EQ(head->ports().size(), row.port_count);
    EXPECT_EQ(port_shapes_at(*head, row.shape_at), row.shapes);
    EXPECT_EQ(port_names_at(*head, row.name_at), row.names);
}

INSTANTIATE_TEST_SUITE_P(Rows, Small_BlockDetectHeadOptions, ::testing::ValuesIn(option_rows()),
                         option_name);

/** @brief Tests that the factory reads the three structural keys: the ports of the same head. */
TEST(Small_BlockDetectHead, TheFactoryBuildsAWideBoxPlainSingleCopyRow)
{
    const std::vector<Shape> levels = level_shapes(K_SIDES, K_WIDTHS);
    const Config::Value attributes = attributes_of(
        R"({"classes": 80, "reg_max": 16, "end2end": false, "class_branch": "plain"})");
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<NN::Node> node,
                            Vision::Blocks::make_detect_node(span_of(levels), attributes,
                                                             K_CONTEXT));

    EXPECT_EQ(node->ports().size(), 2 * K_LEVELS);
    EXPECT_EQ(node->ports()[0].shape, (Shape{80, 80, 4 * K_REG_MAX}));
}

/** @brief Tests that create refuses a reg_max below one. */
TEST(Small_BlockDetectHead, CreateRefusesARegMaxBelowOne)
{
    const std::vector<Shape> levels = level_shapes(K_SIDES, K_WIDTHS);
    DetectOptions no_bins;
    no_bins.reg_max = 0;

    SA_EXPECT_FAILURE(DetectHead::create(span_of(levels), K_CLASSES, K_YOLO_NORM, no_bins),
                      StatusCode::InvalidDescription, "reg_max of at least one", "0");
}

/** @brief Tests that create refuses strides that are not one per level. */
TEST(Small_BlockDetectHead, CreateRefusesStridesOfTheWrongCount)
{
    const std::vector<Shape> levels = level_shapes(K_SIDES, K_WIDTHS);
    DetectOptions short_strides;
    short_strides.strides = {8.0, 16.0};

    SA_EXPECT_FAILURE(
        DetectHead::create(span_of(levels), K_CLASSES, K_YOLO_NORM, short_strides),
        StatusCode::InvalidDescription, "one stride per level", "2 strides", "3 levels");
}

/** @brief Tests that create refuses a stride that is not positive. */
TEST(Small_BlockDetectHead, CreateRefusesAStrideThatIsNotPositive)
{
    const std::vector<Shape> levels = level_shapes(K_SIDES, K_WIDTHS);
    DetectOptions flat_stride;
    flat_stride.strides = {8.0, 0.0, 32.0};

    SA_EXPECT_FAILURE(DetectHead::create(span_of(levels), K_CLASSES, K_YOLO_NORM, flat_stride),
                      StatusCode::InvalidDescription, "positive finite strides");
}

/** @brief Tests that the factory refuses a reg_max that is not a positive whole number. */
TEST_P(Small_BlockDetectHeadRegMax, ARegMaxThatIsNotAPositiveWholeNumberIsRefused)
{
    const std::vector<Shape> levels = level_shapes(K_SIDES, K_WIDTHS);
    const BadValueRow& row = GetParam();
    const Config::Value attributes = attributes_of(row.attributes);

    SA_EXPECT_FAILURE(Vision::Blocks::make_detect_node(span_of(levels), attributes, K_CONTEXT),
                      StatusCode::InvalidDescription, row.key, row.reason, row.extra);
}

INSTANTIATE_TEST_SUITE_P(
    Rows, Small_BlockDetectHeadRegMax,
    ::testing::Values(BadValueRow{"Zero", R"({"classes": 80, "reg_max": 0})", "'reg_max'",
                                  "positive", "layer 23 'detect'"},
                      BadValueRow{"Text", R"({"classes": 80, "reg_max": "16"})", "'reg_max'",
                                  "whole number", ""}),
    bad_value_name);

/** @brief Tests that the factory refuses an end2end flag given as text. */
TEST(Small_BlockDetectHead, TheFactoryRefusesAnEnd2EndFlagThatIsText)
{
    const std::vector<Shape> levels = level_shapes(K_SIDES, K_WIDTHS);
    const Config::Value text_flag = attributes_of(R"({"classes": 80, "end2end": "yes"})");

    SA_EXPECT_FAILURE(Vision::Blocks::make_detect_node(span_of(levels), text_flag, K_CONTEXT),
                      StatusCode::InvalidDescription, "'end2end'", "true or false");
}

/** @brief Tests that the factory refuses a class_branch that names no style, or is a number. */
TEST_P(Small_BlockDetectHeadClassBranch, AClassBranchThatNamesNoStyleIsRefused)
{
    const std::vector<Shape> levels = level_shapes(K_SIDES, K_WIDTHS);
    const BadValueRow& row = GetParam();
    const Config::Value attributes = attributes_of(row.attributes);

    SA_EXPECT_FAILURE(Vision::Blocks::make_detect_node(span_of(levels), attributes, K_CONTEXT),
                      StatusCode::InvalidDescription, row.key, row.reason, row.extra);
}

INSTANTIATE_TEST_SUITE_P(
    Rows, Small_BlockDetectHeadClassBranch,
    ::testing::Values(BadValueRow{"UnknownName", R"({"classes": 80, "class_branch": "wide"})",
                                  "'class_branch'", "depthwise", "plain"},
                      BadValueRow{"Number", R"({"classes": 80, "class_branch": 3})",
                                  "'class_branch'", "depthwise", "plain"}),
    bad_value_name);

/** @brief Tests that without strides every last bias starts from zero, in both forms. */
TEST_P(Small_BlockDetectHeadBiases, WithoutStridesEveryLastBiasStartsFromZero)
{
    DetectOptions options;
    options.class_branch = GetParam();
    const std::unique_ptr<DetectHead> head = make_head(options);
    ASSERT_NE(head, nullptr);

    // A constant initialiser stores the value it was given, so zero compares exactly.
    EXPECT_EQ(last_bias_constants(*head), all_last_biases(0.0));
}

/**
 * @brief Tests that strides set the last box bias to 1 and each last class bias to
 *        log(5 / classes / (640 / stride)^2), on both copies, in one form.
 */
TEST_P(Small_BlockDetectHeadBiases, StridesSetTheLastBiasesOfBothCopies)
{
    DetectOptions options;
    options.class_branch = GetParam();
    options.strides = K_STRIDES;
    const std::unique_ptr<DetectHead> head = make_head(options);
    ASSERT_NE(head, nullptr);

    EXPECT_TRUE(strided_biases_within_bound(last_bias_constants(*head)));
}

INSTANTIATE_TEST_SUITE_P(Styles, Small_BlockDetectHeadBiases,
                         ::testing::Values(ClassBranchStyle::Depthwise, ClassBranchStyle::Plain),
                         style_name);
