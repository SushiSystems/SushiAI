/**************************************************************************/
/* test_import.cpp                                                   */
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
 * @brief Tests the ONNX importer: its refusals, its rows and weights, and the forward pass of the
 *        fixture against the output onnx.reference computed.
 * @author Mustafa Garip
 */

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <initializer_list>
#include <iterator>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>
#include <gtest/gtest.h>
#include <SushiBLAS/engine/engine.hpp>
#include <SushiBLAS/tensor/tensor.hpp>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/allocation/allocation.hpp>
#include <SushiAI/config/model_config.hpp>
#include <SushiAI/config/value.hpp>
#include <SushiAI/dtype/dtype.hpp>
#include <SushiAI/engine_fixture/engine_test.hpp>
#include <SushiAI/execution/execution_backend.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/graph/tensor_pool.hpp>
#include <SushiAI/interchange/onnx_graph.hpp>
#include <SushiAI/interchange/tensor_bytes.hpp>
#include <SushiAI/io/imported_model.hpp>
#include <SushiAI/nn/layer_registry.hpp>
#include <SushiAI/nn/model_graph.hpp>
#include <SushiAI/nn/model_state.hpp>
#include <SushiAI/nn/node_registry.hpp>
#include <SushiAI/shape/shape.hpp>
#include <SushiAI/train/evaluation_step.hpp>
#include <SushiAI/train/steps.hpp>
#include <SushiAI/vision/importers/onnx_import.hpp>
#include <support/tolerance.hpp>

using namespace SushiAI;

namespace
{
    /** @brief Holds the channels of the fixture's output before it is flattened. */
    constexpr std::int64_t K_OUTPUT_CHANNELS = 2;

    /** @brief Holds the rows and the columns of the fixture's output before it is flattened. */
    constexpr std::int64_t K_OUTPUT_SIDE = 8;

    /**
     * @brief Holds the rounded operations on the longest path to one output of the fixture.
     *
     * The 3x3 convolution sums 27 products, the batch normalisation adds 4 operations (subtract,
     * divide, scale, shift), SiLU adds 8 (an exponential, an addition, a division and a
     * product, each counted twice for the library's own rounding), and the 1x1 convolution
     * sums 7 products: 27 + 4 + 8 + 7 = 46.
     */
    constexpr std::size_t K_ROUNDED_OPERATIONS = 46;

    /** @brief Returns the folder of the ONNX fixture and its expected output. */
    std::filesystem::path fixture_folder()
    {
        return std::filesystem::path(SA_REPOSITORY_ROOT) / "tests" / "vision" / "importers" /
               "fixtures" / "onnx";
    }

    /** @brief Holds the input and the output the reference evaluator gave, in ONNX's layout. */
    struct Expected
    {
        std::vector<float> input;  /**< The input, [N, C, H, W] row-major. */
        std::vector<float> output; /**< The output, [N, F] row-major. */
    };

    /** @brief Returns the values of the tensor object @p key of @p record. */
    Result<std::vector<float>> read_values(const Config::Value& record, std::string_view key)
    {
        SA_TRY_ASSIGN(const Config::Value& tensor, record.at(key, "the expected file"));
        SA_TRY_ASSIGN(const Config::Value& list, tensor.at("values", key));
        SA_TRY_ASSIGN(const std::vector<Config::Value>& entries, list.as_array(key));
        std::vector<float> values;
        for (const Config::Value& entry : entries)
        {
            SA_TRY_ASSIGN(const double value, entry.as_real(key));
            values.push_back(static_cast<float>(value));
        }
        return values;
    }

    /** @brief Reads the expected input and output of the fixture. */
    Result<Expected> read_expected()
    {
        SA_TRY_ASSIGN(const Config::Value record,
                      Config::parse_json_file(fixture_folder() / "silu_resize.expected.json"));
        Expected expected;
        SA_TRY_ASSIGN(expected.input, read_values(record, "input"));
        SA_TRY_ASSIGN(expected.output, read_values(record, "output"));
        return expected;
    }

    /** @brief Imports the fixture file. */
    Result<IO::ImportedModel> import_fixture()
    {
        const std::filesystem::path files[] = {fixture_folder() / "silu_resize.onnx"};
        return Vision::Importers::import_onnx_files(
            SushiRuntime::span<const std::filesystem::path>(files, 1));
    }

    /** @brief Returns @p values of a [1, C, H, W] tensor as [1, H, W, C]. */
    std::vector<float> channels_last(const std::vector<float>& values, std::int64_t channels,
                                     std::int64_t height, std::int64_t width)
    {
        std::vector<float> moved(values.size());
        for (std::int64_t c = 0; c < channels; ++c)
        {
            for (std::int64_t h = 0; h < height; ++h)
            {
                for (std::int64_t w = 0; w < width; ++w)
                    moved[static_cast<std::size_t>((h * width + w) * channels + c)] =
                        values[static_cast<std::size_t>((c * height + h) * width + w)];
            }
        }
        return moved;
    }

    /**
     * @brief Builds @p imported, loads its weights, runs it over @p input and returns "output".
     * @param input The sample, [1, H, W, C] row-major.
     */
    Result<std::vector<float>> run_forward(ExecutionBackend& backend,
                                           const IO::ImportedModel& imported,
                                           const std::vector<float>& input)
    {
        SushiBLAS::Engine& engine = backend.engine();
        const Shape& sample = imported.model.input_shape;
        const Shape batch{1, sample[0], sample[1], sample[2]};

        NN::NodeRegistry registry;
        SA_TRY(NN::register_layers(registry));
        SA_TRY_ASSIGN(NN::ModelGraph graph, NN::build_model(registry, imported.model));
        SA_TRY_ASSIGN(const Train::TracedStep traced,
                      Train::build_evaluation_graph(graph, batch));
        SA_TRY_ASSIGN(NN::ModelState state, NN::ModelState::create(engine, graph));

        for (const IO::ImportedTensor& weight : imported.weights)
        {
            SA_TRY_ASSIGN(SushiBLAS::Tensor* const target, state.tensor(weight.record.name));
            SA_FAIL_IF(static_cast<std::size_t>(target->num_elements) * sizeof(float) !=
                           weight.payload.size(),
                       StatusCode::Incompatible, "the weight '{}' does not fit its state entry",
                       weight.record.name);
            std::memcpy(target->data_as<float>(), weight.payload.data(), weight.payload.size());
        }

        Graph::TensorPool pool(engine);
        SA_TRY_ASSIGN(const std::unique_ptr<Train::EvaluationStep> step,
                      Train::EvaluationStep::make(backend, traced, state, pool));
        SA_TRY_ASSIGN(SushiBLAS::Tensor image, allocate_tensor(engine, batch));
        std::memcpy(image.data_as<float>(), input.data(), input.size() * sizeof(float));
        SA_TRY(step->run(image));

        SA_TRY_ASSIGN(const SushiBLAS::Tensor* const published, step->output("output"));
        const float* const values = published->data_as<float>();
        return std::vector<float>(values, values + published->num_elements);
    }

    /** @brief Returns a float32 graph input of the stated extents. */
    Interchange::OnnxValueInfo make_input(const std::string& name,
                                          const std::vector<std::int64_t>& extents)
    {
        Interchange::OnnxValueInfo input;
        input.name = name;
        input.element_type = Interchange::OnnxDataType::Float;
        input.has_shape = true;
        for (const std::int64_t extent : extents)
        {
            Interchange::OnnxDimension dimension;
            dimension.value = extent;
            input.dimensions.push_back(dimension);
        }
        return input;
    }

    /** @brief Returns a node of @p op_type over @p inputs, writing @p outputs. */
    Interchange::OnnxNode make_node(const std::string& op_type, const std::string& name,
                                    std::vector<std::string> inputs,
                                    std::vector<std::string> outputs)
    {
        Interchange::OnnxNode node;
        node.op_type = op_type;
        node.name = name;
        node.inputs = std::move(inputs);
        node.outputs = std::move(outputs);
        return node;
    }

    /** @brief Returns an integer attribute. */
    Interchange::OnnxAttribute make_int(const std::string& name, std::int64_t value)
    {
        Interchange::OnnxAttribute attribute;
        attribute.name = name;
        attribute.type = Interchange::OnnxAttributeType::Int;
        attribute.integer = value;
        return attribute;
    }

    /** @brief Returns an integer list attribute. */
    Interchange::OnnxAttribute make_ints(const std::string& name, std::vector<std::int64_t> values)
    {
        Interchange::OnnxAttribute attribute;
        attribute.name = name;
        attribute.type = Interchange::OnnxAttributeType::Ints;
        attribute.integers = std::move(values);
        return attribute;
    }

    /** @brief Returns the whole numbers @p key of @p row holds: one, or the entries of an array. */
    std::vector<std::int64_t> integers_of(const Config::LayerSpec& row, std::string_view key)
    {
        std::vector<std::int64_t> values;
        if (!row.attributes.has(key))
        {
            ADD_FAILURE() << "the row '" << row.type << "' has no '" << key << "'";
            return values;
        }
        const Config::Value& stated = row.attributes.at(key, "row").value();
        if (stated.kind() == Config::Kind::Int)
        {
            values.push_back(stated.as_int("row").value());
            return values;
        }
        for (const Config::Value& entry : stated.as_array("row").value())
            values.push_back(entry.as_int("row").value());
        return values;
    }

    /** @brief Returns the floats of the weight named @p name; empty when there is none. */
    std::vector<float> floats_of(const IO::ImportedModel& imported, std::string_view name)
    {
        std::vector<float> values;
        for (const IO::ImportedTensor& tensor : imported.weights)
        {
            if (tensor.record.name != name)
                continue;

            values.resize(tensor.payload.size() / sizeof(float));
            std::memcpy(values.data(), tensor.payload.data(), tensor.payload.size());
        }
        return values;
    }

    /** @brief Returns the shape of the weight named @p name; empty when there is none. */
    Shape shape_of_weight(const IO::ImportedModel& imported, std::string_view name)
    {
        for (const IO::ImportedTensor& tensor : imported.weights)
        {
            if (tensor.record.name == name)
                return tensor.record.shape;
        }
        ADD_FAILURE() << "no weight named '" << name << "'";
        return Shape{};
    }

    /** @brief Returns a tensor of @p type whose elements are @p elements, copied byte for byte. */
    template<typename Element>
    Interchange::OnnxTensor make_tensor(const std::string& name, Interchange::OnnxDataType type,
                                        std::vector<std::int64_t> extents,
                                        const std::vector<Element>& elements)
    {
        Interchange::OnnxTensor tensor;
        tensor.name = name;
        tensor.data_type = type;
        tensor.dimensions = std::move(extents);
        std::vector<std::byte> bytes(elements.size() * sizeof(Element));
        std::memcpy(bytes.data(), elements.data(), bytes.size());
        tensor.data = Interchange::TensorBytes::owning(std::move(bytes));
        return tensor;
    }

    /** @brief Returns a graph of opset 17 reading the float image "x" of [1, C, H, W]. */
    Interchange::OnnxGraph make_graph(const std::vector<std::int64_t>& extents)
    {
        Interchange::OnnxGraph graph;
        graph.name = "test";
        graph.opsets.push_back(Interchange::OnnxOpset{"", 17});
        graph.inputs.push_back(make_input("x", extents));
        return graph;
    }

    /** @brief Returns a named output of the graph. */
    Interchange::OnnxValueInfo make_output(const std::string& name)
    {
        Interchange::OnnxValueInfo output;
        output.name = name;
        output.element_type = Interchange::OnnxDataType::Float;
        return output;
    }

    /** @brief Expects row @p index of @p model to be a slice_channels row of the given window. */
    void expect_slice_row(const Config::ModelSpec& model, std::size_t index, std::int64_t offset,
                          std::int64_t count)
    {
        ASSERT_LT(index, model.layers.size());
        const Config::LayerSpec& row = model.layers[index];
        EXPECT_EQ(row.type, "slice_channels");
        SA_ASSERT_OK_AND_ASSIGN(const Config::Value& first, row.attributes.at("offset", "row"));
        SA_ASSERT_OK_AND_ASSIGN(const std::int64_t found_offset, first.as_int("offset"));
        EXPECT_EQ(found_offset, offset);
        SA_ASSERT_OK_AND_ASSIGN(const Config::Value& kept, row.attributes.at("count", "row"));
        SA_ASSERT_OK_AND_ASSIGN(const std::int64_t found_count, kept.as_int("count"));
        EXPECT_EQ(found_count, count);
    }

    /** @brief Expects the rows of @p model to have exactly the types of @p types, in order. */
    void expect_row_types(const Config::ModelSpec& model,
                          std::initializer_list<const char*> types)
    {
        ASSERT_EQ(model.layers.size(), types.size());
        std::size_t at = 0;
        for (const char* const type : types)
        {
            EXPECT_EQ(model.layers[at].type, type) << "row " << at;
            ++at;
        }
    }

    /** @brief Expects the weights of @p imported to have the names of @p names, in order. */
    void expect_weight_names(const IO::ImportedModel& imported,
                             std::initializer_list<const char*> names)
    {
        ASSERT_EQ(imported.weights.size(), names.size());
        std::size_t at = 0;
        for (const char* const name : names)
        {
            EXPECT_EQ(imported.weights[at].record.name, name);
            ++at;
        }
    }

    /**
     * @brief Expects every value of @p output, listed (H, W, C), to equal the value of
     *        @p reference, listed (C, H, W), for the same channel and pixel.
     *
     * The bound of a value is K_ROUNDED_OPERATIONS times the unit roundoff and the largest
     * magnitude of the reference; the helper's threshold multiplies it.
     */
    void expect_matches_reference(const std::vector<float>& output,
                                  const std::vector<float>& reference)
    {
        ASSERT_EQ(output.size(), reference.size());
        double magnitude = 0.0;
        for (const float value : reference)
            magnitude = std::max(magnitude, std::fabs(static_cast<double>(value)));
        const double bound = static_cast<double>(K_ROUNDED_OPERATIONS) *
                             TestSupport::unit_roundoff<float>() * magnitude;
        for (std::int64_t c = 0; c < K_OUTPUT_CHANNELS; ++c)
        {
            for (std::int64_t pixel = 0; pixel < K_OUTPUT_SIDE * K_OUTPUT_SIDE; ++pixel)
            {
                const std::size_t ours = static_cast<std::size_t>(pixel * K_OUTPUT_CHANNELS + c);
                const std::size_t theirs =
                    static_cast<std::size_t>(c * K_OUTPUT_SIDE * K_OUTPUT_SIDE + pixel);
                EXPECT_TRUE(TestSupport::is_within(
                    TestSupport::measure_error(output[ours], reference[theirs], bound)))
                    << "channel " << c << ", pixel " << pixel;
            }
        }
    }

    /** @brief Runs the fixture on the engine of the program. */
    class Medium_OnnxImportFixture : public SushiAI::Test::EngineTest
    {
    };
} // namespace

/** @brief Tests that every node no row can say is listed in one failure, the ones after it too. */
TEST(Small_OnnxImport, EveryUnsupportedNodeIsListedTogether)
{
    Interchange::OnnxGraph graph = make_graph({1, 3, 4, 4});
    graph.nodes.push_back(make_node("Relu", "relu", {"x"}, {"a"}));
    graph.nodes.push_back(make_node("Erf", "gauss", {"a"}, {"b"}));
    graph.nodes.push_back(make_node("Pow", "power", {"b", "a"}, {"c"}));
    graph.nodes.push_back(make_node("Where", "choose", {"a", "a", "a"}, {"d"}));
    graph.outputs = {make_output("c"), make_output("d")};

    SA_EXPECT_FAILURE(Vision::Importers::import_onnx_model(graph, "list.onnx"),
                      StatusCode::Unsupported, "list.onnx", "3 nodes", "'gauss' (Erf)",
                      "'power' (Pow)", "'choose' (Where)");
}

namespace
{
    /** @brief Returns the Conv node "grouped" over the weight "w" with @p group groups. */
    Interchange::OnnxNode grouped_conv(std::int64_t group)
    {
        Interchange::OnnxNode conv = make_node("Conv", "grouped", {"x", "w"}, {"y"});
        conv.attributes.push_back(make_int("group", group));
        return conv;
    }
} // namespace

/**
 * @brief Tests that a convolution of two groups writes the key groups and a weight of in / group
 *        channels.
 *
 * Two filters read two channels each through a window of 1 by 2. The weight [o, i, y, x] holds
 * o * 100 + i * 10 + x; the record [o, y, x, i] reads (o0, x0: 0, 10), (o0, x1: 1, 11),
 * (o1, x0: 100, 110) and (o1, x1: 101, 111).
 */
TEST(Small_OnnxImport, AGroupedConvolutionWritesTheGroupsKeyAndTheGroupWeight)
{
    Interchange::OnnxGraph graph = make_graph({1, 4, 1, 2});
    graph.initializers.push_back(
        make_tensor<float>("w", Interchange::OnnxDataType::Float, {2, 2, 1, 2},
                           {0.0f, 1.0f, 10.0f, 11.0f, 100.0f, 101.0f, 110.0f, 111.0f}));
    graph.nodes.push_back(grouped_conv(2));
    graph.outputs = {make_output("y")};

    SA_ASSERT_OK_AND_ASSIGN(const IO::ImportedModel imported,
                            Vision::Importers::import_onnx_model(graph, "group.onnx"));

    expect_row_types(imported.model, {"conv2d"});
    EXPECT_EQ(integers_of(imported.model.layers[0], "groups"), std::vector<std::int64_t>{2});
    EXPECT_EQ(shape_of_weight(imported, "0.weight"), (Shape{2, 1, 2, 2}));
    EXPECT_EQ(floats_of(imported, "0.weight"),
              (std::vector<float>{0.0f, 10.0f, 1.0f, 11.0f, 100.0f, 110.0f, 101.0f, 111.0f}));
}

/**
 * @brief Tests that a convolution whose group equals both channel counts stores [kh, kw, C].
 *
 * Two channels, two filters of one channel each, a window of 1 by 2: channel 0 holds the taps 10,
 * 11 and channel 1 the taps 20, 21, so tap 0 lists (10, 20) and tap 1 lists (11, 21).
 */
TEST(Small_OnnxImport, ADepthwiseConvolutionStoresItsTapsKernelHeightWidthChannel)
{
    Interchange::OnnxGraph graph = make_graph({1, 2, 1, 3});
    graph.initializers.push_back(make_tensor<float>("w", Interchange::OnnxDataType::Float,
                                                    {2, 1, 1, 2}, {10.0f, 11.0f, 20.0f, 21.0f}));
    graph.nodes.push_back(grouped_conv(2));
    graph.outputs = {make_output("y")};

    SA_ASSERT_OK_AND_ASSIGN(const IO::ImportedModel imported,
                            Vision::Importers::import_onnx_model(graph, "depthwise.onnx"));

    EXPECT_EQ(integers_of(imported.model.layers[0], "groups"), std::vector<std::int64_t>{2});
    EXPECT_EQ(shape_of_weight(imported, "0.weight"), (Shape{1, 2, 2}));
    EXPECT_EQ(floats_of(imported, "0.weight"), (std::vector<float>{10.0f, 20.0f, 11.0f, 21.0f}));
}

/** @brief Tests that a convolution of one group states no groups key. */
TEST(Small_OnnxImport, AConvolutionOfOneGroupStatesNoGroupsKey)
{
    Interchange::OnnxGraph graph = make_graph({1, 2, 1, 2});
    graph.initializers.push_back(make_tensor<float>("w", Interchange::OnnxDataType::Float,
                                                    {2, 2, 1, 1}, {1.0f, 2.0f, 3.0f, 4.0f}));
    graph.nodes.push_back(make_node("Conv", "dense", {"x", "w"}, {"y"}));
    graph.outputs = {make_output("y")};

    SA_ASSERT_OK_AND_ASSIGN(const IO::ImportedModel imported,
                            Vision::Importers::import_onnx_model(graph, "dense.onnx"));

    EXPECT_FALSE(imported.model.layers[0].attributes.has("groups"));
}

/** @brief Tests that filters that the groups do not divide are refused. */
TEST(Small_OnnxImport, AGroupThatDoesNotDivideTheFiltersIsRefused)
{
    Interchange::OnnxGraph graph = make_graph({1, 4, 1, 1});
    graph.initializers.push_back(make_tensor<float>("w", Interchange::OnnxDataType::Float,
                                                    {3, 2, 1, 1},
                                                    {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}));
    graph.nodes.push_back(grouped_conv(2));
    graph.outputs = {make_output("y")};

    SA_EXPECT_FAILURE(Vision::Importers::import_onnx_model(graph, "group.onnx"),
                      StatusCode::InvalidDescription, "'grouped' (Conv)", "3 filters",
                      "2 groups");
}

/** @brief Tests that a weight whose channels times the groups differ from the input's is refused. */
TEST(Small_OnnxImport, AWeightThatDoesNotReadTheChannelsPerGroupIsRefused)
{
    Interchange::OnnxGraph graph = make_graph({1, 4, 1, 1});
    graph.initializers.push_back(make_tensor<float>("w", Interchange::OnnxDataType::Float,
                                                    {2, 3, 1, 1},
                                                    {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}));
    graph.nodes.push_back(grouped_conv(2));
    graph.outputs = {make_output("y")};

    SA_EXPECT_FAILURE(Vision::Importers::import_onnx_model(graph, "group.onnx"),
                      StatusCode::InvalidDescription, "'grouped' (Conv)", "4 channels");
}

/** @brief Tests that a depthwise convolution with unequal pads is refused, since it takes none. */
TEST(Small_OnnxImport, ADepthwiseConvolutionWithUnequalPadsIsRefused)
{
    Interchange::OnnxGraph graph = make_graph({1, 2, 3, 3});
    graph.initializers.push_back(make_tensor<float>("w", Interchange::OnnxDataType::Float,
                                                    {2, 1, 1, 1}, {1.0f, 2.0f}));
    Interchange::OnnxNode conv = grouped_conv(2);
    conv.attributes.push_back(make_ints("pads", {0, 0, 1, 1}));
    graph.nodes.push_back(std::move(conv));
    graph.outputs = {make_output("y")};

    SA_EXPECT_FAILURE(Vision::Importers::import_onnx_model(graph, "depthwise.onnx"),
                      StatusCode::Unsupported, "'grouped' (Conv)", "depthwise");
}

/** @brief Holds one operator over two values of one shape, with the row it becomes. */
struct BinaryRow
{
    std::string op_type; /**< The operator the file spells. */
    std::string row;     /**< The type of the row it becomes. */
};

/** @brief Runs each binary operator the importer translates beside Add, Mul included. */
class Small_OnnxImportBinary : public ::testing::TestWithParam<BinaryRow>
{
};

/** @brief Tests that the operator becomes one row that reads its two operands in order. */
TEST_P(Small_OnnxImportBinary, TheOperatorBecomesOneRowReadingItsOperandsInOrder)
{
    const BinaryRow& binary = GetParam();
    Interchange::OnnxGraph graph = make_graph({1, 3, 4, 4});
    graph.nodes.push_back(make_node("Relu", "relu", {"x"}, {"a"}));
    graph.nodes.push_back(make_node(binary.op_type, "combine", {"a", "x"}, {"y"}));
    graph.outputs = {make_output("y")};

    SA_ASSERT_OK_AND_ASSIGN(const IO::ImportedModel imported,
                            Vision::Importers::import_onnx_model(graph, "binary.onnx"));

    ASSERT_EQ(imported.model.layers.size(), 2u);
    EXPECT_EQ(imported.model.layers[1].type, binary.row);
    ASSERT_EQ(imported.model.layers[1].from.size(), 2u);
    EXPECT_EQ(imported.model.layers[1].from[0].node, 0);
    EXPECT_EQ(imported.model.layers[1].from[1].node, Config::K_MODEL_INPUT);
}

/** @brief Tests that two values of different extents are refused, since the row does not broadcast. */
TEST_P(Small_OnnxImportBinary, ValuesOfDifferentExtentsAreRefused)
{
    const BinaryRow& binary = GetParam();
    Interchange::OnnxGraph graph = make_graph({1, 3, 4, 4});
    Interchange::OnnxNode pool = make_node("MaxPool", "pool", {"x"}, {"p"});
    pool.attributes.push_back(make_ints("kernel_shape", {2, 2}));
    pool.attributes.push_back(make_ints("strides", {2, 2}));
    graph.nodes.push_back(std::move(pool));
    graph.nodes.push_back(make_node(binary.op_type, "combine", {"x", "p"}, {"y"}));
    graph.outputs = {make_output("y")};

    SA_EXPECT_FAILURE(Vision::Importers::import_onnx_model(graph, "binary.onnx"),
                      StatusCode::Unsupported, "'combine' (" + binary.op_type + ")",
                      "different extents", binary.row + " row does not broadcast");
}

/** @brief Tests that a constant operand is refused, since a row reads computed values only. */
TEST_P(Small_OnnxImportBinary, AConstantOperandIsRefused)
{
    const BinaryRow& binary = GetParam();
    Interchange::OnnxGraph graph = make_graph({1, 3, 4, 4});
    graph.initializers.push_back(make_tensor<float>("k", Interchange::OnnxDataType::Float, {1},
                                                    {2.0f}));
    graph.nodes.push_back(make_node(binary.op_type, "combine", {"x", "k"}, {"y"}));
    graph.outputs = {make_output("y")};

    SA_EXPECT_FAILURE(Vision::Importers::import_onnx_model(graph, "binary.onnx"),
                      StatusCode::Unsupported, "'combine' (" + binary.op_type + ")", "constant");
}

INSTANTIATE_TEST_SUITE_P(Operators, Small_OnnxImportBinary,
                         ::testing::Values(BinaryRow{"Sub", "sub"}, BinaryRow{"Div", "div"},
                                           BinaryRow{"Mul", "mul"}),
                         [](const ::testing::TestParamInfo<BinaryRow>& info)
                         { return info.param.op_type; });

/** @brief Holds one Softmax axis over a [1, 3, 4, 4] tensor and the key the row states. */
struct SoftmaxRow
{
    std::string name;                /**< The row's name. */
    std::int64_t axis;               /**< The axis the file states. */
    std::int64_t expected;           /**< The row's axis, or -1 when it states none. */
};

/** @brief Runs each Softmax axis. */
class Small_OnnxImportSoftmax : public ::testing::TestWithParam<SoftmaxRow>
{
};

/** @brief Tests that the axis moves to the channel-last order and is stated only when it is not last. */
TEST_P(Small_OnnxImportSoftmax, TheAxisMovesToTheChannelLastOrder)
{
    const SoftmaxRow& softmax = GetParam();
    Interchange::OnnxGraph graph = make_graph({1, 3, 4, 4});
    Interchange::OnnxNode node = make_node("Softmax", "soft", {"x"}, {"y"});
    node.attributes.push_back(make_int("axis", softmax.axis));
    graph.nodes.push_back(std::move(node));
    graph.outputs = {make_output("y")};

    SA_ASSERT_OK_AND_ASSIGN(const IO::ImportedModel imported,
                            Vision::Importers::import_onnx_model(graph, "softmax.onnx"));

    expect_row_types(imported.model, {"softmax"});
    if (softmax.expected < 0)
        EXPECT_FALSE(imported.model.layers[0].attributes.has("axis"));
    else
        EXPECT_EQ(integers_of(imported.model.layers[0], "axis"),
                  std::vector<std::int64_t>{softmax.expected});
}

/**
 * Over [N, C, H, W] the channels (axis 1 or -3) sit last in a sample [H, W, C], the rows (2 or -2)
 * first and the columns (3 or -1) in the middle at the position 1.
 */
INSTANTIATE_TEST_SUITE_P(
    Axes, Small_OnnxImportSoftmax,
    ::testing::Values(SoftmaxRow{"Channels", 1, -1}, SoftmaxRow{"ChannelsFromTheEnd", -3, -1},
                      SoftmaxRow{"Rows", 2, 0}, SoftmaxRow{"RowsFromTheEnd", -2, 0},
                      SoftmaxRow{"Columns", 3, 1}, SoftmaxRow{"ColumnsFromTheEnd", -1, 1}),
    [](const ::testing::TestParamInfo<SoftmaxRow>& info) { return info.param.name; });

/** @brief Tests that a Softmax with no axis normalises the last axis of the tensor, the columns. */
TEST(Small_OnnxImport, ASoftmaxWithNoAxisNormalisesTheColumns)
{
    Interchange::OnnxGraph graph = make_graph({1, 3, 4, 4});
    graph.nodes.push_back(make_node("Softmax", "soft", {"x"}, {"y"}));
    graph.outputs = {make_output("y")};

    SA_ASSERT_OK_AND_ASSIGN(const IO::ImportedModel imported,
                            Vision::Importers::import_onnx_model(graph, "softmax.onnx"));

    EXPECT_EQ(integers_of(imported.model.layers[0], "axis"), std::vector<std::int64_t>{1});
}

/** @brief Tests that a Softmax over a flattened vector states no axis. */
TEST(Small_OnnxImport, ASoftmaxOverAVectorStatesNoAxis)
{
    Interchange::OnnxGraph graph = make_graph({1, 2, 1, 2});
    graph.nodes.push_back(make_node("Flatten", "flat", {"x"}, {"f"}));
    graph.nodes.push_back(make_node("Softmax", "soft", {"f"}, {"y"}));
    graph.outputs = {make_output("y")};

    SA_ASSERT_OK_AND_ASSIGN(const IO::ImportedModel imported,
                            Vision::Importers::import_onnx_model(graph, "softmax.onnx"));

    expect_row_types(imported.model, {"flatten", "softmax"});
    EXPECT_FALSE(imported.model.layers[1].attributes.has("axis"));
}

/** @brief Tests that a Softmax over the batch axis is refused. */
TEST(Small_OnnxImport, ASoftmaxOverTheBatchAxisIsRefused)
{
    Interchange::OnnxGraph graph = make_graph({1, 3, 4, 4});
    Interchange::OnnxNode node = make_node("Softmax", "soft", {"x"}, {"y"});
    node.attributes.push_back(make_int("axis", 0));
    graph.nodes.push_back(std::move(node));
    graph.outputs = {make_output("y")};

    SA_EXPECT_FAILURE(Vision::Importers::import_onnx_model(graph, "softmax.onnx"),
                      StatusCode::Unsupported, "'soft' (Softmax)", "batch axis");
}

/** @brief Tests that a Softmax over an axis the tensor does not have is refused. */
TEST(Small_OnnxImport, ASoftmaxOverAnAxisBeyondTheTensorIsRefused)
{
    Interchange::OnnxGraph graph = make_graph({1, 3, 4, 4});
    Interchange::OnnxNode node = make_node("Softmax", "soft", {"x"}, {"y"});
    node.attributes.push_back(make_int("axis", 4));
    graph.nodes.push_back(std::move(node));
    graph.outputs = {make_output("y")};

    SA_EXPECT_FAILURE(Vision::Importers::import_onnx_model(graph, "softmax.onnx"),
                      StatusCode::InvalidDescription, "'soft' (Softmax)", "axis 4");
}

/** @brief Holds one Transpose permutation of [1, 2, 3, 4] and the row it becomes. */
struct TransposeRow
{
    std::string name;                    /**< The row's name. */
    std::vector<std::int64_t> perm;      /**< The perm the file states. */
    std::vector<std::int64_t> expected;  /**< The perm of the row, over [H, W, C]. */
};

/** @brief Runs each Transpose permutation. */
class Small_OnnxImportTranspose : public ::testing::TestWithParam<TransposeRow>
{
};

/** @brief Tests that the permutation moves to the channel-last order and the batch axis stays out. */
TEST_P(Small_OnnxImportTranspose, ThePermutationMovesToTheChannelLastOrder)
{
    const TransposeRow& transpose = GetParam();
    Interchange::OnnxGraph graph = make_graph({1, 2, 3, 4});
    Interchange::OnnxNode node = make_node("Transpose", "turn", {"x"}, {"y"});
    node.attributes.push_back(make_ints("perm", transpose.perm));
    graph.nodes.push_back(std::move(node));
    graph.outputs = {make_output("y")};

    SA_ASSERT_OK_AND_ASSIGN(const IO::ImportedModel imported,
                            Vision::Importers::import_onnx_model(graph, "transpose.onnx"));

    expect_row_types(imported.model, {"transpose"});
    EXPECT_EQ(integers_of(imported.model.layers[0], "perm"), transpose.expected);
}

/**
 * The channel-first tensor [N, C, H, W] is the physical [N, H, W, C]. The permutation
 * [0, 2, 3, 1] gives [N, H, W, C] channel-first and so [N, W, C, H] physically, which a sample
 * [H, W, C] reaches by [1, 2, 0]; [0, 1, 3, 2] swaps the rows and the columns, [1, 0, 2]; and
 * [0, 3, 1, 2] gives [N, W, C, H] channel-first and [N, C, H, W] physically, [2, 0, 1].
 */
INSTANTIATE_TEST_SUITE_P(
    Permutations, Small_OnnxImportTranspose,
    ::testing::Values(TransposeRow{"ChannelsLast", {0, 2, 3, 1}, {1, 2, 0}},
                      TransposeRow{"SwapRowsAndColumns", {0, 1, 3, 2}, {1, 0, 2}},
                      TransposeRow{"ChannelsFirstAgain", {0, 3, 1, 2}, {2, 0, 1}}),
    [](const ::testing::TestParamInfo<TransposeRow>& info) { return info.param.name; });

/** @brief Tests that a permutation that moves no axis writes no row. */
TEST(Small_OnnxImport, ATransposeThatMovesNoAxisWritesNoRow)
{
    Interchange::OnnxGraph graph = make_graph({1, 2, 3, 4});
    Interchange::OnnxNode node = make_node("Transpose", "turn", {"x"}, {"a"});
    node.attributes.push_back(make_ints("perm", {0, 1, 2, 3}));
    graph.nodes.push_back(std::move(node));
    graph.nodes.push_back(make_node("Relu", "relu", {"a"}, {"y"}));
    graph.outputs = {make_output("y")};

    SA_ASSERT_OK_AND_ASSIGN(const IO::ImportedModel imported,
                            Vision::Importers::import_onnx_model(graph, "transpose.onnx"));

    expect_row_types(imported.model, {"relu"});
}

/** @brief Tests that a Transpose that gives the next row the permuted extents is read by it. */
TEST(Small_OnnxImport, TheExtentsAfterATransposeAreThePermutedOnes)
{
    Interchange::OnnxGraph graph = make_graph({1, 2, 3, 4});
    Interchange::OnnxNode node = make_node("Transpose", "turn", {"x"}, {"a"});
    node.attributes.push_back(make_ints("perm", {0, 2, 3, 1}));
    graph.nodes.push_back(std::move(node));
    graph.nodes.push_back(make_node("Flatten", "flat", {"a"}, {"f"}));
    graph.initializers.push_back(make_tensor<float>("w", Interchange::OnnxDataType::Float, {1, 24},
                                                    std::vector<float>(24, 1.0f)));
    Interchange::OnnxNode gemm = make_node("Gemm", "dense", {"f", "w"}, {"y"});
    gemm.attributes.push_back(make_int("transB", 1));
    graph.nodes.push_back(std::move(gemm));
    graph.outputs = {make_output("y")};

    SA_ASSERT_OK_AND_ASSIGN(const IO::ImportedModel imported,
                            Vision::Importers::import_onnx_model(graph, "transpose.onnx"));

    expect_row_types(imported.model, {"transpose", "flatten", "linear"});
}

/** @brief Tests that a Transpose that moves the batch axis is refused, the default one included. */
TEST(Small_OnnxImport, ATransposeOfTheBatchAxisIsRefused)
{
    Interchange::OnnxGraph graph = make_graph({1, 2, 3, 4});
    graph.nodes.push_back(make_node("Transpose", "turn", {"x"}, {"y"}));
    graph.outputs = {make_output("y")};

    SA_EXPECT_FAILURE(Vision::Importers::import_onnx_model(graph, "transpose.onnx"),
                      StatusCode::Unsupported, "'turn' (Transpose)", "batch axis");
}

/** @brief Tests that a permutation of the wrong length is refused. */
TEST(Small_OnnxImport, ATransposeThatListsTooFewAxesIsRefused)
{
    Interchange::OnnxGraph graph = make_graph({1, 2, 3, 4});
    Interchange::OnnxNode node = make_node("Transpose", "turn", {"x"}, {"y"});
    node.attributes.push_back(make_ints("perm", {0, 2, 1}));
    graph.nodes.push_back(std::move(node));
    graph.outputs = {make_output("y")};

    SA_EXPECT_FAILURE(Vision::Importers::import_onnx_model(graph, "transpose.onnx"),
                      StatusCode::InvalidDescription, "'turn' (Transpose)", "3 axes");
}

/** @brief Tests that a Transpose of a flattened vector is refused. */
TEST(Small_OnnxImport, ATransposeOfAVectorIsRefused)
{
    Interchange::OnnxGraph graph = make_graph({1, 2, 1, 2});
    graph.nodes.push_back(make_node("Flatten", "flat", {"x"}, {"f"}));
    Interchange::OnnxNode node = make_node("Transpose", "turn", {"f"}, {"y"});
    node.attributes.push_back(make_ints("perm", {0, 1}));
    graph.nodes.push_back(std::move(node));
    graph.outputs = {make_output("y")};

    SA_EXPECT_FAILURE(Vision::Importers::import_onnx_model(graph, "transpose.onnx"),
                      StatusCode::Unsupported, "'turn' (Transpose)", "rank 1");
}

namespace
{
    /** @brief Returns a MaxPool of a window of 3 at stride 1 over "x" with the stated pads. */
    Interchange::OnnxNode padded_pool(std::vector<std::int64_t> pads)
    {
        Interchange::OnnxNode pool = make_node("MaxPool", "pool", {"x"}, {"y"});
        pool.attributes.push_back(make_ints("kernel_shape", {3, 3}));
        pool.attributes.push_back(make_ints("strides", {1, 1}));
        pool.attributes.push_back(make_ints("pads", std::move(pads)));
        return pool;
    }
} // namespace

/** @brief Tests that equal pads on both sides are the key padding. */
TEST(Small_OnnxImport, AMaxPoolWithEqualPadsStatesPadding)
{
    Interchange::OnnxGraph graph = make_graph({1, 2, 5, 5});
    graph.nodes.push_back(padded_pool({1, 1, 1, 1}));
    graph.outputs = {make_output("y")};

    SA_ASSERT_OK_AND_ASSIGN(const IO::ImportedModel imported,
                            Vision::Importers::import_onnx_model(graph, "pool.onnx"));

    EXPECT_EQ(integers_of(imported.model.layers[0], "padding"), std::vector<std::int64_t>{1});
    EXPECT_FALSE(imported.model.layers[0].attributes.has("leading_padding"));
    EXPECT_FALSE(imported.model.layers[0].attributes.has("end_padding"));
}

/**
 * @brief Tests that unequal pads become the keys leading_padding and end_padding.
 *
 * The file lists the pads as (begin rows, begin columns, end rows, end columns) = (1, 0, 0, 2).
 */
TEST(Small_OnnxImport, AMaxPoolWithUnequalPadsStatesLeadingAndEndPadding)
{
    Interchange::OnnxGraph graph = make_graph({1, 2, 5, 5});
    graph.nodes.push_back(padded_pool({1, 0, 0, 2}));
    graph.outputs = {make_output("y")};

    SA_ASSERT_OK_AND_ASSIGN(const IO::ImportedModel imported,
                            Vision::Importers::import_onnx_model(graph, "pool.onnx"));

    EXPECT_FALSE(imported.model.layers[0].attributes.has("padding"));
    EXPECT_EQ(integers_of(imported.model.layers[0], "leading_padding"),
              (std::vector<std::int64_t>{1, 0}));
    EXPECT_EQ(integers_of(imported.model.layers[0], "end_padding"),
              (std::vector<std::int64_t>{0, 2}));
}

/** @brief Tests that a graph of opset 12 is refused with the opset named. */
TEST(Small_OnnxImport, AnOldOpsetIsRefused)
{
    Interchange::OnnxGraph graph = make_graph({1, 3, 4, 4});
    graph.opsets.front().version = 12;
    graph.nodes.push_back(make_node("Relu", "relu", {"x"}, {"y"}));
    graph.outputs = {make_output("y")};

    SA_EXPECT_FAILURE(Vision::Importers::import_onnx_model(graph, "old.onnx"),
                      StatusCode::Unsupported, "opset 12", "13");
}

/** @brief Tests that a call with two files is refused before any file is opened. */
TEST(Small_OnnxImport, TwoFilesAreRefused)
{
    const std::filesystem::path files[] = {"a.onnx", "b.onnx"};

    SA_EXPECT_FAILURE(Vision::Importers::import_onnx_files(
                          SushiRuntime::span<const std::filesystem::path>(files, 2)),
                      StatusCode::InvalidDescription, "'onnx'", "2 files");
}

/** @brief Tests that a file that is not there is FileNotFound. */
TEST(Small_OnnxImport, AMissingFileIsNotFound)
{
    const std::filesystem::path files[] = {fixture_folder() / "absent.onnx"};

    SA_EXPECT_FAILURE(Vision::Importers::import_onnx_files(
                          SushiRuntime::span<const std::filesystem::path>(files, 1)),
                      StatusCode::FileNotFound, "absent.onnx");
}

/** @brief Tests that a Relu graph is one relu row, with the input moved to [H, W, C]. */
TEST(Small_OnnxImport, AReluGraphIsOneRowOverAChannelLastInput)
{
    Interchange::OnnxGraph graph = make_graph({1, 3, 5, 7});
    graph.nodes.push_back(make_node("Relu", "relu", {"x"}, {"y"}));
    graph.outputs = {make_output("y")};

    SA_ASSERT_OK_AND_ASSIGN(const IO::ImportedModel imported,
                            Vision::Importers::import_onnx_model(graph, "relu.onnx"));

    EXPECT_EQ(imported.model.input_shape, (Shape{5, 7, 3}));
    ASSERT_EQ(imported.model.layers.size(), 1u);
    EXPECT_EQ(imported.model.layers[0].type, "relu");
    ASSERT_EQ(imported.model.outputs.size(), 1u);
    EXPECT_EQ(imported.model.outputs[0].name, "y");
    EXPECT_TRUE(imported.weights.empty());
}

/** @brief Tests that a Constant that only feeds a Reshape is folded and the Reshape flattens. */
TEST(Small_OnnxImport, AFoldedConstantShapeMakesTheReshapeAFlatten)
{
    Interchange::OnnxGraph graph = make_graph({1, 2, 3, 3});
    Interchange::OnnxNode constant = make_node("Constant", "shape", {}, {"target"});
    Interchange::OnnxAttribute value;
    value.name = "value";
    value.type = Interchange::OnnxAttributeType::Tensor;
    value.tensor = make_tensor<std::int64_t>("", Interchange::OnnxDataType::Int64, {2}, {1, -1});
    constant.attributes.push_back(std::move(value));
    graph.nodes.push_back(std::move(constant));
    graph.nodes.push_back(make_node("Reshape", "flat", {"x", "target"}, {"y"}));
    graph.outputs = {make_output("y")};

    SA_ASSERT_OK_AND_ASSIGN(const IO::ImportedModel imported,
                            Vision::Importers::import_onnx_model(graph, "reshape.onnx"));

    ASSERT_EQ(imported.model.layers.size(), 1u);
    EXPECT_EQ(imported.model.layers[0].type, "flatten");
}

/** @brief Tests that an equal Split of the channels is one slice_channels row per output. */
TEST(Small_OnnxImport, AnEqualSplitOfTheChannelsIsOneSliceRowPerOutput)
{
    Interchange::OnnxGraph graph = make_graph({1, 4, 2, 2});
    Interchange::OnnxNode split = make_node("Split", "halves", {"x"}, {"p", "q"});
    split.attributes.push_back(make_int("axis", 1));
    graph.nodes.push_back(std::move(split));
    graph.outputs = {make_output("p"), make_output("q")};

    SA_ASSERT_OK_AND_ASSIGN(const IO::ImportedModel imported,
                            Vision::Importers::import_onnx_model(graph, "split.onnx"));

    ASSERT_EQ(imported.model.layers.size(), 2u);
    expect_slice_row(imported.model, 0, 0, 2);
    expect_slice_row(imported.model, 1, 2, 2);
}

/**
 * @brief Tests that the weight of a Gemm after a Flatten moves from (C, H, W) to (H, W, C).
 *
 * The input is 2 channels of 1 row and 2 columns. ONNX orders the four features c0w0, c0w1,
 * c1w0, c1w1 as the weight 10, 20, 30, 40; the flatten row orders them c0w0, c1w0, c0w1,
 * c1w1, so the imported weight reads 10, 30, 20, 40.
 */
TEST(Small_OnnxImport, AGemmWeightAfterAFlattenFollowsTheChannelLastOrder)
{
    Interchange::OnnxGraph graph = make_graph({1, 2, 1, 2});
    graph.initializers.push_back(make_tensor<float>("w", Interchange::OnnxDataType::Float, {1, 4},
                                                    {10.0f, 20.0f, 30.0f, 40.0f}));
    graph.nodes.push_back(make_node("Flatten", "flat", {"x"}, {"f"}));
    Interchange::OnnxNode gemm = make_node("Gemm", "dense", {"f", "w"}, {"y"});
    gemm.attributes.push_back(make_int("transB", 1));
    graph.nodes.push_back(std::move(gemm));
    graph.outputs = {make_output("y")};

    SA_ASSERT_OK_AND_ASSIGN(const IO::ImportedModel imported,
                            Vision::Importers::import_onnx_model(graph, "gemm.onnx"));

    ASSERT_EQ(imported.model.layers.size(), 2u);
    EXPECT_EQ(imported.model.layers[1].type, "linear");
    ASSERT_EQ(imported.weights.size(), 1u);
    EXPECT_EQ(imported.weights[0].record.name, "1.weight");
    EXPECT_EQ(imported.weights[0].record.shape, (Shape{1, 4}));
    float placed[4] = {};
    ASSERT_EQ(imported.weights[0].payload.size(), sizeof(placed));
    std::memcpy(placed, imported.weights[0].payload.data(), sizeof(placed));
    EXPECT_EQ(placed[0], 10.0f);
    EXPECT_EQ(placed[1], 30.0f);
    EXPECT_EQ(placed[2], 20.0f);
    EXPECT_EQ(placed[3], 40.0f);
}

/** @brief Tests that the fixture imports to the rows of its pattern and the weights each owns. */
TEST_F(Medium_OnnxImportFixture, TheFixtureImportsToTheRowsOfItsPattern)
{
    SA_ASSERT_OK_AND_ASSIGN(const IO::ImportedModel imported, import_fixture());

    expect_row_types(imported.model, {"conv2d", "batchnorm2d", "silu", "upsample", "concat",
                                      "conv2d", "flatten"});
    EXPECT_EQ(imported.model.input_shape, (Shape{8, 8, 3}));
    expect_weight_names(imported, {"0.weight", "0.bias", "1.weight", "1.bias", "1.running_mean",
                                   "1.running_var", "5.weight"});
    EXPECT_EQ(imported.weights[0].record.shape, (Shape{4, 3, 3, 3}));
    EXPECT_EQ(imported.weights[6].record.shape, (Shape{2, 1, 1, 7}));
}

/**
 * @brief Tests that the forward pass of the imported fixture equals onnx.reference's output.
 *
 * The reference lists the one output row as (C, H, W) and the model as (H, W, C), so each
 * model value is compared with the reference value of the same channel and pixel.
 */
TEST_F(Medium_OnnxImportFixture, TheForwardEqualsTheReferenceEvaluator)
{
    SA_ASSERT_OK_AND_ASSIGN(const IO::ImportedModel imported, import_fixture());
    SA_ASSERT_OK_AND_ASSIGN(const Expected expected, read_expected());

    SA_ASSERT_OK_AND_ASSIGN(
        const std::vector<float> output,
        run_forward(backend(), imported, channels_last(expected.input, 3, 8, 8)));

    expect_matches_reference(output, expected.output);
}
