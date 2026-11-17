/**************************************************************************/
/* test_constant_folding.cpp                                              */
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
 * @file test_constant_folding.cpp
 * @brief Tests the folding of the shape arithmetic of an ONNX export into constants.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>
#include <gtest/gtest.h>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/interchange/constant_folding.hpp>
#include <SushiAI/interchange/onnx_graph.hpp>
#include <SushiAI/interchange/tensor_bytes.hpp>

using namespace SushiAI;
using Interchange::ConstantKind;
using Interchange::OnnxAttribute;
using Interchange::OnnxAttributeType;
using Interchange::OnnxGraph;
using Interchange::OnnxNode;
using Interchange::OnnxTensor;

namespace
{
    using Integers = std::vector<std::int64_t>;
    using Names = std::vector<std::string>;

    /** @brief Returns an int64 tensor of the given extents holding @p values. */
    OnnxTensor int64_tensor(const std::string& name, const Integers& dimensions,
                            const Integers& values)
    {
        std::vector<std::byte> bytes;
        for (std::int64_t value : values)
            for (unsigned i = 0; i < 8; ++i)
                bytes.push_back(static_cast<std::byte>(
                    (static_cast<std::uint64_t>(value) >> (8u * i)) & 0xFFu));
        OnnxTensor tensor;
        tensor.name = name;
        tensor.dimensions = dimensions;
        tensor.data_type = Interchange::OnnxDataType::Int64;
        tensor.data = Interchange::TensorBytes::owning(std::move(bytes));
        return tensor;
    }

    /** @brief Returns an Int attribute. */
    OnnxAttribute int_attribute(const std::string& name, std::int64_t value)
    {
        OnnxAttribute attribute;
        attribute.name = name;
        attribute.type = OnnxAttributeType::Int;
        attribute.integer = value;
        return attribute;
    }

    /** @brief Returns an Ints attribute. */
    OnnxAttribute ints_attribute(const std::string& name, const Integers& values)
    {
        OnnxAttribute attribute;
        attribute.name = name;
        attribute.type = OnnxAttributeType::Ints;
        attribute.integers = values;
        return attribute;
    }

    /** @brief Returns a Tensor attribute named "value", as a Constant node holds its tensor. */
    OnnxAttribute value_attribute(OnnxTensor tensor)
    {
        OnnxAttribute attribute;
        attribute.name = "value";
        attribute.type = OnnxAttributeType::Tensor;
        attribute.tensor = std::move(tensor);
        return attribute;
    }

    /** @brief Returns a node. */
    OnnxNode make_node(const std::string& op_type, const std::string& name, const Names& inputs,
                       const Names& outputs, std::vector<OnnxAttribute> attributes = {})
    {
        OnnxNode node;
        node.op_type = op_type;
        node.name = name;
        node.inputs = inputs;
        node.outputs = outputs;
        node.attributes = std::move(attributes);
        return node;
    }

    /** @brief Returns a Constant node holding an int64 tensor. */
    OnnxNode constant_node(const std::string& name, const std::string& output,
                           const Integers& dimensions, const Integers& values)
    {
        return make_node("Constant", name, {}, {output},
                         {value_attribute(int64_tensor(output, dimensions, values))});
    }

    /** @brief Returns a graph input of the float type with every extent stated. */
    Interchange::OnnxValueInfo static_input(const std::string& name, const Integers& extents)
    {
        Interchange::OnnxValueInfo info;
        info.name = name;
        info.element_type = Interchange::OnnxDataType::Float;
        info.has_shape = true;
        for (std::int64_t extent : extents)
        {
            Interchange::OnnxDimension dimension;
            dimension.value = extent;
            info.dimensions.push_back(dimension);
        }
        return info;
    }

    /** @brief Returns the names of the nodes of @p nodes, in order. */
    Names names_of(const std::vector<OnnxNode>& nodes)
    {
        Names names;
        for (const OnnxNode& node : nodes)
            names.push_back(node.name);
        return names;
    }

    /** @brief Returns the constant named @p name, or null when none is. */
    const Interchange::ConstantValue* constant_named(const Interchange::FoldedGraph& folded,
                                                     const std::string& name)
    {
        for (const Interchange::NamedConstant& constant : folded.constants)
            if (constant.name == name)
                return &constant.value;
        return nullptr;
    }

    /**
     * @brief Returns a graph that runs @p producers and then a Reshape of "d" by the tensor
     *        @p result, so the folding has a reader to feed.
     */
    OnnxGraph graph_ending_in_reshape(std::vector<OnnxNode> producers, const std::string& result)
    {
        OnnxGraph graph;
        graph.inputs.push_back(static_input("x", {1, 3, 8, 8}));
        graph.inputs.push_back(static_input("d", {6}));
        graph.nodes = std::move(producers);
        graph.nodes.push_back(make_node("Reshape", "reader", {"d", result}, {"r"}));
        return graph;
    }
}

/** @brief Tests that a Shape to Reshape chain folds into one constant and the reader stays. */
TEST(Small_ConstantFolding, FoldsAShapeToReshapeChain)
{
    OnnxGraph graph;
    graph.inputs.push_back(static_input("x", {1, 3, 8, 8}));
    graph.nodes = {
        make_node("Conv", "conv", {"x", "w"}, {"feat"}),
        make_node("Shape", "shape_x", {"x"}, {"shape_x"}),
        constant_node("zero", "zero", {}, {0}),
        make_node("Gather", "gather_x", {"shape_x", "zero"}, {"batch"}),
        make_node("Unsqueeze", "unsqueeze_x", {"batch"}, {"batch1"},
                  {ints_attribute("axes", {0})}),
        constant_node("minus_one", "minus_one", {1}, {-1}),
        make_node("Concat", "concat_x", {"batch1", "minus_one"}, {"target"},
                  {int_attribute("axis", 0)}),
        make_node("Reshape", "reshape", {"feat", "target"}, {"flat"}),
    };

    SA_ASSERT_OK_AND_ASSIGN(const Interchange::FoldedGraph folded,
                            Interchange::fold_constants(graph));

    EXPECT_EQ(names_of(folded.nodes), (Names{"conv", "reshape"}));
    EXPECT_EQ(folded.folded_count, 6u);
    ASSERT_EQ(folded.constants.size(), 1u);
    EXPECT_EQ(folded.constants[0].name, "target");
    EXPECT_EQ(folded.constants[0].value.kind, ConstantKind::Integer);
    EXPECT_EQ(folded.constants[0].value.dimensions, (Integers{2}));
    // Shape gives [1, 3, 8, 8]; element 0 is 1; [1] joined with [-1] is [1, -1].
    EXPECT_EQ(folded.constants[0].value.integers, (Integers{1, -1}));
}

/** @brief Tests that a Constant of ints cast to float folds into the scales a Resize reads. */
TEST(Small_ConstantFolding, FoldsAConstantAndACastIntoTheScalesOfAResize)
{
    OnnxGraph graph;
    graph.inputs.push_back(static_input("x", {1, 3, 8, 8}));
    graph.nodes = {
        make_node("Constant", "scale_ints", {}, {"scale_ints"},
                  {ints_attribute("value_ints", {1, 1, 2, 2})}),
        make_node("Cast", "to_float", {"scale_ints"}, {"scales"}, {int_attribute("to", 1)}),
        make_node("Resize", "resize", {"x", "", "scales"}, {"up"}),
    };

    SA_ASSERT_OK_AND_ASSIGN(const Interchange::FoldedGraph folded,
                            Interchange::fold_constants(graph));

    EXPECT_EQ(names_of(folded.nodes), (Names{"resize"}));
    ASSERT_EQ(folded.constants.size(), 1u);
    EXPECT_EQ(folded.constants[0].name, "scales");
    EXPECT_EQ(folded.constants[0].value.kind, ConstantKind::Real);
    EXPECT_EQ(folded.constants[0].value.dimensions, (Integers{4}));
    EXPECT_EQ(folded.constants[0].value.reals, (std::vector<float>{1.0f, 1.0f, 2.0f, 2.0f}));
}

/** @brief Tests that a chain that starts at a tensor of unstated shape stays whole. */
TEST(Small_ConstantFolding, KeepsEveryNodeOfAChainWithANonConstantInput)
{
    OnnxGraph graph;
    graph.inputs.push_back(static_input("x", {1, 3, 8, 8}));
    graph.nodes = {
        make_node("Conv", "conv", {"x", "w"}, {"feat"}),
        make_node("Shape", "shape_feat", {"feat"}, {"shape_feat"}),
        constant_node("zero_b", "zero_b", {}, {0}),
        make_node("Gather", "gather_dyn", {"shape_feat", "zero_b"}, {"dyn"}),
        make_node("Unsqueeze", "unsqueeze_dyn", {"dyn"}, {"dyn1"}, {ints_attribute("axes", {0})}),
        make_node("Reshape", "reshape_dyn", {"feat", "dyn1"}, {"flat"}),
    };

    SA_ASSERT_OK_AND_ASSIGN(const Interchange::FoldedGraph folded,
                            Interchange::fold_constants(graph));

    EXPECT_EQ(names_of(folded.nodes), names_of(graph.nodes));
    EXPECT_EQ(folded.folded_count, 0u);
    EXPECT_TRUE(folded.constants.empty());
}

/** @brief Tests that a Shape of an input with a symbolic extent does not fold. */
TEST(Small_ConstantFolding, KeepsAShapeWhoseInputHasASymbolicExtent)
{
    OnnxGraph graph = graph_ending_in_reshape({make_node("Shape", "shape_n", {"n"}, {"s"})}, "s");
    Interchange::OnnxValueInfo symbolic = static_input("n", {0, 3});
    symbolic.dimensions[0].value = -1;
    symbolic.dimensions[0].parameter = "batch";
    graph.inputs.push_back(symbolic);

    SA_ASSERT_OK_AND_ASSIGN(const Interchange::FoldedGraph folded,
                            Interchange::fold_constants(graph));

    EXPECT_EQ(names_of(folded.nodes), (Names{"shape_n", "reader"}));
    EXPECT_EQ(folded.folded_count, 0u);
}

/** @brief Tests that a result a graph output or another operator reads is kept. */
TEST(Small_ConstantFolding, KeepsAResultThatAGraphOutputOrAnotherNodeReads)
{
    OnnxGraph as_output =
        graph_ending_in_reshape({make_node("Shape", "shape_x", {"x"}, {"s"})}, "s");
    Interchange::OnnxValueInfo output;
    output.name = "s";
    as_output.outputs.push_back(output);
    OnnxGraph read_by_add = graph_ending_in_reshape(
        {make_node("Shape", "shape_x", {"x"}, {"s"}), make_node("Add", "add", {"s", "s"}, {"t"})},
        "s");

    SA_ASSERT_OK_AND_ASSIGN(const Interchange::FoldedGraph kept_output,
                            Interchange::fold_constants(as_output));
    SA_ASSERT_OK_AND_ASSIGN(const Interchange::FoldedGraph kept_by_reader,
                            Interchange::fold_constants(read_by_add));

    EXPECT_EQ(names_of(kept_output.nodes), (Names{"shape_x", "reader"}));
    EXPECT_EQ(names_of(kept_by_reader.nodes), (Names{"shape_x", "add", "reader"}));
}

/** @brief Tests Shape with start and a negative end, on [1, 3, 8, 8]. */
TEST(Small_ConstantFolding, SlicesAShapeByStartAndEnd)
{
    OnnxGraph graph = graph_ending_in_reshape(
        {make_node("Shape", "shape_x", {"x"}, {"s"},
                   {int_attribute("start", 1), int_attribute("end", -1)})},
        "s");

    SA_ASSERT_OK_AND_ASSIGN(const Interchange::FoldedGraph folded,
                            Interchange::fold_constants(graph));

    ASSERT_NE(constant_named(folded, "s"), nullptr);
    // Elements 1 up to, not including, the last of [1, 3, 8, 8].
    EXPECT_EQ(constant_named(folded, "s")->integers, (Integers{3, 8}));
}

/** @brief Tests Gather along axis 1 of a [2, 3] constant with a negative index. */
TEST(Small_ConstantFolding, GathersAlongAnAxisOfATwoDimensionalConstant)
{
    OnnxGraph graph = graph_ending_in_reshape(
        {constant_node("data", "data", {2, 3}, {1, 2, 3, 4, 5, 6}),
         constant_node("index", "index", {2}, {2, -3}),
         make_node("Gather", "gather", {"data", "index"}, {"g"}, {int_attribute("axis", 1)})},
        "g");

    SA_ASSERT_OK_AND_ASSIGN(const Interchange::FoldedGraph folded,
                            Interchange::fold_constants(graph));

    const Interchange::ConstantValue* value = constant_named(folded, "g");
    ASSERT_NE(value, nullptr);
    // Columns 2 and 0 (-3 wraps to 0) of [[1, 2, 3], [4, 5, 6]].
    EXPECT_EQ(value->dimensions, (Integers{2, 2}));
    EXPECT_EQ(value->integers, (Integers{3, 1, 6, 4}));
}

/** @brief Tests Concat along axis 1 of two [2, 1] constants, which interleaves their rows. */
TEST(Small_ConstantFolding, ConcatenatesAlongTheInnerAxis)
{
    OnnxGraph graph = graph_ending_in_reshape(
        {constant_node("a", "a", {2, 1}, {1, 2}), constant_node("b", "b", {2, 1}, {3, 4}),
         make_node("Concat", "concat", {"a", "b"}, {"c"}, {int_attribute("axis", 1)})},
        "c");

    SA_ASSERT_OK_AND_ASSIGN(const Interchange::FoldedGraph folded,
                            Interchange::fold_constants(graph));

    const Interchange::ConstantValue* value = constant_named(folded, "c");
    ASSERT_NE(value, nullptr);
    // [[1], [2]] beside [[3], [4]] is [[1, 3], [2, 4]].
    EXPECT_EQ(value->dimensions, (Integers{2, 2}));
    EXPECT_EQ(value->integers, (Integers{1, 3, 2, 4}));
}

/** @brief Tests Unsqueeze whose axes come from an initializer, over an initializer's values. */
TEST(Small_ConstantFolding, ReadsInitializersAsConstants)
{
    OnnxGraph graph = graph_ending_in_reshape(
        {make_node("Unsqueeze", "unsqueeze", {"values", "axes"}, {"u"})}, "u");
    graph.initializers.push_back(int64_tensor("values", {3}, {4, 5, 6}));
    graph.initializers.push_back(int64_tensor("axes", {1}, {0}));

    SA_ASSERT_OK_AND_ASSIGN(const Interchange::FoldedGraph folded,
                            Interchange::fold_constants(graph));

    const Interchange::ConstantValue* value = constant_named(folded, "u");
    ASSERT_NE(value, nullptr);
    EXPECT_EQ(value->dimensions, (Integers{1, 3}));
    EXPECT_EQ(value->integers, (Integers{4, 5, 6}));
}

/** @brief Tests that an index past the extent of a foldable Gather fails and names the node. */
TEST(Small_ConstantFolding, RefusesAGatherIndexOutOfRange)
{
    OnnxGraph graph = graph_ending_in_reshape(
        {constant_node("data", "data", {3}, {1, 2, 3}), constant_node("index", "index", {}, {5}),
         make_node("Gather", "gather_bad", {"data", "index"}, {"g"})},
        "g");

    SA_EXPECT_FAILURE(Interchange::fold_constants(graph), StatusCode::InvalidDescription,
                      "gather_bad", "index 5", "extent of 3");
}

/** @brief Tests that a Concat of mismatched extents fails and names the node. */
TEST(Small_ConstantFolding, RefusesAConcatOffTheAxisMismatch)
{
    OnnxGraph graph = graph_ending_in_reshape(
        {constant_node("a", "a", {2, 1}, {1, 2}), constant_node("b", "b", {3, 1}, {3, 4, 5}),
         make_node("Concat", "concat_bad", {"a", "b"}, {"c"}, {int_attribute("axis", 1)})},
        "c");

    SA_EXPECT_FAILURE(Interchange::fold_constants(graph), StatusCode::InvalidDescription,
                      "concat_bad", "extent off the axis");
}
