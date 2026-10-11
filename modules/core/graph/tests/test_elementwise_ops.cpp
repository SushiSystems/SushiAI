/**************************************************************************/
/* test_elementwise_ops.cpp                                               */
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
 * @file test_elementwise_ops.cpp
 * @brief Tests the shape rules and rejections of the elementwise and selection emitters.
 * @author Mustafa Garip
 */

#include <array>
#include <cstddef>
#include <string>
#include <string_view>
#include <vector>
#include <gtest/gtest.h>
#include <SushiAI/core/common.hpp>
#include <SushiAI/core/ops.hpp>
#include <SushiAI/core/shape.hpp>
#include <SushiAI/graph/builder.hpp>
#include <SushiAI/graph/ir.hpp>

using namespace SushiAI;
using Graph::GraphBuilder;
using Graph::ValueID;

namespace
{
    /** @brief Names one two-operand emitter and the identity it must write. */
    struct BinaryEmitter
    {
        std::string_view name;                                   /**< @brief Label for a failure. */
        OpID op;                                                 /**< @brief Expected identity. */
        ValueID (GraphBuilder::*emit)(ValueID, ValueID) const;   /**< @brief The emitter. */
    };

    /** @brief Names one single-operand emitter and the identity it must write. */
    struct UnaryEmitter
    {
        std::string_view name;                          /**< @brief Label for a failure. */
        OpID op;                                        /**< @brief Expected identity. */
        ValueID (GraphBuilder::*emit)(ValueID) const;   /**< @brief The emitter. */
    };

    /** @brief Lists every two-operand emitter under test. */
    const std::array<BinaryEmitter, 4> K_BINARY_EMITTERS = {{
        {"div",        Ops::K_DIV.id,        &GraphBuilder::div},
        {"minimum",    Ops::K_MINIMUM.id,    &GraphBuilder::minimum},
        {"maximum",    Ops::K_MAXIMUM.id,    &GraphBuilder::maximum},
        {"less_equal", Ops::K_LESS_EQUAL.id, &GraphBuilder::less_equal},
    }};

    /** @brief Lists every single-operand emitter under test. */
    const std::array<UnaryEmitter, 4> K_UNARY_EMITTERS = {{
        {"abs",      Ops::K_ABS.id,      &GraphBuilder::abs},
        {"atan",     Ops::K_ATAN.id,     &GraphBuilder::atan},
        {"softplus", Ops::K_SOFTPLUS.id, &GraphBuilder::softplus},
        {"silu",     Ops::K_SILU.id,     &GraphBuilder::silu},
    }};

    /** @brief Lists every activation gradient that reads the forward input. */
    const std::array<OpID, 2> K_INPUT_READING_GRADIENTS = {{
        Ops::K_SILU_BACKWARD.id,
        Ops::K_SOFTPLUS_BACKWARD.id,
    }};

    /** @brief Declares a rank-4 input and converts it to NCHW; the extents survive the permute. */
    ValueID nchw_input(const GraphBuilder& builder, std::string_view debug_name,
                       DType dtype = DType::Float32)
    {
        const ValueID nhwc = builder.input(Shape{2, 2, 2, 2}, debug_name, dtype);
        return builder.layout_convert(nhwc, Graph::TensorLayout::NCHW);
    }

    /** @brief Expects @p out to be the single result of an @p op node that reads @p inputs. */
    void expect_produced_by(const Graph::Graph& graph, ValueID out, OpID op,
                            const std::vector<ValueID>& inputs)
    {
        const Graph::NodeID producer = graph.value(out).producer;
        ASSERT_NE(producer, Graph::K_NO_NODE);

        const Graph::Node& node = graph.node(producer);
        EXPECT_EQ(node.op, op) << "written by " << Ops::name_of(node.op);
        EXPECT_EQ(node.inputs, inputs);
        ASSERT_EQ(node.outputs.size(), 1u);
        EXPECT_EQ(node.outputs[0], out);
    }

    /** @brief Expects @p out to copy the shape, dtype and layout of @p like as an activation. */
    void expect_described_like(const Graph::Graph& graph, ValueID out, ValueID like)
    {
        EXPECT_EQ(graph.value(out).shape, graph.value(like).shape);
        EXPECT_EQ(graph.value(out).dtype, graph.value(like).dtype);
        EXPECT_EQ(graph.value(out).layout, graph.value(like).layout);
        EXPECT_EQ(graph.value(out).lifetime, Graph::Lifetime::Activation);
    }
}

/** @brief Tests that each two-operand emitter writes its own identity over [a, b] in that order. */
TEST(Unit_ElementwiseOps, BinaryEmittersWriteTheirIdentityOverBothOperandsInOrder)
{
    for (const BinaryEmitter& emitter : K_BINARY_EMITTERS)
    {
        SCOPED_TRACE(std::string(emitter.name));

        Graph::Graph graph;
        const GraphBuilder builder(graph);
        const ValueID a = builder.input(Shape{4, 3}, "a");
        const ValueID b = builder.input(Shape{4, 3}, "b");

        const ValueID out = (builder.*emitter.emit)(a, b);

        expect_produced_by(graph, out, emitter.op, {a, b});
    }
}

/** @brief Tests that each two-operand emitter gives its result the first operand's description. */
TEST(Unit_ElementwiseOps, BinaryEmittersKeepShapeDTypeAndLayout)
{
    for (const BinaryEmitter& emitter : K_BINARY_EMITTERS)
    {
        SCOPED_TRACE(std::string(emitter.name));

        Graph::Graph graph;
        const GraphBuilder builder(graph);
        const ValueID a = nchw_input(builder, "a", DType::Float64);
        const ValueID b = nchw_input(builder, "b", DType::Float64);

        const ValueID out = (builder.*emitter.emit)(a, b);

        expect_described_like(graph, out, a);
        EXPECT_EQ(graph.value(out).layout, Graph::TensorLayout::NCHW);
        EXPECT_EQ(graph.value(out).dtype, DType::Float64);
    }
}

/** @brief Tests that each two-operand emitter refuses a broadcastable pair of unequal shapes. */
TEST(Unit_ElementwiseOps, BinaryEmittersRejectUnequalShapesWithoutBroadcasting)
{
    for (const BinaryEmitter& emitter : K_BINARY_EMITTERS)
    {
        SCOPED_TRACE(std::string(emitter.name));

        Graph::Graph graph;
        const GraphBuilder builder(graph);
        const ValueID matrix = builder.input(Shape{4, 3}, "matrix");
        const ValueID column = builder.input(Shape{4, 1}, "column");
        const ValueID row = builder.input(Shape{3}, "row");
        const std::size_t nodes_before = graph.node_count();

        EXPECT_THROW((void)(builder.*emitter.emit)(matrix, column), Error);
        EXPECT_THROW((void)(builder.*emitter.emit)(column, matrix), Error);
        EXPECT_THROW((void)(builder.*emitter.emit)(matrix, row), Error);
        EXPECT_EQ(graph.node_count(), nodes_before) << "a rejected call left a node behind";
    }
}

/** @brief Tests that each two-operand emitter refuses operands of two element types. */
TEST(Unit_ElementwiseOps, BinaryEmittersRejectUnequalDtypes)
{
    for (const BinaryEmitter& emitter : K_BINARY_EMITTERS)
    {
        SCOPED_TRACE(std::string(emitter.name));

        Graph::Graph graph;
        const GraphBuilder builder(graph);
        const ValueID single = builder.input(Shape{4, 3}, "single", DType::Float32);
        const ValueID wide = builder.input(Shape{4, 3}, "wide", DType::Float64);

        EXPECT_THROW((void)(builder.*emitter.emit)(single, wide), Error);
        EXPECT_THROW((void)(builder.*emitter.emit)(wide, single), Error);
    }
}

/** @brief Tests that LESS_EQUAL yields its mask in the operands' element type, not a mask type. */
TEST(Unit_ElementwiseOps, LessEqualYieldsAMaskInTheOperandsOwnDType)
{
    for (DType dtype : {DType::Float32, DType::Float64})
    {
        Graph::Graph graph;
        const GraphBuilder builder(graph);
        const ValueID a = builder.input(Shape{4, 3}, "a", dtype);
        const ValueID b = builder.input(Shape{4, 3}, "b", dtype);

        EXPECT_EQ(graph.value(builder.less_equal(a, b)).dtype, dtype);
    }
}

/** @brief Tests that each single-operand emitter writes its own identity over its operand. */
TEST(Unit_ElementwiseOps, UnaryEmittersWriteTheirIdentityOverTheirOperand)
{
    for (const UnaryEmitter& emitter : K_UNARY_EMITTERS)
    {
        SCOPED_TRACE(std::string(emitter.name));

        Graph::Graph graph;
        const GraphBuilder builder(graph);
        const ValueID x = builder.input(Shape{4, 3}, "x");

        const ValueID out = (builder.*emitter.emit)(x);

        EXPECT_NE(out, x) << "the emitter must write a new value, not its operand";
        expect_produced_by(graph, out, emitter.op, {x});
    }
}

/** @brief Tests that each single-operand emitter gives its result the operand's description. */
TEST(Unit_ElementwiseOps, UnaryEmittersKeepShapeDTypeAndLayout)
{
    for (const UnaryEmitter& emitter : K_UNARY_EMITTERS)
    {
        SCOPED_TRACE(std::string(emitter.name));

        Graph::Graph graph;
        const GraphBuilder builder(graph);
        const ValueID x = nchw_input(builder, "x", DType::Float64);

        const ValueID out = (builder.*emitter.emit)(x);

        expect_described_like(graph, out, x);
        EXPECT_EQ(graph.value(out).layout, Graph::TensorLayout::NCHW);
        EXPECT_EQ(graph.value(out).dtype, DType::Float64);
    }
}

/** @brief Tests that SELECT writes one node reading [condition, a, b] in that order. */
TEST(Unit_ElementwiseOps, SelectReadsItsConditionThenBothBranches)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);
    const ValueID condition = builder.input(Shape{4, 3}, "condition");
    const ValueID a = builder.input(Shape{4, 3}, "a");
    const ValueID b = builder.input(Shape{4, 3}, "b");

    const ValueID out = builder.select(condition, a, b);

    expect_produced_by(graph, out, Ops::K_SELECT.id, {condition, a, b});
}

/** @brief Tests that SELECT gives its result the shape, dtype and layout of its branches. */
TEST(Unit_ElementwiseOps, SelectKeepsShapeDTypeAndLayout)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);
    const ValueID condition = nchw_input(builder, "condition", DType::Float64);
    const ValueID a = nchw_input(builder, "a", DType::Float64);
    const ValueID b = nchw_input(builder, "b", DType::Float64);

    const ValueID out = builder.select(condition, a, b);

    expect_described_like(graph, out, a);
    EXPECT_EQ(graph.value(out).layout, Graph::TensorLayout::NCHW);
}

/** @brief Tests that SELECT refuses a condition whose shape differs from its branches. */
TEST(Unit_ElementwiseOps, SelectRejectsAConditionOfAnotherShape)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);
    const ValueID per_row = builder.input(Shape{4, 1}, "per_row");
    const ValueID a = builder.input(Shape{4, 3}, "a");
    const ValueID b = builder.input(Shape{4, 3}, "b");
    const std::size_t nodes_before = graph.node_count();

    EXPECT_THROW((void)builder.select(per_row, a, b), Error);
    EXPECT_EQ(graph.node_count(), nodes_before) << "a rejected call left a node behind";
}

/** @brief Tests that SELECT refuses a condition whose element type differs from its branches. */
TEST(Unit_ElementwiseOps, SelectRejectsAConditionOfAnotherDType)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);
    const ValueID condition = builder.input(Shape{4, 3}, "condition", DType::Int64);
    const ValueID a = builder.input(Shape{4, 3}, "a", DType::Float32);
    const ValueID b = builder.input(Shape{4, 3}, "b", DType::Float32);

    EXPECT_THROW((void)builder.select(condition, a, b), Error);
}

/** @brief Tests that SELECT refuses two branches of unequal shape. */
TEST(Unit_ElementwiseOps, SelectRejectsBranchesOfUnequalShape)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);
    const ValueID condition = builder.input(Shape{4, 3}, "condition");
    const ValueID a = builder.input(Shape{4, 3}, "a");
    const ValueID narrow = builder.input(Shape{4, 1}, "narrow");

    EXPECT_THROW((void)builder.select(condition, a, narrow), Error);
    EXPECT_THROW((void)builder.select(condition, narrow, a), Error);
}

/** @brief Tests that SELECT refuses two branches of unequal element type. */
TEST(Unit_ElementwiseOps, SelectRejectsBranchesOfUnequalDType)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);
    const ValueID condition = builder.input(Shape{4, 3}, "condition", DType::Float32);
    const ValueID a = builder.input(Shape{4, 3}, "a", DType::Float32);
    const ValueID wide = builder.input(Shape{4, 3}, "wide", DType::Float64);

    EXPECT_THROW((void)builder.select(condition, a, wide), Error);
    EXPECT_THROW((void)builder.select(condition, wide, a), Error);
}

/** @brief Tests that the activation-gradient emitter writes each input-reading gradient. */
TEST(Unit_ElementwiseOps, ActivationBackwardEmitsTheInputReadingGradients)
{
    for (OpID gradient : K_INPUT_READING_GRADIENTS)
    {
        SCOPED_TRACE(std::string(Ops::name_of(gradient)));

        Graph::Graph graph;
        const GraphBuilder builder(graph);
        const ValueID dy = builder.input(Shape{4, 3}, "dy");
        const ValueID x = builder.input(Shape{4, 3}, "x");

        const ValueID dx = builder.activation_backward(gradient, dy, x);

        expect_produced_by(graph, dx, gradient, {dy, x});
        expect_described_like(graph, dx, dy);
    }
}

/** @brief Tests that each input-reading gradient refuses a dy shaped unlike the input. */
TEST(Unit_ElementwiseOps, InputReadingGradientsRejectAGradientOfAnotherShape)
{
    for (OpID gradient : K_INPUT_READING_GRADIENTS)
    {
        SCOPED_TRACE(std::string(Ops::name_of(gradient)));

        Graph::Graph graph;
        const GraphBuilder builder(graph);
        const ValueID dy = builder.input(Shape{4, 1}, "dy");
        const ValueID x = builder.input(Shape{4, 3}, "x");

        EXPECT_THROW((void)builder.activation_backward(gradient, dy, x), Error);
    }
}

/** @brief Tests that each input-reading gradient refuses a dy of another element type. */
TEST(Unit_ElementwiseOps, InputReadingGradientsRejectAGradientOfAnotherDType)
{
    for (OpID gradient : K_INPUT_READING_GRADIENTS)
    {
        SCOPED_TRACE(std::string(Ops::name_of(gradient)));

        Graph::Graph graph;
        const GraphBuilder builder(graph);
        const ValueID single = builder.input(Shape{4, 3}, "single", DType::Float32);
        const ValueID wide = builder.input(Shape{4, 3}, "wide", DType::Float64);

        EXPECT_THROW((void)builder.activation_backward(gradient, single, wide), Error);
        EXPECT_THROW((void)builder.activation_backward(gradient, wide, single), Error);
    }
}

/** @brief Tests that the activation-gradient emitter refuses an identity outside ACTIVATIONS. */
TEST(Unit_ElementwiseOps, ActivationBackwardRejectsAnIdentityThatIsNotAnActivationGradient)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);
    const ValueID dy = builder.input(Shape{4, 3}, "dy");
    const ValueID x = builder.input(Shape{4, 3}, "x");

    EXPECT_THROW((void)builder.activation_backward(Ops::K_SILU.id, dy, x), Error);
    EXPECT_THROW((void)builder.activation_backward(Ops::K_SOFTPLUS.id, dy, x), Error);
    EXPECT_THROW((void)builder.activation_backward(Ops::K_DIV.id, dy, x), Error);
}

/** @brief Tests that Graph::finalise accepts a graph that uses every emitter in one layout. */
TEST(Unit_ElementwiseOps, FinaliseAcceptsAGraphUsingEveryEmitter)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);
    const ValueID a = nchw_input(builder, "a");
    const ValueID b = nchw_input(builder, "b");

    ValueID chained = a;
    for (const BinaryEmitter& emitter : K_BINARY_EMITTERS)
        chained = (builder.*emitter.emit)(chained, b);
    for (const UnaryEmitter& emitter : K_UNARY_EMITTERS)
        chained = (builder.*emitter.emit)(chained);
    (void)builder.select(builder.less_equal(a, b), chained, b);

    EXPECT_NO_THROW(graph.finalise());
}

/** @brief Tests that Graph::finalise refuses an emitted node whose operands sit in two layouts. */
TEST(Unit_ElementwiseOps, FinaliseRejectsOperandsInTwoLayouts)
{
    for (const BinaryEmitter& emitter : K_BINARY_EMITTERS)
    {
        SCOPED_TRACE(std::string(emitter.name));

        Graph::Graph graph;
        const GraphBuilder builder(graph);
        const ValueID nchw = nchw_input(builder, "nchw");
        const ValueID nhwc = builder.input(Shape{2, 2, 2, 2}, "nhwc");

        (void)(builder.*emitter.emit)(nchw, nhwc);

        EXPECT_THROW(graph.finalise(), Error);
    }
}

/** @brief Tests that Graph::finalise refuses a SELECT whose condition sits in another layout. */
TEST(Unit_ElementwiseOps, FinaliseRejectsASelectConditionInAnotherLayout)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);
    const ValueID condition = builder.input(Shape{2, 2, 2, 2}, "condition");
    const ValueID a = nchw_input(builder, "a");
    const ValueID b = nchw_input(builder, "b");

    (void)builder.select(condition, a, b);

    EXPECT_THROW(graph.finalise(), Error);
}

/** @brief Tests that a result computed from a parameter requires a gradient after finalise. */
TEST(Unit_ElementwiseOps, ResultsOfAParameterRequireAGradient)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);
    const ValueID weight = builder.parameter(Shape{4, 3}, "weight");
    const ValueID constant = builder.input(Shape{4, 3}, "constant");

    const ValueID from_parameter = builder.silu(builder.div(weight, constant));
    const ValueID from_inputs = builder.atan(builder.maximum(constant, constant));
    graph.finalise();

    EXPECT_TRUE(graph.value(from_parameter).requires_grad);
    EXPECT_FALSE(graph.value(from_inputs).requires_grad);
}

/** @brief Tests that a comparison of a parameter yields a mask that requires no gradient. */
TEST(Unit_ElementwiseOps, AComparisonOfAParameterRequiresNoGradient)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);
    const ValueID weight = builder.parameter(Shape{4, 3}, "weight");
    const ValueID bound = builder.input(Shape{4, 3}, "bound");

    const ValueID mask = builder.less_equal(weight, bound);
    const ValueID selected = builder.select(mask, bound, bound);
    graph.finalise();

    EXPECT_FALSE(graph.value(mask).requires_grad);
    EXPECT_FALSE(graph.value(selected).requires_grad);
}

/** @brief Tests that constant() writes one operand-free CONSTANT node holding the value. */
TEST(Unit_ElementwiseOps, ConstantWritesAnOperandFreeNodeHoldingItsValue)
{
    constexpr double K_VALUE = 1e-7;
    constexpr double K_VALUE_TOLERANCE = 1e-20;

    Graph::Graph graph;
    const GraphBuilder builder(graph);

    const ValueID out = builder.constant(Shape{4, 3}, K_VALUE, DType::Float64);

    expect_produced_by(graph, out, Ops::K_CONSTANT.id, {});
    EXPECT_NEAR(graph.node(graph.value(out).producer).attributes.get(0), K_VALUE,
                K_VALUE_TOLERANCE);
    EXPECT_EQ(graph.value(out).shape, (Shape{4, 3}));
    EXPECT_EQ(graph.value(out).dtype, DType::Float64);
    EXPECT_EQ(graph.value(out).layout, Graph::TensorLayout::NHWC);
    EXPECT_EQ(graph.value(out).lifetime, Graph::Lifetime::Activation);
}

/** @brief Tests that constant() defaults to FLOAT32, as zeros() does. */
TEST(Unit_ElementwiseOps, ConstantDefaultsToFloat32)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);

    EXPECT_EQ(graph.value(builder.constant(Shape{2}, 1.0)).dtype, DType::Float32);
}

/** @brief Tests that constant() refuses an element type that is not real floating. */
TEST(Unit_ElementwiseOps, ConstantRejectsAnElementTypeThatIsNotRealFloating)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);
    const std::size_t nodes_before = graph.node_count();
    const std::size_t values_before = graph.value_count();

    EXPECT_THROW((void)builder.constant(Shape{2}, 1.0, DType::Int64), Error);
    EXPECT_THROW((void)builder.constant(Shape{2}, 1.0, DType::Complex32), Error);
    EXPECT_EQ(graph.node_count(), nodes_before) << "a rejected call left a node behind";
    EXPECT_EQ(graph.value_count(), values_before) << "a rejected call left a value behind";
}

/** @brief Tests that constant_like() copies the shape, dtype and layout of the value it names. */
TEST(Unit_ElementwiseOps, ConstantLikeCopiesShapeDTypeAndLayoutWithoutReadingItsOperand)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);
    const ValueID x = nchw_input(builder, "x", DType::Float64);

    const ValueID out = builder.constant_like(x, 0.5);

    expect_produced_by(graph, out, Ops::K_CONSTANT.id, {});
    expect_described_like(graph, out, x);
    EXPECT_EQ(graph.value(out).layout, Graph::TensorLayout::NCHW);
    EXPECT_EQ(graph.value(out).dtype, DType::Float64);
}

/** @brief Tests that constant_like() refuses to describe a value that is not real floating. */
TEST(Unit_ElementwiseOps, ConstantLikeRejectsAValueThatIsNotRealFloating)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);
    const ValueID x = builder.input(Shape{4, 3}, "x", DType::Int64);
    const std::size_t nodes_before = graph.node_count();
    const std::size_t values_before = graph.value_count();

    EXPECT_THROW((void)builder.constant_like(x, 1.0), Error);
    EXPECT_EQ(graph.node_count(), nodes_before) << "a rejected call left a node behind";
    EXPECT_EQ(graph.value_count(), values_before) << "a rejected call left a value behind";
}

/** @brief Tests that a CONSTANT joins a binary operation with the value it was shaped like. */
TEST(Unit_ElementwiseOps, AConstantLikeAValueIsAnOperandThatFinaliseAccepts)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);
    const ValueID x = nchw_input(builder, "x");

    const ValueID shifted = builder.add(x, builder.constant_like(x, 1e-7));

    EXPECT_NO_THROW(graph.finalise());
    EXPECT_FALSE(graph.value(shifted).requires_grad);
}

/** @brief Tests that zeros() and zeros_like() still write ZERO and never CONSTANT. */
TEST(Unit_ElementwiseOps, ZerosStillWritesTheZeroIdentity)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);
    const ValueID x = builder.input(Shape{4, 3}, "x");

    expect_produced_by(graph, builder.zeros(Shape{4, 3}), Ops::K_ZERO.id, {});
    expect_produced_by(graph, builder.zeros_like(x, Shape{4, 3}), Ops::K_ZERO.id, {});
}
