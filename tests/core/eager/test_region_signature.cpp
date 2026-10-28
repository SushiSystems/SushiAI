/**************************************************************************/
/* test_region_signature.cpp                                              */
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
 * @file test_region_signature.cpp
 * @brief Tests the signature that signature_of computes for one region of a partition.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstdint>
#include <functional>
#include <initializer_list>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>
#include <gtest/gtest.h>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/autograd/backward.hpp>
#include <SushiAI/autograd/provenance.hpp>
#include <SushiAI/ops/ops.hpp>
#include <SushiAI/eager/partition.hpp>
#include <SushiAI/eager/region_signature.hpp>
#include <SushiAI/eager/scope.hpp>
#include <SushiAI/eager/scope_partitioner.hpp>
#include <SushiAI/eager/scope_trace.hpp>
#include <SushiAI/eager/segment.hpp>
#include <SushiAI/failure_expectations/throw_expectations.hpp>
#include <SushiAI/graph/builder.hpp>
#include <SushiAI/graph/ir.hpp>

using namespace SushiAI;
using Eager::RegionKey;
using Eager::RegionSignature;
using Graph::GraphBuilder;
using Graph::K_NO_VALUE;
using Graph::ValueID;

namespace
{
    /** @brief Holds the key of the one region a hand-built partition has. */
    constexpr RegionKey K_BODY = 1;

    /** @brief Holds the key of the region that reads K_BODY in a two-region partition. */
    constexpr RegionKey K_READER = K_BODY + 1;

    /** @brief Holds the identity the hand-built graph's parameter gets by default. */
    constexpr std::uint64_t K_PARAMETER_IDENTITY = 0x1000;

    /** @brief Holds the identity the value K_READER reads from outside gets by default. */
    constexpr std::uint64_t K_EXTERNAL_IDENTITY = 0x2000;

    /** @brief Holds a graph, its partition and the identity of each of its values. */
    struct Traced
    {
        Graph::Graph graph;                  /**< The graph under test. */
        Eager::Partition partition;          /**< The regions of the graph's nodes. */
        std::vector<std::uint64_t> identity; /**< One identity per value id. */
    };

    /** @brief Returns the signature of @p region of @p traced. */
    [[nodiscard]] RegionSignature signature_of(const Traced& traced, RegionKey region)
    {
        return Eager::signature_of(
            traced.graph, traced.partition, region,
            SushiRuntime::span<const std::uint64_t>(traced.identity.data(),
                                                    traced.identity.size()));
    }

    /** @brief Describes the small graph the hand-built tests vary one property of. */
    struct Recipe
    {
        Shape shape{4, 3};                                  /**< Extents of every value. */
        DType dtype = DType::Float32;                       /**< Element type of every value. */
        double alpha = 2.0;                                 /**< Factor of the closing SCALE. */
        bool subtracts = false;                             /**< Combines with SUB, not ADD. */
        bool swaps_operands = false;                        /**< Combines (tanh, relu). */
        std::uint64_t parameter_identity = K_PARAMETER_IDENTITY; /**< Identity of x. */
    };

    /**
     * @brief Builds scale(combine(relu(x), tanh(x)), alpha) over a parameter x as one region.
     * @return The graph with its four operations under K_BODY and the leaf under no region.
     */
    [[nodiscard]] Traced build(const Recipe& recipe)
    {
        Traced traced;
        const GraphBuilder builder(traced.graph);

        const ValueID x = builder.parameter(recipe.shape, "x", recipe.dtype);
        // The extents are fixed here, so a refusal is a defect: value() stops the test.
        const ValueID rectified = builder.relu(x).value();
        const ValueID squashed = builder.tanh(x).value();
        const ValueID left = recipe.swaps_operands ? squashed : rectified;
        const ValueID right = recipe.swaps_operands ? rectified : squashed;
        // The extents are fixed here, so a refusal is a defect: value() stops the test.
        const ValueID combined =
            (recipe.subtracts ? builder.sub(left, right) : builder.add(left, right)).value();
        SA_EXPECT_OK(builder.scale(combined, recipe.alpha));

        for (const Graph::Node& node : traced.graph.nodes())
            traced.partition.region_of_node.push_back(
                node.op == Ops::K_LEAF.id ? Eager::K_NO_REGION : K_BODY);
        traced.partition.regions = {{K_BODY, "body#0", Eager::Direction::Forward}};
        traced.partition.optimizer_key = K_BODY + 1;

        traced.identity.assign(traced.graph.value_count(), 0);
        traced.identity[x] = recipe.parameter_identity;
        return traced;
    }

    /** @brief Returns the signature of the one region of the graph @p recipe describes. */
    [[nodiscard]] RegionSignature signature_of(const Recipe& recipe)
    {
        return signature_of(build(recipe), K_BODY);
    }

    /** @brief Returns the first node of @p graph that runs @p op, or NO_NODE. */
    [[nodiscard]] Graph::NodeID node_of(const Graph::Graph& graph, OpID op)
    {
        const std::vector<Graph::Node>& nodes = graph.nodes();
        for (std::size_t id = 0; id < nodes.size(); ++id)
            if (nodes[id].op == op)
                return static_cast<Graph::NodeID>(id);
        return Graph::K_NO_NODE;
    }

    /**
     * @brief Gives @p traced a partition and one zero identity per value.
     * @param keys One key per node that is not a LEAF, in node order; a LEAF gets no region.
     * @param regions The regions to list, in ascending key order.
     */
    void assign_regions(Traced& traced,
                        const std::vector<RegionKey>& keys,
                        std::vector<Eager::RegionInfo> regions)
    {
        std::size_t next = 0;
        traced.partition.region_of_node.clear();
        for (const Graph::Node& node : traced.graph.nodes())
            traced.partition.region_of_node.push_back(
                node.op == Ops::K_LEAF.id ? Eager::K_NO_REGION : keys.at(next++));
        traced.partition.regions = std::move(regions);
        traced.partition.optimizer_key = K_READER + 1;
        traced.identity.assign(traced.graph.value_count(), 0);
    }

    /** @brief Describes a graph whose K_READER region reads one value it does not produce. */
    struct ExternalRecipe
    {
        bool reads_leaf = false;                      /**< Reads x, not the K_BODY result. */
        bool lists_producer = true;                   /**< Lists K_BODY among the regions. */
        std::uint64_t identity = K_EXTERNAL_IDENTITY; /**< Identity of the value read. */
    };

    /**
     * @brief Builds relu(x) under K_BODY and, under K_READER, a SCALE of it or of x.
     * @return The graph with the leaf under no region and every other identity zero.
     */
    [[nodiscard]] Traced build(const ExternalRecipe& recipe)
    {
        Traced traced;
        const GraphBuilder builder(traced.graph);

        const ValueID x = builder.parameter(Shape{4, 3}, "x", DType::Float32);
        // The extents are fixed here, so a refusal is a defect: value() stops the test.
        const ValueID rectified = builder.relu(x).value();
        const ValueID read = recipe.reads_leaf ? x : rectified;
        SA_EXPECT_OK(builder.scale(read, 2.0));

        std::vector<Eager::RegionInfo> regions;
        if (recipe.lists_producer)
            regions.push_back({K_BODY, "body#0", Eager::Direction::Forward});
        regions.push_back({K_READER, "reader#0", Eager::Direction::Forward});

        assign_regions(traced, {K_BODY, K_READER}, std::move(regions));
        traced.identity[read] = recipe.identity;
        return traced;
    }

    /** @brief Returns the signature of K_READER in the graph @p recipe describes. */
    [[nodiscard]] RegionSignature signature_of(const ExternalRecipe& recipe)
    {
        return signature_of(build(recipe), K_READER);
    }

    /** @brief Names the three operands the probe node of K_READER reads, in order. */
    enum class ProbeOperands : std::uint8_t
    {
        InsideThenOutside, /**< The K_READER result, then the K_BODY result twice. */
        OutsideThenInside, /**< The K_BODY result twice, then the K_READER result. */
        InsideThenLeaves,  /**< The K_READER result, then the leaf twice. */
        LeavesThenInside,  /**< The leaf twice, then the K_READER result. */
    };

    /**
     * @brief Builds a three-operand node whose operand positions hash as zeros only.
     *
     * Every index and every identity is zero, so the two orders of one set of operands feed
     * the same words and differ in the operand kinds alone.
     *
     * @return relu(x) under K_BODY, then relu(x) and the probe node under K_READER.
     */
    [[nodiscard]] Traced build(ProbeOperands operands)
    {
        Traced traced;
        const GraphBuilder builder(traced.graph);

        const ValueID x = builder.parameter(Shape{4, 3}, "x", DType::Float32);
        // The extents are fixed here, so a refusal is a defect: value() stops the test.
        const ValueID outside = builder.relu(x).value();
        const ValueID inside = builder.relu(x).value();
        const ValueID probed =
            traced.graph.add_value(Shape{4, 3}, DType::Float32, Graph::Lifetime::Activation);

        std::vector<ValueID> read;
        switch (operands)
        {
            case ProbeOperands::InsideThenOutside:
                read = {inside, outside, outside};
                break;
            case ProbeOperands::OutsideThenInside:
                read = {outside, outside, inside};
                break;
            case ProbeOperands::InsideThenLeaves:
                read = {inside, x, x};
                break;
            case ProbeOperands::LeavesThenInside:
                read = {x, x, inside};
                break;
        }
        (void)traced.graph.add_node(Ops::K_ADD.id, std::move(read), {probed});

        assign_regions(traced, {K_BODY, K_READER, K_READER},
                       {{K_BODY, "body#0", Eager::Direction::Forward},
                        {K_READER, "reader#0", Eager::Direction::Forward}});
        return traced;
    }

    /**
     * @brief Builds a node with two results and a TANH that reads result @p index of it.
     * @return The graph with both operations under K_BODY.
     */
    [[nodiscard]] Traced build_reader_of_result(std::size_t index)
    {
        Traced traced;
        const GraphBuilder builder(traced.graph);

        const ValueID x = builder.parameter(Shape{4, 3}, "x", DType::Float32);
        const std::vector<ValueID> results = {
            traced.graph.add_value(Shape{4, 3}, DType::Float32, Graph::Lifetime::Activation),
            traced.graph.add_value(Shape{4, 3}, DType::Float32, Graph::Lifetime::Activation)};
        (void)traced.graph.add_node(Ops::K_RELU.id, {x}, results);
        SA_EXPECT_OK(builder.tanh(results.at(index)));

        assign_regions(traced, {K_BODY, K_BODY},
                       {{K_BODY, "body#0", Eager::Direction::Forward}});
        traced.identity[x] = K_PARAMETER_IDENTITY;
        return traced;
    }

    /** @brief Records @p identity for @p value of @p traced and returns @p value. */
    ValueID with_identity(Traced& traced, ValueID value, std::uint64_t identity)
    {
        traced.identity.resize(traced.graph.value_count(), 0);
        traced.identity[value] = identity;
        return value;
    }

    /** @brief Returns the entry of region @p key of @p partition, or null when unlisted. */
    [[nodiscard]] const Eager::RegionInfo* info_of(const Eager::Partition& partition,
                                                   RegionKey key)
    {
        for (const Eager::RegionInfo& info : partition.regions)
            if (info.key == key)
                return &info;
        return nullptr;
    }

    /**
     * @brief Returns an identity for one result of one node of @p region.
     * @param from_end How many nodes of @p region follow the node.
     * @param result Index of the value among the node's results.
     */
    [[nodiscard]] std::uint64_t role_identity(const Eager::RegionInfo& region,
                                              std::size_t from_end,
                                              std::size_t result)
    {
        const std::string pass =
            region.direction == Eager::Direction::Forward ? "/forward/" : "/backward/";
        return std::hash<std::string>{}(region.name + pass + std::to_string(from_end) + "/"
                                        + std::to_string(result));
    }

    /**
     * @brief Gives every value a node of a region writes the identity of its role.
     * @pre @p traced holds its partition, and every region with a node is listed in it.
     */
    void identify_produced_values(Traced& traced)
    {
        const std::vector<RegionKey>& keys = traced.partition.region_of_node;

        std::unordered_map<RegionKey, std::size_t> unvisited;
        for (RegionKey key : keys)
            ++unvisited[key];

        traced.identity.resize(traced.graph.value_count(), 0);
        for (std::size_t id = 0; id < keys.size(); ++id)
        {
            if (keys[id] == Eager::K_NO_REGION)
                continue;

            // What a later scope reads from "stem" is the result of its last node in both traces.
            const std::size_t from_end = --unvisited[keys[id]];
            const Eager::RegionInfo& region = *info_of(traced.partition, keys[id]);
            const std::vector<ValueID>& results =
                traced.graph.node(static_cast<Graph::NodeID>(id)).outputs;

            for (std::size_t result = 0; result < results.size(); ++result)
                traced.identity[results[result]] = role_identity(region, from_end, result);
        }
    }

    /** @brief Holds a three-scope perceptron traced, differentiated and partitioned. */
    struct ScopedPerceptron
    {
        /**
         * @brief Traces scopes "stem", "body" and "head", then differentiates and partitions.
         * @param stem_layers How many linear layers the "stem" scope holds; at least one.
         */
        explicit ScopedPerceptron(std::size_t stem_layers)
        {
            const GraphBuilder builder(traced.graph);
            Eager::ScopeTrace trace(traced.graph);
            std::vector<ValueID> parameters;

            const ValueID x = with_identity(traced, builder.input(Shape{4, 6}, "x"), 1);
            const ValueID targets =
                with_identity(traced, builder.input(Shape{4, 3}, "targets"), 2);

            ValueID activation = x;
            {
                const Eager::Scope scope(trace, "stem");
                parameters.push_back(
                    with_identity(traced, builder.parameter(Shape{6, 5}, "stem_weight"), 10));
                // The extents are fixed here, so a refusal is a defect: value() stops the test.
                activation = builder.relu(builder.matmul(activation, parameters.back())).value();

                for (std::size_t layer = 1; layer < stem_layers; ++layer)
                {
                    parameters.push_back(with_identity(
                        traced, builder.parameter(Shape{5, 5}, "stem_extra_weight"), 10 + layer));
                    // The extents are fixed here, so a refusal is a defect: value() stops the test.
                    activation =
                        builder.relu(builder.matmul(activation, parameters.back())).value();
                }
            }
            stem_output = activation;
            {
                const Eager::Scope scope(trace, "body");
                parameters.push_back(
                    with_identity(traced, builder.parameter(Shape{5, 4}, "body_weight"), 20));
                // The extents are fixed here, so a refusal is a defect: value() stops the test.
                activation = builder.relu(builder.matmul(activation, parameters.back())).value();
            }

            ValueID loss = K_NO_VALUE;
            {
                const Eager::Scope scope(trace, "head");
                parameters.push_back(
                    with_identity(traced, builder.parameter(Shape{4, 3}, "head_weight"), 30));
                // The extents are fixed here, so a refusal is a defect: value() stops the test.
                const ValueID logits = builder.matmul(activation, parameters.back()).value();
                loss = builder.nll_loss(builder.log_softmax(logits), targets).value();
            }

            const std::vector<Eager::Segment> segments = trace.seal();

            Autograd::Provenance provenance;
            SA_EXPECT_OK(Autograd::differentiate(
                traced.graph, loss,
                SushiRuntime::span<const ValueID>(parameters.data(), parameters.size()),
                K_NO_VALUE, &provenance));

            traced.partition = Eager::ScopePartitioner{}.assign(
                traced.graph,
                SushiRuntime::span<const Eager::Segment>(segments.data(), segments.size()),
                provenance);
            identify_produced_values(traced);
        }

        /** @brief Returns the key of the region of @p name and @p direction, or K_NO_REGION. */
        [[nodiscard]] RegionKey key_of(std::string_view name, Eager::Direction direction) const
        {
            for (const Eager::RegionInfo& info : traced.partition.regions)
                if (info.name == name && info.direction == direction)
                    return info.key;
            return Eager::K_NO_REGION;
        }

        /** @brief Holds the graph, the partition and the identity of every value. */
        Traced traced;

        /** @brief Holds the last result of the "stem" scope, which "body" reads. */
        ValueID stem_output = K_NO_VALUE;
    };
}

/**
 * @brief Tests that the hash of relu(x) over a [4, 3] FLOAT32 parameter of identity 0x2000 keeps
 *        its value, computed apart from SushiAI as 64-bit FNV-1a over the words of the stream:
 *        the operation id, ten zero attributes, one result of rank 2 with extents 4 and 3,
 *        FLOAT32 and NHWC, one operand, and an external operand of identity 0x2000.
 */
TEST(Small_RegionSignature, TheHashOfOneNodeKeepsItsValue)
{
    Traced traced;
    const GraphBuilder builder(traced.graph);
    const ValueID x = builder.parameter(Shape{4, 3}, "x", DType::Float32);
    SA_EXPECT_OK(builder.relu(x));

    traced.partition.region_of_node = {Eager::K_NO_REGION, K_BODY};
    traced.partition.regions = {{K_BODY, "body#0", Eager::Direction::Forward}};
    traced.partition.optimizer_key = K_BODY + 1;
    traced.identity.assign(traced.graph.value_count(), 0);
    traced.identity[x] = K_EXTERNAL_IDENTITY;

    EXPECT_EQ(signature_of(traced, K_BODY).hash, 0x70ba711294da9f98ull);
}

/** @brief Tests that two graphs built by the same code give equal signatures. */
TEST(Small_RegionSignature, TheSameTraceGivesTheSameSignature)
{
    const RegionSignature first = signature_of(Recipe{});
    const RegionSignature second = signature_of(Recipe{});

    EXPECT_EQ(first.hash, second.hash);
    EXPECT_TRUE(first == second);
    EXPECT_EQ(first.ops, (std::vector<OpID>{Ops::K_RELU.id, Ops::K_TANH.id, Ops::K_ADD.id,
                                            Ops::K_SCALE.id}));
}

/** @brief Tests that another SCALE factor gives another hash over the same operations. */
TEST(Small_RegionSignature, AChangedAttributeChangesTheHash)
{
    Recipe changed;
    changed.alpha = 3.0;

    const RegionSignature base = signature_of(Recipe{});
    const RegionSignature other = signature_of(changed);

    EXPECT_EQ(base.ops, other.ops);
    EXPECT_NE(base.hash, other.hash);
}

/** @brief Tests that a value in the last attribute slot gives another hash. */
TEST(Small_RegionSignature, AChangedLastAttributeSlotChangesTheHash)
{
    const Traced base = build(Recipe{});
    Traced changed = build(Recipe{});
    changed.graph.node(node_of(changed.graph, Ops::K_SCALE.id))
        .attributes.values[Graph::Attributes::K_CAPACITY - 1] = 1.0;

    const RegionSignature before = signature_of(base, K_BODY);
    const RegionSignature after = signature_of(changed, K_BODY);

    EXPECT_EQ(before.ops, after.ops);
    EXPECT_NE(before.hash, after.hash);
}

/** @brief Tests that another layout on one result gives another hash. */
TEST(Small_RegionSignature, AChangedLayoutChangesTheHash)
{
    const Traced base = build(Recipe{});
    Traced changed = build(Recipe{});
    const ValueID rectified =
        changed.graph.node(node_of(changed.graph, Ops::K_RELU.id)).outputs.front();
    changed.graph.value(rectified).layout = Graph::TensorLayout::NCHW;

    const RegionSignature before = signature_of(base, K_BODY);
    const RegionSignature after = signature_of(changed, K_BODY);

    EXPECT_EQ(before.ops, after.ops);
    EXPECT_NE(before.hash, after.hash);
}

/** @brief Tests that other extents give another hash over the same operations. */
TEST(Small_RegionSignature, AChangedShapeChangesTheHash)
{
    Recipe changed;
    changed.shape = Shape{4, 5};

    const RegionSignature base = signature_of(Recipe{});
    const RegionSignature other = signature_of(changed);

    EXPECT_EQ(base.ops, other.ops);
    EXPECT_NE(base.hash, other.hash);
}

/** @brief Tests that another element type gives another hash over the same operations. */
TEST(Small_RegionSignature, AChangedDTypeChangesTheHash)
{
    Recipe changed;
    changed.dtype = DType::Float64;

    const RegionSignature base = signature_of(Recipe{});
    const RegionSignature other = signature_of(changed);

    EXPECT_EQ(base.ops, other.ops);
    EXPECT_NE(base.hash, other.hash);
}

/** @brief Tests that SUB in place of ADD changes both the hash and the operation list. */
TEST(Small_RegionSignature, AChangedOpChangesTheHash)
{
    Recipe changed;
    changed.subtracts = true;

    const RegionSignature base = signature_of(Recipe{});
    const RegionSignature other = signature_of(changed);

    EXPECT_NE(base.ops, other.ops);
    EXPECT_NE(base.hash, other.hash);
}

/** @brief Tests that the same structure over a parameter of another identity hashes apart. */
TEST(Small_RegionSignature, ADifferentParameterChangesTheHash)
{
    Recipe changed;
    changed.parameter_identity = K_PARAMETER_IDENTITY + 1;

    const RegionSignature base = signature_of(Recipe{});
    const RegionSignature other = signature_of(changed);

    EXPECT_EQ(base.ops, other.ops);
    EXPECT_NE(base.hash, other.hash);
}

/**
 * @brief Tests that a layer added to "stem" keeps the forward and the backward signature of
 *        "body", which reads the last result of "stem" under one identity in both traces.
 */
TEST(Small_RegionSignature, AnEarlierRegionGrowingDoesNotChangeALaterRegion)
{
    const ScopedPerceptron small(1);
    const ScopedPerceptron grown(2);

    ASSERT_NE(small.stem_output, grown.stem_output);
    ASSERT_EQ(small.traced.identity.at(small.stem_output),
              grown.traced.identity.at(grown.stem_output));
    ASSERT_NE(small.key_of("body#0", Eager::Direction::Forward),
              small.key_of("body#0", Eager::Direction::Backward));

    for (Eager::Direction direction : {Eager::Direction::Forward, Eager::Direction::Backward})
    {
        const RegionKey small_body = small.key_of("body#0", direction);
        const RegionKey grown_body = grown.key_of("body#0", direction);
        ASSERT_NE(small_body, Eager::K_NO_REGION);
        ASSERT_NE(grown_body, Eager::K_NO_REGION);

        const RegionSignature before = signature_of(small.traced, small_body);
        const RegionSignature after = signature_of(grown.traced, grown_body);
        EXPECT_FALSE(before.ops.empty());
        EXPECT_EQ(before.hash, after.hash);
        EXPECT_TRUE(before == after);
    }

    const RegionSignature small_stem =
        signature_of(small.traced, small.key_of("stem#0", Eager::Direction::Forward));
    const RegionSignature grown_stem =
        signature_of(grown.traced, grown.key_of("stem#0", Eager::Direction::Forward));
    EXPECT_TRUE(small_stem != grown_stem);
}

/** @brief Tests that add(a, b) and add(b, a) over two results of the region hash apart. */
TEST(Small_RegionSignature, ARewiredOperandChangesTheHash)
{
    Recipe changed;
    changed.swaps_operands = true;

    const RegionSignature base = signature_of(Recipe{});
    const RegionSignature other = signature_of(changed);

    EXPECT_EQ(base.ops, other.ops);
    EXPECT_NE(base.hash, other.hash);
}

/** @brief Tests that reading the second result of a node hashes apart from reading the first. */
TEST(Small_RegionSignature, AReadOfAnotherResultChangesTheHash)
{
    const RegionSignature first = signature_of(build_reader_of_result(0), K_BODY);
    const RegionSignature second = signature_of(build_reader_of_result(1), K_BODY);

    EXPECT_EQ(first.ops, second.ops);
    EXPECT_NE(first.hash, second.hash);
}

/** @brief Tests that the same reader hashes apart when the value it reads has another identity. */
TEST(Small_RegionSignature, ABoundaryValueOfAnotherIdentityChangesTheHash)
{
    ExternalRecipe changed;
    changed.identity = K_EXTERNAL_IDENTITY + 1;

    const RegionSignature base = signature_of(ExternalRecipe{});
    const RegionSignature other = signature_of(changed);

    EXPECT_EQ(base.ops, other.ops);
    EXPECT_NE(base.hash, other.hash);
}

/** @brief Tests that a leaf and a result of another region hash alike under one identity. */
TEST(Small_RegionSignature, ALeafAndABoundaryValueOfOneIdentityHashAlike)
{
    ExternalRecipe leaf;
    leaf.reads_leaf = true;

    const RegionSignature boundary = signature_of(ExternalRecipe{});
    const RegionSignature from_leaf = signature_of(leaf);

    EXPECT_EQ(boundary.hash, from_leaf.hash);
    EXPECT_TRUE(boundary == from_leaf);
}

/** @brief Tests that a result of a region the partition does not list hashes by its identity. */
TEST(Small_RegionSignature, AnOperandFromAnUnlistedRegionHashesAsExternal)
{
    ExternalRecipe unlisted;
    unlisted.lists_producer = false;

    const RegionSignature listed = signature_of(ExternalRecipe{});
    const RegionSignature other = signature_of(unlisted);

    EXPECT_EQ(listed.hash, other.hash);
    EXPECT_TRUE(listed == other);
}

/** @brief Tests that a result of the region hashes apart from one of another region. */
TEST(Small_RegionSignature, AnOperandInsideTheRegionHashesApartFromOneOutside)
{
    const RegionSignature inside_first =
        signature_of(build(ProbeOperands::InsideThenOutside), K_READER);
    const RegionSignature outside_first =
        signature_of(build(ProbeOperands::OutsideThenInside), K_READER);

    EXPECT_EQ(inside_first.ops, outside_first.ops);
    EXPECT_NE(inside_first.hash, outside_first.hash);
}

/** @brief Tests that a leaf operand hashes apart from an operand of the region. */
TEST(Small_RegionSignature, ALeafOperandHashesApartFromOneInsideTheRegion)
{
    const RegionSignature inside_first =
        signature_of(build(ProbeOperands::InsideThenLeaves), K_READER);
    const RegionSignature leaves_first =
        signature_of(build(ProbeOperands::LeavesThenInside), K_READER);

    EXPECT_EQ(inside_first.ops, leaves_first.ops);
    EXPECT_NE(inside_first.hash, leaves_first.hash);
}

/** @brief Tests that two signatures with one hash and different operations compare unequal. */
TEST(Small_RegionSignature, EqualHashesWithDifferentOpsCompareUnequal)
{
    const RegionSignature added{42, {Ops::K_ADD.id}};
    const RegionSignature subtracted{42, {Ops::K_SUB.id}};
    const RegionSignature added_again{42, {Ops::K_ADD.id}};

    EXPECT_FALSE(added == subtracted);
    EXPECT_TRUE(added != subtracted);
    EXPECT_TRUE(added == added_again);
    EXPECT_FALSE(added != added_again);
}

/** @brief Tests that asking for a key the partition does not list raises an Error. */
TEST(Small_RegionSignature, ARegionThePartitionDoesNotListThrows)
{
    const Traced traced = build(Recipe{});

    SA_EXPECT_ERROR((void)signature_of(traced, K_BODY + 5), "Region 6", "not listed");
    SA_EXPECT_ERROR((void)signature_of(traced, Eager::K_NO_REGION), "Region 0", "not listed");
}

/** @brief Tests that a partition sized for another graph raises an Error. */
TEST(Small_RegionSignature, APartitionOfAnotherNodeCountThrows)
{
    Traced traced = build(Recipe{});
    traced.partition.region_of_node.pop_back();

    SA_EXPECT_ERROR((void)signature_of(traced, K_BODY), "assigns 4 nodes", "holds 5");
}

/** @brief Tests that an external operand whose id lies beyond the identities raises an Error. */
TEST(Small_RegionSignature, AnExternalValueWithoutAnIdentityThrows)
{
    Traced traced = build(Recipe{});
    traced.identity.clear();

    SA_EXPECT_ERROR((void)signature_of(traced, K_BODY), "External value 0", "no identity");
}
