/**************************************************************************/
/* eager_test_support.hpp                                                 */
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
 * @file eager_test_support.hpp
 * @brief Declares the traced steps the host-lane tests of the eager module share.
 * @author Mustafa Garip
 * @see tests/core/eager/README.md
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include <SushiRuntime/core/span.hpp>

#include <SushiAI/autograd/backward.hpp>
#include <SushiAI/autograd/provenance.hpp>
#include <SushiAI/core/ops.hpp>
#include <SushiAI/core/shape.hpp>
#include <SushiAI/eager/partition.hpp>
#include <SushiAI/eager/region_boundary.hpp>
#include <SushiAI/eager/scope.hpp>
#include <SushiAI/eager/scope_partitioner.hpp>
#include <SushiAI/eager/scope_trace.hpp>
#include <SushiAI/eager/segment.hpp>
#include <SushiAI/eager/storage_id.hpp>
#include <SushiAI/graph/builder.hpp>
#include <SushiAI/graph/ir.hpp>

/** @brief Holds what the tests of the eager module share. */
namespace SushiAI::Eager::Support
{
    /** @brief Holds the first storage id a traced step gives its leaves. */
    inline constexpr StorageID K_LEAF_STORAGE = 1000;

    /** @brief Holds the first storage id a traced step gives its parameter gradients. */
    inline constexpr StorageID K_GRADIENT_STORAGE = 2000;

    /** @brief Names the activation that closes the head of encoder_head_loss(). */
    enum class Head : std::uint8_t
    {
        Tanh, /**< The head is tanh(linear(hidden)). */
        ReLU, /**< The head is relu(linear(hidden)). */
    };

    /** @brief Holds one traced step with everything the eager units read from it. */
    struct TracedStep
    {
        Graph::Graph graph;                     /**< The forward and backward nodes. */
        std::vector<Segment> segments;          /**< The segments of the forward nodes. */
        Autograd::Provenance provenance;        /**< The origin of each backward node. */
        Partition partition;                    /**< The region of each node. */
        std::vector<Graph::ValueID> parameters; /**< The parameter leaves, in trace order. */
        std::vector<Graph::ValueID> gradients;  /**< The gradient of each parameter. */
        Graph::ValueID hidden = Graph::K_NO_VALUE; /**< The encoder's result. */
        Graph::ValueID logits = Graph::K_NO_VALUE; /**< The head's result. */
        Graph::ValueID loss = Graph::K_NO_VALUE;   /**< The value differentiated. */
    };

    /** @brief Views @p values as a span. */
    template<typename T>
    [[nodiscard]] SushiRuntime::span<const T> span_of(const std::vector<T>& values) noexcept
    {
        return SushiRuntime::span<const T>(values.data(), values.size());
    }

    /**
     * @brief Emits a linear layer with new parameters and appends them to @p parameters.
     * @return The layer's result, before any activation.
     */
    inline Graph::ValueID linear(const Graph::GraphBuilder& builder, Graph::ValueID x,
                                 int64_t inputs, int64_t outputs,
                                 std::vector<Graph::ValueID>& parameters)
    {
        const Graph::ValueID weight = builder.parameter(Shape{inputs, outputs}, "weight");
        const Graph::ValueID bias = builder.parameter(Shape{outputs}, "bias");
        parameters.push_back(weight);
        parameters.push_back(bias);
        return builder.bias_add(builder.matmul(x, weight), bias);
    }

    /** @brief Seals @p trace, differentiates @p step at its loss and partitions it. */
    inline void finish(TracedStep& step, ScopeTrace& trace)
    {
        step.segments = trace.seal();
        step.gradients = Autograd::differentiate(step.graph, step.loss,
                                                 span_of(step.parameters), Graph::K_NO_VALUE,
                                                 &step.provenance);
        step.partition =
            ScopePartitioner{}.assign(step.graph, span_of(step.segments), step.provenance);
    }

    /** @brief Seals @p trace and partitions @p step without differentiating it. */
    inline void finish_forward(TracedStep& step, ScopeTrace& trace)
    {
        step.segments = trace.seal();
        step.provenance.forward_node_count = step.graph.node_count();
        step.provenance.origin.clear();
        step.partition =
            ScopePartitioner{}.assign(step.graph, span_of(step.segments), step.provenance);
    }

    /**
     * @brief Traces scopes "encoder", "head" and "loss" over an [8, 6] input.
     * @param extra Whether a scope "extra" doubles the logits between the head and the loss.
     * @return A differentiated step of three segments, or four with @p extra.
     */
    inline TracedStep encoder_head_loss(Head head, bool extra = false)
    {
        TracedStep step;
        const Graph::GraphBuilder builder(step.graph);
        ScopeTrace trace(step.graph);

        {
            const Scope scope(trace, "encoder");
            const Graph::ValueID x = builder.input(Shape{8, 6}, "x");
            step.hidden = builder.relu(linear(builder, x, 6, 4, step.parameters));
        }

        {
            const Scope scope(trace, "head");
            const Graph::ValueID projected = linear(builder, step.hidden, 4, 3, step.parameters);
            step.logits = head == Head::Tanh ? builder.tanh(projected) : builder.relu(projected);
        }

        Graph::ValueID scored = step.logits;
        if (extra)
        {
            const Scope scope(trace, "extra");
            scored = builder.scale(step.logits, 2.0);
        }

        {
            const Scope scope(trace, "loss");
            const Graph::ValueID targets = builder.input(Shape{8, 3}, "targets");
            step.loss = builder.nll_loss(scored, targets);
        }

        finish(step, trace);
        return step;
    }

    /**
     * @brief Traces relu(linear(x)) over an [8, 6] input with no backward pass.
     * @param split Whether a scope closes between the linear layer and the activation.
     * @return A step of one segment, or two with @p split.
     */
    inline TracedStep linear_then_relu(bool split)
    {
        TracedStep step;
        const Graph::GraphBuilder builder(step.graph);
        ScopeTrace trace(step.graph);

        Graph::ValueID projected = Graph::K_NO_VALUE;
        {
            const Scope scope(trace, "linear");
            const Graph::ValueID x = builder.input(Shape{8, 6}, "x");
            projected = linear(builder, x, 6, 4, step.parameters);
            if (!split)
                step.hidden = builder.relu(projected);
        }

        if (split)
        {
            const Scope scope(trace, "activation");
            step.hidden = builder.relu(projected);
        }

        finish_forward(step, trace);
        return step;
    }

    /** @brief Returns the loss and every gradient of @p step, the values a session observes. */
    [[nodiscard]] inline std::vector<Graph::ValueID> observed_of(const TracedStep& step)
    {
        std::vector<Graph::ValueID> observed;
        if (step.loss != Graph::K_NO_VALUE)
            observed.push_back(step.loss);
        else
            observed.push_back(step.hidden);

        for (Graph::ValueID gradient : step.gradients)
        {
            if (gradient != Graph::K_NO_VALUE)
                observed.push_back(gradient);
        }

        return observed;
    }

    /**
     * @brief Returns the storage a session would place the leaves and gradients of @p step in.
     * @return One id per value: K_LEAF_STORAGE plus the leaf's position among the leaves,
     *         K_GRADIENT_STORAGE plus the parameter's position, and K_NO_STORAGE elsewhere.
     */
    [[nodiscard]] inline std::vector<StorageID> presets_of(const TracedStep& step)
    {
        std::vector<StorageID> preset(step.graph.value_count(), K_NO_STORAGE);

        StorageID next_leaf = K_LEAF_STORAGE;
        for (const Graph::Node& node : step.graph.nodes())
        {
            if (node.op != Ops::K_LEAF.id)
                continue;

            for (Graph::ValueID leaf : node.outputs)
                preset[leaf] = next_leaf++;
        }

        for (std::size_t i = 0; i < step.gradients.size(); ++i)
        {
            if (step.gradients[i] != Graph::K_NO_VALUE)
                preset[step.gradients[i]] = K_GRADIENT_STORAGE + i;
        }

        return preset;
    }

    /** @brief Returns the boundaries of @p step with its loss and gradients observed. */
    [[nodiscard]] inline BoundaryMap boundaries_of(const TracedStep& step)
    {
        const std::vector<Graph::ValueID> observed = observed_of(step);
        return find_boundaries(step.graph, step.partition, span_of(observed));
    }

    /** @brief Returns the position in Partition::regions of the region named @p name. */
    [[nodiscard]] inline std::size_t region_at(const TracedStep& step, const char* name,
                                               Direction direction)
    {
        for (std::size_t i = 0; i < step.partition.regions.size(); ++i)
        {
            const RegionInfo& info = step.partition.regions[i];
            if (info.name == name && info.direction == direction)
                return i;
        }

        return step.partition.regions.size();
    }

} // namespace SushiAI::Eager::Support
