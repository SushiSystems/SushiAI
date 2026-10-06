/**************************************************************************/
/* ir.hpp                                                                 */
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
 * @file ir.hpp
 * @brief Declares the core Graph intermediate representation types and classes.
 * @author Mustafa Garip
 */

#pragma once

#include <vector>
#include <cstdint>
#include <string_view>
#include <SushiAI/core/ops.hpp>
#include <SushiAI/core/shape.hpp>
#include <SushiAI/core/common.hpp>

namespace SushiAI
{
    namespace Graph
    {
        /** @brief Identifies a value in a Graph; an index, not a pointer. */
        using ValueId = std::uint32_t;

        /** @brief Identifies a node in a Graph; an index, not a pointer. */
        using NodeId = std::uint32_t;

        /** @brief The absent-value sentinel, used for "no producer" and "no gradient". */
        inline constexpr ValueId NO_VALUE = ~ValueId{0};

        /** @brief The absent-node sentinel, marking a value that no node produces. */
        inline constexpr NodeId NO_NODE = ~NodeId{0};

        /**
         * @enum Lifetime
         * @brief Defines the lifetime class and owning arena category for a value.
         */
        enum class Lifetime : std::uint8_t
        {
            /** @brief A learnable weight. Lives for the whole run. */
            PARAMETER,
            /** @brief A gradient of a parameter. Lives for the whole run. */
            GRADIENT,
            /** @brief Optimizer state such as a momentum or variance buffer. */
            OPTIMIZER_STATE,
            /** @brief A minibatch input or target. Lives for the whole run, rewritten per step. */
            INPUT,
            /** @brief An intermediate. Lives only between its producer and its last reader. */
            ACTIVATION,
            /** @brief A transient the planner may overlap freely with any other scratch. */
            SCRATCH,
        };

        /** @brief A human-readable name for @p lifetime, for diagnostics. */
        constexpr const char* lifetime_name(Lifetime lifetime) noexcept
        {
            switch (lifetime)
            {
                case Lifetime::PARAMETER:       return "parameter";
                case Lifetime::GRADIENT:        return "gradient";
                case Lifetime::OPTIMIZER_STATE: return "optimizer_state";
                case Lifetime::INPUT:           return "input";
                case Lifetime::ACTIVATION:      return "activation";
                case Lifetime::SCRATCH:         return "scratch";
            }
            return "unknown";
        }

        /**
         * @brief True when a value of this class outlives a single step.
         *
         * Persistent values are allocated once and never overlapped; transient
         * ones are what the liveness pass actually packs.
         */
        constexpr bool is_persistent(Lifetime lifetime) noexcept
        {
            return lifetime == Lifetime::PARAMETER
                || lifetime == Lifetime::GRADIENT
                || lifetime == Lifetime::OPTIMIZER_STATE
                || lifetime == Lifetime::INPUT;
        }

        /**
         * @struct Attributes
         * @brief Stores inline scalar configuration values for an operation.
         */
        struct Attributes
        {
            /**
             * @brief Maximum number of supported inline scalar attribute slots.
             */
            static constexpr std::size_t CAPACITY = 10;

            /** @brief The scalar values, interpreted per operation. */
            double values[CAPACITY] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

            /** @brief Reads slot @p index, or zero when it is out of range. */
            constexpr double get(std::size_t index) const noexcept
            {
                return index < CAPACITY ? values[index] : 0.0;
            }

            /** @brief Reads slot @p index as a flag. */
            constexpr bool flag(std::size_t index) const noexcept
            {
                return get(index) != 0.0;
            }
        };

        /**
         * @enum TensorLayout
         * @brief Specifies the dimension ordering for a rank-4 tensor value.
         */
        enum class TensorLayout : std::uint8_t
        {
            /** @brief Channels innermost - the single native layout (CV-3). */
            NHWC,
            /** @brief Channels before the spatial axes; a boundary only. */
            NCHW,
        };

        /**
         * @struct Value
         * @brief One single-assignment tensor-shaped value in the graph.
         */
        struct Value
        {
            /** @brief The value's extents. */
            Shape shape{};

            /** @brief The value's element type. */
            DType dtype = DType::FLOAT32;

            /**
             * @brief The axis order this value's extents are in.
             *
             * Graph-wide rather than free-floating: see @ref TensorLayout and
             * @c Graph::finalise(), which proves every node preserves it.
             */
            TensorLayout layout = TensorLayout::NHWC;

            /** @brief Which arena owns this value's storage. */
            Lifetime lifetime = Lifetime::ACTIVATION;

            /** @brief The node that writes this value, or NO_NODE for a leaf. */
            NodeId producer = NO_NODE;

            /**
             * @brief Whether a gradient flows back through this value.
             *
             * Set on parameters by the caller and propagated forward by
             * Graph::finalise(); the differentiation pass uses it to skip
             * whole subgraphs that cannot contribute, which is what makes a
             * frozen layer cost nothing rather than costing a wasted kernel.
             */
            bool requires_grad = false;

            /** @brief A short label carried into the runtime's profiling report. */
            std::string_view debug_name{};
        };

        /**
         * @struct Node
         * @brief One operation: an identity, its operands, its results.
         */
        struct Node
        {
            /** @brief Which operation this is; see SushiAI::Ops. */
            OpID op{};

            /** @brief The values read, in the order the operation expects them. */
            std::vector<ValueId> inputs;

            /** @brief The values written, in the order the operation produces them. */
            std::vector<ValueId> outputs;

            /** @brief The operation's scalar configuration. */
            Attributes attributes{};
        };

        /**
         * @class Graph
         * @brief A single-assignment DAG of tensor operations.
         *
         * Owns values and nodes by index to prevent pointer invalidation.
         */
        class Graph
        {
            public:
                /**
                 * @brief Declares a value.
                 * @param shape The value's extents.
                 * @param dtype The value's element type.
                 * @param lifetime Which arena will own it.
                 * @param debug_name A label for diagnostics; must outlive the graph.
                 * @return The new value's id.
                 */
                ValueId add_value(Shape shape,
                                  DType dtype,
                                  Lifetime lifetime,
                                  std::string_view debug_name = {})
                {
                    values_.push_back(Value{shape, dtype, TensorLayout::NHWC, lifetime,
                                            NO_NODE, false, debug_name});
                    return static_cast<ValueId>(values_.size() - 1);
                }

                /**
                 * @brief Appends an operation and claims its outputs.
                 *
                 * Verifies single-assignment invariants for each output.
                 *
                 * @return The new node index.
                 * @throws Error If an operand is unknown or output already claimed.
                 */
                NodeId add_node(OpID op,
                                std::vector<ValueId> inputs,
                                std::vector<ValueId> outputs,
                                Attributes attributes = {})
                {
                    for (ValueId v : inputs)
                        SA_THROW_IF(v >= values_.size(),
                                    "Operation {} reads value {}, which does not exist",
                                    Ops::name_of(op), v);

                    const NodeId id = static_cast<NodeId>(nodes_.size());

                    for (ValueId v : outputs)
                    {
                        SA_THROW_IF(v >= values_.size(),
                                    "Operation {} writes value {}, which does not exist",
                                    Ops::name_of(op), v);
                        SA_THROW_IF(values_[v].producer != NO_NODE,
                                    "Operation {} writes value {}, which is already written by {}. "
                                    "The IR is single-assignment; emit a new value instead.",
                                    Ops::name_of(op), v, Ops::name_of(nodes_[values_[v].producer].op));
                        values_[v].producer = id;
                    }

                    nodes_.push_back(Node{op, std::move(inputs), std::move(outputs), attributes});
                    return id;
                }

                /**
                 * @brief Inserts an operation before the specified node.
                 *
                 * Maintains topological order and updates subsequent producer indices.
                 *
                 * @return The inserted node index.
                 * @throws Error If out of bounds or topological order is violated.
                 */
                NodeId insert_node(NodeId before, Node node)
                {
                    SA_THROW_IF(before > nodes_.size(),
                                "Cannot insert before node {}; the graph has {} nodes",
                                before, nodes_.size());

                    for (ValueId v : node.inputs)
                    {
                        SA_THROW_IF(v >= values_.size(),
                                    "Operation {} reads value {}, which does not exist",
                                    Ops::name_of(node.op), v);
                        SA_THROW_IF(values_[v].producer != NO_NODE
                                        && values_[v].producer >= before,
                                    "Operation {} would be inserted at {} but reads value {}, "
                                    "which node {} writes. Insertion order must stay a "
                                    "topological order; insert after the operand's producer.",
                                    Ops::name_of(node.op), before, v, values_[v].producer);
                    }

                    for (ValueId v : node.outputs)
                    {
                        SA_THROW_IF(v >= values_.size(),
                                    "Operation {} writes value {}, which does not exist",
                                    Ops::name_of(node.op), v);
                        SA_THROW_IF(values_[v].producer != NO_NODE,
                                    "Operation {} writes value {}, which is already written by {}. "
                                    "The IR is single-assignment; emit a new value instead.",
                                    Ops::name_of(node.op), v,
                                    Ops::name_of(nodes_[values_[v].producer].op));

                        // A result must not already be read before its proposed insertion point.
                        for (NodeId reader = 0; reader < before; ++reader)
                            for (ValueId in : nodes_[reader].inputs)
                                SA_THROW_IF(in == v,
                                            "Operation {} would be inserted at {} but writes "
                                            "value {}, which node {} reads. Insertion order must "
                                            "stay a topological order; insert before the result's "
                                            "first reader.",
                                            Ops::name_of(node.op), before, v, reader);
                    }

                    // Subsequent producer indices are incremented before claiming this node.
                    for (Value& value : values_)
                        if (value.producer != NO_NODE && value.producer >= before)
                            ++value.producer;

                    for (ValueId v : node.outputs)
                        values_[v].producer = before;

                    nodes_.insert(nodes_.begin() + static_cast<std::ptrdiff_t>(before),
                                  std::move(node));
                    return before;
                }

                /** @brief Read-only access to a value. */
                const Value& value(ValueId id) const
                {
                    SA_THROW_IF(id >= values_.size(), "Value {} does not exist", id);
                    return values_[id];
                }

                /** @brief Mutable access to a value, for passes that annotate. */
                Value& value(ValueId id)
                {
                    SA_THROW_IF(id >= values_.size(), "Value {} does not exist", id);
                    return values_[id];
                }

                /** @brief Read-only access to a node. */
                const Node& node(NodeId id) const
                {
                    SA_THROW_IF(id >= nodes_.size(), "Node {} does not exist", id);
                    return nodes_[id];
                }

                /** @brief Mutable access to a node, for passes that rewrite. */
                Node& node(NodeId id)
                {
                    SA_THROW_IF(id >= nodes_.size(), "Node {} does not exist", id);
                    return nodes_[id];
                }

                /** @brief How many values the graph holds. */
                std::size_t value_count() const noexcept { return values_.size(); }

                /** @brief How many nodes the graph holds. */
                std::size_t node_count() const noexcept { return nodes_.size(); }

                /** @brief Every value, in declaration order. */
                const std::vector<Value>& values() const noexcept { return values_; }

                /**
                 * @brief Every node, in insertion order.
                 *
                 * Insertion order is a valid topological order by construction:
                 * a node can only read values that already exist, and a value
                 * only exists once its producer has been appended.
                 */
                const std::vector<Node>& nodes() const noexcept { return nodes_; }

                /**
                 * @brief Completes the forward graph with propagation and proof.
                 *
                 * Propagates gradient requirements and validates layout uniformity.
                 *
                 * @throws Error If an invalid layout conversion is detected.
                 */
                void finalise()
                {
                    propagate_requires_grad();
                    prove_layout_uniformity();
                }

                /**
                 * @brief Proves no node except LAYOUT_CONVERT modifies layout.
                 *
                 * Re-verified across graph rebuilding passes to prevent silent NHWC resets.
                 *
                 * @throws Error If layout mismatch or invalid conversion is found.
                 */
                void prove_layout_uniformity() const
                {
                    for (const Node& n : nodes_)
                    {
                        if (n.op == Ops::LAYOUT_CONVERT.id || n.inputs.empty())
                            continue;

                        const TensorLayout expected = values_[n.inputs.front()].layout;
                        for (ValueId in : n.inputs)
                            SA_THROW_IF(values_[in].layout != expected,
                                        "Node '{}' reads operands in two different layouts; "
                                        "only LAYOUT_CONVERT may cross layouts (CV-4).",
                                        Ops::name_of(n.op));

                        for (ValueId out : n.outputs)
                            SA_THROW_IF(values_[out].layout != expected,
                                        "Node '{}' produces a result whose layout differs "
                                        "from its operands'. Only LAYOUT_CONVERT may do that "
                                        "(CV-4); everything else would silently train the "
                                        "model on transposed data.",
                                        Ops::name_of(n.op));
                    }
                }

            private:
                /**
                 * @brief Propagates @c requires_grad forward from the leaves.
                 *
                 * A value needs a gradient exactly when some input it descends
                 * from needs one. One forward sweep suffices because insertion
                 * order is topological.
                 */
                void propagate_requires_grad() noexcept
                {
                    for (const Node& n : nodes_)
                    {
                        bool any = false;
                        for (ValueId in : n.inputs)
                            if (values_[in].requires_grad) { any = true; break; }

                        if (any)
                            for (ValueId out : n.outputs)
                                values_[out].requires_grad = true;
                    }
                }

                std::vector<Value> values_;
                std::vector<Node> nodes_;
        };

    } // namespace Graph
} // namespace SushiAI
