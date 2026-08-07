/**************************************************************************/
/* ir.hpp                                                                 */
/**************************************************************************/
/*                          This file is part of:                         */
/*                                 SushiAI                                */
/*                https://github.com/SushiSystems/SushiAI                 */
/*                         https://sushisystems.io                        */
/**************************************************************************/
/* Copyright (c) 2026-present Mustafa Garip & Sushi Systems               */
/*                                                                        */
/* Licensed under the Apache License, Version 2.0 (the "License");        */
/* you may not use this file except in compliance with the License.       */
/* You may obtain a copy of the License at                                */
/*                                                                        */
/*     http://www.apache.org/licenses/LICENSE-2.0                         */
/*                                                                        */
/* Unless required by applicable law or agreed to in writing, software    */
/* distributed under the License is distributed on an "AS IS" BASIS,      */
/* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or        */
/* implied. See the License for the specific language governing           */
/* permissions and limitations under the License.                         */
/**************************************************************************/

#pragma once

/**
 * @file ir.hpp
 * @brief SushiAI's own graph IR: the thing autograd, fusion and the memory
 *        planner all transform, before anything is lowered to SushiRuntime.
 *
 * The central design decision of the whole library lives here. A backward
 * rule does not call SushiBLAS and does not touch a @c TaskGraph. It reads
 * this IR and writes more of this IR. Four reasons, in order of how much
 * each would cost to retrofit:
 *
 * 1. A @c TaskGraph node carries @c ResourceRegion byte ranges, and a byte
 *    range cannot be known until the memory planner has assigned an arena
 *    offset - which needs the *whole* forward-and-backward graph to compute
 *    liveness over. Emitting device work during differentiation would force
 *    the offsets to be decided before the information needed to decide them
 *    exists.
 * 2. Fusion partitions a DAG. There is no DAG to partition if differentiation
 *    already flushed its output into the runtime.
 * 3. Differentiating a graph whose nodes are themselves IR keeps the
 *    transform closed over its own output, which is the single property that
 *    makes higher-order derivatives fall out instead of needing a second
 *    mechanism. PyTorch's AOTAutograd cannot do double-backward precisely
 *    because it lacks this closure.
 * 4. @c TaskGraph::compile() drains its builder, so it is not a structure one
 *    can hold, analyse and rewrite. This IR is.
 *
 * The pipeline is therefore: build forward IR -> differentiate (IR to IR) ->
 * fuse (IR to IR) -> plan memory (annotate IR) -> lower to TaskGraph exactly
 * once -> compile once -> replay.
 *
 * Values are single-assignment. No operation mutates its inputs, so no
 * aliasing analysis is needed to differentiate correctly; in-place execution
 * is reintroduced later by the planner, which has the liveness proof that
 * makes it safe. This header is part of the host lane and pulls in no SYCL.
 */

#include <vector>
#include <cstdint>
#include <string_view>
#include <SushiAI/core/ops.hpp>
#include <SushiAI/core/shape.hpp>
#include <SushiAI/core/common.hpp>
#include <SushiRuntime/graph/task_types.hpp>

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
         * @brief How long a value's storage must live, and therefore which arena owns it.
         *
         * The memory planner packs one arena per class rather than one arena
         * for everything. That is not tidiness: SushiRuntime's dependency
         * tracker shards its access history on the base pointer of an
         * allocation, so a single arena collapses every shard into one chain
         * and turns hazard resolution quadratic. Distinct bases never
         * serialise against each other, so splitting by lifetime restores the
         * sharding for free - and the classes below are exactly the groups
         * whose live ranges do not interleave anyway.
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
         * @brief An operation's scalar configuration, stored inline.
         *
         * Deliberately a small fixed array rather than a map: it is copied
         * into the lowered TaskMetadata, which itself holds only eight inline
         * slots, and a node with more than a handful of scalar knobs is a
         * design smell rather than a case to accommodate.
         *
         * @warning Do not put a per-step value here. A learning rate, a loss
         *          scale or an RNG counter baked into an attribute becomes a
         *          compile-time constant, and changing it then forces a
         *          recompile - which is exactly what the compile-once
         *          guarantee exists to prevent. Per-step values are graph
         *          inputs read from device memory.
         */
        struct Attributes
        {
            /**
             * @brief How many slots are meaningful.
             *
             * Eight, because that is @c SushiRuntime::TaskMetadata's
             * @c K_MAX_PARAMS (task_types.hpp:146) - the array an attribute is
             * ultimately copied into. A ninth slot would build here and be
             * silently dropped at lowering, so the two are pinned together by
             * the static_assert below rather than left to agree by memory.
             */
            static constexpr std::size_t CAPACITY = 8;

            /** @brief The scalar values, interpreted per operation. */
            double values[CAPACITY] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

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

        static_assert(Attributes::CAPACITY <= SushiRuntime::Graph::TaskMetadata::K_MAX_PARAMS,
                      "A graph attribute slot has no lowered destination: TaskMetadata "
                      "carries K_MAX_PARAMS scalars and an attribute beyond that would "
                      "be dropped by graph/lowering.cpp without any diagnostic.");

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
         * Owns its values and nodes by index. Indices rather than pointers
         * because every pass here rewrites the graph - appending backward
         * nodes, replacing a fused run with one node, inserting a
         * rematerialised copy - and a vector that reallocates would invalidate
         * every pointer a pass was holding.
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
                    values_.push_back(Value{shape, dtype, lifetime, NO_NODE, false, debug_name});
                    return static_cast<ValueId>(values_.size() - 1);
                }

                /**
                 * @brief Appends an operation and claims its outputs.
                 *
                 * Every output must be a value no other node already produces;
                 * that is what makes the graph single-assignment, and it is
                 * checked rather than assumed because a double-write would
                 * produce a graph that differentiates to silently wrong
                 * gradients rather than to an error.
                 *
                 * @param op The operation's identity.
                 * @param inputs The values it reads.
                 * @param outputs The values it writes.
                 * @param attributes The operation's scalar configuration.
                 * @return The new node's id.
                 * @throws Error If an input or output id is unknown, or an
                 *         output already has a producer.
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
                 * @brief Propagates @c requires_grad forward from the leaves.
                 *
                 * A value needs a gradient exactly when some input it descends
                 * from needs one. One forward sweep suffices because insertion
                 * order is topological. Call once, after the forward graph is
                 * complete and before differentiating.
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

            private:
                std::vector<Value> values_;
                std::vector<Node> nodes_;
        };

    } // namespace Graph
} // namespace SushiAI
