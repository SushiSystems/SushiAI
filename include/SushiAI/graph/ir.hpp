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
         * Deliberately a small fixed array rather than a map: a node with more
         * than a handful of scalar knobs is a design smell rather than a case
         * to accommodate. Nothing downstream truncates it - `lowering.cpp`
         * reads slots with @c get() and hands the values straight to
         * SushiBLAS, so no attribute is ever copied into a fixed-width
         * metadata array. The capacity is simply what the operations need.
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
             * Ten: CV-5's eight geometry slots - kernel_h, kernel_w, stride_h,
             * stride_w, dilation_h, dilation_w, pad_top, pad_left - plus the
             * two output extents, @c out_h in slot 8 and @c out_w in slot 9.
             *
             * The extents have to be carried rather than re-derived. Deriving
             * them from the geometry assumes the padding is symmetric, and an
             * asymmetrically-padded window grid can derive to its own
             * transpose: a [1, 4, 5, C] input with a 3x3 window padded 1 above
             * and 1 left, 2 below and 0 right, is really 5x4 windows but
             * derives to 4x5. Both are 20 windows, so every shape check
             * agrees and the gather silently returns the wrong numbers.
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
         * @brief Which axis order a rank-4 value's extents are in.
         *
         * Deliberately not @c SushiAI::Layout, which aliases SushiBLAS's
         * @c {ROW_MAJOR, COLUMN_MAJOR} - that is a matrix's storage order and a
         * different question. Mixed-layout GEMM is a real BLAS use case;
         * a layout change in the middle of a convolutional network is a bug,
         * which is why this one is a graph-wide invariant (CV-4) rather than a
         * per-operand choice.
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
                    values_.push_back(Value{shape, dtype, TensorLayout::NHWC, lifetime,
                                            NO_NODE, false, debug_name});
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

                /**
                 * @brief Inserts an operation *before* node @p before.
                 *
                 * The entry point a graph-to-graph pass needs and @ref add_node
                 * cannot serve. Appending is enough while a graph is being
                 * built, because a builder emits producers before consumers by
                 * construction. A pass that recomputes an already-consumed
                 * value has the opposite problem: the consumers it is rewiring
                 * are already in @c nodes_, so an appended producer would sit
                 * *after* its own readers. That is not a cosmetic ordering
                 * complaint - @c MemoryPlan::build() rests on "insertion order
                 * is a topological order" and rejects the graph by name, and a
                 * lowering that got past it would record a GEMM reading a
                 * buffer nothing had written yet.
                 *
                 * Every value whose producer sits at or after @p before has its
                 * @c Value::producer bumped, so the ids a caller is holding go
                 * on naming the same values; node ids do not survive, exactly
                 * as they do not survive @c insert_casts().
                 *
                 * The ordering guard runs in both directions, because a
                 * one-sided one would let through exactly the arrangement this
                 * function exists to prevent. Rejecting a result that already
                 * has a producer is the single-assignment check and is not the
                 * ordering check: a result with @e no producer can still be
                 * *read* by a node before @p before, and inserting its producer
                 * after that reader would build the out-of-order graph
                 * @c MemoryPlan::build() then refuses - the guard's own contract
                 * broken by the guard.
                 *
                 * @warning Linear in both tables per call: the producer bump
                 *          sweeps every value and the reader check sweeps every
                 *          node before @p before. That is nothing at one call
                 *          per rewritten operation, which is what the one caller
                 *          does, but it is not the primitive to build a bulk
                 *          rewrite on - a pass inserting many nodes should
                 *          rebuild the node vector once instead.
                 *
                 * @param before The position to insert at; @c node_count()
                 *        appends. Every operand must already be produced
                 *        strictly before it, and no result may already be read
                 *        before it, or the result would be the out-of-order
                 *        graph this function exists to avoid.
                 * @param node The operation, its operands and its results.
                 * @return The inserted node's id, which is @p before.
                 * @throws Error If @p before is past the end, an operand or
                 *         result id is unknown, an operand's producer is not
                 *         strictly before @p before, a result is already read by
                 *         a node before @p before, or a result already has a
                 *         producer.
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

                        // The mirror of the operand check above. A result
                        // nothing produces yet may still already be read, and
                        // inserting its producer after that reader is the same
                        // out-of-order graph from the other side.
                        for (NodeId reader = 0; reader < before; ++reader)
                            for (ValueId in : nodes_[reader].inputs)
                                SA_THROW_IF(in == v,
                                            "Operation {} would be inserted at {} but writes "
                                            "value {}, which node {} reads. Insertion order must "
                                            "stay a topological order; insert before the result's "
                                            "first reader.",
                                            Ops::name_of(node.op), before, v, reader);
                    }

                    // Bump first, then claim: claiming first would bump the
                    // producer this call is establishing.
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
                 * @brief Completes the forward graph: propagation, then proof.
                 *
                 * Call once, after the forward graph is complete and before
                 * differentiating. Named for the phase rather than for one of
                 * its steps, so that gaining the layout proof did not make the
                 * name a lie.
                 *
                 * @throws Error If any node other than @c LAYOUT_CONVERT
                 *         produces a result whose layout differs from its
                 *         operands' (CV-4).
                 */
                void finalise()
                {
                    propagate_requires_grad();
                    prove_layout_uniformity();
                }

                /**
                 * @brief Proves no node but LAYOUT_CONVERT changes the layout.
                 *
                 * Public because @c finalise() is not the only place the
                 * property can be lost. It holds of the graph the builder
                 * emits; the passes that rebuild the graph - @c fuse() and
                 * @c insert_casts() - copy the value table entry by entry, and
                 * a copy that drops the layout tag resets every value to
                 * @c add_value's NHWC default. That failure is silent by
                 * construction: the shapes still line up, every kernel still
                 * runs, and the model trains on transposed data. So each
                 * rebuilding pass re-proves what it claims to have preserved.
                 *
                 * Checked per node rather than by sweeping every value for a
                 * single dominant layout, because the sweep could not express
                 * the one legitimate exemption without also accepting a graph
                 * that is half converted.
                 *
                 * @throws Error If any node other than @c LAYOUT_CONVERT reads
                 *         operands in two layouts, or produces a result whose
                 *         layout differs from its operands' (CV-4).
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
