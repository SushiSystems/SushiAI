/**************************************************************************/
/* cast_insertion.cpp                                                     */
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

#include <algorithm>
#include <cstdint>
#include <unordered_map>
#include <utility>
#include <SushiBLAS/core/format.hpp>
#include <SushiAI/graph/builder.hpp>
#include <SushiAI/graph/cast_insertion.hpp>

namespace SushiAI
{
    namespace Graph
    {
        namespace
        {
            /**
             * @brief Whether the pass leaves @p op's node entirely alone.
             *
             * A leaf is written from outside the step, so its element type is
             * the caller's declaration and not a computed one; a cast is the
             * single node whose operand and result are *meant* to disagree.
             * These are the same two exemptions @c find_dtype_mismatches makes,
             * and they are listed once so the pass and the checker cannot drift.
             */
            bool is_structural(OpID op) noexcept
            {
                return op == Ops::LEAF.id || op == Ops::CAST.id;
            }

            /**
             * @brief The element type @p node computes in, per @p policy.
             *
             * A node's declared precision is read from its first result rather
             * than from its operands, because the result is the value the
             * builder derived from the operation's own shape and dtype rule -
             * and because a node whose operands already disagree is exactly the
             * state this pass exists to fix, so they are not a reliable answer.
             */
            DType computed_dtype(const Node& node,
                                 const std::vector<DType>& declared,
                                 const IPrecisionPolicy& policy)
            {
                SA_THROW_IF(node.outputs.empty(),
                            "{} declares no result, so there is no precision to reason about. "
                            "Every operation in this IR writes at least one value.",
                            Ops::name_of(node.op));

                PrecisionQuery query;
                query.op = node.op;
                query.declared = declared[node.outputs[0]];

                const PrecisionChoice choice = policy.choose(query);

                // A policy that leaves a node where it found it needs no
                // conversion, so it needs no conversion to be *possible* -
                // which matters, because "the real floating types only" is a
                // property of the cast and not of the IR. Checking it
                // unconditionally would refuse an integer-typed operation that
                // this pass was never going to touch.
                if (choice.dtype == query.declared) return choice.dtype;

                SA_THROW_IF(!is_real_float(choice.dtype) || !is_real_float(query.declared),
                            "A precision policy asked {} to move from {} to {}, and the only "
                            "conversion this stack has is between the real floating types "
                            "(half, float32, float64).",
                            Ops::name_of(node.op), dtype_name(query.declared),
                            dtype_name(choice.dtype));

                return choice.dtype;
            }

            /**
             * @brief One (value, target dtype) pair, so a conversion is made once.
             *
             * A parameter read by the forward matmul and again by the backward
             * one needs the same fp16 copy twice; converting it twice would be
             * two kernels and two buffers for one number. Reuse is safe because
             * the IR is single-assignment - the value the second consumer wants
             * is the value the first one already converted.
             */
            struct CastKey
            {
                ValueId value;
                std::int32_t dtype;

                bool operator==(const CastKey& other) const noexcept
                {
                    return value == other.value && dtype == other.dtype;
                }
            };

            struct CastKeyHash
            {
                std::size_t operator()(const CastKey& key) const noexcept
                {
                    return static_cast<std::size_t>(key.value) * 131u
                         + static_cast<std::size_t>(key.dtype);
                }
            };

            /** @brief A stable label for a cast value, chosen by direction. */
            std::string_view cast_label(DType from, DType to) noexcept
            {
                return dtype_size(to) < dtype_size(from) ? "cast.down" : "cast.up";
            }
        }

        std::size_t CastReport::count_into(DType dtype) const noexcept
        {
            return static_cast<std::size_t>(
                std::count_if(casts_.begin(), casts_.end(),
                              [dtype](const InsertedCast& c) { return c.to == dtype; }));
        }

        std::string CastReport::to_string() const
        {
            std::string text = SushiBLAS::Core::format(
                "precision      : {}\n"
                "casts inserted : {} ({} nodes -> {})\n",
                policy_.empty() ? std::string_view{"fp32"} : policy_,
                casts_.size(), before_, after_);

            if (casts_.empty()) return text;

            text += SushiBLAS::Core::format("values retyped : {}\n", retyped_);

            for (const InsertedCast& cast : casts_)
                text += SushiBLAS::Core::format(
                    "  {} {} -> {} for {} ({})\n",
                    cast.feeds_operand ? "operand" : "result ",
                    dtype_name(cast.from), dtype_name(cast.to),
                    Ops::name_of(cast.op),
                    cast.feeds_operand ? "narrowed before the operation"
                                       : "widened into the buffer that outlives the step");

            return text;
        }

        CastReport insert_casts(Graph& graph, const CastOptions& options)
        {
            const UniformPrecisionPolicy fallback;
            const IPrecisionPolicy& policy =
                options.policy != nullptr ? *options.policy : fallback;

            CastReport report;
            report.before_ = graph.node_count();
            report.after_ = graph.node_count();
            report.policy_ = policy.name();

            const std::size_t value_count = graph.value_count();
            const std::size_t node_count = graph.node_count();
            if (value_count == 0) return report;

            // The rewrite emits a cast immediately before the consumer that
            // needs it, which is only a valid position if a producer is already
            // ahead of its consumer. Graph documents that and GraphBuilder makes
            // it true; a hand-assembled graph can break it, and a broken one
            // would place a conversion after the node that reads it - so it is
            // checked here exactly as fuse() and MemoryPlan::build() check it.
            for (NodeId id = 0; id < static_cast<NodeId>(node_count); ++id)
                for (ValueId in : graph.node(id).inputs)
                {
                    const NodeId producer = graph.value(in).producer;
                    SA_THROW_IF(producer != NO_NODE && producer >= id,
                                "Node {} reads value {}, which node {} writes later. The cast "
                                "pass needs the graph's insertion order to be a topological "
                                "order; emit the producer first.",
                                id, in, producer);
                }

            // --- What each value is today, and what each node computes in -----
            // Snapshotted before anything is rewritten: the policy narrows *from*
            // the graph's own precision, and asking it about a value the pass has
            // already retyped would be asking a different question.
            std::vector<DType> declared(value_count);
            for (ValueId id = 0; id < static_cast<ValueId>(value_count); ++id)
                declared[id] = graph.value(id).dtype;

            std::vector<DType> computed(node_count, DType::FLOAT32);
            for (NodeId id = 0; id < static_cast<NodeId>(node_count); ++id)
            {
                const Node& node = graph.node(id);
                computed[id] = is_structural(node.op)
                             ? declared[node.outputs.at(0)]
                             : computed_dtype(node, declared, policy);
            }

            // --- Which values may be retyped ----------------------------------
            // "A value's element type follows its producer, unless something
            // outside the pass depends on it." See the header; the two ways of
            // depending on it are a persistent lifetime class and an explicit
            // mention by the caller.
            std::vector<bool> pinned(value_count, false);
            for (ValueId id = 0; id < static_cast<ValueId>(value_count); ++id)
                pinned[id] = is_persistent(graph.value(id).lifetime);

            for (ValueId id : options.observed)
            {
                SA_THROW_IF(id >= value_count,
                            "CastOptions::observed names value {}, which this graph does not "
                            "have", id);
                pinned[id] = true;
            }

            std::vector<DType> final_dtype = declared;
            for (ValueId id = 0; id < static_cast<ValueId>(value_count); ++id)
            {
                const NodeId producer = graph.value(id).producer;
                if (pinned[id] || producer == NO_NODE) continue;
                if (is_structural(graph.node(producer).op)) continue;

                final_dtype[id] = computed[producer];
                if (final_dtype[id] != declared[id]) ++report.retyped_;
            }

            // --- Rebuild ------------------------------------------------------
            // A new node list, for the reason fuse() needs one: the IR is
            // single-assignment and add_node claims its outputs, so a node
            // cannot be un-produced in place and a cast cannot be spliced in
            // ahead of its consumer. Value ids are preserved entry for entry;
            // the pass's own values are appended after them.
            Graph rebuilt;

            for (ValueId id = 0; id < static_cast<ValueId>(value_count); ++id)
            {
                const Value& value = graph.value(id);
                const ValueId fresh = rebuilt.add_value(value.shape, final_dtype[id],
                                                        value.lifetime, value.debug_name);

                SA_THROW_IF(fresh != id,
                            "The cast pass copied value {} to {}; ids must be preserved because "
                            "callers hold them across the pass", id, fresh);

                rebuilt.value(fresh).requires_grad = value.requires_grad;
            }

            const GraphBuilder builder(rebuilt);
            std::unordered_map<CastKey, ValueId, CastKeyHash> converted;

            for (NodeId id = 0; id < static_cast<NodeId>(node_count); ++id)
            {
                const Node& node = graph.node(id);

                if (is_structural(node.op))
                {
                    rebuilt.add_node(node.op, node.inputs, node.outputs, node.attributes);
                    continue;
                }

                const DType wanted = computed[id];

                std::vector<ValueId> inputs;
                inputs.reserve(node.inputs.size());

                for (ValueId in : node.inputs)
                {
                    if (rebuilt.value(in).dtype == wanted) { inputs.push_back(in); continue; }

                    const CastKey key{in, static_cast<std::int32_t>(wanted)};
                    const auto found = converted.find(key);

                    if (found != converted.end()) { inputs.push_back(found->second); continue; }

                    const DType from = rebuilt.value(in).dtype;
                    const ValueId narrowed = builder.cast(in, wanted, cast_label(from, wanted));

                    converted.emplace(key, narrowed);
                    inputs.push_back(narrowed);

                    report.casts_.push_back(
                        InsertedCast{id, node.op, in, narrowed, from, wanted, true});
                }

                // A result that is pinned keeps its id, its buffer and its
                // element type; the node writes a fresh low-precision value and
                // one cast widens it into the buffer the outside world holds.
                std::vector<ValueId> outputs;
                std::vector<std::pair<ValueId, ValueId>> widenings;
                outputs.reserve(node.outputs.size());

                for (ValueId out : node.outputs)
                {
                    if (rebuilt.value(out).dtype == wanted) { outputs.push_back(out); continue; }

                    // By value, every field of it. `add_value` grows the value
                    // vector, and a reference into a vector that reallocates is
                    // a use-after-free that only appears once a graph outgrows
                    // its initial capacity - the same trap autograd's
                    // Differentiator snapshots its current node to avoid.
                    const Shape shape = rebuilt.value(out).shape;
                    const DType master = rebuilt.value(out).dtype;
                    const bool needs_grad = rebuilt.value(out).requires_grad;

                    const ValueId computed_result =
                        rebuilt.add_value(shape, wanted, Lifetime::ACTIVATION,
                                          cast_label(wanted, master));

                    rebuilt.value(computed_result).requires_grad = needs_grad;

                    outputs.push_back(computed_result);
                    widenings.emplace_back(computed_result, out);

                    report.casts_.push_back(
                        InsertedCast{id, node.op, computed_result, out, wanted, master, false});
                }

                rebuilt.add_node(node.op, std::move(inputs), std::move(outputs), node.attributes);

                for (const std::pair<ValueId, ValueId>& widening : widenings)
                    builder.cast_into(widening.first, widening.second);
            }

            report.after_ = rebuilt.node_count();
            graph = std::move(rebuilt);

            // The pass's whole contract in one line, checked rather than
            // asserted in a comment. It is O(nodes) against a rewrite that is
            // already O(nodes), and every way of getting the placement wrong
            // produces a graph that still trains - so the alternative to
            // checking is finding out from a loss curve.
            const std::vector<DTypeMismatch> mismatches = find_dtype_mismatches(graph, policy);
            SA_THROW_IF(!mismatches.empty(),
                        "The cast pass left {} node(s) whose operands disagree with the "
                        "precision they compute in - the first is: {}. This is a defect in "
                        "insert_casts(), not in the caller's graph.",
                        mismatches.size(), describe(mismatches.front()));

            return report;
        }

        std::vector<DTypeMismatch> find_dtype_mismatches(const Graph& graph,
                                                         const IPrecisionPolicy& policy)
        {
            std::vector<DTypeMismatch> mismatches;

            for (NodeId id = 0; id < static_cast<NodeId>(graph.node_count()); ++id)
            {
                const Node& node = graph.node(id);
                if (is_structural(node.op) || node.outputs.empty()) continue;

                PrecisionQuery query;
                query.op = node.op;
                query.declared = graph.value(node.outputs[0]).dtype;

                const DType wanted = policy.choose(query).dtype;

                for (ValueId in : node.inputs)
                    if (graph.value(in).dtype != wanted)
                        mismatches.push_back(DTypeMismatch{
                            id, node.op, in, graph.value(in).dtype, wanted, false});

                for (ValueId out : node.outputs)
                    if (graph.value(out).dtype != wanted)
                        mismatches.push_back(DTypeMismatch{
                            id, node.op, out, graph.value(out).dtype, wanted, true});
            }

            return mismatches;
        }

        std::string describe(const DTypeMismatch& mismatch)
        {
            return SushiBLAS::Core::format(
                "node {} ({}) computes in {} but its {} value {} is {}",
                mismatch.node, Ops::name_of(mismatch.op), dtype_name(mismatch.expected),
                mismatch.is_result ? "result" : "operand",
                mismatch.value, dtype_name(mismatch.found));
        }

    } // namespace Graph
} // namespace SushiAI
