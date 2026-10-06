/**************************************************************************/
/* cast_insertion.cpp                                                     */
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
 * @file cast_insertion.cpp
 * @brief Implements precision cast insertion and graph rewriting passes.
 * @author Mustafa Garip
 */

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
             * @brief Checks whether the pass leaves the operation unchanged.
             *
             * Leaves and cast nodes are structurally exempt from precision modifications.
             */
            bool is_structural(OpID op) noexcept
            {
                return op == Ops::LEAF.id || op == Ops::CAST.id;
            }

            /**
             * @brief Checks if a data type is eligible for precision narrowing.
             *
             * Non-floating types such as index masks are excluded from cast policies.
             *
             * @param dtype Element data type to inspect.
             * @return True if real floating type.
             */
            bool has_precision(DType dtype) noexcept
            {
                return is_real_float(dtype);
            }

            /**
             * @brief Resolves target computation data type for a node under policy.
             *
             * @param node The graph node to inspect.
             * @param declared Table of declared value data types.
             * @param policy Active precision policy.
             * @return Target computation data type.
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

                // If policy retains declared precision, no conversion check is required.
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
             * @brief Deduplication key pairing a source value with target data type.
             *
             * Ensures identical conversions are materialized at most once per value.
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

            // Verify topological insertion order before emitting preceding cast nodes.
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

            // Snapshot initial declared precision before rewriting graph nodes.
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

            // Retain declared types for persistent or explicitly observed values.
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
                if (!has_precision(declared[id])) continue;

                final_dtype[id] = computed[producer];
                if (final_dtype[id] != declared[id]) ++report.retyped_;
            }

            // Rebuild nodes into a fresh graph to preserve single-assignment invariants.
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

                // Preserving layout tag avoids silent defaults to NHWC across passes.
                rebuilt.value(fresh).layout = value.layout;
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
                    if (!has_precision(rebuilt.value(in).dtype)) { inputs.push_back(in); continue; }

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

                // Pinned results compute in low precision and are widened into external buffers.
                std::vector<ValueId> outputs;
                std::vector<std::pair<ValueId, ValueId>> widenings;
                outputs.reserve(node.outputs.size());

                for (ValueId out : node.outputs)
                {
                    if (rebuilt.value(out).dtype == wanted) { outputs.push_back(out); continue; }
                    if (!has_precision(rebuilt.value(out).dtype)) { outputs.push_back(out); continue; }

                    // Copy fields by value to avoid dangling references during vector reallocation.
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

            // Re-prove layout uniformity across the rebuilt graph.
            graph.prove_layout_uniformity();

            // Validate that every operand matches the computed precision of its node.
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

                // Skip non-floating values exempt from precision policies.
                for (ValueId in : node.inputs)
                    if (has_precision(graph.value(in).dtype) && graph.value(in).dtype != wanted)
                        mismatches.push_back(DTypeMismatch{
                            id, node.op, in, graph.value(in).dtype, wanted, false});

                for (ValueId out : node.outputs)
                    if (has_precision(graph.value(out).dtype) && graph.value(out).dtype != wanted)
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
