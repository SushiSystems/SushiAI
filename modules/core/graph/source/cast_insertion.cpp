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
#include <SushiLog/text/text_format.hpp>
#include <SushiAI/graph/builder.hpp>
#include <SushiAI/graph/cast_insertion.hpp>
#include <SushiAI/failure/check.hpp>

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
                return op == Ops::K_LEAF.id || op == Ops::K_CAST.id;
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
             * @param node The graph node to inspect.
             * @param declared Table of declared value data types.
             * @param policy Active precision policy.
             * @return The type, or InvalidDescription for a non-float conversion.
             * @pre @p node writes at least one value; a node without one stops the process.
             */
            Result<DType> computed_dtype(const Node& node,
                                         const std::vector<DType>& declared,
                                         const PrecisionPolicy& policy)
            {
                SA_FATAL_IF(node.outputs.empty(),
                            "{} declares no result, so there is no precision to reason about. "
                            "Every operation in this IR writes at least one value.",
                            Ops::name_of(node.op));

                PrecisionQuery query;
                query.op = node.op;
                query.declared = declared[node.outputs[0]];

                const PrecisionChoice choice = policy.choose(query);

                // If policy retains declared precision, no conversion check is required.
                if (choice.dtype == query.declared) return choice.dtype;

                SA_FAIL_IF(!is_real_float(choice.dtype) || !is_real_float(query.declared),
                           StatusCode::InvalidDescription,
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
                ValueID value;
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
            std::string text = SushiLog::format_text(
                "precision      : {}\n"
                "casts inserted : {} ({} nodes -> {})\n",
                policy_.empty() ? std::string_view{"fp32"} : policy_,
                casts_.size(), before_, after_);

            if (casts_.empty()) return text;

            text += SushiLog::format_text("values retyped : {}\n", retyped_);

            for (const InsertedCast& cast : casts_)
                text += SushiLog::format_text(
                    "  {} {} -> {} for {} ({})\n",
                    cast.feeds_operand ? "operand" : "result ",
                    dtype_name(cast.from), dtype_name(cast.to),
                    Ops::name_of(cast.op),
                    cast.feeds_operand ? "narrowed before the operation"
                                       : "widened into the buffer that outlives the step");

            return text;
        }

        Result<CastReport> insert_casts(Graph& graph, const CastOptions& options)
        {
            const UniformPrecisionPolicy fallback;
            const PrecisionPolicy& policy =
                options.policy != nullptr ? *options.policy : fallback;

            CastReport report;
            report.before_ = graph.node_count();
            report.after_ = graph.node_count();
            report.policy_ = policy.name();

            const std::size_t value_count = graph.value_count();
            const std::size_t node_count = graph.node_count();
            if (value_count == 0) return std::move(report);

            // Verify topological insertion order before emitting preceding cast nodes.
            for (NodeID id = 0; id < static_cast<NodeID>(node_count); ++id)
                for (ValueID in : graph.node(id).inputs)
                {
                    const NodeID producer = graph.value(in).producer;
                    SA_FATAL_IF(producer != K_NO_NODE && producer >= id,
                                "Node {} reads value {}, which node {} writes later. The cast "
                                "pass needs the graph's insertion order to be a topological "
                                "order; emit the producer first.",
                                id, in, producer);
                }

            // A graph whose operands mix layouts is refused before it is rewritten.
            SA_TRY(graph.prove_layout_uniformity());

            // Snapshot initial declared precision before rewriting graph nodes.
            std::vector<DType> declared(value_count);
            for (ValueID id = 0; id < static_cast<ValueID>(value_count); ++id)
                declared[id] = graph.value(id).dtype;

            std::vector<DType> computed(node_count, DType::Float32);
            for (NodeID id = 0; id < static_cast<NodeID>(node_count); ++id)
            {
                const Node& node = graph.node(id);
                if (is_structural(node.op))
                {
                    computed[id] = declared[node.outputs.at(0)];
                    continue;
                }
                SA_TRY_ASSIGN(computed[id], computed_dtype(node, declared, policy));
            }

            // Retain declared types for persistent or explicitly observed values.
            std::vector<bool> pinned(value_count, false);
            for (ValueID id = 0; id < static_cast<ValueID>(value_count); ++id)
                pinned[id] = is_persistent(graph.value(id).lifetime);

            for (ValueID id : options.observed)
            {
                SA_FATAL_IF(id >= value_count,
                            "CastOptions::observed names value {}, which this graph does not "
                            "have", id);
                pinned[id] = true;
            }

            std::vector<DType> final_dtype = declared;
            for (ValueID id = 0; id < static_cast<ValueID>(value_count); ++id)
            {
                const NodeID producer = graph.value(id).producer;
                if (pinned[id] || producer == K_NO_NODE) continue;
                if (is_structural(graph.node(producer).op)) continue;
                if (!has_precision(declared[id])) continue;

                final_dtype[id] = computed[producer];
                if (final_dtype[id] != declared[id]) ++report.retyped_;
            }

            // Callers hold value ids and state leaves across the pass; only nodes are rebuilt.
            Graph rebuilt = graph.without_nodes();

            for (ValueID id = 0; id < static_cast<ValueID>(value_count); ++id)
                rebuilt.value(id).dtype = final_dtype[id];

            const GraphBuilder builder(rebuilt);
            std::unordered_map<CastKey, ValueID, CastKeyHash> converted;

            for (NodeID id = 0; id < static_cast<NodeID>(node_count); ++id)
            {
                const Node& node = graph.node(id);

                if (is_structural(node.op))
                {
                    rebuilt.add_node(node.op, node.inputs, node.outputs, node.attributes);
                    continue;
                }

                const DType wanted = computed[id];

                std::vector<ValueID> inputs;
                inputs.reserve(node.inputs.size());

                for (ValueID in : node.inputs)
                {
                    if (rebuilt.value(in).dtype == wanted) { inputs.push_back(in); continue; }
                    if (!has_precision(rebuilt.value(in).dtype)) { inputs.push_back(in); continue; }

                    const CastKey key{in, static_cast<std::int32_t>(wanted)};
                    const auto found = converted.find(key);

                    if (found != converted.end()) { inputs.push_back(found->second); continue; }

                    const DType from = rebuilt.value(in).dtype;
                    // Both types carry precision, so a refused cast is a defect: value() is fatal.
                    const ValueID narrowed =
                        builder.cast(in, wanted, cast_label(from, wanted)).value();

                    converted.emplace(key, narrowed);
                    inputs.push_back(narrowed);

                    report.casts_.push_back(
                        InsertedCast{id, node.op, in, narrowed, from, wanted, true});
                }

                // Pinned results compute in low precision and are widened into external buffers.
                std::vector<ValueID> outputs;
                std::vector<std::pair<ValueID, ValueID>> widenings;
                outputs.reserve(node.outputs.size());

                for (ValueID out : node.outputs)
                {
                    if (rebuilt.value(out).dtype == wanted) { outputs.push_back(out); continue; }
                    if (!has_precision(rebuilt.value(out).dtype)) { outputs.push_back(out); continue; }

                    // Copy fields by value to avoid dangling references during vector reallocation.
                    const Shape shape = rebuilt.value(out).shape;
                    const DType master = rebuilt.value(out).dtype;
                    const bool needs_grad = rebuilt.value(out).requires_grad;

                    const ValueID computed_result =
                        rebuilt.add_value(shape, wanted, Lifetime::Activation,
                                          cast_label(wanted, master));

                    rebuilt.value(computed_result).requires_grad = needs_grad;

                    outputs.push_back(computed_result);
                    widenings.emplace_back(computed_result, out);

                    report.casts_.push_back(
                        InsertedCast{id, node.op, computed_result, out, wanted, master, false});
                }

                rebuilt.add_node(node.op, std::move(inputs), std::move(outputs), node.attributes);

                for (const std::pair<ValueID, ValueID>& widening : widenings)
                {
                    const Status widened = builder.cast_into(widening.first, widening.second);
                    SA_FATAL_IF(!widened.is_ok(), "The cast pass could not widen value {}: {}",
                                widening.second, widened.message());
                }
            }

            report.after_ = rebuilt.node_count();
            graph = std::move(rebuilt);

            // The input was proven above, so a rebuilt graph that fails the proof is a defect here.
            const Status proven = graph.prove_layout_uniformity();
            SA_FATAL_IF(!proven.is_ok(),
                        "The cast pass broke the layout uniformity of a proven graph: {}. This "
                        "is a defect in insert_casts(), not in the caller's graph.",
                        proven.message());

            // Validate that every operand matches the computed precision of its node.
            const std::vector<DTypeMismatch> mismatches = find_dtype_mismatches(graph, policy);
            SA_FATAL_IF(!mismatches.empty(),
                        "The cast pass left {} node(s) whose operands disagree with the "
                        "precision they compute in - the first is: {}. This is a defect in "
                        "insert_casts(), not in the caller's graph.",
                        mismatches.size(), describe(mismatches.front()));

            return std::move(report);
        }

        std::vector<DTypeMismatch> find_dtype_mismatches(const Graph& graph,
                                                         const PrecisionPolicy& policy)
        {
            std::vector<DTypeMismatch> mismatches;

            for (NodeID id = 0; id < static_cast<NodeID>(graph.node_count()); ++id)
            {
                const Node& node = graph.node(id);
                if (is_structural(node.op) || node.outputs.empty()) continue;

                PrecisionQuery query;
                query.op = node.op;
                query.declared = graph.value(node.outputs[0]).dtype;

                const DType wanted = policy.choose(query).dtype;

                // Skip non-floating values exempt from precision policies.
                for (ValueID in : node.inputs)
                    if (has_precision(graph.value(in).dtype) && graph.value(in).dtype != wanted)
                        mismatches.push_back(DTypeMismatch{
                            id, node.op, in, graph.value(in).dtype, wanted, false});

                for (ValueID out : node.outputs)
                    if (has_precision(graph.value(out).dtype) && graph.value(out).dtype != wanted)
                        mismatches.push_back(DTypeMismatch{
                            id, node.op, out, graph.value(out).dtype, wanted, true});
            }

            return mismatches;
        }

        std::string describe(const DTypeMismatch& mismatch)
        {
            return SushiLog::format_text(
                "node {} ({}) computes in {} but its {} value {} is {}",
                mismatch.node, Ops::name_of(mismatch.op), dtype_name(mismatch.expected),
                mismatch.is_result ? "result" : "operand",
                mismatch.value, dtype_name(mismatch.found));
        }

    } // namespace Graph
} // namespace SushiAI
