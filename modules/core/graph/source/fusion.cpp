/**************************************************************************/
/* fusion.cpp                                                             */
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
 * @file fusion.cpp
 * @brief Implements graph fusion pattern matching and graph rewriting passes.
 * @author Mustafa Garip
 */

#include <algorithm>
#include <array>
#include <utility>
#include <SushiLog/text/text_format.hpp>
#include <SushiLog/text/value_text.hpp>
#include <SushiAI/graph/fusion.hpp>
#include <SushiAI/failure/error.hpp>

namespace SushiAI
{
    namespace Graph
    {
        namespace
        {
            /** @brief The longest chain any catalog entry replaces. */
            constexpr std::size_t K_MAX_CHAIN = 3;

            /** @brief @c CatalogEntry::spill for an entry that writes only its result. */
            constexpr std::uint8_t K_NO_SPILL = 0xFF;

            /**
             * @struct FusedOperands
             * @brief What the fused node reads, and the scalars it carries.
             */
            struct FusedOperands
            {
                std::vector<ValueID> inputs;
                Attributes attributes{};
            };

            /**
             * @struct CatalogEntry
             * @brief Declarative description of a pre-instantiated fused kernel.
             *
             * @c chain is in producer order with the root last. @c link defines
             * operand positions connecting chain steps to prevent operand inversion.
             */
            struct CatalogEntry
            {
                FusedForm form;
                OpID fused_op;
                std::uint8_t length;
                std::array<OpID, K_MAX_CHAIN> chain;
                std::array<std::uint8_t, K_MAX_CHAIN> link;

                /**
                 * @brief Index of the chain step whose result is preserved as second output.
                 *
                 * Specifies @c K_NO_SPILL when only the root result is written, or an
                 * index into @c chain to retain an auxiliary output across fusion.
                 */
                std::uint8_t spill;

                /** @brief Gathers the fused node's operands from the matched chain. */
                FusedOperands (*compose)(SushiRuntime::span<const Node* const> chain);
            };

            /**
             * @brief Gathers operands for the GEMM-epilogue family: [A, B, bias].
             * @param chain Matched nodes in producer order.
             * @return Fused node operands and preserved matmul attributes.
             */
            FusedOperands compose_gemm_epilogue(SushiRuntime::span<const Node* const> chain)
            {
                const Node& matmul = *chain[0];
                const Node& bias_add = *chain[1];

                FusedOperands operands;
                operands.inputs = {matmul.inputs[0], matmul.inputs[1], bias_add.inputs[1]};
                operands.attributes = matmul.attributes;
                return operands;
            }

            /**
             * @brief Operands for @c K_EXP_SUB_SCALE: [a, b], with alpha in slot 0.
             * @param chain The matched nodes, in producer order.
             * @return The fused node's operands and attributes.
             */
            FusedOperands compose_exp_sub_scale(SushiRuntime::span<const Node* const> chain)
            {
                const Node& exp = *chain[0];
                const Node& sub = *chain[1];
                const Node& scale = *chain[2];

                FusedOperands operands;
                operands.inputs = {exp.inputs[0], sub.inputs[1]};
                operands.attributes = scale.attributes;
                return operands;
            }

            /**
             * @brief Catalog of pre-instantiated fused patterns.
             *
             * Flat constexpr array avoiding static registration constructors
             * in static library builds; matcher sorts by chain length.
             */
            constexpr std::array<CatalogEntry, 6> K_CATALOG = {{
                {FusedForm::GEMMBiasReLU, Ops::K_GEMM_BIAS_RELU.id, 3,
                 {Ops::K_MATMUL.id, Ops::K_BIAS_ADD.id, Ops::K_RELU.id}, {0, 0, 0}, K_NO_SPILL,
                 &compose_gemm_epilogue},

                {FusedForm::GEMMBiasSigmoid, Ops::K_GEMM_BIAS_SIGMOID.id, 3,
                 {Ops::K_MATMUL.id, Ops::K_BIAS_ADD.id, Ops::K_SIGMOID.id}, {0, 0, 0}, K_NO_SPILL,
                 &compose_gemm_epilogue},

                {FusedForm::GEMMBiasTanh, Ops::K_GEMM_BIAS_TANH.id, 3,
                 {Ops::K_MATMUL.id, Ops::K_BIAS_ADD.id, Ops::K_TANH.id}, {0, 0, 0}, K_NO_SPILL,
                 &compose_gemm_epilogue},

                // Spills BIAS_ADD output at step 1 required for GELU backward differentiation.
                {FusedForm::GEMMBiasGELU, Ops::K_GEMM_BIAS_GELU.id, 3,
                 {Ops::K_MATMUL.id, Ops::K_BIAS_ADD.id, Ops::K_GELU.id}, {0, 0, 0}, 1,
                 &compose_gemm_epilogue},

                {FusedForm::GEMMBias, Ops::K_GEMM_BIAS.id, 2,
                 {Ops::K_MATMUL.id, Ops::K_BIAS_ADD.id, Ops::K_LEAF.id}, {0, 0, 0}, K_NO_SPILL,
                 &compose_gemm_epilogue},

                {FusedForm::ExpSubScale, Ops::K_EXP_SUB_SCALE.id, 3,
                 {Ops::K_EXP.id, Ops::K_SUB.id, Ops::K_SCALE.id}, {0, 0, 0}, K_NO_SPILL,
                 &compose_exp_sub_scale},
            }};

            /**
             * @brief Proves at compile time that all folded activations retain backward inputs.
             *
             * Validates that input-differentiating activations specify a valid auxiliary spill.
             * @return True if all catalog activation entries meet backward gradient requirements.
             */
            constexpr bool every_fused_activation_survives()
            {
                for (const CatalogEntry& entry : K_CATALOG)
                {
                    const OpID last = entry.chain[entry.length - 1];
                    if (Ops::activation_of(last) == nullptr) continue;
                    if (Ops::activation_survives_fusion(last)) continue;

                    // Verifies that input-differentiating activation has matching spill.
                    if (entry.spill == K_NO_SPILL) return false;
                    if (entry.spill != entry.length - 2) return false;
                    if (entry.link[entry.length - 1] != 0) return false;
                }
                return true;
            }

            static_assert(every_fused_activation_survives(),
                          "A catalog entry folds an activation whose gradient differentiates at "
                          "its input, without spilling that tensor to an auxiliary output - or it "
                          "spills the wrong step of the chain. Either way the backward pass would "
                          "read something else and the model would train towards the wrong place "
                          "without any test failing. Give the entry a `spill` naming the step that "
                          "produces the activation's operand and an epilogue arm that writes it "
                          "(see GEMM_BIAS_GELU), or do not add the entry.");

            /**
             * @brief Whether @p form is switched off by @p options.
             * @param options The caller's request.
             * @param form The catalog entry.
             * @return True when the caller listed it.
             */
            bool is_disabled(const FusionOptions& options, FusedForm form) noexcept
            {
                return std::find(options.disabled.begin(), options.disabled.end(), form)
                    != options.disabled.end();
            }

            /**
             * @class Candidate
             * @brief One accepted match, staged until the rewrite runs.
             */
            struct Candidate
            {
                const CatalogEntry* entry = nullptr;
                std::array<NodeID, K_MAX_CHAIN> chain{};
            };

            /**
             * @class Matcher
             * @brief One run of the pass: the facts it needs, and the decisions it made.
             *
             * Split out of @c fuse() so the safety rules are readable as
             * predicates rather than as branches nested inside two loops.
             */
            class Matcher
            {
                public:
                    Matcher(const Graph& graph, const FusionOptions& options)
                        : graph_(graph),
                          options_(options),
                          use_count_(graph.value_count(), 0),
                          consumers_(graph.value_count()),
                          observed_(graph.value_count(), false),
                          claimed_(graph.node_count(), false),
                          root_(graph.node_count(), K_NO_NODE)
                    {
                        // Tracks read counts and consumer node IDs for safety and spill ordering.
                        for (NodeID id = 0; id < static_cast<NodeID>(graph.node_count()); ++id)
                            for (ValueID in : graph.node(id).inputs)
                            {
                                ++use_count_[in];
                                consumers_[in].push_back(id);
                            }

                        // Marks persistent values as externally observed to prevent elimination.
                        for (ValueID id = 0; id < graph.value_count(); ++id)
                            observed_[id] = is_persistent(graph.value(id).lifetime);

                        for (ValueID id : options.observed)
                        {
                            SA_THROW_IF(id >= graph.value_count(),
                                        "FusionOptions::observed names value {}, which this "
                                        "graph does not have", id);
                            observed_[id] = true;
                        }

                        candidates_.resize(graph.node_count());
                    }

                    /** @brief Runs the match, filling the candidate table and the declines. */
                    void run();

                    /** @brief The accepted match rooted at @p node, or null. */
                    const Candidate* candidate(NodeID node) const noexcept
                    {
                        return candidates_[node].entry ? &candidates_[node] : nullptr;
                    }

                    /** @brief Whether @p node was absorbed into some other node's match. */
                    bool absorbed(NodeID node) const noexcept
                    {
                        return claimed_[node] && root_[node] != node;
                    }

                    /** @brief Every decline worth reporting. */
                    std::vector<DeclinedFusion>& declined() noexcept { return declined_; }

                private:
                    /**
                     * @brief Evaluates match of @p entry ending at @p root node.
                     * @param entry Catalog entry candidate.
                     * @param root Root node to anchor chain.
                     * @return True if pattern matches and satisfies safety checks.
                     */
                    bool try_entry(const CatalogEntry& entry, NodeID root,
                                   std::array<NodeID, K_MAX_CHAIN>& chain,
                                   ValueID& blocked, FusionDecline& reason) const;

                    /**
                     * @brief Why @p value cannot be fused away, or NONE when it can.
                     * @param value The chain's interior value.
                     * @return The blocking reason.
                     */
                    FusionDecline interior_verdict(ValueID value) const;

                    /**
                     * @brief Checks whether spilled @p value readers execute after @p root.
                     * @param value Spilled value identifier.
                     * @param chain_consumer Absorbed consumer node.
                     * @param root Replacement root node.
                     * @return True if topological ordering is preserved.
                     */
                    bool spill_is_ordered(ValueID value, NodeID chain_consumer,
                                          NodeID root) const;

                    /**
                     * @brief Diagnoses and logs activations excluded from GEMM epilogue fusion.
                     */
                    void explain_unfused_activations();

                    const Graph& graph_;
                    const FusionOptions& options_;
                    std::vector<std::uint32_t> use_count_;
                    std::vector<std::vector<NodeID>> consumers_;
                    std::vector<bool> observed_;
                    std::vector<bool> claimed_;
                    std::vector<NodeID> root_;
                    std::vector<Candidate> candidates_;
                    std::vector<DeclinedFusion> declined_;
            };

            FusionDecline Matcher::interior_verdict(ValueID value) const
            {
                if (observed_[value]) return FusionDecline::ValueIsObserved;
                if (use_count_[value] == 1) return FusionDecline::None;
                return FusionDecline::ValueHasOtherConsumers;
            }

            bool Matcher::spill_is_ordered(ValueID value, NodeID chain_consumer, NodeID root) const
            {
                for (NodeID consumer : consumers_[value])
                    if (consumer != chain_consumer && consumer < root) return false;
                return true;
            }

            void Matcher::explain_unfused_activations()
            {
                for (NodeID id = 0; id < static_cast<NodeID>(graph_.node_count()); ++id)
                {
                    if (claimed_[id]) continue;

                    const Node& node = graph_.node(id);
                    if (Ops::activation_of(node.op) == nullptr) continue;
                    if (node.inputs.empty()) continue;

                    // Reports activation exclusion only when preceding producer successfully fused.
                    const NodeID producer = graph_.value(node.inputs[0]).producer;
                    if (producer == K_NO_NODE || root_[producer] == K_NO_NODE) continue;

                    const Candidate& claimed = candidates_[root_[producer]];
                    if (claimed.entry == nullptr) continue;
                    if (!is_gemm_epilogue(claimed.entry->form)) continue;

                    if (catalog_folds_activation(node.op)) continue;

                    // Categorizes decline between missing catalog row and missing required spill.
                    declined_.push_back(DeclinedFusion{
                        FusedForm::None,
                        node.op,
                        id,
                        node.inputs[0],
                        Ops::activation_survives_fusion(node.op)
                            ? FusionDecline::NoCatalogEntry
                            : FusionDecline::GradientNeedsPreActivation,
                    });
                }
            }

            bool Matcher::try_entry(const CatalogEntry& entry, NodeID root,
                                    std::array<NodeID, K_MAX_CHAIN>& chain,
                                    ValueID& blocked, FusionDecline& reason) const
            {
                blocked = K_NO_VALUE;
                reason = FusionDecline::None;

                const std::size_t length = entry.length;
                chain[length - 1] = root;

                // Walks backwards along declared producer edges to validate chain connectivity.
                for (std::size_t step = length - 1; step > 0; --step)
                {
                    const Node& consumer = graph_.node(chain[step]);
                    const std::size_t position = entry.link[step];

                    if (consumer.inputs.size() <= position) return false;

                    const ValueID edge = consumer.inputs[position];
                    const NodeID producer = graph_.value(edge).producer;

                    if (producer == K_NO_NODE) return false;
                    if (graph_.node(producer).op != entry.chain[step - 1]) return false;
                    if (graph_.node(producer).outputs.size() != 1) return false;
                    if (claimed_[producer]) return false;

                    // Validates topological ordering for preserved auxiliary spill edge.
                    if (entry.spill != K_NO_SPILL && step - 1 == entry.spill)
                    {
                        if (!spill_is_ordered(edge, chain[step], root))
                        {
                            blocked = edge;
                            reason = FusionDecline::SpillReadTooEarly;
                            return false;
                        }
                    }
                    else
                    {
                        const FusionDecline verdict = interior_verdict(edge);
                        if (verdict != FusionDecline::None)
                        {
                            blocked = edge;
                            reason = verdict;
                            return false;
                        }
                    }

                    chain[step - 1] = producer;
                }

                if (is_disabled(options_, entry.form))
                {
                    reason = FusionDecline::Disabled;
                    return false;
                }

                return true;
            }

            void Matcher::run()
            {
                // Sorts catalog entries descending by length using stable sort for determinism.
                std::array<const CatalogEntry*, K_CATALOG.size()> ordered{};
                for (std::size_t i = 0; i < K_CATALOG.size(); ++i) ordered[i] = &K_CATALOG[i];
                std::stable_sort(ordered.begin(), ordered.end(),
                                 [](const CatalogEntry* a, const CatalogEntry* b)
                                 { return a->length > b->length; });

                // Traverses in reverse topological order so longer downstream patterns claim chains first.
                for (std::size_t offset = graph_.node_count(); offset-- > 0; )
                {
                    const NodeID root = static_cast<NodeID>(offset);
                    if (claimed_[root]) continue;

                    for (const CatalogEntry* entry : ordered)
                    {
                        if (graph_.node(root).op != entry->chain[entry->length - 1]) continue;
                        if (graph_.node(root).outputs.size() != 1) continue;

                        std::array<NodeID, K_MAX_CHAIN> chain{};
                        ValueID blocked = K_NO_VALUE;
                        FusionDecline reason = FusionDecline::None;

                        if (try_entry(*entry, root, chain, blocked, reason))
                        {
                            candidates_[root] = Candidate{entry, chain};
                            for (std::size_t step = 0; step < entry->length; ++step)
                            {
                                claimed_[chain[step]] = true;
                                root_[chain[step]] = root;
                            }
                            break;
                        }

                        // Records diagnostic decline when matched candidate violates safety rules.
                        if (reason != FusionDecline::None)
                            declined_.push_back(DeclinedFusion{
                                entry->form, graph_.node(root).op, root, blocked, reason});
                    }
                }

                explain_unfused_activations();
            }
        }

        FusedForm fused_form_from_name(std::string_view name) noexcept
        {
            for (const CatalogEntry& entry : K_CATALOG)
                if (fused_form_name(entry.form) == name) return entry.form;
            return FusedForm::None;
        }

        OpID fused_form_op(FusedForm form) noexcept
        {
            for (const CatalogEntry& entry : K_CATALOG)
                if (entry.form == form) return entry.fused_op;
            return Ops::K_LEAF.id;
        }

        FusedForm fused_form_of(OpID op) noexcept
        {
            for (const CatalogEntry& entry : K_CATALOG)
                if (entry.fused_op == op) return entry.form;
            return FusedForm::None;
        }

        OpID fused_form_activation(FusedForm form) noexcept
        {
            for (const CatalogEntry& entry : K_CATALOG)
            {
                if (entry.form != form) continue;

                const OpID last = entry.chain[entry.length - 1];
                return Ops::activation_of(last) != nullptr ? last : Ops::K_LEAF.id;
            }
            return Ops::K_LEAF.id;
        }

        bool fused_form_spills(FusedForm form) noexcept
        {
            for (const CatalogEntry& entry : K_CATALOG)
                if (entry.form == form) return entry.spill != K_NO_SPILL;
            return false;
        }

        bool catalog_folds_activation(OpID activation) noexcept
        {
            if (Ops::activation_of(activation) == nullptr) return false;

            for (const CatalogEntry& entry : K_CATALOG)
                if (entry.chain[entry.length - 1] == activation) return true;
            return false;
        }

        std::size_t FusionReport::count(FusedForm form) const noexcept
        {
            std::size_t total = 0;
            for (const FusedSubtree& subtree : fused_)
                if (subtree.form == form) ++total;
            return total;
        }

        std::string FusionReport::to_string() const
        {
            if (!enabled_) return "fusion: disabled; the graph was lowered as it was built\n";

            std::string text = SushiLog::format_text(
                "fusion: {} node(s) -> {}, {} subtree(s) fused\n", before_, after_, fused_.size());

            for (const FusedSubtree& subtree : fused_)
            {
                text += SushiLog::format_text("  fused {} nodes [",
                                              SushiLog::left(fused_form_name(subtree.form), 18));
                for (std::size_t i = 0; i < subtree.chain.size(); ++i)
                    text += SushiLog::format_text("{}{}", i ? ", " : "", subtree.chain[i]);
                text += SushiLog::format_text("] -> value {}", subtree.result);

                // Formats spilled auxiliary value alongside primary result.
                if (subtree.spilled != K_NO_VALUE)
                    text += SushiLog::format_text(", spilling the pre-activation to value {}",
                                                    subtree.spilled);

                if (!subtree.eliminated.empty())
                {
                    text += ", dropping value";
                    text += subtree.eliminated.size() > 1 ? "s " : " ";
                    for (std::size_t i = 0; i < subtree.eliminated.size(); ++i)
                        text += SushiLog::format_text("{}{}", i ? ", " : "", subtree.eliminated[i]);
                }
                text += "\n";
            }

            for (const DeclinedFusion& decline : declined_)
            {
                // Formats decline label using fused form name or raw operation name.
                const std::string_view label = decline.form != FusedForm::None
                    ? fused_form_name(decline.form)
                    : Ops::name_of(decline.op);

                text += SushiLog::format_text("  declined {} at node {}",
                                              SushiLog::left(label, 15), decline.root);
                if (decline.blocked_by != K_NO_VALUE)
                    text += SushiLog::format_text(" (value {})", decline.blocked_by);
                text += SushiLog::format_text(": {}\n", fusion_decline_reason(decline.reason));
            }

            return text;
        }

        FusionReport fuse(Graph& graph, const FusionOptions& options)
        {
            FusionReport report;
            report.before_ = graph.node_count();
            report.after_ = graph.node_count();
            report.enabled_ = options.enabled;

            if (!options.enabled) return report;

            // Verifies topological ordering to ensure backward edge matching correctness.
            for (NodeID id = 0; id < static_cast<NodeID>(graph.node_count()); ++id)
                for (ValueID in : graph.node(id).inputs)
                {
                    const NodeID producer = graph.value(in).producer;
                    SA_THROW_IF(producer != K_NO_NODE && producer >= id,
                                "Node {} reads value {}, which node {} writes later. The fusion "
                                "pass needs the graph's insertion order to be a topological "
                                "order; emit the producer first.",
                                id, in, producer);
                }

            Matcher matcher(graph, options);
            matcher.run();

            // Callers hold value ids and state leaves across the pass; only nodes are rebuilt.
            Graph rebuilt = graph.without_nodes();

            for (NodeID id = 0; id < static_cast<NodeID>(graph.node_count()); ++id)
            {
                const Candidate* candidate = matcher.candidate(id);

                if (candidate == nullptr)
                {
                    if (matcher.absorbed(id)) continue;

                    const Node& node = graph.node(id);
                    rebuilt.add_node(node.op, node.inputs, node.outputs, node.attributes);
                    continue;
                }

                const CatalogEntry& entry = *candidate->entry;

                std::array<const Node*, K_MAX_CHAIN> chain{};
                for (std::size_t step = 0; step < entry.length; ++step)
                    chain[step] = &graph.node(candidate->chain[step]);

                const FusedOperands operands = entry.compose(
                    SushiRuntime::span<const Node* const>(chain.data(), entry.length));

                FusedSubtree subtree;
                subtree.form = entry.form;
                subtree.result = chain[entry.length - 1]->outputs[0];
                if (entry.spill != K_NO_SPILL)
                    subtree.spilled = chain[entry.spill]->outputs[0];

                for (std::size_t step = 0; step < entry.length; ++step)
                {
                    subtree.chain.push_back(candidate->chain[step]);

                    // Preserves spilled auxiliary value while recording intermediate eliminations.
                    if (step + 1 < entry.length && step != entry.spill)
                        subtree.eliminated.push_back(chain[step]->outputs[0]);
                }

                // Primary result occupies index 0 with optional spilled value at index 1.
                std::vector<ValueID> outputs{subtree.result};
                if (subtree.spilled != K_NO_VALUE) outputs.push_back(subtree.spilled);

                rebuilt.add_node(entry.fused_op, operands.inputs, std::move(outputs),
                                 operands.attributes);

                report.fused_.push_back(std::move(subtree));
            }

            report.declined_ = std::move(matcher.declined());
            report.after_ = rebuilt.node_count();

            graph = std::move(rebuilt);

            // Validates that rebuilt values preserve required layout uniformity.
            graph.prove_layout_uniformity();

            return report;
        }

    } // namespace Graph
} // namespace SushiAI
