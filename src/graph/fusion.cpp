/**************************************************************************/
/* fusion.cpp                                                             */
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
#include <array>
#include <utility>
#include <SushiBLAS/core/format.hpp>
#include <SushiAI/graph/fusion.hpp>

namespace SushiAI
{
    namespace Graph
    {
        namespace
        {
            /** @brief The longest chain any catalog entry replaces. */
            constexpr std::size_t MAX_CHAIN = 3;

            /** @brief @c CatalogEntry::spill for an entry that writes only its result. */
            constexpr std::uint8_t NO_SPILL = 0xFF;

            /**
             * @struct FusedOperands
             * @brief What the fused node reads, and the scalars it carries.
             */
            struct FusedOperands
            {
                std::vector<ValueId> inputs;
                Attributes attributes{};
            };

            /**
             * @struct CatalogEntry
             * @brief One pre-instantiated fused kernel, described declaratively.
             *
             * @c chain is in producer order, so @c chain[0] writes the value
             * @c chain[1] reads and the last entry is the match's root. @c link
             * says which operand position each step reads its predecessor at,
             * which is what stops @c SUB matching with its operands the wrong
             * way round - @c (exp(a) - b) and @c (b - exp(a)) are different
             * functions and the shapes cannot tell them apart.
             */
            struct CatalogEntry
            {
                FusedForm form;
                OpID fused_op;
                std::uint8_t length;
                std::array<OpID, MAX_CHAIN> chain;
                std::array<std::uint8_t, MAX_CHAIN> link;

                /**
                 * @brief Which chain step's result the fused kernel also writes.
                 *
                 * @c NO_SPILL for an entry that writes only the root's result,
                 * which is every entry but one. Otherwise an index into
                 * @c chain: the value that step used to produce becomes the
                 * fused node's *second output*, keeping its id, rather than
                 * being eliminated. See fusion.hpp's @c spill section for why
                 * that is the whole of what makes a GELU epilogue safe.
                 */
                std::uint8_t spill;

                /** @brief Gathers the fused node's operands from the matched chain. */
                FusedOperands (*compose)(SushiRuntime::span<const Node* const> chain);
            };

            /**
             * @brief Operands for the GEMM-epilogue family: [A, B, bias].
             *
             * The transposition flags come straight off the @c MATMUL, because
             * they are the same flags @c Dispatch::execute_gemm takes - a
             * transposed operand is never a view here, for the reason
             * builder.hpp records.
             *
             * @param chain The matched nodes, in producer order.
             * @return The fused node's operands and attributes.
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
             * @brief Operands for @c EXP_SUB_SCALE: [a, b], with alpha in slot 0.
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
             * @brief The catalog.
             *
             * A flat @c constexpr array of plain data and function pointers,
             * deliberately not a registry something registers into: @c sushiai
             * is a STATIC library, so a translation unit whose only job is to
             * run a registration constructor resolves no symbol and is dropped
             * by the linker - silently, and only in a downstream consumer's
             * build. Same argument, same shape, as @c autograd/backward.cpp's
             * rule table and @c core/ops.hpp's identity table.
             *
             * Order does not matter; the matcher sorts by length itself.
             */
            constexpr std::array<CatalogEntry, 6> CATALOG = {{
                {FusedForm::GEMM_BIAS_RELU, Ops::GEMM_BIAS_RELU.id, 3,
                 {Ops::MATMUL.id, Ops::BIAS_ADD.id, Ops::RELU.id}, {0, 0, 0}, NO_SPILL,
                 &compose_gemm_epilogue},

                {FusedForm::GEMM_BIAS_SIGMOID, Ops::GEMM_BIAS_SIGMOID.id, 3,
                 {Ops::MATMUL.id, Ops::BIAS_ADD.id, Ops::SIGMOID.id}, {0, 0, 0}, NO_SPILL,
                 &compose_gemm_epilogue},

                {FusedForm::GEMM_BIAS_TANH, Ops::GEMM_BIAS_TANH.id, 3,
                 {Ops::MATMUL.id, Ops::BIAS_ADD.id, Ops::TANH.id}, {0, 0, 0}, NO_SPILL,
                 &compose_gemm_epilogue},

                // The one row with a spill. Step 1 is the BIAS_ADD, so the value
                // the fused node keeps writing is exactly the tensor
                // gelu_backward differentiates at - which the assertion below
                // checks rather than trusts, because spilling the MATMUL's
                // result instead would be as silently wrong as spilling nothing.
                {FusedForm::GEMM_BIAS_GELU, Ops::GEMM_BIAS_GELU.id, 3,
                 {Ops::MATMUL.id, Ops::BIAS_ADD.id, Ops::GELU.id}, {0, 0, 0}, 1,
                 &compose_gemm_epilogue},

                {FusedForm::GEMM_BIAS, Ops::GEMM_BIAS.id, 2,
                 {Ops::MATMUL.id, Ops::BIAS_ADD.id, Ops::LEAF.id}, {0, 0, 0}, NO_SPILL,
                 &compose_gemm_epilogue},

                {FusedForm::EXP_SUB_SCALE, Ops::EXP_SUB_SCALE.id, 3,
                 {Ops::EXP.id, Ops::SUB.id, Ops::SCALE.id}, {0, 0, 0}, NO_SPILL,
                 &compose_exp_sub_scale},
            }};

            /**
             * @brief Proves every folded activation still has the tensor its gradient reads.
             *
             * This is the whole of Task 2's trap, closed at compile time - and
             * it is a *conditional* proof, not a blanket refusal, because the
             * condition is what the fix consists of. An activation whose
             * gradient differentiates at its **output** may be folded by any
             * entry: the pre-activation was dead already. One that
             * differentiates at its **input** may be folded only by an entry
             * that writes that input to an auxiliary buffer, and only if the
             * value it writes is the one the gradient actually reads - which,
             * with @c link all zero, is the result of the step immediately
             * before the activation.
             *
             * Both halves are checked. A row for a new @c INPUT-gradient
             * activation with @c NO_SPILL fails here; so does one that spills
             * the wrong step. Either way the build stops, rather than the model
             * training towards the wrong place in silence - and "must not"
             * enforced by review is "will, eventually".
             *
             * @c constexpr rather than @c consteval, which is C++20; the
             * static_assert below is the only caller and forces constant
             * evaluation, so the proof still happens at compile time.
             */
            constexpr bool every_fused_activation_survives()
            {
                for (const CatalogEntry& entry : CATALOG)
                {
                    const OpID last = entry.chain[entry.length - 1];
                    if (Ops::activation_of(last) == nullptr) continue;
                    if (Ops::activation_survives_fusion(last)) continue;

                    // From here on: the gradient reads the activation's input,
                    // so the entry has to hand it back.
                    if (entry.spill == NO_SPILL) return false;
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
                std::array<NodeId, MAX_CHAIN> chain{};
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
                          root_(graph.node_count(), NO_NODE)
                    {
                        // Both, and they are not the same fact. `use_count_`
                        // counts reads, so a node reading one value twice
                        // refuses the fold exactly as two nodes would;
                        // `consumers_` names *who*, which is what an entry with
                        // a spill has to ask about - see spill_is_ordered().
                        for (NodeId id = 0; id < static_cast<NodeId>(graph.node_count()); ++id)
                            for (ValueId in : graph.node(id).inputs)
                            {
                                ++use_count_[in];
                                consumers_[in].push_back(id);
                            }

                        // A persistent value's buffer outlives the step, so
                        // something outside the graph is going to read or write
                        // it whatever the graph says. The caller need not list
                        // parameters, gradients, optimizer state or inputs.
                        for (ValueId id = 0; id < graph.value_count(); ++id)
                            observed_[id] = is_persistent(graph.value(id).lifetime);

                        for (ValueId id : options.observed)
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
                    const Candidate* candidate(NodeId node) const noexcept
                    {
                        return candidates_[node].entry ? &candidates_[node] : nullptr;
                    }

                    /** @brief Whether @p node was absorbed into some other node's match. */
                    bool absorbed(NodeId node) const noexcept
                    {
                        return claimed_[node] && root_[node] != node;
                    }

                    /** @brief Every decline worth reporting. */
                    std::vector<DeclinedFusion>& declined() noexcept { return declined_; }

                private:
                    /**
                     * @brief Tries @p entry rooted at @p root.
                     *
                     * @param entry The catalog entry to try.
                     * @param root The node that would be the chain's last.
                     * @param chain Receives the matched node ids on success.
                     * @param blocked Receives the value that refused, if any.
                     * @param reason Receives why, if a structural match was refused.
                     * @return True when the chain matched and every safety rule passed.
                     */
                    bool try_entry(const CatalogEntry& entry, NodeId root,
                                   std::array<NodeId, MAX_CHAIN>& chain,
                                   ValueId& blocked, FusionDecline& reason) const;

                    /**
                     * @brief Why @p value cannot be fused away, or NONE when it can.
                     * @param value The chain's interior value.
                     * @return The blocking reason.
                     */
                    FusionDecline interior_verdict(ValueId value) const;

                    /**
                     * @brief Whether a spilled @p value's readers all run after the fused node.
                     *
                     * The fused node is emitted at @p root's position, later
                     * than the absorbed producer's, so a reader sitting between
                     * the two would read a value written after it. Nothing
                     * about the *numbers* would be wrong; the graph would
                     * simply stop being in topological order, which every pass
                     * below this one takes as given. The chain's own consumer
                     * is excluded because it is absorbed and disappears.
                     *
                     * @param value The value the entry would spill.
                     * @param chain_consumer The chain node that reads it.
                     * @param root The node the fused one will replace.
                     * @return True when the fold keeps the order valid.
                     */
                    bool spill_is_ordered(ValueId value, NodeId chain_consumer,
                                          NodeId root) const;

                    /**
                     * @brief Records the activations that stayed out of a fused GEMM.
                     *
                     * The loop above cannot: an activation with no catalog entry
                     * never starts a match, so nothing declines and the report
                     * would be silent about the single most important decision
                     * the pass makes. Runs after the matching, because it has to
                     * see which chains were claimed.
                     */
                    void explain_unfused_activations();

                    const Graph& graph_;
                    const FusionOptions& options_;
                    std::vector<std::uint32_t> use_count_;
                    std::vector<std::vector<NodeId>> consumers_;
                    std::vector<bool> observed_;
                    std::vector<bool> claimed_;
                    std::vector<NodeId> root_;
                    std::vector<Candidate> candidates_;
                    std::vector<DeclinedFusion> declined_;
            };

            FusionDecline Matcher::interior_verdict(ValueId value) const
            {
                if (observed_[value]) return FusionDecline::VALUE_IS_OBSERVED;
                if (use_count_[value] == 1) return FusionDecline::NONE;
                return FusionDecline::VALUE_HAS_OTHER_CONSUMERS;
            }

            bool Matcher::spill_is_ordered(ValueId value, NodeId chain_consumer, NodeId root) const
            {
                for (NodeId consumer : consumers_[value])
                    if (consumer != chain_consumer && consumer < root) return false;
                return true;
            }

            void Matcher::explain_unfused_activations()
            {
                for (NodeId id = 0; id < static_cast<NodeId>(graph_.node_count()); ++id)
                {
                    if (claimed_[id]) continue;

                    const Node& node = graph_.node(id);
                    if (Ops::activation_of(node.op) == nullptr) continue;
                    if (node.inputs.empty()) continue;

                    // Only worth a line when the thing feeding it *did* fuse:
                    // "the GEMM collapsed and the activation did not" is the
                    // decision a reader is looking for. An activation sitting
                    // anywhere else was never a candidate.
                    const NodeId producer = graph_.value(node.inputs[0]).producer;
                    if (producer == NO_NODE || root_[producer] == NO_NODE) continue;

                    const Candidate& claimed = candidates_[root_[producer]];
                    if (claimed.entry == nullptr) continue;
                    if (!is_gemm_epilogue(claimed.entry->form)) continue;

                    // An activation the catalog *does* compose with a GEMM was
                    // not skipped for want of a row, so whatever stopped it was
                    // situational - disabled, an interior value with a second
                    // consumer, a spill read too early - and the loop above
                    // recorded that at this very node. Adding a second line
                    // saying "no catalog entry composes it" would contradict
                    // the first one; both of the sentences below are about a
                    // composition that does not exist.
                    if (catalog_folds_activation(node.op)) continue;

                    declined_.push_back(DeclinedFusion{
                        FusedForm::NONE,
                        node.op,
                        id,
                        node.inputs[0],
                        // The split that matters to whoever reads this: one is
                        // a row somebody forgot, the other is a row that cannot
                        // be written without an epilogue that spills - which is
                        // the run-time half of every_fused_activation_survives().
                        Ops::activation_survives_fusion(node.op)
                            ? FusionDecline::NO_CATALOG_ENTRY
                            : FusionDecline::GRADIENT_NEEDS_PRE_ACTIVATION,
                    });
                }
            }

            bool Matcher::try_entry(const CatalogEntry& entry, NodeId root,
                                    std::array<NodeId, MAX_CHAIN>& chain,
                                    ValueId& blocked, FusionDecline& reason) const
            {
                blocked = NO_VALUE;
                reason = FusionDecline::NONE;

                const std::size_t length = entry.length;
                chain[length - 1] = root;

                // Walk backwards along the chain. Every step has to be a real
                // producer edge at the declared operand position, and the value
                // on that edge has to be one this pass is allowed to stop
                // writing - or one the entry has undertaken to keep writing.
                for (std::size_t step = length - 1; step > 0; --step)
                {
                    const Node& consumer = graph_.node(chain[step]);
                    const std::size_t position = entry.link[step];

                    if (consumer.inputs.size() <= position) return false;

                    const ValueId edge = consumer.inputs[position];
                    const NodeId producer = graph_.value(edge).producer;

                    if (producer == NO_NODE) return false;
                    if (graph_.node(producer).op != entry.chain[step - 1]) return false;
                    if (graph_.node(producer).outputs.size() != 1) return false;
                    if (claimed_[producer]) return false;

                    // The spilled edge is the one exception to rule 1, and it is
                    // an exception for the only reason that could justify one:
                    // the fused node goes on writing it, under the same id. So
                    // a second consumer is fine and so is an outside reader -
                    // there is a buffer for both. What still has to hold is the
                    // *order*.
                    if (entry.spill != NO_SPILL && step - 1 == entry.spill)
                    {
                        if (!spill_is_ordered(edge, chain[step], root))
                        {
                            blocked = edge;
                            reason = FusionDecline::SPILL_READ_TOO_EARLY;
                            return false;
                        }
                    }
                    else
                    {
                        const FusionDecline verdict = interior_verdict(edge);
                        if (verdict != FusionDecline::NONE)
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
                    reason = FusionDecline::DISABLED;
                    return false;
                }

                return true;
            }

            void Matcher::run()
            {
                // Longest first, so a chain that could be matched by two entries
                // becomes the bigger one. Stable, so the catalog's own order
                // breaks ties and two runs of the same graph fuse identically.
                std::array<const CatalogEntry*, CATALOG.size()> ordered{};
                for (std::size_t i = 0; i < CATALOG.size(); ++i) ordered[i] = &CATALOG[i];
                std::stable_sort(ordered.begin(), ordered.end(),
                                 [](const CatalogEntry* a, const CatalogEntry* b)
                                 { return a->length > b->length; });

                // Roots first. A chain's root is its *last* node, so visiting in
                // reverse insertion order reaches `relu` before `bias_add` and
                // the three-node form claims the chain before the two-node one
                // can take half of it. Forwards, `gemm+bias` would win every
                // time and `gemm+bias+relu` would never match at all.
                for (std::size_t offset = graph_.node_count(); offset-- > 0; )
                {
                    const NodeId root = static_cast<NodeId>(offset);
                    if (claimed_[root]) continue;

                    for (const CatalogEntry* entry : ordered)
                    {
                        if (graph_.node(root).op != entry->chain[entry->length - 1]) continue;
                        if (graph_.node(root).outputs.size() != 1) continue;

                        std::array<NodeId, MAX_CHAIN> chain{};
                        ValueId blocked = NO_VALUE;
                        FusionDecline reason = FusionDecline::NONE;

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

                        // A chain that is simply not in the graph is not worth
                        // reporting; a chain that is there and was refused is
                        // the only thing anyone reads this list for.
                        if (reason != FusionDecline::NONE)
                            declined_.push_back(DeclinedFusion{
                                entry->form, graph_.node(root).op, root, blocked, reason});
                    }
                }

                explain_unfused_activations();
            }
        }

        // =====================================================================
        // Catalog lookups
        // =====================================================================

        FusedForm fused_form_from_name(std::string_view name) noexcept
        {
            for (const CatalogEntry& entry : CATALOG)
                if (fused_form_name(entry.form) == name) return entry.form;
            return FusedForm::NONE;
        }

        OpID fused_form_op(FusedForm form) noexcept
        {
            for (const CatalogEntry& entry : CATALOG)
                if (entry.form == form) return entry.fused_op;
            return Ops::LEAF.id;
        }

        FusedForm fused_form_of(OpID op) noexcept
        {
            for (const CatalogEntry& entry : CATALOG)
                if (entry.fused_op == op) return entry.form;
            return FusedForm::NONE;
        }

        OpID fused_form_activation(FusedForm form) noexcept
        {
            for (const CatalogEntry& entry : CATALOG)
            {
                if (entry.form != form) continue;

                const OpID last = entry.chain[entry.length - 1];
                return Ops::activation_of(last) != nullptr ? last : Ops::LEAF.id;
            }
            return Ops::LEAF.id;
        }

        bool fused_form_spills(FusedForm form) noexcept
        {
            for (const CatalogEntry& entry : CATALOG)
                if (entry.form == form) return entry.spill != NO_SPILL;
            return false;
        }

        bool catalog_folds_activation(OpID activation) noexcept
        {
            if (Ops::activation_of(activation) == nullptr) return false;

            for (const CatalogEntry& entry : CATALOG)
                if (entry.chain[entry.length - 1] == activation) return true;
            return false;
        }

        // =====================================================================
        // FusionReport
        // =====================================================================

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

            std::string text = SushiBLAS::Core::format("fusion: {} node(s) -> {}, {} subtree(s) fused\n",
                                                       before_, after_, fused_.size());

            for (const FusedSubtree& subtree : fused_)
            {
                text += SushiBLAS::Core::format("  fused {:<18} nodes [", fused_form_name(subtree.form));
                for (std::size_t i = 0; i < subtree.chain.size(); ++i)
                    text += SushiBLAS::Core::format("{}{}", i ? ", " : "", subtree.chain[i]);
                text += SushiBLAS::Core::format("] -> value {}", subtree.result);

                // Printed next to the result rather than folded into the
                // dropped list, because it is the opposite of a dropped value
                // and the difference is the whole of what makes this form safe.
                if (subtree.spilled != NO_VALUE)
                    text += SushiBLAS::Core::format(", spilling the pre-activation to value {}",
                                                    subtree.spilled);

                if (!subtree.eliminated.empty())
                {
                    text += ", dropping value";
                    text += subtree.eliminated.size() > 1 ? "s " : " ";
                    for (std::size_t i = 0; i < subtree.eliminated.size(); ++i)
                        text += SushiBLAS::Core::format("{}{}", i ? ", " : "", subtree.eliminated[i]);
                }
                text += "\n";
            }

            for (const DeclinedFusion& decline : declined_)
            {
                // A decline with no form is one the catalog had no entry for,
                // so the operation's own name is the only thing that identifies
                // it - and is what a reader is looking for anyway.
                const std::string_view label = decline.form != FusedForm::NONE
                    ? fused_form_name(decline.form)
                    : Ops::name_of(decline.op);

                text += SushiBLAS::Core::format("  declined {:<15} at node {}", label, decline.root);
                if (decline.blocked_by != NO_VALUE)
                    text += SushiBLAS::Core::format(" (value {})", decline.blocked_by);
                text += SushiBLAS::Core::format(": {}\n", fusion_decline_reason(decline.reason));
            }

            return text;
        }

        // =====================================================================
        // The pass
        // =====================================================================

        FusionReport fuse(Graph& graph, const FusionOptions& options)
        {
            FusionReport report;
            report.before_ = graph.node_count();
            report.after_ = graph.node_count();
            report.enabled_ = options.enabled;

            if (!options.enabled) return report;

            // The matcher walks producer edges and trusts that a producer was
            // appended before its consumer, which Graph documents and
            // GraphBuilder makes true. A hand-assembled graph can break it, and
            // a broken one would silently match a chain that does not execute
            // in that order, so it is checked here rather than assumed.
            for (NodeId id = 0; id < static_cast<NodeId>(graph.node_count()); ++id)
                for (ValueId in : graph.node(id).inputs)
                {
                    const NodeId producer = graph.value(in).producer;
                    SA_THROW_IF(producer != NO_NODE && producer >= id,
                                "Node {} reads value {}, which node {} writes later. The fusion "
                                "pass needs the graph's insertion order to be a topological "
                                "order; emit the producer first.",
                                id, in, producer);
                }

            Matcher matcher(graph, options);
            matcher.run();

            // --- Rebuild ------------------------------------------------------
            // Value ids are preserved entry for entry so every id a caller holds
            // keeps meaning what it meant; see the header. Only the node list is
            // rewritten, which it has to be: the IR is single-assignment and
            // add_node claims its outputs, so a node cannot be un-produced in
            // place.
            Graph rebuilt;

            for (ValueId id = 0; id < static_cast<ValueId>(graph.value_count()); ++id)
            {
                const Value& value = graph.value(id);
                const ValueId fresh = rebuilt.add_value(value.shape, value.dtype,
                                                        value.lifetime, value.debug_name);

                SA_THROW_IF(fresh != id,
                            "The fusion pass copied value {} to {}; ids must be preserved "
                            "because callers hold them across the pass", id, fresh);

                rebuilt.value(fresh).requires_grad = value.requires_grad;
            }

            for (NodeId id = 0; id < static_cast<NodeId>(graph.node_count()); ++id)
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

                std::array<const Node*, MAX_CHAIN> chain{};
                for (std::size_t step = 0; step < entry.length; ++step)
                    chain[step] = &graph.node(candidate->chain[step]);

                const FusedOperands operands = entry.compose(
                    SushiRuntime::span<const Node* const>(chain.data(), entry.length));

                FusedSubtree subtree;
                subtree.form = entry.form;
                subtree.result = chain[entry.length - 1]->outputs[0];
                if (entry.spill != NO_SPILL)
                    subtree.spilled = chain[entry.spill]->outputs[0];

                for (std::size_t step = 0; step < entry.length; ++step)
                {
                    subtree.chain.push_back(candidate->chain[step]);

                    // The spilled step's value is not eliminated - the fused
                    // node keeps writing it, which is the whole point of the
                    // entry having a spill at all.
                    if (step + 1 < entry.length && step != entry.spill)
                        subtree.eliminated.push_back(chain[step]->outputs[0]);
                }

                // Two outputs for a spilling entry, and in this order: the
                // result first, so `outputs[0]` still means "what this node
                // computes" for every walker that reads only the first one -
                // the cast pass's precision query and the planner's alias
                // merge both do.
                std::vector<ValueId> outputs{subtree.result};
                if (subtree.spilled != NO_VALUE) outputs.push_back(subtree.spilled);

                rebuilt.add_node(entry.fused_op, operands.inputs, std::move(outputs),
                                 operands.attributes);

                report.fused_.push_back(std::move(subtree));
            }

            report.declined_ = std::move(matcher.declined());
            report.after_ = rebuilt.node_count();

            graph = std::move(rebuilt);
            return report;
        }

    } // namespace Graph
} // namespace SushiAI
