/**************************************************************************/
/* fusion.hpp                                                             */
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
 * @file fusion.hpp
 * @brief The static tree pass: which subtrees of the IR become one kernel.
 *
 * A pure IR-to-IR function, in the host lane, with no engine and no SYCL - the
 * same shape as @c graph/memory_plan.hpp and for the same reason. It runs
 * after differentiation and before planning, on the joint forward-and-backward
 * DAG, because a forward node's fusability depends on what the backward pass
 * still reads.
 *
 * @section catalog Fusion here is a selector over a catalog, not a code generator
 *
 * This is the constraint the whole design hangs off, so it is stated first.
 * @c sycl_ext_codeplay_kernel_fusion has been **removed** from intel-llvm:
 * nothing in this stack can fuse two kernels at run time. A fused kernel is a
 * C++ template instantiation of a functor composition, so *which compositions
 * exist* is decided when SushiAI is compiled, and a pass running at graph-build
 * time can only **choose among** the ones already instantiated.
 *
 * So this pass is a matcher, not a synthesiser. @c FusedForm enumerates the
 * catalog; @c CATALOG (in fusion.cpp) says which chain of IR operations each
 * entry replaces; the pass rewrites matched chains into single nodes carrying
 * the fused identity. Adding a form means four edits, all of them mechanical:
 * an @c Ops identity, a @c FusedForm entry, a @c CATALOG row, and a recorder in
 * @c graph/lowering.cpp holding the functor itself.
 *
 * @section safety The three ways a fusion is silently wrong
 *
 * Each of these produces a graph that still runs, still converges and still
 * looks healthy, which is why each is a checked rule rather than an assumption.
 *
 * 1. **A value consumed more than once cannot be fused away.** The fused kernel
 *    never writes the chain's interior values. If an interior value has a
 *    second consumer, fusing means either recomputing it or losing it, so the
 *    pass requires a use count of exactly one. It does not assume a chain and
 *    it does not "chase the last use"; it counts.
 *
 * 2. **A value the backward pass differentiates at must survive.** This is
 *    rule 1 again, and deliberately so. The backward pass is *in the graph* by
 *    the time this runs, so a pre-activation the gradient reads is simply a
 *    second consumer, and the use count already refuses. What makes that work
 *    out correctly per activation is @c Ops::ACTIVATIONS: ReLU's gradient reads
 *    its output (bit-identically - see that table), so its pre-activation has
 *    one consumer and folds; GELU's reads its input, so its pre-activation has
 *    two and does not. The catalog is @c static_assert-ed against that table,
 *    so a fused form for a non-recoverable activation cannot be added by
 *    accident.
 *
 * 3. **A value someone outside the graph reads must survive.** The loss and the
 *    logits are ordinary activations to this pass and are read by the training
 *    loop. @c FusionOptions::observed is how a caller says so; the persistent
 *    lifetime classes are protected without being listed.
 *
 * @section exact What fusion is allowed to change about the numbers
 *
 * Nothing, for every form in today's catalog. The stack builds with
 * @c -ffp-contract=off through SushiRuntime's @c sushiruntime_deterministic_fp
 * contract, so collapsing kernels introduces no FMA the separate spelling did
 * not have, and every fused functor is written in the operand order of the
 * operations it replaces. @c FusionOptions::enabled keeps the unfused path
 * reachable so the two can be compared byte for byte, exactly as
 * @c Optim::UpdateKernel::SEPARATE does for the optimizer;
 * @c tests/functional/regression/test_fusion.cpp is that comparison.
 *
 * @section ids Value ids survive the rewrite
 *
 * The IR is single-assignment and @c Graph::add_node claims its outputs, so a
 * node cannot be un-produced in place: the pass builds a new node list. It
 * copies the **value table entry for entry**, so every @c ValueId means what it
 * meant before - which is what lets a caller hold on to @c Train::StepSpec, a
 * parameter's id inside an @c NN::Parameter, or a gradient id across the pass
 * with no remapping anywhere. A value whose only producer was absorbed simply
 * ends up with no producer and no consumer; nothing materialises it, and
 * @c MemoryPlan already declines to pack a value nothing writes.
 */

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <vector>
#include <SushiAI/core/common.hpp>
#include <SushiAI/core/ops.hpp>
#include <SushiAI/graph/ir.hpp>

namespace SushiAI
{
    namespace Graph
    {
        /**
         * @enum FusedForm
         * @brief The catalog: every fused kernel this build actually contains.
         *
         * Not a set of flags to combine - each entry is a separate template
         * instantiation that exists in the binary, and a combination not listed
         * here does not exist. See the file comment.
         */
        enum class FusedForm : std::uint8_t
        {
            /** @brief Not a fused node. */
            NONE,

            /** @brief @c MATMUL then @c BIAS_ADD, as one GEMM with a bias epilogue. */
            GEMM_BIAS,

            /** @brief @c MATMUL, @c BIAS_ADD then @c RELU, as one GEMM. */
            GEMM_BIAS_RELU,

            /** @brief @c MATMUL, @c BIAS_ADD then @c SIGMOID, as one GEMM. */
            GEMM_BIAS_SIGMOID,

            /** @brief @c MATMUL, @c BIAS_ADD then @c TANH, as one GEMM. */
            GEMM_BIAS_TANH,

            /** @brief @c EXP, @c SUB then @c SCALE, as one elementwise pass. */
            EXP_SUB_SCALE,
        };

        /** @brief A human-readable name for @p form, for diagnostics. */
        constexpr std::string_view fused_form_name(FusedForm form) noexcept
        {
            switch (form)
            {
                case FusedForm::NONE:              return "none";
                case FusedForm::GEMM_BIAS:         return "gemm+bias";
                case FusedForm::GEMM_BIAS_RELU:    return "gemm+bias+relu";
                case FusedForm::GEMM_BIAS_SIGMOID: return "gemm+bias+sigmoid";
                case FusedForm::GEMM_BIAS_TANH:    return "gemm+bias+tanh";
                case FusedForm::EXP_SUB_SCALE:     return "exp-sub-scale";
            }
            return "unknown";
        }

        /**
         * @brief The catalog entry @p name refers to.
         *
         * The inverse of @ref fused_form_name, for a caller naming a form from
         * outside the program - a CLI flag, a config file. Exists so that
         * "which forms are switched off" can be said in the same words the
         * report prints, rather than through an index nobody can read.
         *
         * @param name A name as @ref fused_form_name spells it.
         * @return The matching form, or @c FusedForm::NONE when there is none.
         */
        FusedForm fused_form_from_name(std::string_view name) noexcept;

        /**
         * @brief The operation identity a node of @p form carries.
         * @param form The catalog entry.
         * @return Its @c Ops identity, or @c Ops::LEAF.id for @c FusedForm::NONE.
         */
        OpID fused_form_op(FusedForm form) noexcept;

        /**
         * @brief The catalog entry behind an operation identity.
         * @param op The identity to look up.
         * @return The form, or @c FusedForm::NONE when @p op is not a fused one.
         */
        FusedForm fused_form_of(OpID op) noexcept;

        /**
         * @brief Whether @p form is one of the GEMM-epilogue entries.
         *
         * They share a lowering, an operand order ([A, B, bias]) and the
         * ordering obligation their undeclared bias read carries, so asking
         * once is better than four comparisons at each site.
         *
         * @param form The catalog entry.
         * @return True for the @c GEMM_BIAS family.
         */
        constexpr bool is_gemm_epilogue(FusedForm form) noexcept
        {
            return form == FusedForm::GEMM_BIAS
                || form == FusedForm::GEMM_BIAS_RELU
                || form == FusedForm::GEMM_BIAS_SIGMOID
                || form == FusedForm::GEMM_BIAS_TANH;
        }

        /**
         * @brief The activation a GEMM-epilogue form folds in, if any.
         * @param form The catalog entry.
         * @return The forward activation's identity, or @c Ops::LEAF.id when
         *         the form folds no activation.
         */
        OpID fused_form_activation(FusedForm form) noexcept;

        /**
         * @enum FusionDecline
         * @brief Why a structurally matching chain was not fused.
         *
         * Only reasons a reader would want to act on. A chain that simply is
         * not in the graph is not a decline and is not recorded - the report
         * would be nothing but noise.
         */
        enum class FusionDecline : std::uint8_t
        {
            /** @brief Placeholder; never recorded. */
            NONE,

            /** @brief An interior value feeds a consumer outside the chain. */
            VALUE_HAS_OTHER_CONSUMERS,

            /**
             * @brief An interior value is read from outside the graph.
             *
             * Named by @c FusionOptions::observed, or a persistent lifetime
             * class whose buffer outlives the step.
             */
            VALUE_IS_OBSERVED,

            /**
             * @brief The activation's gradient differentiates at its input.
             *
             * The one decline whose alternative is silently wrong rather than
             * merely wasteful, and the reason the catalog has no entry to try
             * in the first place: a fused kernel never writes the
             * pre-activation, so the gradient would be taken at the wrong
             * point, nothing would fail, and the model would train towards the
             * wrong place. See @c Ops::Activation.
             */
            GRADIENT_NEEDS_PRE_ACTIVATION,

            /**
             * @brief The activation could be folded and the catalog has no entry.
             *
             * Not a safety refusal - a missing row. It is reported because a
             * new activation whose gradient reads its output is fusable the day
             * it is added, and the only thing standing between it and a free
             * launch is a line in @c CATALOG that somebody forgot.
             */
            NO_CATALOG_ENTRY,

            /** @brief The caller switched this form off through @c FusionOptions. */
            DISABLED,
        };

        /** @brief A sentence explaining @p reason, for the report. */
        constexpr std::string_view fusion_decline_reason(FusionDecline reason) noexcept
        {
            switch (reason)
            {
                case FusionDecline::NONE:
                    return "no reason recorded";
                case FusionDecline::VALUE_HAS_OTHER_CONSUMERS:
                    return "an intermediate is read by a node outside the chain, and a fused "
                           "kernel never writes it";
                case FusionDecline::VALUE_IS_OBSERVED:
                    return "an intermediate is read from outside the graph, so it must reach "
                           "memory";
                case FusionDecline::GRADIENT_NEEDS_PRE_ACTIVATION:
                    return "this activation's gradient differentiates at its input, which a "
                           "fused kernel would never write - fusing anyway trains towards the "
                           "wrong place without failing";
                case FusionDecline::NO_CATALOG_ENTRY:
                    return "this activation is safe to fold but no catalog entry composes it "
                           "with a GEMM epilogue; add one";
                case FusionDecline::DISABLED:
                    return "the caller switched this form off";
            }
            return "unknown";
        }

        /**
         * @struct FusionOptions
         * @brief What the pass is allowed to do.
         */
        struct FusionOptions
        {
            /**
             * @brief Whether to fuse at all.
             *
             * False leaves the graph exactly as it arrived, which is the
             * control an A/B measurement needs and the reference a byte-for-byte
             * comparison compares against.
             */
            bool enabled = true;

            /**
             * @brief Values something outside the graph reads.
             *
             * The pass knows what a value *is* and who reads it *inside* the
             * graph, and nothing about who holds its id. A training loop reads
             * the loss and the logits by id after every replay; an activation
             * that is fused away has no buffer for them to read. Persistent
             * lifetime classes need not be listed - a parameter, a gradient, an
             * optimizer's state and an input are protected by their class.
             *
             * Read only during @c fuse(); it need not outlive the call.
             */
            std::span<const ValueId> observed{};

            /**
             * @brief Catalog entries to leave unmatched.
             *
             * For a test that wants one form at a time, and for a caller who
             * has measured that a particular form does not pay on their model.
             * A disabled form is reported as declined rather than ignored.
             *
             * Read only during @c fuse().
             */
            std::span<const FusedForm> disabled{};
        };

        /**
         * @struct FusedSubtree
         * @brief One chain the pass replaced with a single node.
         *
         * Every node id here indexes the graph **as it arrived**, because that
         * is the graph whoever is debugging the pass is looking at.
         */
        struct FusedSubtree
        {
            /** @brief Which catalog entry was instantiated. */
            FusedForm form = FusedForm::NONE;

            /** @brief The chain's nodes, in producer order; the last one is its root. */
            std::vector<NodeId> chain;

            /** @brief The value the fused node writes - the root's own result. */
            ValueId result = NO_VALUE;

            /** @brief The values that stopped being written; they have no buffer now. */
            std::vector<ValueId> eliminated;
        };

        /**
         * @struct DeclinedFusion
         * @brief One chain that matched structurally and was refused anyway.
         */
        struct DeclinedFusion
        {
            /**
             * @brief The catalog entry that would have applied.
             *
             * @c FusedForm::NONE when there was no entry to try at all, which
             * is itself the finding - see @c NO_CATALOG_ENTRY and
             * @c GRADIENT_NEEDS_PRE_ACTIVATION. @c op names the operation in
             * that case.
             */
            FusedForm form = FusedForm::NONE;

            /** @brief The operation that stayed on its own. */
            OpID op{};

            /** @brief That operation's node, in the graph as it arrived. */
            NodeId root = NO_NODE;

            /** @brief The value that blocked it, or @c NO_VALUE when none did. */
            ValueId blocked_by = NO_VALUE;

            /** @brief Why. */
            FusionDecline reason = FusionDecline::NONE;
        };

        /**
         * @class FusionReport
         * @brief What the pass did, and what it refused to do.
         *
         * A fusion pass nobody can interrogate is one nobody will trust: every
         * mistake it can make produces a graph that still trains, so "it got
         * faster and the loss still falls" is not evidence. This is the record
         * that makes a decision arguable - which subtrees collapsed, which did
         * not, and for each of those, which value stopped it and why.
         */
        class FusionReport
        {
            public:
                /** @brief Every chain that became one node. */
                std::span<const FusedSubtree> fused() const noexcept { return fused_; }

                /** @brief Every chain that matched and was refused. */
                std::span<const DeclinedFusion> declined() const noexcept { return declined_; }

                /** @brief How many nodes the graph had before the pass. */
                std::size_t nodes_before() const noexcept { return before_; }

                /** @brief How many it has after. */
                std::size_t nodes_after() const noexcept { return after_; }

                /** @brief Whether the pass was allowed to run at all. */
                bool enabled() const noexcept { return enabled_; }

                /**
                 * @brief How many nodes of @p form were produced.
                 * @param form The catalog entry to count.
                 * @return The number of matches.
                 */
                std::size_t count(FusedForm form) const noexcept;

                /** @brief A multi-line human-readable summary; empty when nothing happened. */
                std::string to_string() const;

            private:
                friend FusionReport fuse(Graph& graph, const FusionOptions& options);

                std::vector<FusedSubtree> fused_;
                std::vector<DeclinedFusion> declined_;
                std::size_t before_ = 0;
                std::size_t after_ = 0;
                bool enabled_ = false;
        };

        /**
         * @brief Partitions @p graph into maximal fusable subtrees and rewrites them.
         *
         * Matches longest-chain-first from each node, walking backwards through
         * interior values that have exactly one consumer, and rewrites every
         * accepted match into one node carrying the catalog entry's identity.
         * A longer form that is refused falls back to a shorter one rooted
         * further up the same chain, so a GELU layer still gets its bias folded
         * into the GEMM even though its activation cannot be.
         *
         * Value ids are preserved exactly; node ids are not. See the file
         * comment for both.
         *
         * @param graph The graph to rewrite, in place.
         * @param options What the pass is allowed to do.
         * @return What it did, and what it refused.
         * @throws Error If the graph is not in topological insertion order.
         */
        FusionReport fuse(Graph& graph, const FusionOptions& options = {});

    } // namespace Graph
} // namespace SushiAI
