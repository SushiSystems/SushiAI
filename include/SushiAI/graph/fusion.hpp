/****************************************************************/
/* fusion.hpp                                                   */
/* SushiAI - https://github.com/SushiSystems/SushiAI            */
/* Copyright (c) 2026 Sushi Systems                             */
/* Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.    */
/* Commercial use requires a licence from Sushi Systems.        */
/****************************************************************/

/**
 * @file fusion.hpp
 * @brief Declares the static subtree fusion pass and fused catalog forms.
 * @author Mustafa Garip
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>
#include <SushiRuntime/core/span.hpp>
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

            /**
             * @brief @c MATMUL, @c BIAS_ADD then @c GELU, spilling the pre-activation.
             *
             * The only entry that writes two values. See @ref spill; without
             * the spill this row could not exist, and the build says so.
             */
            GEMM_BIAS_GELU,

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
                case FusedForm::GEMM_BIAS_GELU:    return "gemm+bias+gelu";
                case FusedForm::EXP_SUB_SCALE:     return "exp-sub-scale";
            }
            return "unknown";
        }

        /**
         * @brief Returns the catalog entry @p name refers to.
         * @param name Name as @ref fused_form_name spells it.
         * @return Matching form, or @c FusedForm::NONE when unmatched.
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
         * @brief Returns whether @p form is a GEMM-epilogue entry.
         * @param form The catalog entry.
         * @return True for the GEMM_BIAS family.
         */
        constexpr bool is_gemm_epilogue(FusedForm form) noexcept
        {
            return form == FusedForm::GEMM_BIAS
                || form == FusedForm::GEMM_BIAS_RELU
                || form == FusedForm::GEMM_BIAS_SIGMOID
                || form == FusedForm::GEMM_BIAS_TANH
                || form == FusedForm::GEMM_BIAS_GELU;
        }

        /**
         * @brief The activation a GEMM-epilogue form folds in, if any.
         * @param form The catalog entry.
         * @return The forward activation's identity, or @c Ops::LEAF.id when
         *         the form folds no activation.
         */
        OpID fused_form_activation(FusedForm form) noexcept;

        /**
         * @brief Returns whether a node of @p form writes an auxiliary second value.
         * @param form The catalog entry.
         * @return True when a node of @p form declares two outputs.
         */
        bool fused_form_spills(FusedForm form) noexcept;

        /**
         * @brief Returns whether any catalog entry folds @p activation into a GEMM.
         * @param activation A forward activation identity.
         * @return True when some entry's chain ends in @p activation.
         */
        bool catalog_folds_activation(OpID activation) noexcept;

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
             * @brief Indicates the activation gradient differentiates at its input.
             */
            GRADIENT_NEEDS_PRE_ACTIVATION,

            /**
             * @brief Indicates a spilled value is read before the fused node writes it.
             */
            SPILL_READ_TOO_EARLY,

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
                    return "this activation's gradient differentiates at its input and no "
                           "catalog entry spills that tensor - fusing anyway trains towards "
                           "the wrong place without failing";
                case FusionDecline::SPILL_READ_TOO_EARLY:
                    return "the pre-activation this form would spill is read by a node that "
                           "runs before the fused GEMM would write it";
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
             * @brief Values read externally whose buffers must not be eliminated.
             */
            SushiRuntime::span<const ValueId> observed{};

            /**
             * @brief Catalog entries to leave unmatched during fusion.
             */
            SushiRuntime::span<const FusedForm> disabled{};
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

            /**
             * @brief Pre-activation value the fused node also writes, or @c NO_VALUE.
             */
            ValueId spilled = NO_VALUE;

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
         * @brief Records which subtrees collapsed and which were refused.
         */
        class FusionReport
        {
            public:
                /** @brief Every chain that became one node. */
                SushiRuntime::span<const FusedSubtree> fused() const noexcept { return fused_; }

                /** @brief Every chain that matched and was refused. */
                SushiRuntime::span<const DeclinedFusion> declined() const noexcept { return declined_; }

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
         * @param graph The graph to rewrite in place.
         * @param options What the pass is allowed to do.
         * @return What it did and what it refused.
         * @throws Error If the graph is not in topological insertion order.
         */
        FusionReport fuse(Graph& graph, const FusionOptions& options = {});

    } // namespace Graph
} // namespace SushiAI
