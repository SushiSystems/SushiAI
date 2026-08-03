/**************************************************************************/
/* cast_insertion.hpp                                                     */
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
 * @file cast_insertion.hpp
 * @brief The pass that makes every node's operands agree with what it computes in.
 *
 * A pure IR-to-IR function in the host lane, with no engine and no SYCL - the
 * same shape as @c graph/fusion.hpp and @c graph/memory_plan.hpp, and for the
 * same reason. It reads a precision policy (@c graph/precision.hpp) and
 * rewrites the graph so that the invariant in its name holds **by
 * construction**: every node's operands and results carry the element type
 * that node computes in, and every place where two precisions meet is an
 * explicit @c Ops::CAST node with its own value, its own buffer and its own
 * dependency edges.
 *
 * @section position Where it runs, and why exactly there
 *
 * The pipeline is: build forward -> @c differentiate() -> **insert_casts()**
 * -> @c fuse() -> plan memory -> lower. Both neighbours are load-bearing.
 *
 * - **After differentiation**, because a backward node needs casting exactly
 *   as much as a forward one does, and there are no backward nodes to cast
 *   before the transform has run. Running earlier would leave every gradient
 *   matmul reading operands of the wrong type, and SushiBLAS would throw at
 *   record time - loudly, but only after a whole training script had been
 *   written against a pipeline that cannot work.
 *
 * - **Before fusion**, because the fusion matcher walks producer edges through
 *   values with exactly one consumer, and a cast dropped into the middle of a
 *   @c MATMUL -> @c BIAS_ADD -> @c RELU chain would break the chain. Casting
 *   first means the matcher sees operands that already agree, so the chains
 *   that survive are the chains that were really there. It also means the
 *   *policy* is what decides whether a chain survives: @c BIAS_ADD and the
 *   foldable activations compute in the GEMM's precision precisely so that no
 *   cast lands inside a fusable chain - see @c graph/precision.hpp rule 2.
 *
 * @section derived Master weights are not rewritten; the fp16 copy is derived
 *
 * A parameter keeps its @c FLOAT32 dtype and its @c Lifetime::PARAMETER, and
 * so does its gradient. What a low-precision GEMM reads is a *different*
 * value - an activation-lifetime copy produced by a cast node, planned and
 * packed with the rest of the step's scratch and rewritten on every replay
 * because the parameter it reads changed. The rule the pass applies is one
 * sentence: **a value's element type follows its producer's computed type,
 * unless something outside the pass depends on it.** "Something outside"
 * means one of two things, and both are checked rather than assumed:
 *
 * 1. A persistent lifetime class - @c PARAMETER, @c GRADIENT,
 *    @c OPTIMIZER_STATE, @c INPUT. The optimizer reads these by tensor, not
 *    through the graph, and it is not part of this transform.
 * 2. A value the caller named in @c CastOptions::observed - the loss and the
 *    logits, which the training loop reads back by id. Exactly the list
 *    @c FusionOptions::observed takes, and for the same reason.
 *
 * Everything else is an intermediate that nothing outside the graph can name,
 * so the pass is free to change its type.
 *
 * @section backward The direction that goes silently wrong
 *
 * A missing cast fails loudly: SushiBLAS refuses a mixed-dtype gemm, axpy or
 * copy by name. A cast in the *wrong place* does not - the model trains and
 * merely learns worse - and the wrong place with real consequences is the
 * backward one. If a matmul computes in fp16, the gradient it produces is
 * fp16, and it must be widened before it reaches the fp32 gradient buffer the
 * optimizer accumulates into. That happens here as an ordinary consequence of
 * the rule above: the gradient value is @c Lifetime::GRADIENT, so it is
 * pinned, so the matmul writes a fresh low-precision result and a cast widens
 * it into the pinned buffer.
 *
 * @c Ops::ACCUMULATE is the one operation where getting this wrong would also
 * corrupt memory rather than only numbers, because its result *aliases its
 * first operand's buffer* (@c TensorPool::alias). The IR is safe against that
 * by construction rather than by a special case: @c GraphBuilder::accumulate
 * calls @c require_same, which compares dtype as well as shape, so a
 * mismatched accumulate cannot be built in the first place; and the pass gives
 * a node's operands and result one type, so it cannot introduce one either.
 *
 * @section ids Value ids survive the rewrite
 *
 * Like @c fuse(), the pass copies the value table entry for entry and appends
 * its new values after it, so every @c ValueId a caller holds - a
 * @c Train::StepSpec, an @c NN::Parameter's id, a gradient id - keeps meaning
 * what it meant. What changes about an existing value is at most its
 * @c Value::dtype, and only when nothing outside the graph depends on it.
 */

#include <cstddef>
#include <string>
#include <string_view>
#include <vector>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/core/common.hpp>
#include <SushiAI/core/ops.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/graph/precision.hpp>

namespace SushiAI
{
    namespace Graph
    {
        /**
         * @struct CastOptions
         * @brief What the pass is allowed to change.
         */
        struct CastOptions
        {
            /**
             * @brief Who decides each node's precision, or null for uniform.
             *
             * Null is spelled out as @c UniformPrecisionPolicy internally
             * rather than short-circuiting the pass, so "no policy" and "the
             * fp32 policy" go down the same code path and produce the same
             * graph. That matters for the control in a test: a zero-cast
             * result that was *computed* is evidence, and one that was
             * returned early is not.
             *
             * Read only during @c insert_casts(); it need not outlive the call.
             */
            const IPrecisionPolicy* policy = nullptr;

            /**
             * @brief Values something outside the graph reads by id.
             *
             * Their element type is left exactly as it was. The persistent
             * lifetime classes need not be listed - a parameter, a gradient,
             * an optimizer's state and an input are pinned by their class.
             *
             * Read only during @c insert_casts().
             */
            SushiRuntime::span<const ValueId> observed{};
        };

        /**
         * @struct InsertedCast
         * @brief One conversion the pass added.
         */
        struct InsertedCast
        {
            /** @brief The node whose dtype boundary this cast resolves. */
            NodeId node = NO_NODE;

            /** @brief That node's operation, in the graph as it arrived. */
            OpID op{};

            /** @brief The value read. */
            ValueId source = NO_VALUE;

            /** @brief The value written. */
            ValueId destination = NO_VALUE;

            /** @brief The source's element type. */
            DType from = DType::FLOAT32;

            /** @brief The destination's element type. */
            DType to = DType::FLOAT32;

            /**
             * @brief Whether this cast feeds an operand or widens a result.
             *
             * The two directions fail differently and are worth telling apart
             * in a report: an operand cast that is missing throws, a result
             * cast that is missing writes a low-precision answer into a
             * high-precision buffer and trains on it.
             */
            bool feeds_operand = true;
        };

        /**
         * @class CastReport
         * @brief What the pass did, so a mixed-precision run can be interrogated.
         *
         * The same argument @c FusionReport makes: every mistake this pass can
         * make produces a graph that still runs and still converges, so "the
         * loss fell" is not evidence that the casts landed where they should.
         * This is the record that makes the placement arguable.
         */
        class CastReport
        {
            public:
                /** @brief Every conversion added, in graph order. */
                SushiRuntime::span<const InsertedCast> casts() const noexcept
                {
                    return SushiRuntime::span<const InsertedCast>(casts_.data(), casts_.size());
                }

                /** @brief How many conversions were added. */
                std::size_t count() const noexcept { return casts_.size(); }

                /** @brief How many conversions into @p dtype were added. */
                std::size_t count_into(DType dtype) const noexcept;

                /** @brief How many nodes the graph had before the pass. */
                std::size_t nodes_before() const noexcept { return before_; }

                /** @brief How many it has after. */
                std::size_t nodes_after() const noexcept { return after_; }

                /** @brief How many values changed element type without changing id. */
                std::size_t retyped_values() const noexcept { return retyped_; }

                /** @brief The policy's name, as it named itself. */
                std::string_view policy() const noexcept { return policy_; }

                /** @brief A multi-line human-readable summary; short when nothing happened. */
                std::string to_string() const;

            private:
                friend CastReport insert_casts(Graph& graph, const CastOptions& options);

                std::vector<InsertedCast> casts_;
                std::size_t before_ = 0;
                std::size_t after_ = 0;
                std::size_t retyped_ = 0;
                std::string_view policy_{};
        };

        /**
         * @brief Rewrites @p graph so every node's operands match its computed dtype.
         *
         * Walks the graph in insertion order, asks @p options.policy what each
         * node computes in, and emits an @c Ops::CAST wherever an operand or a
         * result disagrees. Value ids are preserved; node ids are not, because
         * the IR is single-assignment and a cast has to be *inserted* ahead of
         * its consumer rather than appended.
         *
         * Idempotent for any conforming policy: running it twice adds nothing
         * the second time, which is what @c IPrecisionPolicy's idempotence
         * contract buys (see @c graph/precision.hpp).
         *
         * @param graph The graph to rewrite, in place.
         * @param options The policy and the values it may not retype.
         * @return What it inserted, and where.
         * @throws Error If the graph is not in topological insertion order, if
         *         a policy asks for a dtype no conversion can produce, or if
         *         the resulting graph fails its own invariant - which is a
         *         defect in this pass and says so.
         */
        CastReport insert_casts(Graph& graph, const CastOptions& options = {});

        /**
         * @struct DTypeMismatch
         * @brief One place a graph disagrees with a precision policy.
         */
        struct DTypeMismatch
        {
            /** @brief The offending node. */
            NodeId node = NO_NODE;

            /** @brief Its operation. */
            OpID op{};

            /** @brief The operand or result that disagrees. */
            ValueId value = NO_VALUE;

            /** @brief That value's element type. */
            DType found = DType::FLOAT32;

            /** @brief The element type the policy says the node computes in. */
            DType expected = DType::FLOAT32;

            /** @brief Whether @c value is one of the node's results. */
            bool is_result = false;
        };

        /**
         * @brief Every node of @p graph whose operands or results disagree with @p policy.
         *
         * The invariant @c insert_casts() establishes, checkable from outside
         * and *after the fact* - which is the point. A spot assertion that one
         * particular matmul got its cast is worth much less than a walk of the
         * whole graph asserting the rule, because the walk keeps holding as
         * operations are added and as later passes rewrite nodes. Running it
         * after @c fuse() is what proves fusion did not swallow a master
         * weight or re-pair operands of two precisions.
         *
         * @c Ops::LEAF and @c Ops::CAST are exempt and are the only exemptions:
         * a leaf's type is the caller's to declare, and a cast is by definition
         * the node whose operand and result disagree.
         *
         * @param graph The graph to check.
         * @param policy The policy it should conform to.
         * @return Every disagreement found; empty when the invariant holds.
         */
        std::vector<DTypeMismatch> find_dtype_mismatches(const Graph& graph,
                                                         const IPrecisionPolicy& policy);

        /** @brief A one-line description of @p mismatch, for a test failure or a report. */
        std::string describe(const DTypeMismatch& mismatch);

    } // namespace Graph
} // namespace SushiAI
