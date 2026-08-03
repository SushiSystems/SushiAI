/**************************************************************************/
/* precision.hpp                                                          */
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
 * @file precision.hpp
 * @brief Which element type each operation computes in, as an injected policy.
 *
 * @c graph/cast_insertion.hpp decides **where the conversions go** - a
 * mechanical question about the graph, answerable once someone has said what
 * each node computes in. This file answers *that*: it is where the machine
 * learning knowledge lives, and it is deliberately separate for the same
 * reason @c graph/fusion_selector.hpp is separate from @c graph/fusion.hpp.
 * The pass is a rewrite anybody can verify; the policy is a set of judgements
 * about numerics that has to be argued rather than checked, so it is stated in
 * one small place where the argument can be read.
 *
 * @section masters What mixed precision actually is here
 *
 * fp16 compute with fp32 master weights. The parameters, their gradients and
 * the optimizer's state stay @c FLOAT32 for the whole run; the fp16 tensor a
 * GEMM reads is a *derived* value with an activation lifetime, produced by a
 * cast node and thrown away with the rest of the step. That is not a
 * stylistic preference: an fp16 weight update loses the update entirely
 * whenever it is small relative to the weight - `1.0f16 + 1e-4f16 == 1.0f16`,
 * because fp16's spacing at 1.0 is 9.8e-4 - and a weight that stops moving
 * while the loss still falls is the failure this whole arrangement exists to
 * prevent. Master weights are the reason mixed precision converges at all.
 *
 * @section rules The rules, and why each one is what it is
 *
 * @c MixedPrecisionPolicy narrows exactly two families and leaves everything
 * else in the graph's declared precision.
 *
 * 1. **@c MATMUL and the fused GEMM forms compute low.** This is the entire
 *    point: it is where the FLOPs are, and it is the only place a machine with
 *    a low-precision matrix unit would go faster. Nothing else in the catalog
 *    would.
 *
 * 2. **@c BIAS_ADD and the *foldable* activations compute low too, and that is
 *    forced rather than chosen.** A @c GEMM_BIAS_RELU node applies its
 *    epilogue inside the GEMM's own kernel, in the GEMM's element type. So if
 *    the IR said @c BIAS_ADD computes in fp32, then fusing it into an fp16
 *    GEMM would change the numbers - and @c graph/fusion.hpp promises fusion
 *    changes none. The epilogue's precision is decided by the kernel that will
 *    absorb it, not by this policy; all this policy can do is agree with it.
 *    "Foldable" is @c Ops::activation_survives_fusion, which is the same table
 *    the fusion catalog is @c static_assert-ed against.
 *
 * 3. **Everything that sums many terms stays high: @c SUM_ROWS,
 *    @c LOG_SOFTMAX, @c NLL_LOSS, @c ACCUMULATE.** fp16 carries about 11 bits
 *    of significand, so a sum over a few hundred terms loses the tail of the
 *    sum to the rounding of the running total - quietly, and worse the longer
 *    the reduction. SushiBLAS's own reduction header
 *    (<SushiBLAS/engine/math/reductions.hpp>) makes the same point in its
 *    sharpest form: a *count* in HALF is exact only below 2048, which is
 *    2^11 - the magnitude at which a running total stops being able to resolve
 *    an increment of one at all. A reduction is also never the expensive part
 *    of a step, so there is nothing to trade.
 *
 * 4. **The cross-entropy tail stays high: @c EXP, @c SUB, @c SCALE.** Its
 *    gradient is `(softmax(x) - y) / N`, and for the correct class that is a
 *    subtraction of two nearly-equal numbers. fp16's spacing just below 1.0 is
 *    4.9e-4, so `p - 1` for a confident, correct prediction keeps barely a
 *    couple of significant digits - and unlike underflow, cancellation is not
 *    something loss scaling can rescue, because the information is destroyed
 *    by the subtraction and not by the exponent range. @c EXP is on this list
 *    in every mainstream autocast policy for the same family of reasons.
 *
 * 5. **@c GELU stays high, unlike the other activations.** Two reasons, and
 *    the first is rule 2 read backwards: no catalog entry folds GELU (its
 *    gradient differentiates at its *input*, see @c Ops::Activation), so
 *    nothing forces its precision and the policy is free to pick. Given the
 *    freedom, the tanh approximation computes `x*x*x`, which overflows fp16 at
 *    |x| ≈ 40.3 and arrives at the tanh as an infinity; fp32 has room to spare
 *    there. Two casts is a cheap price for removing an overflow the model
 *    would never report.
 *
 * 6. **The optimizer is not on this list at all**, because its update is not
 *    an IR node - @c Optim records it against the fp32 parameter and the fp32
 *    gradient directly. Stated here anyway, because "the update runs in fp32"
 *    is the load-bearing half of rule 0 and a reader looking for it should
 *    find out where it is enforced rather than conclude it is missing.
 *
 * @section idempotence The one contract an implementation must satisfy
 *
 * @c choose() must be **idempotent**: asking it about a node whose declared
 * type is already its own answer must return that answer again. That is what
 * makes the pass's invariant checkable *after the fact* - see
 * @c find_dtype_mismatches - rather than only observable while the pass is
 * running, and a policy that failed it would produce a graph that re-casts
 * itself on every run of the pass. Both policies here satisfy it trivially:
 * they only ever narrow, and never below their own floor.
 *
 * Host lane: no SushiBLAS engine, no SYCL. A policy is a statement about
 * numerics, not about a device, and the fp32/fp16 trade-off does not change
 * shape between two machines - only its payoff does, which is
 * @c graph/fusion_selector.hpp's kind of question and not this one.
 */

#include <string_view>
#include <SushiAI/core/common.hpp>
#include <SushiAI/core/ops.hpp>

namespace SushiAI
{
    namespace Graph
    {
        /**
         * @struct PrecisionQuery
         * @brief One operation, described in the terms a precision policy reasons in.
         */
        struct PrecisionQuery
        {
            /** @brief Which operation this is; see SushiAI::Ops. */
            OpID op{};

            /**
             * @brief The element type the node's result carries today.
             *
             * The graph's own precision for this node, before any narrowing -
             * which is what a policy narrows *from*, so that a model traced in
             * fp64 is not silently dropped to fp32 by a policy whose author was
             * thinking about fp32 models.
             */
            DType declared = DType::FLOAT32;
        };

        /**
         * @struct PrecisionChoice
         * @brief What the operation should compute in, and why.
         */
        struct PrecisionChoice
        {
            /** @brief The element type every operand and result must agree on. */
            DType dtype = DType::FLOAT32;

            /** @brief A sentence for the report; a static string, never owned. */
            std::string_view reason{};
        };

        /**
         * @class IPrecisionPolicy
         * @brief Per-operation precision, as an interface rather than a table.
         *
         * An interface for the same reason @c IFusionSelector is one: the
         * decision is a *policy*, so a test has to be able to drive the pass
         * with one it wrote - including deliberately silly ones, which is how
         * the cast pass gets exercised on shapes no shipped policy produces.
         *
         * @warning Implementations must be idempotent; see the file comment.
         */
        class IPrecisionPolicy
        {
            public:
                virtual ~IPrecisionPolicy() = default;

                /**
                 * @brief Chooses the element type @p query's operation computes in.
                 * @param query The operation and its declared precision.
                 * @return The element type and the reason for it.
                 */
                virtual PrecisionChoice choose(const PrecisionQuery& query) const = 0;

                /**
                 * @brief A short name for the report and the demo's banner.
                 *
                 * The only thing besides @ref choose that an implementation
                 * owes a caller. There is deliberately no "is this policy
                 * mixed?" predicate to go with it: the one caller that would
                 * want one is the pass, and the pass must not have it. A
                 * control that short-circuits proves nothing - "the fp32
                 * policy inserted no casts" is evidence only if the pass
                 * actually looked.
                 *
                 * @return A static string, never owned; must outlive the report.
                 */
                virtual std::string_view name() const noexcept = 0;
        };

        /**
         * @class UniformPrecisionPolicy
         * @brief The default: every operation computes in the graph's own dtype.
         *
         * Answers @c query.declared to everything, so the cast pass inserts
         * nothing and every graph built before mixed precision existed is
         * bit-for-bit what it was. It is a real object rather than a null
         * pointer check so that "no policy" and "the fp32 policy" are the same
         * thing to every reader, and so a test can assert the *control* -
         * zero casts - against the same interface the mixed case goes through.
         */
        class UniformPrecisionPolicy final : public IPrecisionPolicy
        {
            public:
                PrecisionChoice choose(const PrecisionQuery& query) const override;

                std::string_view name() const noexcept override { return "fp32"; }
        };

        /**
         * @class MixedPrecisionPolicy
         * @brief fp16 where the FLOPs are, the graph's own precision everywhere else.
         *
         * The five rules and their reasoning are in the file comment; this
         * class is only their spelling. The low type is a constructor
         * parameter rather than a hard-coded @c HALF because the enum is
         * expected to grow a bfloat16 (see the master plan's AI-4 entry), and
         * the *shape* of the policy does not change when it does - only which
         * narrow type it names.
         */
        class MixedPrecisionPolicy final : public IPrecisionPolicy
        {
            public:
                /**
                 * @brief Builds a policy narrowing the compute-bound ops to @p low.
                 * @param low The narrow element type; must be a real float.
                 * @throws Error If @p low is complex or integral.
                 */
                explicit MixedPrecisionPolicy(DType low = DType::HALF);

                /** @brief The narrow element type this policy narrows to. */
                DType low() const noexcept { return low_; }

                PrecisionChoice choose(const PrecisionQuery& query) const override;

                std::string_view name() const noexcept override { return "mixed"; }

            private:
                DType low_;
        };

        /**
         * @brief Whether @p op is one of the operations a mixed policy narrows.
         *
         * Exposed because it is the one fact a reader has to be able to check
         * against @c Ops::ALL when adding an operation: a new op that nothing
         * lists here stays in the graph's declared precision, which is the
         * safe answer and not a silent one.
         *
         * @param op The operation's identity.
         * @return True for MATMUL, the fused GEMM forms, BIAS_ADD and the
         *         foldable activations and their gradients.
         */
        bool narrows_under_mixed_precision(OpID op) noexcept;

    } // namespace Graph
} // namespace SushiAI
