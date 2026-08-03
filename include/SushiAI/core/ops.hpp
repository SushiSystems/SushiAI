/**************************************************************************/
/* ops.hpp                                                                */
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
 * @file ops.hpp
 * @brief The identity of every SushiAI operation, as compile-time constants.
 *
 * SushiRuntime identifies an operation by @c OpID, a 32-bit FNV-1a hash of a
 * name, and offers no registry to register into outside the distributed build.
 * So the compiler is the registry: every identity is a @c constexpr constant
 * declared here, and a @c static_assert at the bottom of this file proves the
 * whole set is collision-free. Thirty-two bits is not many — an accidental
 * collision between two ops would silently route one to the other's profiling
 * row, so it is checked rather than hoped for.
 *
 * Every name carries the @c sushiai. prefix so it cannot collide with
 * SushiBLAS's own identities (@c blas.lvl3.gemm and friends) or a future
 * sibling's.
 *
 * This header is part of the host lane: it pulls in no SYCL and can be
 * included by a device-free unit test.
 */

#include <span>
#include <array>
#include <cstddef>
#include <cstdint>
#include <string_view>
#include <SushiRuntime/graph/task_types.hpp>

namespace SushiAI
{
    /** @brief The runtime's operation-identity type, aliased for brevity. */
    using OpID = SushiRuntime::Graph::OpID;

    namespace Ops
    {
        using namespace SushiRuntime::Graph::Literals;

        /**
         * @struct OpName
         * @brief One operation's identity: the string and its hash, kept together.
         *
         * Pairing them is what lets the uniqueness check report *which* two
         * names collided, and lets a diagnostic print a name for an id.
         */
        struct OpName
        {
            /** @brief The dotted, prefixed name; also what the profiler shows. */
            std::string_view name;

            /** @brief The hash of @c name, as SushiRuntime identifies it. */
            OpID id;
        };

        /** @brief Builds an OpName, hashing the name at compile time. */
        consteval OpName op(std::string_view n)
        {
            return OpName{n, OpID(OpID::hash_str(n.data(), n.size()))};
        }

        // --- Structural -----------------------------------------------------
        /** @brief A graph leaf: a parameter or an input, produced by nothing. */
        inline constexpr OpName LEAF = op("sushiai.leaf");
        /** @brief Accumulates a gradient contribution into an existing buffer. */
        inline constexpr OpName ACCUMULATE = op("sushiai.accumulate");

        // --- Linear algebra -------------------------------------------------
        /** @brief 2-D matrix product, transposition expressed by flags not views. */
        inline constexpr OpName MATMUL = op("sushiai.matmul");
        /** @brief Adds a row vector to every row of a matrix. */
        inline constexpr OpName BIAS_ADD = op("sushiai.bias_add");

        // --- Elementwise ----------------------------------------------------
        /** @brief Elementwise sum of two equally shaped tensors. */
        inline constexpr OpName ADD = op("sushiai.add");
        /** @brief Elementwise difference of two equally shaped tensors. */
        inline constexpr OpName SUB = op("sushiai.sub");
        /** @brief Elementwise product of two equally shaped tensors. */
        inline constexpr OpName MUL = op("sushiai.mul");
        /** @brief Multiplies every element by a host-side constant. */
        inline constexpr OpName SCALE = op("sushiai.scale");

        /** @brief Elementwise natural exponential, out of place. */
        inline constexpr OpName EXP = op("sushiai.exp");
        /** @brief An all-zero tensor, read by nothing upstream. */
        inline constexpr OpName ZERO = op("sushiai.zero");

        // --- Activations ----------------------------------------------------
        /** @brief Rectified linear unit. */
        inline constexpr OpName RELU = op("sushiai.relu");
        /** @brief Gaussian error linear unit, tanh approximation. */
        inline constexpr OpName GELU = op("sushiai.gelu");
        /** @brief Logistic sigmoid. */
        inline constexpr OpName SIGMOID = op("sushiai.sigmoid");
        /** @brief Hyperbolic tangent. */
        inline constexpr OpName TANH = op("sushiai.tanh");

        // --- Activation gradients -------------------------------------------
        // Each takes [dy, primal] and yields [dx]. Which tensor "primal" is
        // differs by activation and is not a detail a caller may guess; the
        // ACTIVATIONS table below is the one place that says which, and
        // @ref Activation explains why that one fact decides both how the
        // backward pass is emitted and which activations may be folded into
        // the kernel that produced their input.

        /** @brief Gradient of RELU; second operand is the forward output. */
        inline constexpr OpName RELU_BACKWARD = op("sushiai.relu_backward");
        /** @brief Gradient of GELU; second operand is the forward input. */
        inline constexpr OpName GELU_BACKWARD = op("sushiai.gelu_backward");
        /** @brief Gradient of SIGMOID; second operand is the forward output. */
        inline constexpr OpName SIGMOID_BACKWARD = op("sushiai.sigmoid_backward");
        /** @brief Gradient of TANH; second operand is the forward output. */
        inline constexpr OpName TANH_BACKWARD = op("sushiai.tanh_backward");

        // --- Losses and normalisation ---------------------------------------
        /** @brief Row-wise log-softmax, computed max-shifted for stability. */
        inline constexpr OpName LOG_SOFTMAX = op("sushiai.log_softmax");
        /** @brief Mean negative log-likelihood against integer-valued targets. */
        inline constexpr OpName NLL_LOSS = op("sushiai.nll_loss");

        // --- Reductions -----------------------------------------------------
        /** @brief Sums a matrix along its rows, yielding one value per column. */
        inline constexpr OpName SUM_ROWS = op("sushiai.sum_rows");

        // --- Optimizer updates ----------------------------------------------
        // Not IR operations: no node carries these and no backward rule looks
        // them up. They are here because this file is the registry of every
        // identity SushiAI puts into a SushiRuntime task, and the collision
        // proof below is only worth anything if it covers all of them - a fused
        // update that happened to hash onto MATMUL would silently take over its
        // profiling row.

        /** @brief One parameter's whole AdamW update, fused into one kernel. */
        inline constexpr OpName ADAMW_UPDATE = op("sushiai.optim.adamw");
        /** @brief One parameter's whole SGD update, fused into one kernel. */
        inline constexpr OpName SGD_UPDATE = op("sushiai.optim.sgd");

        // --- Fused forms ------------------------------------------------------
        // These *are* IR operations: the fusion pass rewrites a matched subtree
        // into one node carrying one of these identities, and the lowering has a
        // recorder for each.
        //
        // Each name is a distinct kernel instantiation rather than a flag on a
        // generic one, and that is forced rather than chosen.
        // @c sycl_ext_codeplay_kernel_fusion has been removed from intel-llvm, so
        // there is no runtime IR-level fusion anywhere in this stack: a functor
        // composition is a C++ template instantiation, and which compositions
        // exist is decided when *this library* is compiled. The set below is
        // therefore a **catalog**, and a pass running at graph-build time can only
        // choose among its entries - it cannot synthesise a new one. See
        // graph/fusion.hpp for the matcher and graph/fusion_selector.hpp for the
        // bind-time choice between entries.

        /** @brief GEMM with a bias broadcast folded into its store. */
        inline constexpr OpName GEMM_BIAS = op("sushiai.fused.gemm_bias");
        /** @brief GEMM with a bias broadcast and a ReLU folded into its store. */
        inline constexpr OpName GEMM_BIAS_RELU = op("sushiai.fused.gemm_bias_relu");
        /** @brief GEMM with a bias broadcast and a sigmoid folded into its store. */
        inline constexpr OpName GEMM_BIAS_SIGMOID = op("sushiai.fused.gemm_bias_sigmoid");
        /** @brief GEMM with a bias broadcast and a tanh folded into its store. */
        inline constexpr OpName GEMM_BIAS_TANH = op("sushiai.fused.gemm_bias_tanh");

        /**
         * @brief One pass computing @c out = (exp(a) - b) * alpha.
         *
         * The cross-entropy gradient's whole elementwise tail: five recorded
         * operations - a copy, an exp, a subtraction, a copy and a scal - become
         * one, with both temporaries living in registers.
         */
        inline constexpr OpName EXP_SUB_SCALE = op("sushiai.fused.exp_sub_scale");

        /**
         * @brief The ordering node a fused epilogue's undeclared reads need.
         *
         * @c Dispatch::execute_gemm declares A, B and C to the runtime's
         * dependency tracker and nothing the epilogue itself touches - the
         * functor is an opaque callable to that layer. A bias vector the
         * epilogue dereferences is therefore invisible, and SushiBLAS says in
         * its own documentation that the caller owns ordering it. This is that
         * ordering, made explicit; see @c Lowering::declare_epilogue_operand.
         */
        inline constexpr OpName EPILOGUE_OPERANDS = op("sushiai.fused.epilogue_operands");

        /**
         * @brief Every identity declared above, in one enumerable table.
         *
         * The uniqueness proof reads this, and so does the diagnostic that
         * turns an OpID back into a printable name.
         */
        inline constexpr std::array ALL = {
            LEAF, ACCUMULATE,
            MATMUL, BIAS_ADD,
            ADD, SUB, MUL, SCALE, EXP, ZERO,
            RELU, GELU, SIGMOID, TANH,
            RELU_BACKWARD, GELU_BACKWARD, SIGMOID_BACKWARD, TANH_BACKWARD,
            LOG_SOFTMAX, NLL_LOSS,
            SUM_ROWS,
            ADAMW_UPDATE, SGD_UPDATE,
            GEMM_BIAS, GEMM_BIAS_RELU, GEMM_BIAS_SIGMOID, GEMM_BIAS_TANH,
            EXP_SUB_SCALE, EPILOGUE_OPERANDS,
        };

        /**
         * @brief Reports whether every identity in @p table hashes distinctly.
         *
         * Quadratic, but it runs at compile time over a table of a few dozen
         * entries, so the simple form is the right one.
         *
         * @param table The identities to check.
         * @return True when no two entries share an OpID.
         */
        consteval bool all_ids_unique(std::span<const OpName> table)
        {
            for (std::size_t i = 0; i < table.size(); ++i)
                for (std::size_t j = i + 1; j < table.size(); ++j)
                    if (table[i].id == table[j].id)
                        return false;
            return true;
        }

        static_assert(all_ids_unique(std::span<const OpName>(ALL.data(), ALL.size())),
                      "Two SushiAI operation names hash to the same 32-bit OpID. "
                      "Rename one of them - a collision would silently merge their "
                      "profiling rows and confuse any downstream dispatch on id.");

        /**
         * @brief The declared name behind an OpID, for diagnostics and reports.
         * @param id The identity to look up.
         * @return The matching name, or "sushiai.unknown" when @p id is not ours.
         */
        constexpr std::string_view name_of(OpID id) noexcept
        {
            for (const OpName& entry : ALL)
                if (entry.id == id) return entry.name;
            return "sushiai.unknown";
        }

        // =====================================================================
        // Activations: the one fact that decides both differentiation and fusion
        // =====================================================================

        /**
         * @enum GradientSource
         * @brief Which tensor an activation's gradient rule differentiates at.
         */
        enum class GradientSource : std::uint8_t
        {
            /** @brief The forward input; the pre-activation must survive the step. */
            INPUT,
            /** @brief The forward output; the pre-activation is dead on arrival. */
            OUTPUT,
        };

        /**
         * @struct Activation
         * @brief One activation, its gradient, and where that gradient reads.
         *
         * @c gradient_source is the load-bearing field, and it decides two
         * things that look unrelated and are the same question:
         *
         * 1. **What @c autograd/backward.cpp emits.** The gradient operation
         *    takes @c [dy, primal]; getting @c primal wrong is silent, because
         *    both candidates have the same shape, so the graph builds and the
         *    model simply learns the wrong thing. SushiBLAS fixes the
         *    convention per activation and this table mirrors it.
         *
         * 2. **Whether the activation may be folded into the kernel that
         *    produced its input.** A fused form never writes the
         *    pre-activation. If the gradient reads the *output*, nothing was
         *    destroyed and the fusion is exact. If it reads the *input*, the
         *    fused form would train towards the wrong place - nothing throws,
         *    nothing goes NaN, the loss still falls. That is the trap
         *    cuBLASLt's @c GELU_AUX_BIAS exists to solve, and it is why
         *    @c graph/fusion.hpp static_asserts that every fused-activation
         *    catalog entry names an activation whose source is @c OUTPUT.
         *
         * @c RELU reads the output rather than the input, which is a
         * substitution and not an approximation: SushiBLAS's @c relu_backward
         * is @c dy*(primal>0?1:0), and @c relu(x)>0 exactly when @c x>0, so the
         * two spellings are **bit-identical**. Making that the unconditional
         * rule - not a special case the fusion pass applies - is what leaves the
         * pre-activation with a single consumer in the plain graph too, so the
         * fusion pass needs no activation-specific knowledge to license the
         * fold: an ordinary use count already says yes for ReLU and no for GELU.
         */
        struct Activation
        {
            /** @brief The forward operation's identity. */
            OpID forward;

            /** @brief The gradient operation's identity. */
            OpID backward;

            /** @brief Which tensor the gradient differentiates at. */
            GradientSource gradient_source;
        };

        /** @brief Every activation this library differentiates, in one table. */
        inline constexpr std::array<Activation, 4> ACTIVATIONS = {{
            {RELU.id,    RELU_BACKWARD.id,    GradientSource::OUTPUT},
            {GELU.id,    GELU_BACKWARD.id,    GradientSource::INPUT},
            {SIGMOID.id, SIGMOID_BACKWARD.id, GradientSource::OUTPUT},
            {TANH.id,    TANH_BACKWARD.id,    GradientSource::OUTPUT},
        }};

        /**
         * @brief The activation entry for a forward operation.
         * @param forward The forward operation's identity.
         * @return A pointer into @ref ACTIVATIONS, or null when @p forward is
         *         not an activation.
         */
        constexpr const Activation* activation_of(OpID forward) noexcept
        {
            for (const Activation& entry : ACTIVATIONS)
                if (entry.forward == forward) return &entry;
            return nullptr;
        }

        /**
         * @brief Whether folding @p forward into its producer keeps backward exact.
         *
         * True exactly when the gradient differentiates at the output, because
         * then the pre-activation a fused form never writes was not needed. See
         * @ref Activation for why this is the whole of the question.
         *
         * @param forward The forward activation's identity.
         * @return True when the activation is safe to fuse in a training pass.
         */
        constexpr bool activation_survives_fusion(OpID forward) noexcept
        {
            const Activation* entry = activation_of(forward);
            return entry != nullptr && entry->gradient_source == GradientSource::OUTPUT;
        }

    } // namespace Ops
} // namespace SushiAI
