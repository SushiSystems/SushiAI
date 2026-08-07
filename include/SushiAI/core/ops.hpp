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

#include <array>
#include <cstddef>
#include <cstdint>
#include <string_view>
#include <SushiRuntime/core/span.hpp>
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

        /**
         * @brief Builds an OpName, hashing the name at compile time.
         *
         * @c constexpr rather than @c consteval because @c consteval is C++20 and
         * this library is C++17. Nothing is lost: every caller below initialises an
         * @c inline @c constexpr variable, which forces constant evaluation anyway,
         * so no name is hashed at run time.
         */
        constexpr OpName op(std::string_view n)
        {
            return OpName{n, OpID(OpID::hash_str(n.data(), n.size()))};
        }

        // --- Structural -----------------------------------------------------
        /** @brief A graph leaf: a parameter or an input, produced by nothing. */
        inline constexpr OpName LEAF = op("sushiai.leaf");
        /** @brief Accumulates a gradient contribution into an existing buffer. */
        inline constexpr OpName ACCUMULATE = op("sushiai.accumulate");

        /**
         * @brief Copies a value into another element type, out of place.
         *
         * The only operation in this library whose operand and result are
         * *allowed* to disagree on dtype - every other one is checked against
         * a single element type by @c GraphBuilder, and SushiBLAS throws on a
         * mismatch below that. It is emitted by exactly one place,
         * @c graph/cast_insertion.hpp, and it is what makes mixed precision a
         * property of the graph rather than a runtime reinterpretation: a
         * value computed in fp16 and a value stored in fp32 are two values
         * with two buffers and a node between them, so the memory planner, the
         * dependency tracker and any reader of the IR all see the conversion
         * instead of having to infer it.
         */
        inline constexpr OpName CAST = op("sushiai.cast");

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

        /**
         * @brief Multiplies every element by a **device-resident** scalar.
         *
         * @c SCALE's factor is an @c Attributes slot, which is a compile-time
         * constant baked into the recorded task - see the warning on
         * @c Graph::Attributes. That is right for `1/N` and wrong for anything
         * that changes between replays, because changing it forces a recompile
         * and @c CompiledStep::compile_count() is a pinned guarantee. This
         * operation takes its factor as a **second operand**: a 1-element
         * value the host rewrites between replays, exactly as
         * @c Optim::StepScalars does for the optimizer's per-step numbers.
         *
         * Its one caller today is gradient loss scaling
         * (@c optim/loss_scaler.hpp), which needs both directions - scaling the
         * loss gradient up before it narrows to fp16, and dividing each
         * parameter gradient back down before the update.
         */
        inline constexpr OpName DYNAMIC_SCALE = op("sushiai.dynamic_scale");

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

        // --- Structural -----------------------------------------------------

        /** @brief Reinterprets extents without moving data. */
        inline constexpr OpName RESHAPE = op("sushiai.reshape");
        /** @brief The only node permitted to change a value's TensorLayout (CV-4). */
        inline constexpr OpName LAYOUT_CONVERT = op("sushiai.layout_convert");

        // --- Convolution, as CV-1 spells it ---------------------------------
        //
        // There is deliberately no CONV2D. A convolution is IM2COL followed by
        // MATMUL, which is why its weight gradient needs no rule of its own -
        // matmul_backward already produces it.

        /** @brief Gathers each window into a row of a column matrix. */
        inline constexpr OpName IM2COL = op("sushiai.im2col");
        /** @brief The adjoint of IM2COL; a gather, never an atomic scatter (CV-6). */
        inline constexpr OpName COL2IM = op("sushiai.col2im");

        // --- Pooling --------------------------------------------------------

        /** @brief Windowed maximum; also writes the argmax indices. */
        inline constexpr OpName MAXPOOL2D = op("sushiai.maxpool2d");
        /** @brief Routes each gradient back through the recorded argmax. */
        inline constexpr OpName MAXPOOL2D_BACKWARD = op("sushiai.maxpool2d_backward");
        /** @brief Windowed mean, dividing by the window area (count_include_pad). */
        inline constexpr OpName AVGPOOL2D = op("sushiai.avgpool2d");
        /** @brief Spreads each gradient evenly over the window it came from. */
        inline constexpr OpName AVGPOOL2D_BACKWARD = op("sushiai.avgpool2d_backward");

        // --- Whole-plane spatial --------------------------------------------

        /** @brief Global average over H and W: [N,H,W,C] -> [N,C]. */
        inline constexpr OpName SPATIAL_MEAN = op("sushiai.spatial_mean");
        /**
         * @brief Broadcasts an [N,C] gradient over [N,H,W,C], scaled by 1/(H*W).
         *
         * A distinct operation rather than a reuse of BIAS_ADD's or
         * CHANNEL_AFFINE's broadcast: this one is per-(n,c), those are per-row
         * and per-channel. Its absence was a real gap in an earlier draft.
         */
        inline constexpr OpName SPATIAL_MEAN_BACKWARD = op("sushiai.spatial_mean_backward");
        /** @brief Nearest-neighbour upsample by an integer scale. */
        inline constexpr OpName UPSAMPLE2D = op("sushiai.upsample2d");
        /** @brief Sums each source pixel's replicas back together. */
        inline constexpr OpName UPSAMPLE2D_BACKWARD = op("sushiai.upsample2d_backward");

        // --- Structure ------------------------------------------------------

        /** @brief Joins several values along the channel axis. */
        inline constexpr OpName CONCAT_CHANNELS = op("sushiai.concat_channels");
        /**
         * @brief Extracts a channel range.
         *
         * Declared as an operation in its own right rather than as
         * CONCAT_CHANNELS's private backward, because it is independently
         * reachable: Darknet's route layer with a group index is exactly this.
         */
        inline constexpr OpName SLICE_CHANNELS = op("sushiai.slice_channels");

        // --- Normalisation --------------------------------------------------

        /** @brief Batch normalisation's forward pass over an NHWC value. */
        inline constexpr OpName BATCHNORM2D = op("sushiai.batchnorm2d");
        /** @brief Batch normalisation's gradient, for input, gamma and beta. */
        inline constexpr OpName BATCHNORM2D_BACKWARD = op("sushiai.batchnorm2d_backward");
        /** @brief Per-channel scale and shift: y = gamma * x + beta. */
        inline constexpr OpName CHANNEL_AFFINE = op("sushiai.channel_affine");
        /**
         * @brief Updates a BatchNorm's running statistics.
         *
         * Not part of the differentiated graph (CV-7): it reads batch
         * statistics and writes buffers no gradient flows through, so it is
         * emitted into the training step beside the graph rather than into it.
         */
        inline constexpr OpName BATCHNORM_UPDATE_STATS = op("sushiai.batchnorm_update_stats");

        // --- Activation -----------------------------------------------------

        /** @brief Leaky ReLU; its slope is attribute slot 0. */
        inline constexpr OpName LEAKY_RELU = op("sushiai.leaky_relu");
        /** @brief Gradient of LEAKY_RELU; second operand is the forward output. */
        inline constexpr OpName LEAKY_RELU_BACKWARD = op("sushiai.leaky_relu_backward");

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
         * @brief GEMM with a bias broadcast and a GELU, **spilling the pre-activation**.
         *
         * The one fused form that writes two values, and the only reason it is
         * allowed to exist at all. GELU's gradient differentiates at its
         * *input* (see @ref Activation), so the other three epilogues' argument
         * - "the fused kernel destroyed nothing the backward pass wanted" -
         * does not apply to it. This form makes it apply: the epilogue stores
         * @c value+bias[col] to a second buffer on its way past and then
         * returns @c gelu of it, so the tensor the backward rule reads is
         * written after all. It is cuBLASLt's @c GELU_AUX_BIAS, and the second
         * output *is* the value the unfused @c BIAS_ADD would have produced -
         * same id, same bytes - which is why no backward rule changes.
         *
         * @c graph/fusion.cpp's @c every_fused_activation_survives() is what
         * ties the two halves together: an entry folding an @c INPUT-gradient
         * activation without declaring that spill fails the build.
         */
        inline constexpr OpName GEMM_BIAS_GELU = op("sushiai.fused.gemm_bias_gelu");

        /**
         * @brief One pass computing @c out = (exp(a) - b) * alpha.
         *
         * The cross-entropy gradient's whole elementwise tail: five recorded
         * operations - a copy, an exp, a subtraction, a copy and a scal - become
         * one, with both temporaries living in registers.
         */
        inline constexpr OpName EXP_SUB_SCALE = op("sushiai.fused.exp_sub_scale");

        // There was an EPILOGUE_OPERANDS identity here, naming a host-side node
        // that submitted no work and carried the dependency edges a fused GEMM's
        // epilogue could not declare for itself. SushiBLAS's seam takes the
        // epilogue's access set now
        // (`Dispatch::gemm_epilogue(f).reads(t).writes(t)`), so the edges land on
        // the GEMM task and the node has no producer. Removed rather than left as
        // an unused constant: an operation identity nothing emits is a name a
        // reader has to chase before finding out it means nothing.

        /**
         * @brief Every identity declared above, in one enumerable table.
         *
         * The uniqueness proof reads this, and so does the diagnostic that
         * turns an OpID back into a printable name.
         */
        inline constexpr std::array ALL = {
            LEAF, ACCUMULATE, CAST,
            MATMUL, BIAS_ADD,
            ADD, SUB, MUL, SCALE, DYNAMIC_SCALE, EXP, ZERO,
            RELU, GELU, SIGMOID, TANH, LEAKY_RELU,
            RELU_BACKWARD, GELU_BACKWARD, SIGMOID_BACKWARD, TANH_BACKWARD,
            LEAKY_RELU_BACKWARD,
            LOG_SOFTMAX, NLL_LOSS,
            SUM_ROWS,
            ADAMW_UPDATE, SGD_UPDATE,
            GEMM_BIAS, GEMM_BIAS_RELU, GEMM_BIAS_SIGMOID, GEMM_BIAS_TANH, GEMM_BIAS_GELU,
            EXP_SUB_SCALE,
            RESHAPE, LAYOUT_CONVERT,
            IM2COL, COL2IM,
            MAXPOOL2D, MAXPOOL2D_BACKWARD, AVGPOOL2D, AVGPOOL2D_BACKWARD,
            SPATIAL_MEAN, SPATIAL_MEAN_BACKWARD, UPSAMPLE2D, UPSAMPLE2D_BACKWARD,
            CONCAT_CHANNELS, SLICE_CHANNELS,
            BATCHNORM2D, BATCHNORM2D_BACKWARD, CHANNEL_AFFINE, BATCHNORM_UPDATE_STATS,
        };

        /**
         * @brief Reports whether every identity in @p table hashes distinctly.
         *
         * Quadratic, but it runs at compile time over a table of a few dozen
         * entries, so the simple form is the right one.
         *
         * @c constexpr rather than @c consteval (C++20), and taking
         * @c SushiRuntime::span rather than @c std::span (also C++20) - the shim
         * behind that name marks its constructors, @c size() and @c operator[]
         * @c constexpr, so the whole walk is still a constant expression. The proof
         * has not degraded to a run-time check: its only caller is the static_assert
         * below, and a static_assert forces constant evaluation, so a collision is
         * still a compile error rather than something nobody runs.
         *
         * @param table The identities to check.
         * @return True when no two entries share an OpID.
         */
        constexpr bool all_ids_unique(SushiRuntime::span<const OpName> table)
        {
            for (std::size_t i = 0; i < table.size(); ++i)
                for (std::size_t j = i + 1; j < table.size(); ++j)
                    if (table[i].id == table[j].id)
                        return false;
            return true;
        }

        static_assert(all_ids_unique(SushiRuntime::span<const OpName>(ALL.data(), ALL.size())),
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
         * 2. **What a fused form has to do to be allowed to fold the
         *    activation.** A fused form writes only its final result unless it
         *    is built to do otherwise. If the gradient reads the *output*,
         *    nothing was destroyed and folding is exact and free. If it reads
         *    the *input*, a form that writes only its result would train
         *    towards the wrong place - nothing throws, nothing goes NaN, the
         *    loss still falls. The fix is not to forbid the fold but to make
         *    the fused kernel spill the pre-activation to a second buffer,
         *    which is what cuBLASLt's @c GELU_AUX_BIAS does and what
         *    @c Ops::GEMM_BIAS_GELU does here. So this field does not decide
         *    "foldable or not" on its own; it decides **which of the two kinds
         *    of catalog entry** the activation needs, and
         *    @c graph/fusion.cpp's @c every_fused_activation_survives()
         *    static_asserts that an @c INPUT-gradient activation only ever
         *    appears in an entry that declares the spill.
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
        inline constexpr std::array<Activation, 5> ACTIVATIONS = {{
            {RELU.id,       RELU_BACKWARD.id,       GradientSource::OUTPUT},
            {GELU.id,       GELU_BACKWARD.id,       GradientSource::INPUT},
            {SIGMOID.id,    SIGMOID_BACKWARD.id,    GradientSource::OUTPUT},
            {TANH.id,       TANH_BACKWARD.id,       GradientSource::OUTPUT},
            {LEAKY_RELU.id, LEAKY_RELU_BACKWARD.id, GradientSource::OUTPUT},
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
         * @brief Whether folding @p forward needs **no** auxiliary spill.
         *
         * True exactly when the gradient differentiates at the output, because
         * then the pre-activation a result-only fused form never writes was not
         * needed in the first place.
         *
         * False is not "unfusable": it is "unfusable *for free*". An
         * @c INPUT-gradient activation is still foldable by a form that spills
         * the pre-activation to a second buffer - see @c Ops::GEMM_BIAS_GELU -
         * and @c graph/fusion.hpp's catalog is what says which entries do. Read
         * this predicate as a question about the *entry* an activation needs,
         * not about whether one can exist; @ref Activation argues both halves.
         *
         * @param forward The forward activation's identity.
         * @return True when folding @p forward destroys nothing the backward
         *         pass reads, so no spill is required.
         */
        constexpr bool activation_survives_fusion(OpID forward) noexcept
        {
            const Activation* entry = activation_of(forward);
            return entry != nullptr && entry->gradient_source == GradientSource::OUTPUT;
        }

    } // namespace Ops
} // namespace SushiAI
