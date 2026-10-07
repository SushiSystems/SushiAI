/**************************************************************************/
/* ops.hpp                                                                */
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
 * @file ops.hpp
 * @brief Defines compile-time operation identities and metadata for SushiAI.
 * @author Mustafa Garip
 */

#pragma once

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

            /**
             * @brief Whether a backward rule must exist for this operation.
             *
             * Proved by @c autograd/rules/all.hpp, which is where the rule
             * table is visible; this file only declares the claim.
             */
            bool differentiable = false;

            /**
             * @brief Whether the result needs no gradient, whatever the operands need.
             *
             * Declared through @ref gradient_barrier_op, which states the rule.
             */
            bool blocks_gradient = false;
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
            return OpName{n, OpID(OpID::hash_str(n.data(), n.size())), false, false};
        }

        /**
         * @brief Builds an OpName that declares itself differentiable.
         *
         * The claim is checked, not trusted: @c autograd/rules/all.hpp's
         * static_assert walks @ref ALL and fails the build if any entry built
         * this way has no matching backward rule.
         */
        constexpr OpName differentiable_op(std::string_view n)
        {
            return OpName{n, OpID(OpID::hash_str(n.data(), n.size())), true, false};
        }

        /**
         * @brief Builds an OpName whose result lets no gradient reach its operands.
         *
         * This is the one way an operation passes no gradient. @c Graph::finalise leaves the
         * result without @c requires_grad, so no rule contributes to it, and the reverse walk
         * steps over the node. A barrier has no backward rule; the static_assert in
         * @c autograd/rules/all.hpp fails the build if one gains a row.
         */
        constexpr OpName gradient_barrier_op(std::string_view n)
        {
            return OpName{n, OpID(OpID::hash_str(n.data(), n.size())), false, true};
        }

        /** @brief A graph leaf: a parameter or an input, produced by nothing. */
        inline constexpr OpName LEAF = op("sushiai.leaf");
        /** @brief Accumulates a gradient contribution into an existing buffer. */
        inline constexpr OpName ACCUMULATE = differentiable_op("sushiai.accumulate");

        /**
         * @brief Copies a value into another element type, out of place.
         */
        inline constexpr OpName CAST = op("sushiai.cast");

        /** @brief 2-D matrix product, transposition expressed by flags not views. */
        inline constexpr OpName MATMUL = differentiable_op("sushiai.matmul");
        /** @brief Adds a row vector to every row of a matrix. */
        inline constexpr OpName BIAS_ADD = differentiable_op("sushiai.bias_add");

        /** @brief Elementwise sum of two equally shaped tensors. */
        inline constexpr OpName ADD = differentiable_op("sushiai.add");
        /** @brief Elementwise difference of two equally shaped tensors. */
        inline constexpr OpName SUB = differentiable_op("sushiai.sub");
        /** @brief Elementwise product of two equally shaped tensors. */
        inline constexpr OpName MUL = differentiable_op("sushiai.mul");
        /** @brief Multiplies every element by a host-side constant. */
        inline constexpr OpName SCALE = differentiable_op("sushiai.scale");

        /**
         * @brief Multiplies every element by a device-resident scalar.
         */
        inline constexpr OpName DYNAMIC_SCALE = op("sushiai.dynamic_scale");

        /** @brief Elementwise natural exponential, out of place. */
        inline constexpr OpName EXP = differentiable_op("sushiai.exp");
        /** @brief An all-zero tensor, read by nothing upstream. */
        inline constexpr OpName ZERO = gradient_barrier_op("sushiai.zero");
        /** @brief Fills a tensor with the value in attribute slot 0; reads no operand. */
        inline constexpr OpName CONSTANT = gradient_barrier_op("sushiai.constant");

        /** @brief Divides two equally shaped tensors elementwise: a / b. */
        inline constexpr OpName DIV = differentiable_op("sushiai.div");
        /** @brief Takes the elementwise smaller of two equally shaped tensors. */
        inline constexpr OpName MINIMUM = differentiable_op("sushiai.minimum");
        /** @brief Takes the elementwise larger of two equally shaped tensors. */
        inline constexpr OpName MAXIMUM = differentiable_op("sushiai.maximum");
        /** @brief Takes the elementwise absolute value, out of place. */
        inline constexpr OpName ABS = differentiable_op("sushiai.abs");
        /** @brief Takes the elementwise arctangent, out of place. */
        inline constexpr OpName ATAN = differentiable_op("sushiai.atan");

        /** @brief Compares elementwise: 1 where a <= b and 0 elsewhere, in the operands' dtype. */
        inline constexpr OpName LESS_EQUAL = gradient_barrier_op("sushiai.less_equal");
        /** @brief Selects elementwise: a where the condition is non-zero, b elsewhere. */
        inline constexpr OpName SELECT = differentiable_op("sushiai.select");

        /** @brief Rectified linear unit. */
        inline constexpr OpName RELU = differentiable_op("sushiai.relu");
        /** @brief Gaussian error linear unit, tanh approximation. */
        inline constexpr OpName GELU = differentiable_op("sushiai.gelu");
        /** @brief Logistic sigmoid. */
        inline constexpr OpName SIGMOID = differentiable_op("sushiai.sigmoid");
        /** @brief Hyperbolic tangent. */
        inline constexpr OpName TANH = differentiable_op("sushiai.tanh");
        /** @brief Computes the sigmoid linear unit, x * sigmoid(x). */
        inline constexpr OpName SILU = differentiable_op("sushiai.silu");
        /** @brief Computes softplus, log(1 + exp(x)). */
        inline constexpr OpName SOFTPLUS = differentiable_op("sushiai.softplus");

        // Each activation gradient takes [dy, primal] and yields [dx].
        /** @brief Gradient of RELU; second operand is the forward output. */
        inline constexpr OpName RELU_BACKWARD = op("sushiai.relu_backward");
        /** @brief Gradient of GELU; second operand is the forward input. */
        inline constexpr OpName GELU_BACKWARD = op("sushiai.gelu_backward");
        /** @brief Gradient of SIGMOID; second operand is the forward output. */
        inline constexpr OpName SIGMOID_BACKWARD = op("sushiai.sigmoid_backward");
        /** @brief Gradient of TANH; second operand is the forward output. */
        inline constexpr OpName TANH_BACKWARD = op("sushiai.tanh_backward");
        /** @brief Computes the gradient of SILU; second operand is the forward input. */
        inline constexpr OpName SILU_BACKWARD = op("sushiai.silu_backward");
        /** @brief Computes the gradient of SOFTPLUS; second operand is the forward input. */
        inline constexpr OpName SOFTPLUS_BACKWARD = op("sushiai.softplus_backward");

        /** @brief Reinterprets extents without moving data. */
        inline constexpr OpName RESHAPE = differentiable_op("sushiai.reshape");
        /** @brief The only node permitted to change a value's TensorLayout (CV-4). */
        inline constexpr OpName LAYOUT_CONVERT = differentiable_op("sushiai.layout_convert");
        /** @brief Passes a value through unchanged and lets no gradient reach its operand. */
        inline constexpr OpName STOP_GRADIENT = gradient_barrier_op("sushiai.stop_gradient");

        // A convolution is IM2COL followed by MATMUL.
        /** @brief Gathers each window into a row of a column matrix. */
        inline constexpr OpName IM2COL = differentiable_op("sushiai.im2col");
        /** @brief The adjoint of IM2COL; a gather, never an atomic scatter (CV-6). */
        inline constexpr OpName COL2IM = differentiable_op("sushiai.col2im");

        /** @brief Windowed maximum; also writes the argmax indices. */
        inline constexpr OpName MAXPOOL2D = differentiable_op("sushiai.maxpool2d");
        /** @brief Routes each gradient back through the recorded argmax. */
        inline constexpr OpName MAXPOOL2D_BACKWARD = op("sushiai.maxpool2d_backward");
        /** @brief Windowed mean, dividing by the window area (count_include_pad). */
        inline constexpr OpName AVGPOOL2D = differentiable_op("sushiai.avgpool2d");
        /** @brief Spreads each gradient evenly over the window it came from. */
        inline constexpr OpName AVGPOOL2D_BACKWARD = op("sushiai.avgpool2d_backward");

        /** @brief Global average over H and W: [N,H,W,C] -> [N,C]. */
        inline constexpr OpName SPATIAL_MEAN = differentiable_op("sushiai.spatial_mean");
        /** @brief Broadcasts an [N,C] gradient over [N,H,W,C], scaled by 1/(H*W). */
        inline constexpr OpName SPATIAL_MEAN_BACKWARD = op("sushiai.spatial_mean_backward");
        /** @brief Nearest-neighbour upsample by an integer scale. */
        inline constexpr OpName UPSAMPLE2D = differentiable_op("sushiai.upsample2d");
        /** @brief Sums each source pixel's replicas back together. */
        inline constexpr OpName UPSAMPLE2D_BACKWARD = op("sushiai.upsample2d_backward");

        /** @brief Joins several values along the channel axis. */
        inline constexpr OpName CONCAT_CHANNELS = differentiable_op("sushiai.concat_channels");
        /** @brief Extracts a channel range. */
        inline constexpr OpName SLICE_CHANNELS = differentiable_op("sushiai.slice_channels");

        /** @brief Batch normalisation's forward pass over an NHWC value. */
        inline constexpr OpName BATCHNORM2D = differentiable_op("sushiai.batchnorm2d");
        /** @brief Batch normalisation's gradient, for input, gamma and beta. */
        inline constexpr OpName BATCHNORM2D_BACKWARD = op("sushiai.batchnorm2d_backward");
        /** @brief Per-channel scale and shift: y = gamma * x + beta. */
        inline constexpr OpName CHANNEL_AFFINE = differentiable_op("sushiai.channel_affine");
        /**
         * @brief Updates running statistics for batch normalisation.
         */
        inline constexpr OpName BATCHNORM_UPDATE_STATS = op("sushiai.batchnorm_update_stats");

        /** @brief Leaky ReLU; its slope is attribute slot 0. */
        inline constexpr OpName LEAKY_RELU = differentiable_op("sushiai.leaky_relu");
        /** @brief Gradient of LEAKY_RELU; second operand is the forward output. */
        inline constexpr OpName LEAKY_RELU_BACKWARD = op("sushiai.leaky_relu_backward");

        /** @brief Row-wise log-softmax, computed max-shifted for stability. */
        inline constexpr OpName LOG_SOFTMAX = op("sushiai.log_softmax");
        /** @brief Mean negative log-likelihood against integer-valued targets. */
        inline constexpr OpName NLL_LOSS = differentiable_op("sushiai.nll_loss");

        /** @brief Sums a matrix along its rows, yielding one value per column. */
        inline constexpr OpName SUM_ROWS = differentiable_op("sushiai.sum_rows");

        // Optimizer updates are tracked in tasks for collision avoidance and profiling.
        /** @brief One parameter's whole AdamW update, fused into one kernel. */
        inline constexpr OpName ADAMW_UPDATE = op("sushiai.optim.adamw");
        /** @brief One parameter's whole SGD update, fused into one kernel. */
        inline constexpr OpName SGD_UPDATE = op("sushiai.optim.sgd");

        // Fused operation identities represent compiled kernel compositions.
        /** @brief GEMM with a bias broadcast folded into its store. */
        inline constexpr OpName GEMM_BIAS = op("sushiai.fused.gemm_bias");
        /** @brief GEMM with a bias broadcast and a ReLU folded into its store. */
        inline constexpr OpName GEMM_BIAS_RELU = op("sushiai.fused.gemm_bias_relu");
        /** @brief GEMM with a bias broadcast and a sigmoid folded into its store. */
        inline constexpr OpName GEMM_BIAS_SIGMOID = op("sushiai.fused.gemm_bias_sigmoid");
        /** @brief GEMM with a bias broadcast and a tanh folded into its store. */
        inline constexpr OpName GEMM_BIAS_TANH = op("sushiai.fused.gemm_bias_tanh");

        /**
         * @brief Fused GEMM with bias broadcast and GELU, spilling pre-activation.
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

        /**
         * @brief Every identity declared above, in one enumerable table.
         *
         * The uniqueness proof reads this, and so does the diagnostic that
         * turns an OpID back into a printable name.
         */
        inline constexpr std::array ALL = {
            LEAF, ACCUMULATE, CAST,
            MATMUL, BIAS_ADD,
            ADD, SUB, MUL, SCALE, DYNAMIC_SCALE, EXP, ZERO, CONSTANT,
            DIV, MINIMUM, MAXIMUM, ABS, ATAN,
            LESS_EQUAL, SELECT,
            RELU, GELU, SIGMOID, TANH, LEAKY_RELU, SILU, SOFTPLUS,
            RELU_BACKWARD, GELU_BACKWARD, SIGMOID_BACKWARD, TANH_BACKWARD,
            LEAKY_RELU_BACKWARD, SILU_BACKWARD, SOFTPLUS_BACKWARD,
            LOG_SOFTMAX, NLL_LOSS,
            SUM_ROWS,
            ADAMW_UPDATE, SGD_UPDATE,
            GEMM_BIAS, GEMM_BIAS_RELU, GEMM_BIAS_SIGMOID, GEMM_BIAS_TANH, GEMM_BIAS_GELU,
            EXP_SUB_SCALE,
            RESHAPE, LAYOUT_CONVERT, STOP_GRADIENT,
            IM2COL, COL2IM,
            MAXPOOL2D, MAXPOOL2D_BACKWARD, AVGPOOL2D, AVGPOOL2D_BACKWARD,
            SPATIAL_MEAN, SPATIAL_MEAN_BACKWARD, UPSAMPLE2D, UPSAMPLE2D_BACKWARD,
            CONCAT_CHANNELS, SLICE_CHANNELS,
            BATCHNORM2D, BATCHNORM2D_BACKWARD, CHANNEL_AFFINE, BATCHNORM_UPDATE_STATS,
        };

        /**
         * @brief Reports whether every identity in @p table hashes distinctly.
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

        /**
         * @brief Reports whether any identity in @p table is both differentiable and a barrier.
         * @param table The identities to check.
         * @return True when some entry carries both flags.
         */
        constexpr bool any_barrier_is_differentiable(SushiRuntime::span<const OpName> table)
        {
            for (const OpName& entry : table)
            {
                if (entry.differentiable && entry.blocks_gradient)
                    return true;
            }
            return false;
        }

        static_assert(!any_barrier_is_differentiable(
                          SushiRuntime::span<const OpName>(ALL.data(), ALL.size())),
                      "A SushiAI operation is declared both differentiable and a gradient "
                      "barrier. A barrier's rule would never run; pick one of the two.");

        /**
         * @brief Reports whether @p op cuts the gradient path between its result and its operands.
         * @return True when @p op was declared through @ref gradient_barrier_op.
         */
        constexpr bool is_gradient_barrier(OpID op) noexcept
        {
            for (const OpName& entry : ALL)
            {
                if (entry.id == op)
                    return entry.blocks_gradient;
            }
            return false;
        }

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
         * @brief Describes an activation, its gradient operation, and gradient source.
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
        inline constexpr std::array<Activation, 7> ACTIVATIONS = {{
            {RELU.id,       RELU_BACKWARD.id,       GradientSource::OUTPUT},
            {GELU.id,       GELU_BACKWARD.id,       GradientSource::INPUT},
            {SIGMOID.id,    SIGMOID_BACKWARD.id,    GradientSource::OUTPUT},
            {TANH.id,       TANH_BACKWARD.id,       GradientSource::OUTPUT},
            {LEAKY_RELU.id, LEAKY_RELU_BACKWARD.id, GradientSource::OUTPUT},
            {SILU.id,       SILU_BACKWARD.id,       GradientSource::INPUT},
            {SOFTPLUS.id,   SOFTPLUS_BACKWARD.id,   GradientSource::INPUT},
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
         * @brief Reports whether @p op is the gradient operation of a row in @ref ACTIVATIONS.
         */
        constexpr bool is_activation_backward(OpID op) noexcept
        {
            for (const Activation& entry : ACTIVATIONS)
            {
                if (entry.backward == op)
                    return true;
            }
            return false;
        }

        /**
         * @brief Reports whether folding @p forward needs no auxiliary spill.
         * @param forward The forward activation identity.
         * @return True when folding @p forward requires no auxiliary spill.
         */
        constexpr bool activation_survives_fusion(OpID forward) noexcept
        {
            const Activation* entry = activation_of(forward);
            return entry != nullptr && entry->gradient_source == GradientSource::OUTPUT;
        }

    } // namespace Ops
} // namespace SushiAI
