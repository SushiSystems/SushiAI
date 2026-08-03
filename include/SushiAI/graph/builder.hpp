/**************************************************************************/
/* builder.hpp                                                            */
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
 * @file builder.hpp
 * @brief The one place that knows an operation's output shape.
 *
 * Graph::add_node takes ids and asks no questions: it enforces
 * single-assignment and nothing else, because it is the substrate every pass
 * rewrites and a pass that already knows its own shapes should not pay for a
 * re-derivation. That leaves shape inference homeless, and shape inference is
 * exactly the rule that must be stated once - a layer, a backward rule and a
 * fusion pass all emit MATMUL, and three copies of "the output is [m, n]"
 * would be three chances to disagree.
 *
 * So every emitter lives here, each one validating its operands and declaring
 * its result. Nothing in the library calls @c Graph::add_node directly with a
 * hand-computed shape.
 *
 * Host lane: this header pulls in no SushiBLAS and no SYCL.
 */

#include <cstdint>
#include <string_view>
#include <SushiAI/core/ops.hpp>
#include <SushiAI/core/shape.hpp>
#include <SushiAI/core/common.hpp>
#include <SushiAI/graph/ir.hpp>

namespace SushiAI
{
    namespace Graph
    {
        /**
         * @class GraphBuilder
         * @brief Emits shape-checked IR nodes into a Graph.
         *
         * A non-owning view over a Graph. Copyable and cheap; hold one for as
         * long as you are appending, and never past the graph's lifetime.
         */
        class GraphBuilder
        {
            public:
                /**
                 * @brief Wraps @p graph for emission.
                 * @param graph The graph to append to; must outlive this builder.
                 */
                explicit GraphBuilder(Graph& graph) noexcept : graph_(&graph) {}

                /** @brief The graph being built. */
                Graph& graph() const noexcept { return *graph_; }

                // --- Leaves -----------------------------------------------------

                /**
                 * @brief Declares a minibatch input: a leaf rewritten every step.
                 * @param shape The value's extents.
                 * @param debug_name A label for diagnostics; must outlive the graph.
                 * @param dtype The element type.
                 * @return The new value's id.
                 */
                ValueId input(Shape shape, std::string_view debug_name,
                              DType dtype = DType::FLOAT32) const;

                /**
                 * @brief Declares a learnable parameter: a leaf that needs a gradient.
                 * @param shape The value's extents.
                 * @param debug_name A label for diagnostics; must outlive the graph.
                 * @param dtype The element type.
                 * @return The new value's id.
                 */
                ValueId parameter(Shape shape, std::string_view debug_name,
                                  DType dtype = DType::FLOAT32) const;

                // --- Linear algebra ---------------------------------------------

                /**
                 * @brief Emits @c MATMUL: C = op(A) * op(B).
                 *
                 * Transposition is a flag, never a view: SushiBLAS derives a
                 * matrix's leading dimension from its @c shape rather than its
                 * @c strides, so a transposed Tensor view is silently
                 * mis-indexed. The flag reaches gemm's own @c transA / @c transB,
                 * which is the only correct route.
                 *
                 * @param a Left operand.
                 * @param b Right operand.
                 * @param transpose_a Whether to transpose @p a.
                 * @param transpose_b Whether to transpose @p b.
                 * @return The product's id.
                 * @throws Error If either operand is not rank 2, the dtypes
                 *         differ, or the contracted extents disagree.
                 */
                ValueId matmul(ValueId a, ValueId b,
                               bool transpose_a = false, bool transpose_b = false) const;

                /**
                 * @brief Emits @c BIAS_ADD: Y = X + 1 * bias^T, over an [N, C] X.
                 * @param x The [N, C] matrix.
                 * @param bias The [C] row added to every row of @p x.
                 * @return The sum's id.
                 * @throws Error If the ranks or extents do not line up.
                 */
                ValueId bias_add(ValueId x, ValueId bias) const;

                /**
                 * @brief Emits @c SUM_ROWS: the column sums of an [N, C] matrix.
                 * @param x The [N, C] matrix.
                 * @return The [C] result's id.
                 * @throws Error If @p x is not rank 2.
                 */
                ValueId sum_rows(ValueId x) const;

                // --- Elementwise -------------------------------------------------

                /** @brief Emits @c ADD: elementwise A + B. @throws Error on mismatch. */
                ValueId add(ValueId a, ValueId b) const;

                /** @brief Emits @c SUB: elementwise A - B. @throws Error on mismatch. */
                ValueId sub(ValueId a, ValueId b) const;

                /** @brief Emits @c MUL: elementwise A * B. @throws Error on mismatch. */
                ValueId mul(ValueId a, ValueId b) const;

                /**
                 * @brief Emits @c SCALE: Y = alpha * X.
                 * @param x The operand.
                 * @param alpha The host-side constant factor.
                 * @return The scaled value's id.
                 */
                ValueId scale(ValueId x, double alpha) const;

                /** @brief Emits @c EXP: elementwise exp(X), out of place. */
                ValueId exp(ValueId x) const;

                /**
                 * @brief Emits @c ZERO: a fresh all-zero tensor with no operands.
                 * @param shape The value's extents.
                 * @param dtype The element type.
                 * @return The new value's id.
                 */
                ValueId zeros(Shape shape, DType dtype = DType::FLOAT32) const;

                // --- Activations ---------------------------------------------------

                /** @brief Emits @c RELU. */
                ValueId relu(ValueId x) const;

                /** @brief Emits @c GELU (tanh approximation). */
                ValueId gelu(ValueId x) const;

                /** @brief Emits @c SIGMOID. */
                ValueId sigmoid(ValueId x) const;

                /** @brief Emits @c TANH. */
                ValueId tanh(ValueId x) const;

                /**
                 * @brief Emits one of the activation-gradient operations.
                 *
                 * @param op @c RELU_BACKWARD, @c GELU_BACKWARD,
                 *        @c SIGMOID_BACKWARD or @c TANH_BACKWARD.
                 * @param dy The incoming gradient.
                 * @param primal The forward input (relu/gelu) or the forward
                 *        output (sigmoid/tanh) - see ops.hpp.
                 * @return The incoming-side gradient's id.
                 * @throws Error If @p op is not an activation gradient, or the
                 *         operands disagree in shape or dtype.
                 */
                ValueId activation_backward(OpID op, ValueId dy, ValueId primal) const;

                // --- Losses ----------------------------------------------------------

                /**
                 * @brief Emits @c LOG_SOFTMAX over the rows of an [N, C] matrix.
                 * @param x The logits.
                 * @return The [N, C] log-probabilities' id.
                 * @throws Error If @p x is not rank 2.
                 */
                ValueId log_softmax(ValueId x) const;

                /**
                 * @brief Emits @c NLL_LOSS: the mean negative log-likelihood.
                 *
                 * @param log_probs An [N, C] value, normally a @c LOG_SOFTMAX result.
                 * @param one_hot An [N, C] one-hot target matrix.
                 * @return A 1-element loss value's id.
                 * @throws Error If the two operands disagree in shape or dtype.
                 */
                ValueId nll_loss(ValueId log_probs, ValueId one_hot) const;

                // --- Structural --------------------------------------------------------

                /**
                 * @brief Emits @c ACCUMULATE: acc_out = acc + delta, in @p acc's buffer.
                 *
                 * The output aliases @p acc at lowering time, so this is the
                 * ordered, allocation-free way to sum gradient contributions at
                 * a fan-out. It is ordered by the graph's own edges rather than
                 * by atomics, which is what keeps replay byte-identical under
                 * SushiRuntime's @c sushiruntime_deterministic_fp contract.
                 *
                 * @param acc The buffer accumulated into.
                 * @param delta The contribution added.
                 * @return The accumulated value's id.
                 * @throws Error If the operands disagree in shape or dtype.
                 */
                ValueId accumulate(ValueId acc, ValueId delta) const;

                /**
                 * @brief Declares a value with the same shape, dtype and lifetime as @p like.
                 * @param like The value to copy the description from.
                 * @param debug_name A label for diagnostics; must outlive the graph.
                 * @return The new value's id.
                 */
                ValueId like(ValueId like, std::string_view debug_name = {}) const;

            private:
                ValueId declare(Shape shape, DType dtype, Lifetime lifetime,
                                std::string_view debug_name = {}) const;

                ValueId elementwise(OpID op, ValueId a, ValueId b, std::string_view what) const;

                ValueId unary(OpID op, ValueId x) const;

                void require_same(ValueId a, ValueId b, std::string_view what) const;

                Graph* graph_;
        };

    } // namespace Graph
} // namespace SushiAI
