/**************************************************************************/
/* rules.hpp                                                              */
/**************************************************************************/
/*                          This file is part of:                         */
/*                                 SushiAI                                */
/*                 https://github.com/SushiSystems/SushiAI                */
/*                         https://sushisystems.io                        */
/**************************************************************************/
/* Copyright (c) 2026-present Mustafa Garip & Sushi Systems               */
/*                                                                        */
/* Licensed under the Apache License, Version 2.0 (the "License"); you    */
/* may not use this file except in compliance with the License. You may   */
/* obtain a copy of the License at                                        */
/*                                                                        */
/*     https://www.apache.org/licenses/LICENSE-2.0                        */
/*                                                                        */
/* Unless required by applicable law or agreed to in writing, software    */
/* distributed under the License is distributed on an "AS IS" BASIS,      */
/* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or        */
/* implied. See the License for the specific language governing           */
/* permissions and limitations under the License.                         */
/**************************************************************************/

/**
 * @file rules.hpp
 * @brief Declares the host rules HostEvaluator runs, one per operation, and what they read.
 * @author Mustafa Garip
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>
#include <SushiAI/ops/ops.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/graph/ir.hpp>

namespace SushiAI
{
    namespace Graph
    {
        namespace Detail
        {
            /**
             * @class HostNodeContext
             * @brief Hands one node's description and host buffers to the rule that computes it.
             */
            class HostNodeContext final
            {
                public:
                    /**
                     * @brief Binds @p node of @p graph to @p buffers.
                     * @pre Every input of @p node holds its elements in @p buffers, and every
                     *      output buffer is sized to its value's element count.
                     */
                    HostNodeContext(const Graph& graph, const Node& node,
                                    std::vector<std::vector<double>>& buffers) noexcept
                        : graph_(graph), node_(node), buffers_(buffers)
                    {

                    }

                    HostNodeContext(const HostNodeContext&) = delete;
                    HostNodeContext& operator=(const HostNodeContext&) = delete;
                    HostNodeContext(HostNodeContext&&) = delete;
                    HostNodeContext& operator=(HostNodeContext&&) = delete;
                    ~HostNodeContext() = default;

                    /** @brief Returns the node being computed. */
                    [[nodiscard]] const Node& node() const noexcept
                    {
                        return node_;
                    }

                    /** @brief Returns the number of operands the node reads. */
                    [[nodiscard]] std::size_t input_count() const noexcept
                    {
                        return node_.inputs.size();
                    }

                    /** @brief Returns the shape of operand @p at. */
                    [[nodiscard]] const Shape& input_shape(std::size_t at) const noexcept
                    {
                        return graph_.value(node_.inputs[at]).shape;
                    }

                    /** @brief Returns the elements of operand @p at. */
                    [[nodiscard]] const std::vector<double>& input(std::size_t at) const noexcept
                    {
                        return buffers_[node_.inputs[at]];
                    }

                    /** @brief Returns the description of result @p at. */
                    [[nodiscard]] const Value& output_value(std::size_t at) const noexcept
                    {
                        return graph_.value(node_.outputs[at]);
                    }

                    /** @brief Returns the buffer of result @p at, sized to its element count. */
                    [[nodiscard]] std::vector<double>& output(std::size_t at) const noexcept
                    {
                        return buffers_[node_.outputs[at]];
                    }

                    /** @brief Reads attribute slot @p slot as an extent or an index. */
                    [[nodiscard]] int64_t attribute_int(std::size_t slot) const noexcept
                    {
                        return static_cast<int64_t>(node_.attributes.get(slot));
                    }

                private:
                    /// @brief The graph the node belongs to.
                    const Graph& graph_;

                    /// @brief The node being computed.
                    const Node& node_;

                    /// @brief The elements of every value of the graph, by ValueID.
                    std::vector<std::vector<double>>& buffers_;
            };

            /** @brief Computes the results of one node from its operands. */
            using HostRule = Status (*)(const HostNodeContext&);

            /** @brief Pairs an operation with the host rule that computes it. */
            struct HostRuleEntry
            {
                OpID op;       /**< The operation. */
                HostRule rule; /**< What computes it. */
            };

            /** @brief Writes zero into every element of the result. */
            [[nodiscard]] Status host_zero(const HostNodeContext& context) noexcept;
            /** @brief Writes attribute slot 0 into every element of the result. */
            [[nodiscard]] Status host_constant(const HostNodeContext& context) noexcept;
            /** @brief Copies the operand, rounded to float for a FLOAT32 result. */
            [[nodiscard]] Status host_cast(const HostNodeContext& context) noexcept;
            /** @brief Writes the elementwise sum of two operands. */
            [[nodiscard]] Status host_add(const HostNodeContext& context) noexcept;
            /** @brief Writes the elementwise difference of two operands. */
            [[nodiscard]] Status host_sub(const HostNodeContext& context) noexcept;
            /** @brief Writes the elementwise product of two operands. */
            [[nodiscard]] Status host_mul(const HostNodeContext& context) noexcept;
            /** @brief Writes the operand times attribute slot 0. */
            [[nodiscard]] Status host_scale(const HostNodeContext& context) noexcept;
            /** @brief Writes the elementwise exponential of the operand. */
            [[nodiscard]] Status host_exp(const HostNodeContext& context) noexcept;
            /** @brief Writes 1 where the first operand is at most the second and 0 elsewhere. */
            [[nodiscard]] Status host_less_equal(const HostNodeContext& context) noexcept;
            /** @brief Writes operand 1 where operand 0 is non-zero and operand 2 elsewhere. */
            [[nodiscard]] Status host_select(const HostNodeContext& context) noexcept;
            /** @brief Writes the tanh approximation of GELU of the operand. */
            [[nodiscard]] Status host_gelu(const HostNodeContext& context) noexcept;
            /** @brief Writes x times the logistic function of x for each element. */
            [[nodiscard]] Status host_silu(const HostNodeContext& context) noexcept;
            /** @brief Writes the operand where it is positive and 0 elsewhere. */
            [[nodiscard]] Status host_relu(const HostNodeContext& context) noexcept;
            /** @brief Writes the absolute value of the operand. */
            [[nodiscard]] Status host_abs(const HostNodeContext& context) noexcept;
            /** @brief Writes log(1 + exp(x)) of each element without overflow. */
            [[nodiscard]] Status host_softplus(const HostNodeContext& context) noexcept;

            /** @brief Writes op(A) * op(B) of rank 2 or 3, transposing per slots 0 and 1. */
            [[nodiscard]] Status host_matmul(const HostNodeContext& context) noexcept;
            /** @brief Writes an [N, C] operand plus a [C] bias added to every row. */
            [[nodiscard]] Status host_bias_add(const HostNodeContext& context) noexcept;
            /** @brief Writes the column sums of an [N, C] operand. */
            [[nodiscard]] Status host_sum_rows(const HostNodeContext& context) noexcept;
            /** @brief Writes a zero [V, D] result with each row added to the row its index names. */
            [[nodiscard]] Status host_scatter_add_rows(const HostNodeContext& context) noexcept;
            /** @brief Writes the scores with each entry above the diagonal set to slot 0. */
            [[nodiscard]] Status host_causal_mask(const HostNodeContext& context) noexcept;

            /** @brief Copies the operand, whose elements a reshape keeps in order. */
            [[nodiscard]] Status host_reshape(const HostNodeContext& context) noexcept;
            /** @brief Writes the rows of a [V, D] table that an index operand names. */
            [[nodiscard]] Status host_gather_rows(const HostNodeContext& context) noexcept;
            /** @brief Writes the last-axis slice that starts at attribute slot 0. */
            [[nodiscard]] Status host_slice_channels(const HostNodeContext& context) noexcept;
            /** @brief Writes the operands joined along the last axis. */
            [[nodiscard]] Status host_concat_channels(const HostNodeContext& context) noexcept;
            /** @brief Writes gamma times the operand plus beta, per last-axis channel. */
            [[nodiscard]] Status host_channel_affine(const HostNodeContext& context) noexcept;

            /** @brief Writes the log-softmax of each row of an [N, C] operand. */
            [[nodiscard]] Status host_log_softmax(const HostNodeContext& context) noexcept;
            /** @brief Writes the mean negative log-likelihood of the one-hot targets. */
            [[nodiscard]] Status host_nll_loss(const HostNodeContext& context) noexcept;
            /** @brief Writes the layer normalisation of each last-axis row and its statistics. */
            [[nodiscard]] Status host_layer_norm(const HostNodeContext& context) noexcept;
            /** @brief Writes the batch normalisation of an NHWC operand and its statistics. */
            [[nodiscard]] Status host_batchnorm2d(const HostNodeContext& context) noexcept;
            /** @brief Writes the scale and the shift that fold a normalisation into a conv. */
            [[nodiscard]] Status host_batchnorm_fold(const HostNodeContext& context) noexcept;

            /** @brief Writes each NHWC window as a row of the column matrix, padding with zeros. */
            [[nodiscard]] Status host_im2col(const HostNodeContext& context) noexcept;
            /** @brief Writes each channel of an NHWC operand filtered by its own taps. */
            [[nodiscard]] Status host_depthwise_conv2d(const HostNodeContext& context) noexcept;
            /** @brief Writes the maximum of each NHWC window and the flat index it came from. */
            [[nodiscard]] Status host_maxpool2d(const HostNodeContext& context) noexcept;
            /** @brief Writes each NHWC pixel repeated by the scales in slots 0 and 1. */
            [[nodiscard]] Status host_upsample2d(const HostNodeContext& context) noexcept;
        } // namespace Detail
    } // namespace Graph
} // namespace SushiAI
