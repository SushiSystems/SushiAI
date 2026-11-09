/**************************************************************************/
/* host_evaluator.cpp                                                     */
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
 * @file host_evaluator.cpp
 * @brief Defines HostEvaluator and the table of the host rules it runs.
 * @author Mustafa Garip
 */

#include <SushiAI/graph/host_evaluator.hpp>

#include <array>
#include <cstddef>
#include <utility>
#include <vector>
#include <SushiAI/shape/shape.hpp>
#include "host_evaluation/rules.hpp"

namespace SushiAI
{
    namespace Graph
    {
        namespace
        {
            /** @brief Lists every operation HostEvaluator computes, with its rule. */
            constexpr std::array<Detail::HostRuleEntry, 31> K_HOST_RULES = {{
                {Ops::K_ZERO.id, &Detail::host_zero},
                {Ops::K_CONSTANT.id, &Detail::host_constant},
                {Ops::K_CAST.id, &Detail::host_cast},
                {Ops::K_ADD.id, &Detail::host_add},
                {Ops::K_SUB.id, &Detail::host_sub},
                {Ops::K_MUL.id, &Detail::host_mul},
                {Ops::K_SCALE.id, &Detail::host_scale},
                {Ops::K_EXP.id, &Detail::host_exp},
                {Ops::K_LESS_EQUAL.id, &Detail::host_less_equal},
                {Ops::K_SELECT.id, &Detail::host_select},
                {Ops::K_GELU.id, &Detail::host_gelu},
                {Ops::K_SILU.id, &Detail::host_silu},
                {Ops::K_RELU.id, &Detail::host_relu},
                {Ops::K_ABS.id, &Detail::host_abs},
                {Ops::K_SOFTPLUS.id, &Detail::host_softplus},
                {Ops::K_MATMUL.id, &Detail::host_matmul},
                {Ops::K_BIAS_ADD.id, &Detail::host_bias_add},
                {Ops::K_SUM_ROWS.id, &Detail::host_sum_rows},
                {Ops::K_RESHAPE.id, &Detail::host_reshape},
                {Ops::K_GATHER_ROWS.id, &Detail::host_gather_rows},
                {Ops::K_SLICE_CHANNELS.id, &Detail::host_slice_channels},
                {Ops::K_CONCAT_CHANNELS.id, &Detail::host_concat_channels},
                {Ops::K_CHANNEL_AFFINE.id, &Detail::host_channel_affine},
                {Ops::K_LOG_SOFTMAX.id, &Detail::host_log_softmax},
                {Ops::K_LAYER_NORM.id, &Detail::host_layer_norm},
                {Ops::K_BATCHNORM2D.id, &Detail::host_batchnorm2d},
                {Ops::K_BATCHNORM_FOLD.id, &Detail::host_batchnorm_fold},
                {Ops::K_IM2COL.id, &Detail::host_im2col},
                {Ops::K_DEPTHWISE_CONV2D.id, &Detail::host_depthwise_conv2d},
                {Ops::K_MAXPOOL2D.id, &Detail::host_maxpool2d},
                {Ops::K_UPSAMPLE2D.id, &Detail::host_upsample2d},
            }};

            /** @brief Returns the rule of @p op, or nullptr when the table holds none. */
            Detail::HostRule rule_of(OpID op) noexcept
            {
                for (const Detail::HostRuleEntry& entry : K_HOST_RULES)
                {
                    if (entry.op == op)
                        return entry.rule;
                }

                return nullptr;
            }

            /** @brief Returns the element count of @p value. */
            std::size_t element_count(const Value& value) noexcept
            {
                return static_cast<std::size_t>(value.shape.num_elements());
            }
        } // namespace

        bool host_evaluates(OpID op) noexcept
        {
            return op == Ops::K_LEAF.id || rule_of(op) != nullptr;
        }

        HostEvaluator::HostEvaluator(const Graph& graph) noexcept
            : graph_(graph), buffers_(graph.value_count()), written_(graph.value_count(), false)
        {

        }

        Status HostEvaluator::write(ValueID value, std::vector<double> elements) noexcept
        {
            SA_FAIL_IF(value >= graph_.value_count(), StatusCode::InvalidDescription,
                       "HostEvaluator cannot write value {}; the graph holds {} values", value,
                       graph_.value_count());
            const Value& described = graph_.value(value);
            SA_FAIL_IF(described.producer != K_NO_NODE
                           && graph_.node(described.producer).op != Ops::K_LEAF.id,
                       StatusCode::InvalidDescription,
                       "HostEvaluator cannot write value {} ('{}'); {} computes it", value,
                       described.debug_name, Ops::name_of(graph_.node(described.producer).op));
            SA_FAIL_IF(elements.size() != element_count(described),
                       StatusCode::InvalidDescription,
                       "HostEvaluator needs {} elements for value {} ('{}' of {}); got {}",
                       element_count(described), value, described.debug_name,
                       described.shape.to_string(), elements.size());

            buffers_[value] = std::move(elements);
            written_[value] = true;
            return Status();
        }

        Status HostEvaluator::run() noexcept
        {
            for (const Node& node : graph_.nodes())
            {
                for (ValueID input : node.inputs)
                    SA_FAIL_IF(!written_[input], StatusCode::InvalidDescription,
                               "{} reads value {} ('{}'), which holds no elements; write each "
                               "leaf before run()",
                               Ops::name_of(node.op), input, graph_.value(input).debug_name);

                if (node.op == Ops::K_LEAF.id)
                {
                    for (ValueID output : node.outputs)
                        SA_FAIL_IF(!written_[output], StatusCode::InvalidDescription,
                                   "Leaf {} ('{}') holds no elements; write it before run()",
                                   output, graph_.value(output).debug_name);
                    continue;
                }

                const Detail::HostRule rule = rule_of(node.op);
                SA_FAIL_IF(rule == nullptr, StatusCode::Unsupported,
                           "HostEvaluator has no rule for {}", Ops::name_of(node.op));

                for (ValueID output : node.outputs)
                    buffers_[output].assign(element_count(graph_.value(output)), 0.0);

                SA_TRY(rule(Detail::HostNodeContext(graph_, node, buffers_)));
                for (ValueID output : node.outputs)
                    written_[output] = true;
            }

            return Status();
        }

        Result<std::vector<double>> HostEvaluator::read(ValueID value) const noexcept
        {
            SA_FAIL_IF(value >= graph_.value_count() || !written_[value],
                       StatusCode::InvalidDescription,
                       "HostEvaluator holds no elements for value {}; write it or run() first",
                       value);
            return buffers_[value];
        }
    } // namespace Graph
} // namespace SushiAI
