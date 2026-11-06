/**************************************************************************/
/* normalised_branch.hpp                                                  */
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
 * @file normalised_branch.hpp
 * @brief Defines the trace of one residual branch of a transformer block.
 * @author Mustafa Garip
 */

#pragma once

#include <SushiAI/failure/result.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/graph/builder.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/nn/layer_norm.hpp>
#include <SushiAI/nn/trace.hpp>

namespace SushiAI
{
    namespace Language::Blocks
    {
        namespace Detail
        {
            /**
             * @brief Traces @p trace_child over @p norm, then over @p branch, and adds @p x.
             * @param trace_child Called as trace_child(layer, value) for each of the two; it
             *        returns the id of the layer's result.
             * @return The sum's id, or the failure @p norm or @p branch recorded on the trace.
             */
            template<typename Branch, typename TraceChild>
            [[nodiscard]] Result<Graph::ValueID> add_normalised_branch(NN::LayerNorm& norm,
                                                                       Branch& branch,
                                                                       const NN::Trace& trace,
                                                                       Graph::ValueID x,
                                                                       TraceChild&& trace_child)
            {
                const Graph::ValueID normalised = trace_child(norm, x);
                SA_TRY(trace.status());

                const Graph::ValueID mapped = trace_child(branch, normalised);
                SA_TRY(trace.status());

                SA_TRY_ASSIGN(const Graph::ValueID sum, trace.builder().add(x, mapped));
                return sum;
            }

        } // namespace Detail
    } // namespace Language::Blocks
} // namespace SushiAI
