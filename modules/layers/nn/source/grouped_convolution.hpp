/**************************************************************************/
/* grouped_convolution.hpp                                                */
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
 * @file grouped_convolution.hpp
 * @brief Declares the trace of a convolution whose filters each read one slice of the channels.
 * @author Mustafa Garip
 */

#pragma once

#include <cstdint>
#include <SushiAI/graph/builder.hpp>

namespace SushiAI
{
    namespace NN
    {
        namespace Detail
        {
            /**
             * @brief Traces one convolution per group over slices of @p x, @p weight and
             *        @p bias, and joins the results along the channels.
             * @param weight The [out, kernel_h, kernel_w, in / groups] filters.
             * @param bias The [out] bias, or an absent value to add none.
             * @return The [N, out_height, out_width, out] value, or the failure of an emitter.
             * @pre groups divides both channel counts.
             */
            [[nodiscard]] Graph::Traced trace_grouped_convolution(
                const Graph::GraphBuilder& builder, Graph::ValueID x, Graph::ValueID weight,
                Graph::ValueID bias, const Graph::ConvGeometry& geometry, int64_t out_height,
                int64_t out_width, int64_t groups);
        } // namespace Detail
    } // namespace NN
} // namespace SushiAI
