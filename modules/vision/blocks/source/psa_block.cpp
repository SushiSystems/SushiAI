/**************************************************************************/
/* psa_block.cpp                                                          */
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
 * @file psa_block.cpp
 * @brief Implements PSABlock's constructor, its trace and its shape rule.
 * @author Mustafa Garip
 */

#include <SushiAI/vision/blocks/psa_block.hpp>

#include <SushiAI/graph/builder.hpp>
#include "block_attributes.hpp"

namespace SushiAI
{
    namespace Vision::Blocks
    {
        PSABlock::PSABlock(int64_t channels, int64_t heads, NN::NormOptions norm)
            : attn(channels, heads, norm),
              ffn(Conv(channels, 2 * channels, Detail::K_POINTWISE, Detail::K_POINTWISE, norm),
                  ConvLinear(2 * channels, channels, Detail::K_POINTWISE, Detail::K_POINTWISE,
                             norm))
        {

        }

        Graph::ValueId PSABlock::forward(const NN::Trace& trace, Graph::ValueId x)
        {
            const Graph::GraphBuilder& builder = trace.builder();
            const Graph::ValueId attended = NN::trace_layer(attn, trace, x);
            const Graph::ValueId mixed = builder.add(x, attended);
            const Graph::ValueId fed = NN::trace_layer(ffn, trace, mixed);

            return builder.add(mixed, fed);
        }

        Shape PSABlock::output_shape(const Shape& input) const
        {
            // attn and ffn each keep the per-sample shape, and each refuses a width it cannot read.
            return ffn.output_shape(attn.output_shape(input));
        }

    } // namespace Vision::Blocks
} // namespace SushiAI
