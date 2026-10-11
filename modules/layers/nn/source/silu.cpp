/**************************************************************************/
/* silu.cpp                                                               */
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
 * @file silu.cpp
 * @brief Implements tracing and shape inference for the SiLU layer.
 * @author Mustafa Garip
 */

#include <SushiAI/nn/silu.hpp>

namespace SushiAI
{
    namespace NN
    {
        Graph::ValueID SiLU::forward(const Graph::GraphBuilder& builder, Graph::ValueID x) const
        {
            return builder.silu(x);
        }

        Shape SiLU::output_shape(const Shape& input) const
        {
            return input;
        }

    } // namespace NN
} // namespace SushiAI
