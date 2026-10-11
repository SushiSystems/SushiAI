/**************************************************************************/
/* mish.cpp                                                               */
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
 * @file mish.cpp
 * @brief Implements tracing and shape inference for the Mish layer.
 * @author Mustafa Garip
 */

#include <SushiAI/nn/mish.hpp>

namespace SushiAI
{
    namespace NN
    {
        Graph::ValueID Mish::forward(const Graph::GraphBuilder& builder, Graph::ValueID x) const
        {
            const Graph::ValueID softplus_x = builder.softplus(x);
            const Graph::ValueID tanh_softplus_x = builder.tanh(softplus_x);
            return builder.mul(x, tanh_softplus_x);
        }

        Shape Mish::output_shape(const Shape& input) const
        {
            return input;
        }

    } // namespace NN
} // namespace SushiAI
