/**************************************************************************/
/* identity.cpp                                                           */
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
 * @file identity.cpp
 * @brief Implements tracing and shape inference for the Identity layer.
 * @author Mustafa Garip
 */

#include <SushiAI/nn/identity.hpp>

namespace SushiAI
{
    namespace NN
    {
        Graph::Traced Identity::forward(const Graph::GraphBuilder&,
                                        const Graph::Traced& input) const noexcept
        {
            return input;
        }

        Result<Shape> Identity::output_shape(const Shape& input) const noexcept
        {
            return input;
        }

    } // namespace NN
} // namespace SushiAI
