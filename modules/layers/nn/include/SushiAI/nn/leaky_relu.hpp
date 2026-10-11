/**************************************************************************/
/* leaky_relu.hpp                                                         */
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
 * @file leaky_relu.hpp
 * @brief Declares the LeakyReLU activation layer.
 * @author Mustafa Garip
 */

#pragma once

#include <tuple>
#include <SushiAI/core/shape.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/graph/builder.hpp>
#include <SushiAI/nn/module.hpp>

namespace SushiAI
{
    namespace NN
    {
        /** @brief Applies x where x is positive and slope * x elsewhere, elementwise. */
        struct LeakyReLU
        {
            /** @brief Holds the factor applied to negative inputs; must lie in (0, 1). */
            double slope = 0.01;

            /** @brief Lists this layer's parameters, of which there are none. */
            static constexpr auto fields()
            {
                return std::tuple<>{};
            }

            /**
             * @brief Traces the activation over an input of any rank.
             * @return The result's id, shaped and laid out like @p x.
             */
            Graph::ValueID forward(const Graph::GraphBuilder& builder, Graph::ValueID x) const;

            /**
             * @brief Computes the per-sample shape forward() produces.
             * @return The shape of @p input, which the activation keeps.
             */
            Shape output_shape(const Shape& input) const;
        };

        static_assert(Module<LeakyReLU>, "LeakyReLU must satisfy the Module contract");

    } // namespace NN
} // namespace SushiAI
