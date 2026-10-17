/**************************************************************************/
/* avg_pool2d.hpp                                                         */
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
 * @file avg_pool2d.hpp
 * @brief Declares the 2-D average pooling layer that traces into graph IR.
 * @author Mustafa Garip
 */

#pragma once

#include <tuple>
#include <SushiAI/shape/shape.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/graph/builder.hpp>
#include <SushiAI/nn/module.hpp>

namespace SushiAI
{
    namespace NN
    {
        /** @brief Averages every window of an NHWC image, dividing by the full window area. */
        struct AvgPool2d
        {
            /** @brief Holds the window; its top and left pads apply to both sides of an axis. */
            Graph::ConvGeometry geometry{};

            /** @brief Lists this layer's parameters, of which there are none. */
            static constexpr auto fields()
            {
                return std::tuple<>{};
            }

            /**
             * @brief Traces the pooling node.
             * @param x An NHWC [N, H, W, C] value that the window fits.
             * @return The pooled [N, out_h, out_w, C] value's id.
             */
            Graph::ValueID forward(const Graph::GraphBuilder& builder, Graph::ValueID x) const;

            /**
             * @brief Computes the per-sample shape forward() produces.
             * @param input A per-sample [H, W, C] shape that the window fits.
             * @return The per-sample [out_h, out_w, C] shape.
             */
            Shape output_shape(const Shape& input) const;
        };

        static_assert(Module<AvgPool2d>, "AvgPool2d must satisfy the Module contract");

    } // namespace NN
} // namespace SushiAI
