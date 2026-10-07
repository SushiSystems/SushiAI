/**************************************************************************/
/* global_avg_pool.hpp                                                    */
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
 * @file global_avg_pool.hpp
 * @brief Declares the layer that averages every spatial plane down to one value per channel.
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
        /** @brief Averages an [N, H, W, C] activation over H and W, giving [N, C]. */
        struct GlobalAvgPool
        {
            /** @brief Lists this layer's parameters, of which there are none. */
            static constexpr auto fields()
            {
                return std::tuple<>{};
            }

            /**
             * @brief Traces the spatial mean of @p x.
             * @param x An NHWC [N, H, W, C] value.
             * @return The [N, C] result's id.
             */
            Graph::ValueId forward(const Graph::GraphBuilder& builder, Graph::ValueId x) const;

            /**
             * @brief Computes the per-sample shape forward() produces.
             * @param input A per-sample [H, W, C] shape.
             * @return The per-sample [C] shape.
             */
            Shape output_shape(const Shape& input) const;
        };

        static_assert(Module<GlobalAvgPool>, "GlobalAvgPool must satisfy the Module contract");

    } // namespace NN
} // namespace SushiAI
