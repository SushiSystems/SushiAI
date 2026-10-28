/**************************************************************************/
/* upsample2d.hpp                                                         */
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
 * @file upsample2d.hpp
 * @brief Declares the nearest-neighbour upsampling layer.
 * @author Mustafa Garip
 */

#pragma once

#include <cstdint>
#include <tuple>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/shape/shape.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/graph/builder.hpp>
#include <SushiAI/nn/module.hpp>

namespace SushiAI
{
    namespace NN
    {
        /** @brief Repeats every pixel of an NHWC image over a scale-by-scale block. */
        struct Upsample2d
        {
            /**
             * @brief Holds the replication factor applied to both height and width; must be at
             * least 1.
             */
            int64_t scale = 2;

            /** @brief Lists this layer's parameters, of which there are none. */
            static constexpr auto fields() noexcept
            {
                return std::tuple<>{};
            }

            /**
             * @brief Traces the upsampling of @p input.
             * @param input An NHWC [N, H, W, C] value.
             * @return The [N, H * scale, W * scale, C] result; the failure of @p input,
             *         or InvalidDescription for an input the layer refuses.
             */
            Graph::Traced forward(const Graph::GraphBuilder& builder,
                                  const Graph::Traced& input) const noexcept;

            /**
             * @brief Computes the per-sample shape forward() produces.
             * @param input A per-sample [H, W, C] shape.
             * @return The per-sample [H * scale, W * scale, C] shape, or
             *         InvalidDescription for a scale or a shape the layer refuses.
             */
            Result<Shape> output_shape(const Shape& input) const noexcept;
        };

        static_assert(Module<Upsample2d>, "Upsample2d must satisfy the Module contract");

    } // namespace NN
} // namespace SushiAI
