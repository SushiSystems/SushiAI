/**************************************************************************/
/* flatten.hpp                                                            */
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
 * @file flatten.hpp
 * @brief Declares the Flatten layer, which folds every per-sample axis into one.
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
        /** @brief Folds [N, d1, ..., dk] into [N, d1 * ... * dk] without moving data. */
        struct Flatten
        {
            /** @brief Lists this layer's parameters, of which there are none. */
            static constexpr auto fields()
            {
                return std::tuple<>{};
            }

            /**
             * @brief Traces the layer as one RESHAPE, or as nothing for a rank-2 input.
             * @param x An NHWC input of rank 2 or more, batch axis first.
             * @return The [N, d1 * ... * dk] result's id; @p x itself when it is rank 2.
             */
            Graph::ValueID forward(const Graph::GraphBuilder& builder, Graph::ValueID x) const;

            /**
             * @brief Computes the per-sample shape forward() produces.
             * @param input A per-sample shape of rank 1 or more.
             * @return The rank-1 shape holding the element count of @p input.
             */
            Shape output_shape(const Shape& input) const;
        };

        static_assert(Module<Flatten>, "Flatten must satisfy the Module contract");

    } // namespace NN
} // namespace SushiAI
