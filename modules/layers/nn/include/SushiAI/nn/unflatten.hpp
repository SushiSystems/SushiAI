/**************************************************************************/
/* unflatten.hpp                                                          */
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
 * @file unflatten.hpp
 * @brief Declares the layer that gives a flat feature vector its image shape.
 * @author Mustafa Garip
 */

#pragma once

#include <cstdint>
#include <tuple>
#include <SushiAI/core/shape.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/graph/builder.hpp>
#include <SushiAI/nn/module.hpp>

namespace SushiAI
{
    namespace NN
    {
        /** @brief Views an [N, height * width * channels] value as [N, height, width, channels]. */
        struct Unflatten
        {
            /** @brief Holds the image height each sample is given. */
            int64_t height = 0;

            /** @brief Holds the image width each sample is given. */
            int64_t width = 0;

            /** @brief Holds the channel count each sample is given. */
            int64_t channels = 0;

            /** @brief Lists this layer's parameters, of which there are none. */
            static constexpr auto fields()
            {
                return std::tuple<>{};
            }

            /**
             * @brief Traces a RESHAPE of @p x to [N, height, width, channels].
             * @param x An NHWC [N, height * width * channels] value.
             * @return The reshaped value's id.
             */
            Graph::ValueID forward(const Graph::GraphBuilder& builder, Graph::ValueID x) const;

            /**
             * @brief Computes the per-sample shape forward() produces.
             * @param input The per-sample [height * width * channels] shape.
             * @return The per-sample [height, width, channels] shape.
             */
            Shape output_shape(const Shape& input) const;
        };

        static_assert(Module<Unflatten>, "Unflatten must satisfy the Module contract");

    } // namespace NN
} // namespace SushiAI
