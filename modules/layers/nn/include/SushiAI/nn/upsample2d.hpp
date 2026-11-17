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
/* Licensed under the Apache License, Version 2.0 (the "License"); you    */
/* may not use this file except in compliance with the License. You may   */
/* obtain a copy of the License at                                        */
/*                                                                        */
/*     https://www.apache.org/licenses/LICENSE-2.0                        */
/*                                                                        */
/* Unless required by applicable law or agreed to in writing, software    */
/* distributed under the License is distributed on an "AS IS" BASIS,      */
/* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or        */
/* implied. See the License for the specific language governing           */
/* permissions and limitations under the License.                         */
/**************************************************************************/

/**
 * @file upsample2d.hpp
 * @brief Declares the upsampling layer, nearest-neighbour or bilinear.
 * @author Mustafa Garip
 */

#pragma once

#include <cstdint>
#include <string_view>
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
        /**
         * @brief Scales an NHWC image by an integer factor: nearest repeats every pixel over a
         *        scale-by-scale block, bilinear interpolates each output pixel from the four
         *        input pixels around its half-pixel centre.
         */
        struct Upsample2d
        {
            /** @brief Names how the layer fills the pixels it adds. */
            enum class Mode : std::uint8_t
            {
                Nearest,  /**< Repeats the nearest input pixel; the default. */
                Bilinear, /**< Interpolates between the four input pixels around the centre. */
            };

            /** @brief Names the value of the "mode" key that selects Mode::Nearest. */
            static constexpr std::string_view K_NEAREST_NAME = "nearest";

            /** @brief Names the value of the "mode" key that selects Mode::Bilinear. */
            static constexpr std::string_view K_BILINEAR_NAME = "bilinear";

            /**
             * @brief Reads the name a model file gives a mode.
             * @return The mode, or InvalidDescription naming the two it knows.
             */
            [[nodiscard]] static Result<Mode> mode_from_name(std::string_view name) noexcept;

            /**
             * @brief Holds the replication factor applied to both height and width; must be at
             * least 1.
             */
            int64_t scale = 2;

            /** @brief Holds how the added pixels are filled. */
            Mode mode = Mode::Nearest;

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
