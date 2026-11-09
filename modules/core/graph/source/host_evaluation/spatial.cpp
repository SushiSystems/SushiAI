/**************************************************************************/
/* spatial.cpp                                                            */
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
 * @file spatial.cpp
 * @brief Defines the host rules of the NHWC max pool and nearest-neighbour upsample.
 * @author Mustafa Garip
 */

#include "rules.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>
#include <SushiAI/shape/shape.hpp>

namespace SushiAI
{
    namespace Graph
    {
        namespace Detail
        {
            namespace
            {
                /** @brief Holds the extents of one NHWC value. */
                struct Extents
                {
                    int64_t batch = 0;    /**< N. */
                    int64_t height = 0;   /**< H. */
                    int64_t width = 0;    /**< W. */
                    int64_t channels = 0; /**< C. */
                };

                /** @brief Reads the four extents of an NHWC @p shape. */
                Extents extents_of(const Shape& shape) noexcept
                {
                    return Extents{shape[0], shape[1], shape[2], shape[3]};
                }

                /** @brief Returns the flat offset of (n, h, w, c) in an NHWC value of @p e. */
                int64_t offset_of(const Extents& e, int64_t n, int64_t h, int64_t w,
                                  int64_t c) noexcept
                {
                    return ((n * e.height + h) * e.width + w) * e.channels + c;
                }

                /** @brief Reports whether @p candidate replaces @p best; a NaN wins and stays. */
                bool takes(int64_t best_index, double best, double candidate) noexcept
                {
                    if (best_index < 0)
                        return true;

                    if (std::isnan(best))
                        return false;

                    return std::isnan(candidate) || candidate > best;
                }
            } // namespace

            Status host_im2col(const HostNodeContext& context) noexcept
            {
                const Extents in = extents_of(context.input_shape(0));
                const int64_t kernel_h = context.attribute_int(0);
                const int64_t kernel_w = context.attribute_int(1);
                const int64_t stride_h = context.attribute_int(2);
                const int64_t stride_w = context.attribute_int(3);
                const int64_t dilation_h = context.attribute_int(4);
                const int64_t dilation_w = context.attribute_int(5);
                const int64_t pad_top = context.attribute_int(6);
                const int64_t pad_left = context.attribute_int(7);
                const int64_t out_h = context.attribute_int(8);
                const int64_t out_w = context.attribute_int(9);
                const std::vector<double>& x = context.input(0);
                std::vector<double>& cols = context.output(0);

                // Row (n, h, w) holds taps (r, s, c) in the order (r * kernel_w + s) * C + c.
                std::size_t at = 0;
                for (int64_t n = 0; n < in.batch; ++n)
                    for (int64_t h = 0; h < out_h; ++h)
                        for (int64_t w = 0; w < out_w; ++w)
                            for (int64_t r = 0; r < kernel_h; ++r)
                                for (int64_t s = 0; s < kernel_w; ++s)
                                    for (int64_t c = 0; c < in.channels; ++c, ++at)
                                    {
                                        const int64_t h_in =
                                            h * stride_h - pad_top + r * dilation_h;
                                        const int64_t w_in =
                                            w * stride_w - pad_left + s * dilation_w;
                                        if (h_in < 0 || h_in >= in.height || w_in < 0
                                            || w_in >= in.width)
                                            continue;

                                        cols[at] = x[static_cast<std::size_t>(
                                            offset_of(in, n, h_in, w_in, c))];
                                    }

                return Status();
            }

            Status host_depthwise_conv2d(const HostNodeContext& context) noexcept
            {
                const Extents in = extents_of(context.input_shape(0));
                const Extents out = extents_of(context.output_value(0).shape);
                const int64_t kernel_h = context.attribute_int(0);
                const int64_t kernel_w = context.attribute_int(1);
                const int64_t stride_h = context.attribute_int(2);
                const int64_t stride_w = context.attribute_int(3);
                const int64_t dilation_h = context.attribute_int(4);
                const int64_t dilation_w = context.attribute_int(5);
                const int64_t pad_top = context.attribute_int(6);
                const int64_t pad_left = context.attribute_int(7);
                const std::vector<double>& x = context.input(0);
                const std::vector<double>& taps = context.input(1);
                std::vector<double>& y = context.output(0);

                for (int64_t n = 0; n < out.batch; ++n)
                    for (int64_t h = 0; h < out.height; ++h)
                        for (int64_t w = 0; w < out.width; ++w)
                            for (int64_t c = 0; c < out.channels; ++c)
                            {
                                double sum = 0.0;
                                for (int64_t r = 0; r < kernel_h; ++r)
                                    for (int64_t s = 0; s < kernel_w; ++s)
                                    {
                                        const int64_t h_in =
                                            h * stride_h - pad_top + r * dilation_h;
                                        const int64_t w_in =
                                            w * stride_w - pad_left + s * dilation_w;
                                        if (h_in < 0 || h_in >= in.height || w_in < 0
                                            || w_in >= in.width)
                                            continue;

                                        sum += x[static_cast<std::size_t>(
                                                   offset_of(in, n, h_in, w_in, c))]
                                            * taps[static_cast<std::size_t>(
                                                (r * kernel_w + s) * in.channels + c)];
                                    }

                                y[static_cast<std::size_t>(offset_of(out, n, h, w, c))] = sum;
                            }

                return Status();
            }

            Status host_maxpool2d(const HostNodeContext& context) noexcept
            {
                const Extents in = extents_of(context.input_shape(0));
                const Extents out = extents_of(context.output_value(0).shape);
                const int64_t kernel_h = context.attribute_int(0);
                const int64_t kernel_w = context.attribute_int(1);
                const int64_t stride_h = context.attribute_int(2);
                const int64_t stride_w = context.attribute_int(3);
                const int64_t dilation_h = context.attribute_int(4);
                const int64_t dilation_w = context.attribute_int(5);
                const int64_t pad_top = context.attribute_int(6);
                const int64_t pad_left = context.attribute_int(7);
                const std::vector<double>& x = context.input(0);
                std::vector<double>& y = context.output(0);
                std::vector<double>& indices = context.output(1);

                for (int64_t n = 0; n < out.batch; ++n)
                    for (int64_t h = 0; h < out.height; ++h)
                        for (int64_t w = 0; w < out.width; ++w)
                            for (int64_t c = 0; c < out.channels; ++c)
                            {
                                double best = 0.0;
                                int64_t best_index = -1;
                                for (int64_t r = 0; r < kernel_h; ++r)
                                {
                                    const int64_t h_in = h * stride_h - pad_top + r * dilation_h;
                                    if (h_in < 0 || h_in >= in.height)
                                        continue;

                                    for (int64_t s = 0; s < kernel_w; ++s)
                                    {
                                        const int64_t w_in =
                                            w * stride_w - pad_left + s * dilation_w;
                                        if (w_in < 0 || w_in >= in.width)
                                            continue;

                                        const int64_t at = offset_of(in, n, h_in, w_in, c);
                                        const double candidate = x[static_cast<std::size_t>(at)];
                                        if (takes(best_index, best, candidate))
                                        {
                                            best = candidate;
                                            best_index = at;
                                        }
                                    }
                                }

                                const std::size_t at =
                                    static_cast<std::size_t>(offset_of(out, n, h, w, c));
                                y[at] = best;
                                indices[at] = static_cast<double>(best_index);
                            }

                return Status();
            }

            Status host_upsample2d(const HostNodeContext& context) noexcept
            {
                const Extents in = extents_of(context.input_shape(0));
                const Extents out = extents_of(context.output_value(0).shape);
                const int64_t scale_h = context.attribute_int(0);
                const int64_t scale_w = context.attribute_int(1);
                const std::vector<double>& x = context.input(0);
                std::vector<double>& y = context.output(0);

                for (int64_t n = 0; n < out.batch; ++n)
                    for (int64_t h = 0; h < out.height; ++h)
                        for (int64_t w = 0; w < out.width; ++w)
                            for (int64_t c = 0; c < out.channels; ++c)
                                y[static_cast<std::size_t>(offset_of(out, n, h, w, c))] =
                                    x[static_cast<std::size_t>(
                                        offset_of(in, n, h / scale_h, w / scale_w, c))];

                return Status();
            }
        } // namespace Detail
    } // namespace Graph
} // namespace SushiAI
