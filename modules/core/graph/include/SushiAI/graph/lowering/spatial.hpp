/**************************************************************************/
/* spatial.hpp                                                            */
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
 * @file spatial.hpp
 * @brief Declares the lowering rows of the spatial operations and their table.
 * @author Mustafa Garip
 */

#pragma once

#include <array>
#include <SushiAI/graph/lowering/rule.hpp>

namespace SushiAI
{
    namespace Graph
    {
        /** @brief Records IM2COL, which gathers every window of an NHWC image into rows. */
        [[nodiscard]] Status lower_im2col(const LoweringContext& context) noexcept;

        /** @brief Records COL2IM, which scatters window rows back onto an NHWC image. */
        [[nodiscard]] Status lower_col2im(const LoweringContext& context) noexcept;

        /** @brief Records MAXPOOL2D, which writes the window maxima and their argmax indices. */
        [[nodiscard]] Status lower_maxpool2d(const LoweringContext& context) noexcept;

        /** @brief Records MAXPOOL2D_BACKWARD, which routes gradients to the argmax elements. */
        [[nodiscard]] Status lower_maxpool2d_backward(const LoweringContext& context) noexcept;

        /** @brief Records AVGPOOL2D, which writes the mean of each window. */
        [[nodiscard]] Status lower_avgpool2d(const LoweringContext& context) noexcept;

        /** @brief Records AVGPOOL2D_BACKWARD, which spreads each gradient over its window. */
        [[nodiscard]] Status lower_avgpool2d_backward(const LoweringContext& context) noexcept;

        /** @brief Records UPSAMPLE2D, which repeats each pixel over a scale block. */
        [[nodiscard]] Status lower_upsample2d(const LoweringContext& context) noexcept;

        /** @brief Records UPSAMPLE2D_BACKWARD, which sums each scale block onto its pixel. */
        [[nodiscard]] Status lower_upsample2d_backward(const LoweringContext& context) noexcept;

        /** @brief Records SPATIAL_MEAN, which averages an NHWC tensor over H and W. */
        [[nodiscard]] Status lower_spatial_mean(const LoweringContext& context) noexcept;

        /** @brief Records SPATIAL_MEAN_BACKWARD, which spreads an [N,C] gradient over H and W. */
        [[nodiscard]] Status lower_spatial_mean_backward(const LoweringContext& context) noexcept;

        /** @brief Records CONCAT_CHANNELS, which copies each operand into a channel range. */
        [[nodiscard]] Status lower_concat_channels(const LoweringContext& context) noexcept;

        /** @brief Records SLICE_CHANNELS, which copies a channel range out of its operand. */
        [[nodiscard]] Status lower_slice_channels(const LoweringContext& context) noexcept;

        /** @brief Records DEPTHWISE_CONV2D, which filters each channel with its own taps. */
        [[nodiscard]] Status lower_depthwise_conv2d(const LoweringContext& context) noexcept;

        /** @brief Records DEPTHWISE_CONV2D_BACKWARD_INPUT, which gathers each input gradient. */
        [[nodiscard]] Status lower_depthwise_conv2d_backward_input(const LoweringContext& context) noexcept;

        /** @brief Records DEPTHWISE_CONV2D_BACKWARD_WEIGHT, which sums each tap's gradient. */
        [[nodiscard]] Status lower_depthwise_conv2d_backward_weight(const LoweringContext& context) noexcept;

        /** @brief Pairs each spatial operation with the row that lowers it. */
        inline constexpr std::array<LoweringEntry, 15> K_SPATIAL_LOWERINGS = {{
            {Ops::K_IM2COL.id, &lower_im2col},
            {Ops::K_COL2IM.id, &lower_col2im},
            {Ops::K_MAXPOOL2D.id, &lower_maxpool2d},
            {Ops::K_MAXPOOL2D_BACKWARD.id, &lower_maxpool2d_backward},
            {Ops::K_AVGPOOL2D.id, &lower_avgpool2d},
            {Ops::K_AVGPOOL2D_BACKWARD.id, &lower_avgpool2d_backward},
            {Ops::K_UPSAMPLE2D.id, &lower_upsample2d},
            {Ops::K_UPSAMPLE2D_BACKWARD.id, &lower_upsample2d_backward},
            {Ops::K_SPATIAL_MEAN.id, &lower_spatial_mean},
            {Ops::K_SPATIAL_MEAN_BACKWARD.id, &lower_spatial_mean_backward},
            {Ops::K_CONCAT_CHANNELS.id, &lower_concat_channels},
            {Ops::K_SLICE_CHANNELS.id, &lower_slice_channels},
            {Ops::K_DEPTHWISE_CONV2D.id, &lower_depthwise_conv2d},
            {Ops::K_DEPTHWISE_CONV2D_BACKWARD_INPUT.id, &lower_depthwise_conv2d_backward_input},
            {Ops::K_DEPTHWISE_CONV2D_BACKWARD_WEIGHT.id, &lower_depthwise_conv2d_backward_weight},
        }};
    } // namespace Graph
} // namespace SushiAI
