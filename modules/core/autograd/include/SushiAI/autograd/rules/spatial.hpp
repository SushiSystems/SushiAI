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
 * @brief Declares backward differentiation rules for spatial and pooling operations.
 * @author Mustafa Garip
 */

#pragma once

#include <array>
#include <SushiAI/autograd/rule.hpp>

namespace SushiAI
{
    namespace Autograd
    {
        /**
         * @brief IM2COL's gradient: col2im, scattering each row back onto the
         *        pixels its window read, summing where windows overlapped.
         * @param tape The differentiation state for this node.
         */
        void im2col_backward(const GradientTape& tape) noexcept;

        /**
         * @brief COL2IM's gradient: im2col with the forward's own geometry,
         *        the adjoint of the adjoint.
         * @param tape The differentiation state for this node.
         */
        void col2im_backward(const GradientTape& tape) noexcept;

        /**
         * @brief MAXPOOL2D's gradient: routes each gradient through the
         *        argmax indices its own forward wrote.
         * @param tape The differentiation state for this node.
         */
        void maxpool2d_backward(const GradientTape& tape) noexcept;

        /**
         * @brief AVGPOOL2D's gradient: spreads each gradient evenly over the
         *        window it came from, dividing by the constant window area.
         * @param tape The differentiation state for this node.
         */
        void avgpool2d_backward(const GradientTape& tape) noexcept;

        /**
         * @brief UPSAMPLE2D's gradient: sums each source pixel's replicas.
         * @param tape The differentiation state for this node.
         */
        void upsample2d_backward(const GradientTape& tape) noexcept;

        /**
         * @brief SPATIAL_MEAN's gradient: broadcasts the [N,C] gradient back
         *        over [N,H,W,C], scaled by 1/(H*W).
         * @param tape The differentiation state for this node.
         */
        void spatial_mean_backward(const GradientTape& tape) noexcept;

        /**
         * @brief CONCAT_CHANNELS's gradient: each operand receives the slice
         *        of the incoming gradient at that operand's own running
         *        channel offset.
         * @param tape The differentiation state for this node.
         */
        void concat_channels_backward(const GradientTape& tape) noexcept;

        /**
         * @brief SLICE_CHANNELS's gradient: a scatter into zeros, expressed as
         *        a concat of the zero ranges before and after the slice with
         *        the incoming gradient itself.
         * @param tape The differentiation state for this node.
         */
        void slice_channels_backward(const GradientTape& tape) noexcept;

        /**
         * @brief DEPTHWISE_CONV2D's gradient: the input gradient and the taps gradient,
         *        each emitted only when its operand needs one.
         * @param tape The differentiation state for this node.
         */
        void depthwise_conv2d_backward(const GradientTape& tape) noexcept;

        /** @brief Spatial operations' backward rules. */
        inline constexpr std::array<RuleEntry, 9> K_SPATIAL_RULES = {{
            {Ops::K_IM2COL.id, &im2col_backward},
            {Ops::K_COL2IM.id, &col2im_backward},
            {Ops::K_MAXPOOL2D.id, &maxpool2d_backward},
            {Ops::K_AVGPOOL2D.id, &avgpool2d_backward},
            {Ops::K_UPSAMPLE2D.id, &upsample2d_backward},
            {Ops::K_SPATIAL_MEAN.id, &spatial_mean_backward},
            {Ops::K_CONCAT_CHANNELS.id, &concat_channels_backward},
            {Ops::K_SLICE_CHANNELS.id, &slice_channels_backward},
            {Ops::K_DEPTHWISE_CONV2D.id, &depthwise_conv2d_backward},
        }};
    }
}
