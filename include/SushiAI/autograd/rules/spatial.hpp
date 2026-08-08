/**************************************************************************/
/* spatial.hpp                                                           */
/**************************************************************************/
/*                          This file is part of:                         */
/*                                 SushiAI                                */
/*                https://github.com/SushiSystems/SushiAI                 */
/*                         https://sushisystems.io                        */
/**************************************************************************/
/* Copyright (c) 2026-present Mustafa Garip & Sushi Systems               */
/*                                                                        */
/* Licensed under the Apache License, Version 2.0 (the "License");        */
/* you may not use this file except in compliance with the License.       */
/* You may obtain a copy of the License at                                */
/*                                                                        */
/*     http://www.apache.org/licenses/LICENSE-2.0                         */
/*                                                                        */
/* Unless required by applicable law or agreed to in writing, software    */
/* distributed under the License is distributed on an "AS IS" BASIS,      */
/* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or        */
/* implied. See the License for the specific language governing           */
/* permissions and limitations under the License.                         */
/**************************************************************************/

#pragma once

// Convolutional and pooling operations' backward rules: IM2COL, COL2IM,
// MAXPOOL2D, AVGPOOL2D, SPATIAL_MEAN, UPSAMPLE2D and the channel-structural
// ops. Empty until Tasks 11-13.

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
        void im2col_backward(const GradientTape& tape);

        /**
         * @brief COL2IM's gradient: im2col with the forward's own geometry,
         *        the adjoint of the adjoint.
         * @param tape The differentiation state for this node.
         */
        void col2im_backward(const GradientTape& tape);

        /**
         * @brief MAXPOOL2D's gradient: routes each gradient through the
         *        argmax indices its own forward wrote.
         * @param tape The differentiation state for this node.
         */
        void maxpool2d_backward(const GradientTape& tape);

        /**
         * @brief AVGPOOL2D's gradient: spreads each gradient evenly over the
         *        window it came from, dividing by the constant window area.
         * @param tape The differentiation state for this node.
         */
        void avgpool2d_backward(const GradientTape& tape);

        /**
         * @brief UPSAMPLE2D's gradient: sums each source pixel's replicas.
         * @param tape The differentiation state for this node.
         */
        void upsample2d_backward(const GradientTape& tape);

        /**
         * @brief SPATIAL_MEAN's gradient: broadcasts the [N,C] gradient back
         *        over [N,H,W,C], scaled by 1/(H*W).
         * @param tape The differentiation state for this node.
         */
        void spatial_mean_backward(const GradientTape& tape);

        /** @brief Spatial operations' backward rules. */
        inline constexpr std::array<RuleEntry, 6> SPATIAL_RULES = {{
            {Ops::IM2COL.id, &im2col_backward},
            {Ops::COL2IM.id, &col2im_backward},
            {Ops::MAXPOOL2D.id, &maxpool2d_backward},
            {Ops::AVGPOOL2D.id, &avgpool2d_backward},
            {Ops::UPSAMPLE2D.id, &upsample2d_backward},
            {Ops::SPATIAL_MEAN.id, &spatial_mean_backward},
        }};
    }
}
