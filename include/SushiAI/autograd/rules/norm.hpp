/**************************************************************************/
/* norm.hpp                                                               */
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

/**
 * @file norm.hpp
 * @brief Declares backward differentiation rules for normalization operations.
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
         * @brief Differentiates 2D batch normalization across input, scale, and shift.
         * @param tape Traversal context providing forward nodes and gradient emitters.
         */
        void batchnorm2d_backward(const GradientTape& tape);

        /**
         * @brief Differentiates channel affine transformation with per-channel parameter sums.
         * @param tape Traversal context providing forward nodes and gradient emitters.
         */
        void channel_affine_backward(const GradientTape& tape);

        /** @brief Holds backward rule entries for normalization operations. */
        inline constexpr std::array<RuleEntry, 2> NORM_RULES = {{
            {Ops::BATCHNORM2D.id, &batchnorm2d_backward},
            {Ops::CHANNEL_AFFINE.id, &channel_affine_backward},
        }};
    }
}
