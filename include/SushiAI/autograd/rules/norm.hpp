/****************************************************************/
/* norm.hpp                                                     */
/* SushiAI - https://github.com/SushiSystems/SushiAI            */
/* Copyright (c) 2026 Sushi Systems                             */
/* Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.    */
/* Commercial use requires a licence from Sushi Systems.        */
/****************************************************************/

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
