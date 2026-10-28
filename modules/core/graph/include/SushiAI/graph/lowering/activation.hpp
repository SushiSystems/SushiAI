/**************************************************************************/
/* activation.hpp                                                         */
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
 * @file activation.hpp
 * @brief Declares the lowering rows of the activation operations and their table.
 * @author Mustafa Garip
 */

#pragma once

#include <array>
#include <SushiAI/graph/lowering/rule.hpp>

namespace SushiAI
{
    namespace Graph
    {
        /** @brief Records the ReLU of one operand. */
        [[nodiscard]] Status lower_relu(const LoweringContext& context) noexcept;

        /** @brief Records the GELU of one operand. */
        [[nodiscard]] Status lower_gelu(const LoweringContext& context) noexcept;

        /** @brief Records the logistic sigmoid of one operand. */
        [[nodiscard]] Status lower_sigmoid(const LoweringContext& context) noexcept;

        /** @brief Records the tanh of one operand. */
        [[nodiscard]] Status lower_tanh(const LoweringContext& context) noexcept;

        /** @brief Records the SiLU of one operand. */
        [[nodiscard]] Status lower_silu(const LoweringContext& context) noexcept;

        /** @brief Records the softplus of one operand. */
        [[nodiscard]] Status lower_softplus(const LoweringContext& context) noexcept;

        /** @brief Records the leaky ReLU of one operand, with the slope of attribute 0. */
        [[nodiscard]] Status lower_leaky_relu(const LoweringContext& context) noexcept;

        /** @brief Records the ReLU gradient from the upstream gradient and a forward tensor. */
        [[nodiscard]] Status lower_relu_backward(const LoweringContext& context) noexcept;

        /** @brief Records the GELU gradient from the upstream gradient and a forward tensor. */
        [[nodiscard]] Status lower_gelu_backward(const LoweringContext& context) noexcept;

        /** @brief Records the sigmoid gradient from the upstream gradient and a forward tensor. */
        [[nodiscard]] Status lower_sigmoid_backward(const LoweringContext& context) noexcept;

        /** @brief Records the tanh gradient from the upstream gradient and a forward tensor. */
        [[nodiscard]] Status lower_tanh_backward(const LoweringContext& context) noexcept;

        /** @brief Records the SiLU gradient from the upstream gradient and a forward tensor. */
        [[nodiscard]] Status lower_silu_backward(const LoweringContext& context) noexcept;

        /** @brief Records the softplus gradient from the upstream gradient and a forward tensor. */
        [[nodiscard]] Status lower_softplus_backward(const LoweringContext& context) noexcept;

        /** @brief Records the leaky ReLU gradient, with the slope of attribute 0. */
        [[nodiscard]] Status lower_leaky_relu_backward(const LoweringContext& context) noexcept;

        /** @brief Pairs each activation operation with the row that lowers it. */
        inline constexpr std::array<LoweringEntry, 14> K_ACTIVATION_LOWERINGS = {{
            {Ops::K_RELU.id, &lower_relu},
            {Ops::K_GELU.id, &lower_gelu},
            {Ops::K_SIGMOID.id, &lower_sigmoid},
            {Ops::K_TANH.id, &lower_tanh},
            {Ops::K_SILU.id, &lower_silu},
            {Ops::K_SOFTPLUS.id, &lower_softplus},
            {Ops::K_LEAKY_RELU.id, &lower_leaky_relu},
            {Ops::K_RELU_BACKWARD.id, &lower_relu_backward},
            {Ops::K_GELU_BACKWARD.id, &lower_gelu_backward},
            {Ops::K_SIGMOID_BACKWARD.id, &lower_sigmoid_backward},
            {Ops::K_TANH_BACKWARD.id, &lower_tanh_backward},
            {Ops::K_SILU_BACKWARD.id, &lower_silu_backward},
            {Ops::K_SOFTPLUS_BACKWARD.id, &lower_softplus_backward},
            {Ops::K_LEAKY_RELU_BACKWARD.id, &lower_leaky_relu_backward},
        }};
    } // namespace Graph
} // namespace SushiAI
