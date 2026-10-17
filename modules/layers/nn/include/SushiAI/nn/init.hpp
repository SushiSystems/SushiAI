/**************************************************************************/
/* init.hpp                                                               */
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
 * @file init.hpp
 * @brief Declares parameter initialization schemes and fan-in calculation helpers.
 * @author Mustafa Garip
 */

#pragma once

#include <cstdint>
#include <SushiBLAS/engine/engine.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/nn/model_state.hpp>
#include <SushiAI/nn/module.hpp>

namespace SushiAI
{
    namespace NN
    {
        /**
         * @enum Nonlinearity
         * @brief The activation a layer feeds, which sets its initialisation gain.
         */
        enum class Nonlinearity : std::uint8_t
        {
            Linear,     /**< No activation, or an identity one. */
            Tanh,       /**< Hyperbolic tangent. */
            ReLU,       /**< Rectified linear unit; also GELU and SiLU in practice. */
            LeakyReLU, /**< Leaky rectified linear unit. */
            Sigmoid,    /**< Logistic sigmoid. */
        };

        /**
         * @brief The variance-preserving gain for @p nonlinearity.
         * @param nonlinearity Which activation follows the layer.
         * @param negative_slope The leaky-ReLU slope; ignored otherwise.
         * @return The multiplicative gain; always positive.
         */
        double calculate_gain(Nonlinearity nonlinearity, double negative_slope = 0.01) noexcept;

        /**
         * @brief Records a fill of @p tensor with @p value.
         * @return Success, or the seam's failure of the recording.
         */
        [[nodiscard]] Status constant(SushiBLAS::Engine& engine, SushiBLAS::Tensor& tensor,
                                      double value);

        /**
         * @brief Records a fill of @p tensor from N(0, deviation^2).
         * @param engine The engine whose Philox stream supplies the numbers.
         * @return Success, InvalidDescription for a deviation that is not positive and
         *         finite, or the seam's failure of the recording.
         */
        [[nodiscard]] Status normal(SushiBLAS::Engine& engine, SushiBLAS::Tensor& tensor,
                                    double deviation);

        /**
         * @brief Records a fill of @p tensor from N(0, (gain / sqrt(fan_in))^2).
         * @param engine The engine whose Philox stream supplies the numbers.
         * @param fan_in Inputs feeding one output unit; must be positive.
         * @return Success, InvalidDescription for a fan-in that is not positive, or the seam's
         *         failure of the recording.
         */
        [[nodiscard]] Status kaiming_normal(SushiBLAS::Engine& engine, SushiBLAS::Tensor& tensor,
                                            int64_t fan_in, double gain);

        /**
         * @brief Records a fill of @p tensor uniformly on [-b, b] with b = gain * sqrt(3 / fan_in).
         * @param engine The engine whose Philox stream supplies the numbers.
         * @param fan_in Inputs feeding one output unit; must be positive.
         * @return Success, InvalidDescription for a fan-in that is not positive, or the seam's
         *         failure of the recording.
         */
        [[nodiscard]] Status kaiming_uniform(SushiBLAS::Engine& engine, SushiBLAS::Tensor& tensor,
                                             int64_t fan_in, double gain);

        /**
         * @brief Records a fill of @p tensor from N(0, (gain * sqrt(2 / (fan_in + fan_out)))^2).
         * @param fan_in Inputs feeding one output unit.
         * @param fan_out Outputs one input unit feeds; the two fans must sum to a positive value.
         * @return Success, InvalidDescription when the fans do not sum to a positive value, or
         *         the seam's failure of the recording.
         */
        [[nodiscard]] Status xavier_normal(SushiBLAS::Engine& engine, SushiBLAS::Tensor& tensor,
                                           int64_t fan_in, int64_t fan_out, double gain);

        /**
         * @brief Records a uniform fill of @p tensor on [-a, a].
         * @param fan_in Inputs feeding one output unit.
         * @param fan_out Outputs one input unit feeds; the two fans must sum to a positive value.
         * @param gain Scales the bound: a = gain * sqrt(6 / (fan_in + fan_out)).
         * @return Success, InvalidDescription when the fans do not sum to a positive value, or
         *         the seam's failure of the recording.
         */
        [[nodiscard]] Status xavier_uniform(SushiBLAS::Engine& engine, SushiBLAS::Tensor& tensor,
                                            int64_t fan_in, int64_t fan_out, double gain);

        /**
         * @brief Records a fill of @p tensor with zeros.
         * @return Success, or the seam's failure of the recording.
         */
        [[nodiscard]] Status zeros(SushiBLAS::Engine& engine, SushiBLAS::Tensor& tensor);

        /**
         * @enum InitFamily
         * @brief Which variance rule to apply to weights.
         */
        enum class InitFamily : std::uint8_t
        {
            Kaiming, /**< Scale by fan-in; the right default for a rectifier stack. */
            Xavier,  /**< Scale by both fans; the right default for tanh or sigmoid. */
        };

        /**
         * @struct InitPolicy
         * @brief How a whole model is initialised.
         */
        struct InitPolicy
        {
            /** @brief Which variance rule the weights follow. */
            InitFamily family = InitFamily::Kaiming;

            /** @brief The activation the hidden layers feed. */
            Nonlinearity nonlinearity = Nonlinearity::ReLU;

            /** @brief Draw uniformly rather than normally. */
            bool uniform = false;

            /** @brief The leaky-ReLU slope, when that is the nonlinearity. */
            double negative_slope = 0.01;

            /** @brief Zero the last fan-scaled entry. */
            bool zero_final_layer = true;

            /** @brief The Philox seed; the same seed gives the same weights. */
            std::uint64_t seed = 0x51757368'69414921ull;
        };

        namespace Detail
        {
            /**
             * @brief Records the initial fill of one tensor.
             * @param init The fill; a Constant or a Normal one is tested before the zeroing.
             * @param fan_in Inputs feeding one output unit, as the parameter declared them.
             * @param fan_out Outputs one input unit feeds, as the parameter declared them.
             * @param is_final_layer True if the tensor belongs to the final classifier layer.
             * @return Success, or the failure of the fill it records.
             */
            [[nodiscard]] Status apply_initializer(SushiBLAS::Engine& engine,
                                                   SushiBLAS::Tensor& tensor, ParameterInit init,
                                                   int64_t fan_in, int64_t fan_out,
                                                   const InitPolicy& policy,
                                                   bool is_final_layer);
        }

        /**
         * @brief Records the initial fill of every entry of @p state and runs the recording.
         * @param state The state whose entries were listed by ModelState::create().
         * @param policy How the fills are drawn and which entry is zeroed.
         * @return Success, or the failure of a fill or of the run.
         * @pre The engine's graph holds no other recorded task; the run executes all of it.
         */
        [[nodiscard]] Status initialize(SushiBLAS::Engine& engine, ModelState& state,
                                        const InitPolicy& policy = {});

    } // namespace NN
} // namespace SushiAI
