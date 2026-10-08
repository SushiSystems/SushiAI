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

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>
#include <SushiBLAS/engine.hpp>
#include <SushiBLAS/engine/math/random.hpp>
#include <SushiAI/core/common.hpp>
#include <SushiAI/core/seam.hpp>
#include <SushiAI/core/status.hpp>
#include <SushiAI/graph/tensor_pool.hpp>
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
            Relu,       /**< Rectified linear unit; also GELU and SiLU in practice. */
            LeakyRelu, /**< Leaky rectified linear unit. */
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
            Nonlinearity nonlinearity = Nonlinearity::Relu;

            /** @brief Draw uniformly rather than normally. */
            bool uniform = false;

            /** @brief The leaky-ReLU slope, when that is the nonlinearity. */
            double negative_slope = 0.01;

            /** @brief Zero the last weight and bias, so the initial loss is exactly ln(C). */
            bool zero_final_layer = true;

            /** @brief The Philox seed; the same seed gives the same weights. */
            std::uint64_t seed = 0x51757368'69414921ull;
        };

        namespace Detail
        {
            /**
             * @brief Records the initial fill of one parameter tensor.
             * @param parameter Parameter whose init() and fans select the fill.
             * @param is_final_layer True if parameter belongs to final classifier layer.
             * @return Success, or the failure of the fill it records.
             */
            [[nodiscard]] Status apply_initializer(SushiBLAS::Engine& engine,
                                                   SushiBLAS::Tensor& tensor,
                                                   const Parameter& parameter,
                                                   const InitPolicy& policy,
                                                   bool is_final_layer);
        }

        /**
         * @brief Records the initial fill of every parameter of module and runs the recording.
         * @param module Model whose parameters were declared during trace.
         * @return Success, or the failure of a fill or of the run.
         * @pre Every parameter of module is bound in pool, and the engine's graph holds no
         *      other recorded task; the run executes all of it.
         */
        template<typename M>
        [[nodiscard]] Status initialize(SushiBLAS::Engine& engine, M& module,
                                        Graph::TensorPool& pool,
                                        const InitPolicy& policy = {})
        {
            SA_TRY(SushiAI::Detail::guard(SushiAI::Detail::SeamPhase::Record, [&]() -> Status
            {
                SushiBLAS::RandomOps random(engine);
                random.set_seed(policy.seed);
                return Status();
            }));

            std::vector<Parameter*> parameters;
            for_each_parameter(module, [&parameters](const std::string&, Parameter& parameter)
            {
                parameters.push_back(&parameter);
            });

            std::size_t final_weight = parameters.size();
            for (std::size_t i = 0; i < parameters.size(); ++i)
                if (parameters[i]->role() == ParameterRole::Weight) final_weight = i;

            for (std::size_t i = 0; i < parameters.size(); ++i)
            {
                const bool is_final = policy.zero_final_layer && i >= final_weight;
                SA_TRY(Detail::apply_initializer(engine, pool.tensor(parameters[i]->id()),
                                                 *parameters[i], policy, is_final));
            }

            return SushiAI::Detail::run_recorded(engine);
        }

    } // namespace NN
} // namespace SushiAI
