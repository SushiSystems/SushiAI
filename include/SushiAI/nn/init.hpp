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
#include <string>
#include <SushiBLAS/engine.hpp>
#include <SushiAI/core/common.hpp>
#include <SushiAI/graph/lowering.hpp>
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
         * @brief Fills @p tensor from N(0, (gain / sqrt(fan_in))^2).
         * @param engine The engine whose Philox stream supplies the numbers.
         * @param tensor The tensor to fill.
         * @param fan_in Inputs feeding one output unit.
         * @param gain The nonlinearity's gain.
         */
        void kaiming_normal(SushiBLAS::Engine& engine, SushiBLAS::Tensor& tensor,
                            int64_t fan_in, double gain);

        /**
         * @brief Fills @p tensor uniformly on [-b, b] with b = gain * sqrt(3 / fan_in).
         * @param engine The engine whose Philox stream supplies the numbers.
         * @param tensor The tensor to fill.
         * @param fan_in Inputs feeding one output unit.
         * @param gain The nonlinearity's gain.
         */
        void kaiming_uniform(SushiBLAS::Engine& engine, SushiBLAS::Tensor& tensor,
                             int64_t fan_in, double gain);

        /**
         * @brief Fills @p tensor from N(0, (gain * sqrt(2 / (fan_in + fan_out)))^2).
         * @param engine The engine whose Philox stream supplies the numbers.
         * @param tensor The tensor to fill.
         * @param fan_in Inputs feeding one output unit.
         * @param fan_out Outputs one input unit feeds.
         * @param gain The nonlinearity's gain.
         */
        void xavier_normal(SushiBLAS::Engine& engine, SushiBLAS::Tensor& tensor,
                           int64_t fan_in, int64_t fan_out, double gain);

        /**
         * @brief Fills @p tensor uniformly on [-a, a] with a = gain * sqrt(6 / (fan_in + fan_out)).
         * @param engine The engine whose Philox stream supplies the numbers.
         * @param tensor The tensor to fill.
         * @param fan_in Inputs feeding one output unit.
         * @param fan_out Outputs one input unit feeds.
         * @param gain The nonlinearity's gain.
         */
        void xavier_uniform(SushiBLAS::Engine& engine, SushiBLAS::Tensor& tensor,
                            int64_t fan_in, int64_t fan_out, double gain);

        /**
         * @brief Fills @p tensor with zeros.
         * @param engine The engine to record the fill on.
         * @param tensor The tensor to zero.
         */
        void zeros(SushiBLAS::Engine& engine, SushiBLAS::Tensor& tensor);

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
             * @brief Applies policy initialization to one parameter tensor.
             * @param engine Engine used to execute tensor fill operations.
             * @param tensor Destination parameter tensor to fill.
             * @param parameter Parameter metadata containing shape and role.
             * @param policy Initialization policy configuration.
             * @param is_final_layer True if parameter belongs to final classifier layer.
             */
            void apply_initializer(SushiBLAS::Engine& engine,
                                   SushiBLAS::Tensor& tensor,
                                   const Parameter& parameter,
                                   const InitPolicy& policy,
                                   bool is_final_layer);
        }

        /**
         * @brief Initializes every parameter of module and flushes the device.
         * @param engine Engine managing parameter tensor memory.
         * @param module Model to initialize whose parameters were declared during trace.
         * @param pool Tensor pool binding parameter values.
         * @param policy Initialization policy configuration.
         * @throws Error If a parameter tensor is not bound in pool.
         */
        template<typename M>
        void initialize(SushiBLAS::Engine& engine, M& module, Graph::TensorPool& pool,
                        const InitPolicy& policy = {})
        {
            engine.random().set_seed(policy.seed);

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
                Detail::apply_initializer(engine, pool.tensor(parameters[i]->id()),
                                          *parameters[i], policy, is_final);
            }

            engine.execute();
            engine.get_context().wait_all();
        }

    } // namespace NN
} // namespace SushiAI
