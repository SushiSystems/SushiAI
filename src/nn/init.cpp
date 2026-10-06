/**************************************************************************/
/* init.cpp                                                               */
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
 * @file init.cpp
 * @brief Implements neural network parameter initializers and gain calculations.
 * @author Mustafa Garip
 */

#include <cmath>
#include <SushiAI/nn/init.hpp>

namespace SushiAI
{
    namespace NN
    {
        double calculate_gain(Nonlinearity nonlinearity, double negative_slope) noexcept
        {
            switch (nonlinearity)
            {
                case Nonlinearity::LINEAR:     return 1.0;
                case Nonlinearity::SIGMOID:    return 1.0;
                case Nonlinearity::TANH:       return 5.0 / 3.0;
                case Nonlinearity::RELU:       return std::sqrt(2.0);
                case Nonlinearity::LEAKY_RELU: return std::sqrt(2.0 / (1.0 + negative_slope * negative_slope));
            }
            return 1.0;
        }

        void kaiming_normal(SushiBLAS::Engine& engine, SushiBLAS::Tensor& tensor,
                            int64_t fan_in, double gain)
        {
            SA_THROW_IF(fan_in <= 0, "Kaiming initialisation needs a positive fan-in; got {}", fan_in);
            engine.random().normal(tensor, 0.0, gain / std::sqrt(static_cast<double>(fan_in)));
        }

        void kaiming_uniform(SushiBLAS::Engine& engine, SushiBLAS::Tensor& tensor,
                             int64_t fan_in, double gain)
        {
            SA_THROW_IF(fan_in <= 0, "Kaiming initialisation needs a positive fan-in; got {}", fan_in);

            // Bound matches normal standard deviation: b = gain * sqrt(3 / fan_in).
            const double bound = gain * std::sqrt(3.0 / static_cast<double>(fan_in));
            engine.random().uniform(tensor, -bound, bound);
        }

        void xavier_normal(SushiBLAS::Engine& engine, SushiBLAS::Tensor& tensor,
                           int64_t fan_in, int64_t fan_out, double gain)
        {
            SA_THROW_IF(fan_in + fan_out <= 0,
                        "Xavier initialisation needs positive fans; got {} and {}", fan_in, fan_out);

            const double deviation =
                gain * std::sqrt(2.0 / static_cast<double>(fan_in + fan_out));
            engine.random().normal(tensor, 0.0, deviation);
        }

        void xavier_uniform(SushiBLAS::Engine& engine, SushiBLAS::Tensor& tensor,
                            int64_t fan_in, int64_t fan_out, double gain)
        {
            SA_THROW_IF(fan_in + fan_out <= 0,
                        "Xavier initialisation needs positive fans; got {} and {}", fan_in, fan_out);

            const double bound = gain * std::sqrt(6.0 / static_cast<double>(fan_in + fan_out));
            engine.random().uniform(tensor, -bound, bound);
        }

        void zeros(SushiBLAS::Engine& engine, SushiBLAS::Tensor& tensor)
        {
            engine.random().constant(tensor, 0.0);
        }

        namespace Detail
        {
            void apply_initializer(SushiBLAS::Engine& engine,
                                   SushiBLAS::Tensor& tensor,
                                   const Parameter& parameter,
                                   const InitPolicy& policy,
                                   bool is_final_layer)
            {
                // Invariant: Biases and final classifier layers initialize to zero.
                if (is_final_layer || parameter.role() == ParameterRole::BIAS)
                {
                    zeros(engine, tensor);
                    return;
                }

                const double gain = calculate_gain(policy.nonlinearity, policy.negative_slope);
                const int64_t fan_in = parameter.fan_in();
                const int64_t fan_out = parameter.fan_out();

                if (policy.family == InitFamily::KAIMING)
                {
                    if (policy.uniform) kaiming_uniform(engine, tensor, fan_in, gain);
                    else                kaiming_normal(engine, tensor, fan_in, gain);
                }
                else
                {
                    if (policy.uniform) xavier_uniform(engine, tensor, fan_in, fan_out, gain);
                    else                xavier_normal(engine, tensor, fan_in, fan_out, gain);
                }
            }
        }

    } // namespace NN
} // namespace SushiAI
