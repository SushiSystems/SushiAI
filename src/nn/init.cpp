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

#include <SushiAI/nn/init.hpp>

#include <cmath>
#include <SushiBLAS/engine/math/random.hpp>
#include <SushiAI/core/seam.hpp>

namespace SushiAI
{
    namespace NN
    {
        double calculate_gain(Nonlinearity nonlinearity, double negative_slope) noexcept
        {
            switch (nonlinearity)
            {
                case Nonlinearity::Linear:    return 1.0;
                case Nonlinearity::Sigmoid:   return 1.0;
                case Nonlinearity::Tanh:      return 5.0 / 3.0;
                case Nonlinearity::Relu:      return std::sqrt(2.0);
                case Nonlinearity::LeakyRelu:
                    return std::sqrt(2.0 / (1.0 + negative_slope * negative_slope));
            }
            return 1.0;
        }

        Status constant(SushiBLAS::Engine& engine, SushiBLAS::Tensor& tensor, double value)
        {
            return SushiAI::Detail::guard(SushiAI::Detail::SeamPhase::Record, [&]() -> Status
            {
                SushiBLAS::RandomOps random(engine);
                random.constant(tensor, value);
                return Status();
            });
        }

        Status kaiming_normal(SushiBLAS::Engine& engine, SushiBLAS::Tensor& tensor,
                              int64_t fan_in, double gain)
        {
            SA_FAIL_IF(fan_in <= 0, StatusCode::InvalidDescription,
                       "Kaiming initialisation needs a positive fan-in; got {}", fan_in);

            const double deviation = gain / std::sqrt(static_cast<double>(fan_in));
            return SushiAI::Detail::guard(SushiAI::Detail::SeamPhase::Record, [&]() -> Status
            {
                SushiBLAS::RandomOps random(engine);
                random.normal(tensor, 0.0, deviation);
                return Status();
            });
        }

        Status kaiming_uniform(SushiBLAS::Engine& engine, SushiBLAS::Tensor& tensor,
                               int64_t fan_in, double gain)
        {
            SA_FAIL_IF(fan_in <= 0, StatusCode::InvalidDescription,
                       "Kaiming initialisation needs a positive fan-in; got {}", fan_in);

            // Bound matches normal standard deviation: b = gain * sqrt(3 / fan_in).
            const double bound = gain * std::sqrt(3.0 / static_cast<double>(fan_in));
            return SushiAI::Detail::guard(SushiAI::Detail::SeamPhase::Record, [&]() -> Status
            {
                SushiBLAS::RandomOps random(engine);
                random.uniform(tensor, -bound, bound);
                return Status();
            });
        }

        Status xavier_normal(SushiBLAS::Engine& engine, SushiBLAS::Tensor& tensor,
                             int64_t fan_in, int64_t fan_out, double gain)
        {
            SA_FAIL_IF(fan_in + fan_out <= 0, StatusCode::InvalidDescription,
                       "Xavier initialisation needs positive fans; got {} and {}", fan_in, fan_out);

            const double deviation =
                gain * std::sqrt(2.0 / static_cast<double>(fan_in + fan_out));
            return SushiAI::Detail::guard(SushiAI::Detail::SeamPhase::Record, [&]() -> Status
            {
                SushiBLAS::RandomOps random(engine);
                random.normal(tensor, 0.0, deviation);
                return Status();
            });
        }

        Status xavier_uniform(SushiBLAS::Engine& engine, SushiBLAS::Tensor& tensor,
                              int64_t fan_in, int64_t fan_out, double gain)
        {
            SA_FAIL_IF(fan_in + fan_out <= 0, StatusCode::InvalidDescription,
                       "Xavier initialisation needs positive fans; got {} and {}", fan_in, fan_out);

            const double bound = gain * std::sqrt(6.0 / static_cast<double>(fan_in + fan_out));
            return SushiAI::Detail::guard(SushiAI::Detail::SeamPhase::Record, [&]() -> Status
            {
                SushiBLAS::RandomOps random(engine);
                random.uniform(tensor, -bound, bound);
                return Status();
            });
        }

        Status zeros(SushiBLAS::Engine& engine, SushiBLAS::Tensor& tensor)
        {
            return constant(engine, tensor, 0.0);
        }

        namespace Detail
        {
            Status apply_initializer(SushiBLAS::Engine& engine,
                                     SushiBLAS::Tensor& tensor,
                                     const Parameter& parameter,
                                     const InitPolicy& policy,
                                     bool is_final_layer)
            {
                // A constant fill named by init() is tested before the final-layer zeroing.
                const ParameterInit init = parameter.init();
                if (init.kind == ParameterInit::Kind::Constant)
                    return constant(engine, tensor, init.constant);

                if (is_final_layer)
                    return zeros(engine, tensor);

                const double gain = calculate_gain(policy.nonlinearity, policy.negative_slope);
                const int64_t fan_in = parameter.fan_in();
                const int64_t fan_out = parameter.fan_out();

                if (policy.family == InitFamily::Kaiming)
                {
                    if (policy.uniform) return kaiming_uniform(engine, tensor, fan_in, gain);
                    return kaiming_normal(engine, tensor, fan_in, gain);
                }

                if (policy.uniform) return xavier_uniform(engine, tensor, fan_in, fan_out, gain);
                return xavier_normal(engine, tensor, fan_in, fan_out, gain);
            }
        }

    } // namespace NN
} // namespace SushiAI
