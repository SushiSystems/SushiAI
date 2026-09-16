/**************************************************************************/
/* optimizer_registry.cpp                                                 */
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
 * @file optimizer_registry.cpp
 * @brief Implements registry entries and factory functions for optimizer instantiation.
 * @author Mustafa Garip
 */

#include <SushiAI/optim/optimizer_registry.hpp>

#include <string>
#include <utility>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/config/registry.hpp>

namespace SushiAI
{
    namespace Optim
    {
        namespace
        {
            /** @brief Every key an "adamw" optimizer accepts. */
            constexpr std::string_view ADAMW_KEYS[] = {
                "learning_rate", "beta1", "beta2", "eps", "weight_decay",
            };

            /** @brief Every key an "sgd" optimizer accepts. */
            constexpr std::string_view SGD_KEYS[] = {
                "learning_rate", "momentum", "dampening", "weight_decay", "nesterov",
            };

            /** @brief Wraps a literal key list as a span, for reject_unknown_keys. */
            template<std::size_t N>
            SushiRuntime::span<const std::string_view> as_span(const std::string_view (&keys)[N])
            {
                return SushiRuntime::span<const std::string_view>(keys, N);
            }

            /**
             * @brief Reads an optional real from an optimizer's attributes.
             * @param attributes The optimizer's keys.
             * @param key Which one.
             * @param prefix What to call this optimizer in a diagnostic.
             * @param fallback Returned when @p key is absent.
             * @return The value, or @p fallback.
             */
            double read_real(const Config::Value& attributes, std::string_view key,
                             const std::string& prefix, double fallback)
            {
                const Config::Value spelled = Config::Value::real(fallback);
                return attributes.at_or(key, spelled)
                                 .as_real(prefix + " '" + std::string(key) + "'");
            }

            /**
             * @brief Reads an optional boolean from an optimizer's attributes.
             * @param attributes The optimizer's keys.
             * @param key Which one.
             * @param prefix What to call this optimizer in a diagnostic.
             * @param fallback Returned when @p key is absent.
             * @return The value, or @p fallback.
             */
            bool read_bool(const Config::Value& attributes, std::string_view key,
                           const std::string& prefix, bool fallback)
            {
                const Config::Value spelled = Config::Value::boolean(fallback);
                return attributes.at_or(key, spelled)
                                 .as_bool(prefix + " '" + std::string(key) + "'");
            }
        }

        std::unique_ptr<Optimizer> make_adamw(SushiBLAS::Engine& engine,
                                              std::vector<ParameterSlot> parameters,
                                              const Config::Value& attributes,
                                              std::string_view context)
        {
            attributes.reject_unknown_keys(context, as_span(ADAMW_KEYS));

            const std::string prefix(context);
            AdamWOptions options;
            options.learning_rate = read_real(attributes, "learning_rate", prefix,
                                              options.learning_rate);
            options.beta1 = read_real(attributes, "beta1", prefix, options.beta1);
            options.beta2 = read_real(attributes, "beta2", prefix, options.beta2);
            options.eps = read_real(attributes, "eps", prefix, options.eps);
            options.weight_decay = read_real(attributes, "weight_decay", prefix,
                                             options.weight_decay);

            SA_THROW_IF(options.learning_rate <= 0.0,
                        "{} 'learning_rate' must be positive, but it is {}",
                        prefix, options.learning_rate);

            return std::make_unique<AdamW>(engine, std::move(parameters), options);
        }

        std::unique_ptr<Optimizer> make_sgd(SushiBLAS::Engine& engine,
                                            std::vector<ParameterSlot> parameters,
                                            const Config::Value& attributes,
                                            std::string_view context)
        {
            attributes.reject_unknown_keys(context, as_span(SGD_KEYS));

            const std::string prefix(context);
            SGDOptions options;
            options.learning_rate = read_real(attributes, "learning_rate", prefix,
                                              options.learning_rate);
            options.momentum = read_real(attributes, "momentum", prefix, options.momentum);
            options.dampening = read_real(attributes, "dampening", prefix, options.dampening);
            options.weight_decay = read_real(attributes, "weight_decay", prefix,
                                             options.weight_decay);
            options.nesterov = read_bool(attributes, "nesterov", prefix, options.nesterov);

            SA_THROW_IF(options.learning_rate <= 0.0,
                        "{} 'learning_rate' must be positive, but it is {}",
                        prefix, options.learning_rate);

            return std::make_unique<SGD>(engine, std::move(parameters), options);
        }

        std::unique_ptr<Optimizer> make_optimizer(SushiBLAS::Engine& engine,
                                                  std::string_view type,
                                                  std::vector<ParameterSlot> parameters,
                                                  const Config::Value& attributes,
                                                  std::string_view context)
        {
            return Config::find_entry(OPTIMIZER_REGISTRY, type, "optimizer type")
                       .make(engine, std::move(parameters), attributes, context);
        }

    } // namespace Optim
} // namespace SushiAI
