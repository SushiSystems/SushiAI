/**************************************************************************/
/* optimizer_registry.cpp                                                 */
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
 * @file optimizer_registry.cpp
 * @brief Implements registry entries and factory functions for optimizer instantiation.
 * @author Mustafa Garip
 */

#include <SushiAI/optim/optimizer_registry.hpp>

#include <cstddef>
#include <string>
#include <utility>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/config/registry.hpp>
#include <SushiAI/core/status.hpp>

namespace SushiAI
{
    namespace Optim
    {
        namespace
        {
            /** @brief Names the key that decays every slot, whatever its role. */
            constexpr std::string_view K_DECAY_ALL_KEY = "decay_all";

            /** @brief Names the key that clips the gradients to a global norm; absent, none. */
            constexpr std::string_view K_CLIP_NORM_KEY = "clip_norm";

            /** @brief Every key an "adamw" optimizer accepts. */
            constexpr std::string_view K_ADAMW_KEYS[] = {
                "learning_rate", "beta1", "beta2", "eps", "weight_decay",
                K_DECAY_ALL_KEY, K_CLIP_NORM_KEY,
            };

            /** @brief Every key an "sgd" optimizer accepts. */
            constexpr std::string_view K_SGD_KEYS[] = {
                "learning_rate", "momentum", "dampening", "weight_decay", "nesterov",
                K_DECAY_ALL_KEY, K_CLIP_NORM_KEY,
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

            /**
             * @brief Enables the clip of @p optimizer when @p attributes state "clip_norm".
             * @param prefix What to call this optimizer in a diagnostic.
             * @return Success, also when the key is absent and nothing is enabled, or the
             *         refusal of Optimizer::enable_gradient_clip named as the key's.
             */
            Status apply_clip_norm(Optimizer& optimizer, const Config::Value& attributes,
                                   const std::string& prefix)
            {
                if (!attributes.has(K_CLIP_NORM_KEY))
                    return Status();

                const std::string where = prefix + " '" + std::string(K_CLIP_NORM_KEY) + "'";
                const double max_norm = attributes.at(K_CLIP_NORM_KEY, where).as_real(where);
                return SA_ANNOTATE(optimizer.enable_gradient_clip(max_norm), "{}", where);
            }
        }

        Result<std::unique_ptr<Optimizer>> make_adamw(SushiBLAS::Engine& engine,
                                                      std::vector<ParameterSlot> parameters,
                                                      const Config::Value& attributes,
                                                      std::string_view context)
        {
            attributes.reject_unknown_keys(context, as_span(K_ADAMW_KEYS));

            const std::string prefix(context);
            AdamWOptions options;
            options.learning_rate = read_real(attributes, "learning_rate", prefix,
                                              options.learning_rate);
            options.beta1 = read_real(attributes, "beta1", prefix, options.beta1);
            options.beta2 = read_real(attributes, "beta2", prefix, options.beta2);
            options.eps = read_real(attributes, "eps", prefix, options.eps);
            options.weight_decay = read_real(attributes, "weight_decay", prefix,
                                             options.weight_decay);
            options.decay_all = read_bool(attributes, K_DECAY_ALL_KEY, prefix,
                                          options.decay_all);

            SA_FAIL_IF(options.learning_rate <= 0.0, StatusCode::InvalidDescription,
                       "{} 'learning_rate' must be positive, but it is {}",
                       prefix, options.learning_rate);

            SA_TRY_ASSIGN(std::unique_ptr<AdamW> made,
                          AdamW::make(engine, std::move(parameters), options));
            SA_TRY(apply_clip_norm(*made, attributes, prefix));
            std::unique_ptr<Optimizer> base = std::move(made);
            return std::move(base);
        }

        Result<std::unique_ptr<Optimizer>> make_sgd(SushiBLAS::Engine& engine,
                                                    std::vector<ParameterSlot> parameters,
                                                    const Config::Value& attributes,
                                                    std::string_view context)
        {
            attributes.reject_unknown_keys(context, as_span(K_SGD_KEYS));

            const std::string prefix(context);
            SGDOptions options;
            options.learning_rate = read_real(attributes, "learning_rate", prefix,
                                              options.learning_rate);
            options.momentum = read_real(attributes, "momentum", prefix, options.momentum);
            options.dampening = read_real(attributes, "dampening", prefix, options.dampening);
            options.weight_decay = read_real(attributes, "weight_decay", prefix,
                                             options.weight_decay);
            options.nesterov = read_bool(attributes, "nesterov", prefix, options.nesterov);
            options.decay_all = read_bool(attributes, K_DECAY_ALL_KEY, prefix,
                                          options.decay_all);

            SA_FAIL_IF(options.learning_rate <= 0.0, StatusCode::InvalidDescription,
                       "{} 'learning_rate' must be positive, but it is {}",
                       prefix, options.learning_rate);

            SA_TRY_ASSIGN(std::unique_ptr<SGD> made,
                          SGD::make(engine, std::move(parameters), options));
            SA_TRY(apply_clip_norm(*made, attributes, prefix));
            std::unique_ptr<Optimizer> base = std::move(made);
            return std::move(base);
        }

        Result<std::unique_ptr<Optimizer>> make_optimizer(SushiBLAS::Engine& engine,
                                                          std::string_view type,
                                                          std::vector<ParameterSlot> parameters,
                                                          const Config::Value& attributes,
                                                          std::string_view context)
        {
            return Config::find_entry(K_OPTIMIZER_REGISTRY, type, "optimizer type")
                       .make(engine, std::move(parameters), attributes, context);
        }

    } // namespace Optim
} // namespace SushiAI
