/**************************************************************************/
/* optimizer_registry.hpp                                                 */
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
 * @file optimizer_registry.hpp
 * @brief Declares registry entries and factory functions for optimizer instantiation.
 * @author Mustafa Garip
 */

#pragma once

#include <array>
#include <memory>
#include <string_view>
#include <vector>
#include <SushiBLAS/engine/engine.hpp>
#include <SushiAI/config/value.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/optim/adamw.hpp>
#include <SushiAI/optim/optimizer.hpp>
#include <SushiAI/optim/sgd.hpp>

namespace SushiAI
{
    namespace Optim
    {
        /**
         * @brief Builds one optimizer over @p parameters from its configuration.
         * @param engine The engine its state buffers are allocated from.
         * @param parameters The parameter/gradient pairs to update; moved in.
         * @param attributes The optimizer's own keys; the factory rejects any it does not know.
         * @param context What to call this optimizer in a diagnostic.
         * @return The optimizer, or the failure that refused its configuration.
         */
        using OptimizerFactory = Result<std::unique_ptr<Optimizer>> (*)(
            SushiBLAS::Engine& engine,
            std::vector<ParameterSlot> parameters,
            const Config::Value& attributes,
            std::string_view context);

        /**
         * @struct OptimizerEntry
         * @brief One row of @ref K_OPTIMIZER_REGISTRY.
         */
        struct OptimizerEntry
        {
            /** @brief The "type" a configuration file spells. */
            std::string_view type;

            /** @brief What builds it. */
            OptimizerFactory make;
        };

        /**
         * @brief Builds an AdamW optimizer from configuration attributes.
         * @param engine Engine used to allocate optimizer moment buffers.
         * @param parameters Parameter and gradient slots moved into optimizer.
         * @param attributes Optional hyperparameters (learning_rate, beta1, beta2, etc.).
         * @param context Descriptive string for diagnostic error messages.
         * @return Constructed AdamW optimizer instance, or a failure when an option is refused.
         */
        [[nodiscard]] Result<std::unique_ptr<Optimizer>> make_adamw(
            SushiBLAS::Engine& engine,
            std::vector<ParameterSlot> parameters,
            const Config::Value& attributes,
            std::string_view context);

        /**
         * @brief Builds an SGD optimizer from configuration attributes.
         * @param engine Engine used to allocate optimizer momentum buffers.
         * @param parameters Parameter and gradient slots moved into optimizer.
         * @param attributes Optional hyperparameters (learning_rate, momentum, etc.).
         * @param context Descriptive string for diagnostic error messages.
         * @return Constructed SGD optimizer instance, or a failure when an option is refused.
         */
        [[nodiscard]] Result<std::unique_ptr<Optimizer>> make_sgd(
            SushiBLAS::Engine& engine,
            std::vector<ParameterSlot> parameters,
            const Config::Value& attributes,
            std::string_view context);

        /** @brief Every optimizer a configuration file may name. */
        inline constexpr std::array<OptimizerEntry, 2> K_OPTIMIZER_REGISTRY = {
            OptimizerEntry{"adamw", &make_adamw},
            OptimizerEntry{"sgd",   &make_sgd},
        };

        /**
         * @brief Builds the optimizer corresponding to the specified type key.
         * @param engine Engine used to allocate optimizer state.
         * @param type Registry key identifying optimizer type.
         * @param parameters Parameter and gradient slots moved into optimizer.
         * @param attributes Optimizer configuration attributes.
         * @return Constructed optimizer instance, or a failure when its configuration is refused.
         */
        [[nodiscard]] Result<std::unique_ptr<Optimizer>> make_optimizer(
            SushiBLAS::Engine& engine,
            std::string_view type,
            std::vector<ParameterSlot> parameters,
            const Config::Value& attributes,
            std::string_view context);

    } // namespace Optim
} // namespace SushiAI
