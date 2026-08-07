/**************************************************************************/
/* optimizer_registry.hpp                                                 */
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

#pragma once

/**
 * @file optimizer_registry.hpp
 * @brief Which optimizers a configuration file may name, and what builds each.
 *
 * A file of its own rather than a section of optimizer.hpp, because the table
 * has to name the concrete rules and both @c adamw.hpp and @c sgd.hpp already
 * include @c optimizer.hpp - putting the table there would close the cycle.
 * This header sits above all three and nothing in the library includes it.
 *
 * @c UpdateKernel is deliberately **not** a configurable key. It selects the
 * reference spelling a byte-for-byte regression test compares the fused kernel
 * against, not a hyperparameter, and @c SEPARATE additionally cannot join the
 * compiled plan - so a configuration file that reached it could silently
 * change what a run is, or be refused for a reason its author never asked
 * about.
 *
 * Nothing self-registers; see config/registry.hpp for why that is forced
 * rather than preferred in a static library.
 */

#include <array>
#include <memory>
#include <string_view>
#include <vector>
#include <SushiBLAS/engine.hpp>
#include <SushiAI/config/value.hpp>
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
         * @return The optimizer.
         */
        using OptimizerFactory = std::unique_ptr<Optimizer> (*)(
            SushiBLAS::Engine& engine,
            std::vector<ParameterSlot> parameters,
            const Config::Value& attributes,
            std::string_view context);

        /**
         * @struct OptimizerEntry
         * @brief One row of @ref OPTIMIZER_REGISTRY.
         */
        struct OptimizerEntry
        {
            /** @brief The "type" a configuration file spells. */
            std::string_view type;

            /** @brief What builds it. */
            OptimizerFactory make;
        };

        /**
         * @brief Builds an @ref AdamW from its configuration attributes.
         * @param engine The engine its moments are allocated from.
         * @param parameters The parameter/gradient pairs; moved in.
         * @param attributes learning_rate, beta1, beta2, eps and weight_decay, all optional.
         * @param context What to call this optimizer in a diagnostic.
         * @return The optimizer.
         * @throws Error If a key is unknown or a value is out of range.
         */
        std::unique_ptr<Optimizer> make_adamw(SushiBLAS::Engine& engine,
                                              std::vector<ParameterSlot> parameters,
                                              const Config::Value& attributes,
                                              std::string_view context);

        /**
         * @brief Builds an @ref SGD from its configuration attributes.
         * @param engine The engine its momentum buffers are allocated from.
         * @param parameters The parameter/gradient pairs; moved in.
         * @param attributes learning_rate, momentum, dampening, weight_decay and
         *        nesterov, all optional.
         * @param context What to call this optimizer in a diagnostic.
         * @return The optimizer.
         * @throws Error If a key is unknown or a value is out of range.
         */
        std::unique_ptr<Optimizer> make_sgd(SushiBLAS::Engine& engine,
                                            std::vector<ParameterSlot> parameters,
                                            const Config::Value& attributes,
                                            std::string_view context);

        /** @brief Every optimizer a configuration file may name. */
        inline constexpr std::array<OptimizerEntry, 2> OPTIMIZER_REGISTRY = {
            OptimizerEntry{"adamw", &make_adamw},
            OptimizerEntry{"sgd",   &make_sgd},
        };

        /**
         * @brief Builds the optimizer named @p type.
         * @param engine The engine its state is allocated from.
         * @param type The registry key.
         * @param parameters The parameter/gradient pairs; moved in.
         * @param attributes The optimizer's own keys.
         * @param context What to call it in a diagnostic.
         * @return The optimizer.
         * @throws Error If @p type is unknown; the message lists every type
         *         this build has.
         */
        std::unique_ptr<Optimizer> make_optimizer(SushiBLAS::Engine& engine,
                                                  std::string_view type,
                                                  std::vector<ParameterSlot> parameters,
                                                  const Config::Value& attributes,
                                                  std::string_view context);

    } // namespace Optim
} // namespace SushiAI
