/**************************************************************************/
/* darknet_recipe.hpp                                                     */
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
 * @file darknet_recipe.hpp
 * @brief Declares the training settings of a .cfg in the shape of a training file.
 * @author Mustafa Garip
 * @see docs/design/MODEL_IMPORT/README.md
 */

#pragma once

#include <string>
#include <vector>
#include <SushiAI/config/value.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/vision/importers/darknet_config.hpp>
#include <SushiAI/vision/importers/darknet_training.hpp>

namespace SushiAI
{
    namespace Vision::Importers
    {
        /** @brief Holds the training settings of a .cfg as a training file states them. */
        struct DarknetRecipe
        {
            /** @brief Holds an object with the keys "objective", "optimizer" and "schedule". */
            Config::Value training;

            /** @brief Lists each key of the .cfg that no key of a training file holds. */
            std::vector<std::string> notes;
        };

        /**
         * @brief Writes the settings of @p training as the sections of a training file.
         * @param config The .cfg @p training was read from; its lines are quoted.
         * @return The "anchor" objective of the heads, "sgd" with the solver's momentum and
         *         decay, and the "step" schedule of its rate, burn-in, steps and scales.
         *         InvalidDescription naming two heads and the key they differ in, or for what
         *         the writer of a section refuses.
         */
        [[nodiscard]] Result<DarknetRecipe> write_darknet_recipe(const DarknetConfig& config,
                                                                 const DarknetTraining& training) noexcept;

    } // namespace Vision::Importers
} // namespace SushiAI
