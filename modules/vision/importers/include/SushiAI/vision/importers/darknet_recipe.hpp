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
/* Licensed under the Apache License, Version 2.0 (the "License"); you    */
/* may not use this file except in compliance with the License. You may   */
/* obtain a copy of the License at                                        */
/*                                                                        */
/*     https://www.apache.org/licenses/LICENSE-2.0                        */
/*                                                                        */
/* Unless required by applicable law or agreed to in writing, software    */
/* distributed under the License is distributed on an "AS IS" BASIS,      */
/* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or        */
/* implied. See the License for the specific language governing           */
/* permissions and limitations under the License.                         */
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

            /**
             * @brief Holds the keys of the training file's "data" block the .cfg states: the
             *        "pipeline" "darknet" and its "jitter", "hue", "saturation" and "exposure".
             */
            Config::Value data;

            /** @brief Lists each key of the .cfg that no key of a training file holds. */
            std::vector<std::string> notes;
        };

        /**
         * @brief Writes the settings of @p training as the sections of a training file.
         * @param config The .cfg @p training was read from; its lines are quoted.
         * @return The "anchor" objective, "sgd" and the "step" schedule as training, and the
         *         "darknet" pipeline of the augmentation keys as data, in float32. A head's
         *         jitter other than the last head's is noted. InvalidDescription naming two
         *         heads and the key they differ in, or what a section writer refuses.
         */
        [[nodiscard]] Result<DarknetRecipe> write_darknet_recipe(const DarknetConfig& config,
                                                                 const DarknetTraining& training) noexcept;

    } // namespace Vision::Importers
} // namespace SushiAI
