/**************************************************************************/
/* registries.hpp                                                         */
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
 * @file registries.hpp
 * @brief Declares the registries an app looks its types up in and the function that fills them.
 * @author Mustafa Garip
 */

#pragma once

#include <SushiAI/core/result.hpp>
#include <SushiAI/data/dataset_registry.hpp>
#include <SushiAI/nn/node_registry.hpp>

namespace SushiAI
{
    namespace Apps
    {
        /**
         * @struct Registries
         * @brief Holds the rows a configuration file may name in an app.
         */
        struct Registries
        {
            /** @brief The node and layer types a model file may name. */
            NN::NodeRegistry nodes;

            /** @brief The dataset types a data block may name. */
            Data::DatasetRegistry datasets;
        };

        /**
         * @brief Builds the registries of an app: the layers, the blocks, "rings" and "mnist".
         * @return The filled registries, or the first failure a register function returned.
         */
        [[nodiscard]] Result<Registries> compose_registries();

    } // namespace Apps
} // namespace SushiAI
