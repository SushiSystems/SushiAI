/**************************************************************************/
/* domains.hpp                                                            */
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
 * @file domains.hpp
 * @brief Declares the function that adds the rows of the linked domain to the app registries.
 * @author Mustafa Garip
 */

#pragma once

#include <SushiAI/core/status.hpp>
#include <SushiAI/data/dataset_registry.hpp>
#include <SushiAI/nn/node_registry.hpp>

namespace SushiAI
{
    namespace Apps
    {
        /**
         * @brief Adds the node and dataset rows of the domain this build links.
         * @param nodes The node registry the domain's node rows are added to.
         * @param datasets The dataset registry the domain's dataset rows are added to.
         * @return Ok, or the first failure a register function returned.
         */
        [[nodiscard]] Status register_domains(NN::NodeRegistry& nodes,
                                              Data::DatasetRegistry& datasets);

    } // namespace Apps
} // namespace SushiAI
