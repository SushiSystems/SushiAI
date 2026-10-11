/**************************************************************************/
/* registries.cpp                                                         */
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
 * @file registries.cpp
 * @brief Implements compose_registries over the register functions of the library.
 * @author Mustafa Garip
 */

#include "registries.hpp"

#include <SushiAI/core/result.hpp>
#include <SushiAI/core/status.hpp>
#include <SushiAI/data/dataset.hpp>
#include <SushiAI/data/dataset_registry.hpp>
#include <SushiAI/nn/layer_registry.hpp>
#include <SushiAI/nn/node_registry.hpp>
#include <SushiAI/train/objective_registry.hpp>

#include "domains.hpp"
#include "trainer_run.hpp"

namespace SushiAI
{
    namespace Apps
    {
        Result<Registries> compose_registries()
        {
            Registries registries;

            SA_TRY(NN::register_layers(registries.nodes));
            SA_TRY(Data::register_datasets(registries.datasets));
            SA_TRY(Train::register_objectives(registries.objectives));
            SA_TRY(register_runs(registries.runs));
            SA_TRY(register_vision_domain(registries));
            SA_TRY(register_language_domain(registries));

            return registries;
        }

    } // namespace Apps
} // namespace SushiAI
