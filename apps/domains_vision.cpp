/**************************************************************************/
/* domains_vision.cpp                                                     */
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
 * @file domains_vision.cpp
 * @brief Adds the vision blocks and the MNIST dataset to the app registries.
 * @author Mustafa Garip
 */

#include "domains.hpp"

#include <SushiAI/core/status.hpp>
#include <SushiAI/vision/registration.hpp>

namespace SushiAI
{
    namespace Apps
    {
        Status register_domains(NN::NodeRegistry& nodes, Data::DatasetRegistry& datasets)
        {
            SA_TRY(Vision::register_nodes(nodes));
            SA_TRY(Vision::register_datasets(datasets));

            return Status();
        }

    } // namespace Apps
} // namespace SushiAI
