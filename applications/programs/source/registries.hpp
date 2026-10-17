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

#include <SushiAI/failure/result.hpp>
#include <SushiAI/data/dataset_registry.hpp>
#include <SushiAI/io/importer_registry.hpp>
#include <SushiAI/nn/node_registry.hpp>
#include <SushiAI/train/objective_registry.hpp>

#include "inference_run.hpp"
#include "training_run.hpp"

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

            /** @brief The objective types an objective block may name. */
            Train::ObjectiveRegistry objectives;

            /** @brief The training runs, by the objective type each trains. */
            RunRegistry runs;

            /** @brief The reports of `sushiai inference`, by the task type each reads. */
            InferenceRegistry inferences;

            /** @brief The importers `sushiai import` may name, by format. */
            IO::ImporterRegistry importers;
        };

        /**
         * @brief Builds the registries of an app: the rows of the library, then of the domain.
         * @return The filled registries, or the first failure a register function returned.
         */
        [[nodiscard]] Result<Registries> compose_registries();

    } // namespace Apps
} // namespace SushiAI
