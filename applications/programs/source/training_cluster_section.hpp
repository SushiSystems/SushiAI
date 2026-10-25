/**************************************************************************/
/* training_cluster_section.hpp                                           */
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
 * @file training_cluster_section.hpp
 * @brief Declares read_training_cluster, the cluster section read with "deterministic".
 * @author Mustafa Garip
 * @see modules/core/distribution/README.md
 */

#pragma once

#include <SushiAI/config/train_config.hpp>
#include <SushiAI/distribution/cluster_section.hpp>
#include <SushiAI/failure/result.hpp>

namespace SushiAI
{
    namespace Apps
    {
        /**
         * @brief Reads the cluster section of @p spec, refusing "if_absent": "local" while
         *        "deterministic" is true.
         *
         * A local fallback runs on the master's device, whose bits differ. ClusterSection
         * does not know "deterministic", so this function checks the pair.
         * @return The section, or InvalidDescription naming both keys and the replica.
         */
        [[nodiscard]] Result<Distribution::ClusterSection> read_training_cluster(
            const Config::TrainSpec& spec);

    } // namespace Apps
} // namespace SushiAI
