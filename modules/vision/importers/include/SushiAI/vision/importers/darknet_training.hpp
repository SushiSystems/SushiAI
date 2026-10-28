/**************************************************************************/
/* darknet_training.hpp                                                   */
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
 * @file darknet_training.hpp
 * @brief Declares the solver settings and the per-head training options a .cfg holds.
 * @author Mustafa Garip
 */

#pragma once

#include <cstdint>
#include <vector>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/vision/detect/anchor/head_training_options.hpp>
#include <SushiAI/vision/importers/darknet_config.hpp>

namespace SushiAI
{
    namespace Vision::Importers
    {
        /** @brief Holds the solver settings of a .cfg's [net] section. */
        struct DarknetSolver
        {
            /** @brief Holds learning_rate; 0.001 when the file does not spell it. */
            double learning_rate = 0.001;

            /** @brief Holds momentum; 0.9 when the file does not spell it. */
            double momentum = 0.9;

            /** @brief Holds decay; 0.0005 when the file does not spell it. */
            double decay = 0.0005;

            /** @brief Holds burn_in, the iterations of the warm-up; 0 when absent. */
            std::int64_t burn_in = 0;

            /** @brief Holds steps, the iterations at which the rate is scaled, in file order. */
            std::vector<std::int64_t> steps;

            /** @brief Holds scales, the factor each entry of steps applies, in file order. */
            std::vector<double> scales;

            /** @brief Holds batch, the images of one iteration; 1 when absent, at least 1. */
            std::int64_t batch = 1;

            /** @brief Holds subdivisions, the parts a batch is split into; 1 when absent. */
            std::int64_t subdivisions = 1;

            /** @brief Holds max_batches, the iteration at which training stops; 0 when absent. */
            std::int64_t max_batches = 0;
        };

        /** @brief Holds the solver settings and the training options of every [yolo] section. */
        struct DarknetTraining
        {
            /** @brief Holds the settings read from the [net] section. */
            DarknetSolver solver;

            /// @brief Holds one entry per [yolo] section, in file order; each has the box
            ///        gradient BoxGradient::DarknetCompatible.
            std::vector<Detect::Anchor::HeadTrainingOptions> heads;
        };

        /**
         * @brief Reads the solver settings and the training options of the heads of a .cfg.
         * @return The settings, an absent key keeping its default. InvalidDescription names the
         *         line of a malformed value, a batch or subdivisions below 1, steps and scales
         *         of different lengths, or a head validate_head_training refuses. Unsupported
         *         names the line of a policy other than steps, a truth_thresh other than 1, a
         *         label_smooth_eps other than 0, or an iou_loss other than mse and ciou.
         */
        [[nodiscard]] Result<DarknetTraining> read_darknet_training(const DarknetConfig& config) noexcept;

    } // namespace Vision::Importers
} // namespace SushiAI
