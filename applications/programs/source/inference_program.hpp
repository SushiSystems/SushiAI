/**************************************************************************/
/* inference_program.hpp                                                  */
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
 * @file inference_program.hpp
 * @brief Declares the assembly `sushiai inference` runs once it holds the parsed files.
 * @author Mustafa Garip
 * @see applications/programs/README.md, "inference_program.hpp".
 */

#pragma once

#include <filesystem>
#include <SushiBLAS/engine/engine.hpp>
#include <SushiAI/config/model_config.hpp>
#include <SushiAI/config/train_config.hpp>
#include <SushiAI/failure/status.hpp>

namespace SushiAI
{
    namespace Apps
    {
        /**
         * @brief Evaluates a checkpoint over a dataset and prints its report.
         * @param model_path The model file, as the command line names it.
         * @param model_spec The parsed model file.
         * @param data_spec The parsed evaluation file.
         * @return Success, or the first failure; InvalidDescription when the data and the model
         *         disagree or the step compiled more than once.
         */
        [[nodiscard]] Status run_inference_file(const std::filesystem::path& model_path,
                                                const std::filesystem::path& checkpoint_path,
                                                const Config::ModelSpec& model_spec,
                                                const Config::DataSpec& data_spec,
                                                SushiBLAS::Engine& engine);
    } // namespace Apps
} // namespace SushiAI
