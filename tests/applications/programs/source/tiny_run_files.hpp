/**************************************************************************/
/* tiny_run_files.hpp                                                     */
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
 * @file tiny_run_files.hpp
 * @brief Declares the training file the split-step tests share over the tiny run's model.
 * @author Mustafa Garip
 */

#pragma once

#include <filesystem>
#include <string>

#include <SushiAI/config/train_config.hpp>
#include <SushiAI/config/value.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>

/** @brief Holds the training file the split-step tests share over the tiny run's model. */
namespace SushiAI::Apps::TinyRunTest
{
    /** @brief The rows of each batch of the file's data. */
    inline constexpr int K_BATCH_ROWS = 8;

    /** @brief Returns the directory of the tiny run's model file. */
    inline std::filesystem::path tiny_run_directory()
    {
#ifdef SA_TEST_FIXTURE_DIR
        return std::filesystem::path(SA_TEST_FIXTURE_DIR) / "tiny_run";
#else
        return {};
#endif
    }

    /**
     * @brief Returns the tiny run's training file at version 5: `rings` in two batches of
     *        K_BATCH_ROWS, `adamw`, two epochs, no checkpoint, and @p members after the epochs.
     */
    inline Config::TrainSpec tiny_spec(const std::string& members)
    {
        const std::string text = R"({
            "sushiai_train": 5,
            "model": "tiny_model.json",
            "objective": {"type": "cross_entropy"},
            "data": {"type": "rings", "batch_size": 8, "batches": 2, "classes": 4,
                     "seed": 3735928559},
            "optimizer": {"type": "adamw", "learning_rate": 3e-3},
            "epochs": 2)" + members + "}";
        const Config::Value root = Test::require_value(Config::parse_json(text, "<test>"));
        return Test::require_value(Config::parse_train(root, "<test>", tiny_run_directory()));
    }
} // namespace SushiAI::Apps::TinyRunTest
