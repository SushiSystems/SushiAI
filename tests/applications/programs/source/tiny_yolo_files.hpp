/**************************************************************************/
/* tiny_yolo_files.hpp                                                    */
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
 * @file tiny_yolo_files.hpp
 * @brief Declares the tiny_yolo files and runs the program tests share.
 * @author Mustafa Garip
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <SushiBLAS/engine/engine.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/config/model_config.hpp>
#include <SushiAI/config/train_config.hpp>
#include <SushiAI/config/value.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/failure/status.hpp>
#include "inference_program.hpp"
#include "training_program.hpp"

/** @brief Holds what the tests of `sushiai train` and `sushiai inference` share over tiny_yolo. */
namespace SushiAI::Apps::TinyYoloTest
{
    /** @brief Holds the images of the fixture: two batches of two. */
    inline constexpr std::size_t K_IMAGES = 4;

    /** @brief Holds the seed the training file states. */
    inline constexpr std::uint64_t K_SEED = 7;

    /** @brief Holds the updates of the run: one epoch of two batches. */
    inline constexpr std::uint64_t K_STEPS = 2;

    /** @brief Returns the directory of the fixture: the model file and the YOLO-layout dataset. */
    inline std::filesystem::path fixture_directory()
    {
#ifdef SA_TEST_FIXTURE_DIR
        return std::filesystem::path(SA_TEST_FIXTURE_DIR) / "tiny_yolo";
#else
        return {};
#endif
    }

    /** @brief Returns the data block both files state: the fixture read as a detection task. */
    inline std::string data_block()
    {
        return R"({"reader": "yolo_txt", "task": "detection", "root": ")"
               + fixture_directory().generic_string()
               + R"(", "split": "train", "batch_size": 2, "input": [16, 16],
                   "max_objects": 8, "classes": 2})";
    }

    /** @brief Returns the training file: the anchor objective, one epoch, @p checkpoint. */
    inline std::string training_text(const std::filesystem::path& checkpoint)
    {
        return R"({
            "sushiai_train": 3,
            "model": "model.json",
            "objective": {"type": "anchor"},
            "data": )" + data_block() + R"(,
            "optimizer": {"type": "sgd", "momentum": 0.9, "learning_rate": 0.001},
            "epochs": 1,
            "seed": 7,
            "checkpoint": ")" + checkpoint.generic_string() + R"("
        })";
    }

    /** @brief Returns the parsed training file, which writes its checkpoint to @p checkpoint. */
    inline Config::TrainSpec training_spec(const std::filesystem::path& checkpoint)
    {
        const Config::Value root =
            Test::require_value(Config::parse_json(training_text(checkpoint), "<test>"));
        return Test::require_value(Config::parse_train(root, "<test>", fixture_directory()));
    }

    /**
     * @brief Runs `sushiai train` over the fixture through the assembly the program runs.
     * @return The outcome of the run, or the first failure of the setup, the run or the save.
     */
    inline Result<TrainingOutcome> train_the_fixture(SushiBLAS::Engine& engine,
                                                     const std::filesystem::path& checkpoint)
    {
        return run_training_file("<test>", training_spec(checkpoint), engine);
    }

    /**
     * @brief Runs `sushiai inference` over the fixture through the assembly the program runs,
     *        which prints the report of the task type of the model file.
     * @return The first failure of the build, the load or the report.
     */
    inline Status report_the_fixture(SushiBLAS::Engine& engine,
                                     const std::filesystem::path& checkpoint)
    {
        const std::filesystem::path model_path = fixture_directory() / "model.json";
        const Config::ModelSpec model_spec = Test::require_value(Config::load_model(model_path));
        const Config::DataSpec data_spec = Test::require_value(Config::parse_data(
            Test::require_value(Config::parse_json(R"({"sushiai_data": 1, "data": )" + data_block() + "}",
                                             "<test>")),
            "<test>", {}));
        return run_inference_file(model_path, checkpoint, model_spec, data_spec, engine);
    }
} // namespace SushiAI::Apps::TinyYoloTest
