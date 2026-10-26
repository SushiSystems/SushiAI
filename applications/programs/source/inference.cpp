/**************************************************************************/
/* inference.cpp                                                          */
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
 * @file inference.cpp
 * @brief Runs `sushiai inference`: evaluates a checkpoint over a dataset, forward pass only.
 * @author Mustafa Garip
 * @see applications/programs/README.md, "inference.cpp".
 */

#include <cstdlib>
#include <exception>
#include <filesystem>
#include <iostream>
#include <string>
#include <string_view>
#include <SushiAI/SushiAI.hpp>
#include <SushiAI/execution/execution.hpp>
#include <SushiBLAS/engine/engine.hpp>
#include <SushiRuntime/core/span.hpp>
#include "inference_program.hpp"
#include "profile_arguments.hpp"
#include "profile_session.hpp"

/** @brief Shortens the SushiBLAS namespace. */
namespace sb = SushiBLAS;

namespace
{
    /** @brief Holds the text `--help` prints. */
    constexpr std::string_view K_USAGE =
        "sushiai inference - evaluate a trained checkpoint over a dataset.\n"
        "\n"
        "Usage: sushiai inference <model.json> <model.sushi> <eval.json>\n"
        "                         [--profile <profile.json>]\n"
        "\n"
        "  model.json   the architecture the checkpoint was trained under\n"
        "  model.sushi  the weights `sushiai train` or `sushiai import` wrote\n"
        "  eval.json    the data to evaluate on; the architecture file carries none\n"
        "  --profile    starts a SushiProfiler session from profile.json for the run and\n"
        "               writes the record that file names, whether the run succeeds or not\n"
        "\n"
        "  A model file without a task section is reported as a classifier. One with\n"
        "  a task section is reported by the row of its task type; the data of a\n"
        "  `next_token_generation` model is a `prompt` block, and the report prints\n"
        "  the prompt and its continuation.\n";

    /** @brief Evaluates the checkpoint the arguments name and prints the report. */
    [[nodiscard]] sa::Status run(int argc, char** argv)
    {
        SA_TRY_ASSIGN(const sa::Apps::ProfileArguments arguments,
                      sa::Apps::read_profile_arguments(
                          SushiRuntime::span<const char* const>(argv + 1, argc - 1)));
        if (arguments.positional.size() == 1
            && (arguments.positional[0] == "-h" || arguments.positional[0] == "--help"))
        {
            std::cout << K_USAGE;
            return sa::Status();
        }

        SA_FAIL_IF(arguments.positional.size() != 3, sa::StatusCode::InvalidDescription,
                   "usage: sushiai inference <model.json> <model.sushi> <eval.json> "
                   "[--profile <profile.json>]");

        const std::filesystem::path model_path = arguments.positional[0];
        const std::filesystem::path checkpoint_path = arguments.positional[1];
        const std::filesystem::path data_path = arguments.positional[2];

        const sa::Config::ModelSpec model_spec = sa::Config::load_model(model_path);
        const sa::Config::DataSpec data_spec = sa::Config::load_data(data_path);

        SA_TRY_ASSIGN(sr::Runtime runtime, sa::Detail::create_runtime());
        sb::Engine engine(runtime);

        // The session opens after the engine and closes before it, so both outlive it.
        std::unique_ptr<sa::Apps::ProfileSession> profile;
        if (arguments.config)
        {
            SA_TRY_ASSIGN(profile, sa::Apps::ProfileSession::open(*arguments.config, runtime));
        }

        return sa::Apps::end_profiled_run(
            profile,
            sa::Apps::run_inference_file(model_path, checkpoint_path, model_spec, data_spec,
                                         engine));
    }
} // namespace

/** @brief Runs the inference driver and returns its exit code. */
int main(int argc, char** argv)
{
    try
    {
        const sa::Status status = run(argc, argv);
        if (!status.is_ok())
        {
            std::cerr << "sushiai inference failed: " << status.message() << "\n";
            return EXIT_FAILURE;
        }

        return EXIT_SUCCESS;
    }
    catch (const std::exception& error)
    {
        std::cerr << "sushiai inference failed: " << error.what() << "\n";
        return EXIT_FAILURE;
    }
}
