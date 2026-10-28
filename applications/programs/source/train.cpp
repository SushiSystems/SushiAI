/**************************************************************************/
/* train.cpp                                                              */
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
 * @file train.cpp
 * @brief Runs `sushiai train`: trains the model a JSON configuration describes, by registry lookup.
 * @author Mustafa Garip
 * @see applications/programs/README.md, "train.cpp".
 */

#include <cstdlib>
#include <iostream>
#include <string>
#include <string_view>
#include <SushiAI/SushiAI.hpp>
#include <SushiAI/execution/execution.hpp>
#include <SushiBLAS/engine/engine.hpp>
#include <SushiRuntime/core/span.hpp>
#include "cluster_training.hpp"
#include "profile_arguments.hpp"
#include "profile_session.hpp"
#include "training_program.hpp"

namespace sb = SushiBLAS;

namespace
{
    /** @brief Holds the text `--help` prints. */
    constexpr std::string_view K_USAGE =
        "sushiai train - train a model described by a JSON configuration.\n"
        "\n"
        "Usage: sushiai train <train.json> [--profile <profile.json>]\n"
        "\n"
        "  train.json names the architecture file, the objective, the data, the\n"
        "  optimizer, the epoch count, the seed, the weights to start from or the\n"
        "  checkpoint to resume, and where to write the checkpoint. The model, weights,\n"
        "  resume and checkpoint paths resolve against its own directory.\n"
        "\n"
        "  --profile starts a SushiProfiler session from profile.json for the run and\n"
        "  writes the record that file names, whether the run succeeds or not.\n"
        "\n"
        "  A file with a 'cluster' section makes this program the master: it trains over\n"
        "  the replicas the section names, each served by `sushiai_worker` on its machine.\n"
        "  A build without `sa build --distributed` refuses such a file.\n";

    /**
     * @brief Trains the model the configuration at `argv[1]` describes.
     * @return Success, or the first failure a step of the run returned.
     */
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

        SA_FAIL_IF(arguments.positional.size() != 1, sa::StatusCode::InvalidDescription,
                   "usage: sushiai train <train.json> [--profile <profile.json>]");
        const std::string& spec_path = arguments.positional[0];

        SA_TRY_ASSIGN(const sa::Config::TrainSpec spec, sa::Config::load_train(spec_path));

        SA_TRY_ASSIGN(const sr::RuntimeConfig runtime_config,
                      sa::Apps::training_runtime_config(spec));
        SA_TRY_ASSIGN(sr::Runtime runtime, sa::Detail::create_runtime(runtime_config));
        sb::Engine engine(runtime);

        // The session opens after the engine and closes before it, so both outlive it.
        std::unique_ptr<sa::Apps::ProfileSession> profile;
        if (arguments.config)
        {
            SA_TRY_ASSIGN(profile, sa::Apps::ProfileSession::open(*arguments.config, runtime));
        }

        const sa::Result<sa::Apps::TrainingOutcome> trained =
            sa::Apps::run_training_file(spec_path, spec, engine);
        return sa::Apps::end_profiled_run(profile, trained.status());
    }
} // namespace

int main(int argc, char** argv)
{
    const sa::Status status = run(argc, argv);
    if (status.is_ok())
        return EXIT_SUCCESS;

    std::cerr << "sushiai train failed: " << status.message() << "\n"
              << "  kind: " << sa::status_code_name(status.code()) << "\n";
    return EXIT_FAILURE;
}
