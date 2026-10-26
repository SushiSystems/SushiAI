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
#include <exception>
#include <iostream>
#include <string>
#include <string_view>
#include <SushiAI/SushiAI.hpp>
#include <SushiAI/execution/execution.hpp>
#include <SushiBLAS/engine/engine.hpp>
#include "cluster_training.hpp"
#include "training_program.hpp"

namespace sb = SushiBLAS;

namespace
{
    /** @brief Holds the text `--help` prints. */
    constexpr std::string_view K_USAGE =
        "sushiai train - train a model described by a JSON configuration.\n"
        "\n"
        "Usage: sushiai train <train.json>\n"
        "\n"
        "  train.json names the architecture file, the objective, the data, the\n"
        "  optimizer, the epoch count, the seed, the weights to start from or the\n"
        "  checkpoint to resume, and where to write the checkpoint. The model, weights,\n"
        "  resume and checkpoint paths resolve against its own directory.\n"
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
        if (argc == 2 && (std::string(argv[1]) == "-h" || std::string(argv[1]) == "--help"))
        {
            std::cout << K_USAGE;
            return sa::Status();
        }

        SA_FAIL_IF(argc != 2, sa::StatusCode::InvalidDescription,
                   "usage: sushiai train <train.json>");

        const sa::Config::TrainSpec spec = sa::Config::load_train(argv[1]);

        SA_TRY_ASSIGN(const sr::RuntimeConfig runtime_config,
                      sa::Apps::training_runtime_config(spec));
        SA_TRY_ASSIGN(sr::Runtime runtime, sa::Detail::create_runtime(runtime_config));
        sb::Engine engine(runtime);

        SA_TRY(sa::Apps::run_training_file(argv[1], spec, engine));

        return sa::Status();
    }
} // namespace

int main(int argc, char** argv)
{
    try
    {
        const sa::Status status = run(argc, argv);
        if (!status.is_ok())
        {
            std::cerr << "sushiai train failed: " << status.message() << "\n";
            return EXIT_FAILURE;
        }

        return EXIT_SUCCESS;
    }
    catch (const std::exception& error)
    {
        std::cerr << "sushiai train failed: " << error.what() << "\n";
        return EXIT_FAILURE;
    }
}
