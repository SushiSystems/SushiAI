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

#include <cstdint>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>
#include <SushiAI/SushiAI.hpp>
#include <SushiAI/execution/execution.hpp>
#include <SushiAI/io/run_checkpoint.hpp>
#include <SushiAI/optim/cosine_schedule.hpp>
#include <SushiAI/optim/rate_schedule.hpp>
#include <SushiAI/optim/run_schedule.hpp>
#include <SushiAI/optim/step_schedule.hpp>
#include <SushiBLAS/SushiBLAS.hpp>
#include <SushiLog/text/text_format.hpp>
#include <SushiLog/text/value_text.hpp>
#include "cluster_training.hpp"
#include "registries.hpp"
#include "run_start.hpp"
#include "training_run.hpp"
#include "training_setup.hpp"

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

    /** @brief Returns what the header prints for a step schedule. */
    [[nodiscard]] std::string describe_schedule(const sa::Optim::StepSchedule& schedule)
    {
        return SushiLog::format_text("scheduled from {}, {} warm-up iterations, {} steps",
                                       SushiLog::general(schedule.base_rate, 6),
                                       schedule.warmup_iterations,
                                       schedule.steps.size());
    }

    /** @brief Returns what the header prints for a cosine schedule. */
    [[nodiscard]] std::string describe_schedule(const sa::Optim::CosineSchedule& schedule)
    {
        return SushiLog::format_text(
            "scheduled from {}, {} warm-up iterations, cosine to {} at iteration {}",
            SushiLog::general(schedule.base_rate, 6), schedule.warmup_iterations,
            SushiLog::general(schedule.final_rate, 6),
            schedule.total_iterations);
    }

    /** @brief Returns what the header prints for the rate of the run. */
    [[nodiscard]] std::string describe_rate(const sa::Optim::RateSchedule* schedule)
    {
        if (schedule == nullptr)
            return "the optimizer's own";

        return std::visit([](const auto& held)
        {
            return describe_schedule(held);
        }, *schedule);
    }

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
        SA_TRY(sa::Config::check_start(spec, argv[1]));
        const sa::Config::ModelSpec model_spec = sa::Config::load_model(spec.model_path);

        SA_TRY_ASSIGN(const sr::RuntimeConfig runtime_config,
                      sa::Apps::training_runtime_config(spec));
        SA_TRY_ASSIGN(sr::Runtime runtime, sa::Detail::create_runtime(runtime_config));
        sb::Engine engine(runtime);

        // Everything below is picked by string, not by type; see applications/programs/README.md.

        SA_TRY_ASSIGN(const sa::Apps::Registries registries, sa::Apps::compose_registries());

        SA_TRY_ASSIGN(const std::uint64_t seed, sa::Config::read_run_seed(spec, argv[1]));
        // A cluster's master runs no graph, so its pool holds the results the exchange writes.
        const sa::Apps::SetupScope scope =
            spec.cluster.is_null() ? sa::Apps::SetupScope::Step : sa::Apps::SetupScope::Results;
        SA_TRY_ASSIGN(const std::unique_ptr<sa::Apps::TrainingSetup> setup,
                      sa::Apps::TrainingSetup::make(registries, spec, model_spec, engine, seed,
                                                    std::nullopt, scope));
        const sa::Data::Dataset& dataset = setup->dataset();
        const sa::NN::ModelGraph& model = setup->model();
        const sa::Train::TrainingGraph& built = setup->built();

        SA_TRY_ASSIGN(const std::optional<sa::Optim::RateSchedule> read_schedule,
                      sa::Optim::read_rate_schedule(spec.schedule, spec.optimizer_attributes));
        const sa::Optim::RateSchedule* const schedule =
            read_schedule ? &*read_schedule : nullptr;

        SA_TRY_ASSIGN(const std::unique_ptr<sa::Optim::Optimizer> optimizer,
                      sa::Optim::make_optimizer(engine, spec.optimizer_type, setup->parameters(),
                                                spec.optimizer_attributes, "the optimizer block"));

        sa::IO::RunProgress start;
        SA_TRY(sa::Apps::start_state(spec, engine, setup->state(), *optimizer, &start));

        std::cout << "sushiai train\n"
                  << "----------------------------------------------------------------\n"
                  << "SushiAI    : " << SushiAI::version_string()
                  << " on SushiBLAS " << SUSHIBLAS_VERSION_STRING << "\n"
                  << SushiLog::format_text("Config     : {}\n", argv[1])
                  << SushiLog::format_text("Model      : {} ({} layers, {} learnable scalars)\n",
                                             spec.model_path.string(), model.size(),
                                             sa::NN::parameter_elements(model))
                  << SushiLog::format_text("Weights    : {}\n",
                                             sa::Apps::describe_start(spec))
                  << SushiLog::format_text("Objective  : {}\n", setup->objective().name())
                  << SushiLog::format_text("Data       : {} ({} x {})\n",
                                             dataset.name(), dataset.batch_count(),
                                             dataset.batch_size())
                  << SushiLog::format_text("Optimizer  : {}\n", spec.optimizer_type)
                  << SushiLog::format_text("Rate       : {}\n", describe_rate(schedule))
                  << SushiLog::format_text("Graph      : {} IR nodes over {} values\n",
                                             built.graph.node_count(), built.graph.value_count())
                  << SushiLog::format_text("Seed       : {}\n", start.seed)
                  << SushiLog::format_text("Schedule   : {} epochs\n\n", spec.epochs);

        // The run is a row or the fallback run; see applications/programs/README.md.
        sa::Apps::PreparedRun prepared{
            engine, spec, dataset, setup->given(), setup->objective(), built, setup->state(),
            setup->pool(), *optimizer, setup->take_updates(), schedule, start,
        };
        // A file with a cluster section trains over its replicas; see cluster_training.hpp.
        const sr::span<const sa::Optim::ParameterSlot> slots(setup->parameters().data(),
                                                             setup->parameters().size());
        SA_TRY_ASSIGN(const sa::Apps::RunSummary summary,
                      spec.cluster.is_null()
                          ? sa::Apps::run_training(registries.runs, spec.objective_type, prepared)
                          : sa::Apps::run_cluster_training(prepared, slots));

        // CTest pins this line's shape, as in `sa demo mlp`; see applications/programs/README.md.
        std::cout << SushiLog::format_text("\ncompiles: {}\n", summary.compile_count);
        if (summary.compile_count != 1)
            std::cerr << "The step must be compiled exactly once.\n";

        SA_FAIL_IF(summary.compile_count != 1, sa::StatusCode::InvalidDescription,
                   "the step was compiled {} times", summary.compile_count);

        if (!spec.checkpoint.empty())
        {
            SA_TRY(sa::IO::save_run_checkpoint(spec.checkpoint, setup->state(), *optimizer,
                                               summary.progress));
            std::cout << SushiLog::format_text("wrote {} ({} scalars)\n",
                                                 spec.checkpoint.string(),
                                                 sa::NN::parameter_elements(model));
        }

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
