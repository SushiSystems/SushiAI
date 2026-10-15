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
#include <SushiAI/SushiAI.h>
#include <SushiAI/core/seam.hpp>
#include <SushiAI/data/roles.hpp>
#include <SushiAI/data/target.hpp>
#include <SushiAI/io/run_checkpoint.hpp>
#include <SushiAI/optim/cosine_schedule.hpp>
#include <SushiAI/optim/rate_schedule.hpp>
#include <SushiAI/optim/run_schedule.hpp>
#include <SushiAI/optim/step_schedule.hpp>
#include <SushiAI/train/objective_context.hpp>
#include <SushiAI/train/objective_registry.hpp>
#include <SushiAI/train/role_match.hpp>
#include <SushiBLAS/SushiBLAS.hpp>
#include <SushiBLAS/core/format.hpp>
#include "registries.hpp"
#include "run_start.hpp"
#include "training_run.hpp"

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
        "  resume and checkpoint paths resolve against its own directory.\n";

    /** @brief Lists the roles whose description states the class count of a dataset. */
    constexpr std::string_view K_CLASS_ROLES[] = {sa::Data::K_ROLE_LABELS,
                                                  sa::Data::K_ROLE_CLASSES};

    /** @brief Returns the class count the first class role of @p dataset states, else 0. */
    [[nodiscard]] int64_t class_count_of(const sa::Data::Dataset& dataset) noexcept
    {
        for (const std::string_view role : K_CLASS_ROLES)
        {
            const sa::Data::TargetDescription* const described =
                sa::Data::find_target(dataset.targets(), role);
            if (described != nullptr)
                return described->classes;
        }

        return 0;
    }

    /** @brief Returns what the header prints for a step schedule. */
    [[nodiscard]] std::string describe_schedule(const sa::Optim::StepSchedule& schedule)
    {
        return SushiBLAS::Core::format("scheduled from {:.6g}, {} warm-up iterations, {} steps",
                                       schedule.base_rate, schedule.warmup_iterations,
                                       schedule.steps.size());
    }

    /** @brief Returns what the header prints for a cosine schedule. */
    [[nodiscard]] std::string describe_schedule(const sa::Optim::CosineSchedule& schedule)
    {
        return SushiBLAS::Core::format(
            "scheduled from {:.6g}, {} warm-up iterations, cosine to {:.6g} at iteration {}",
            schedule.base_rate, schedule.warmup_iterations, schedule.final_rate,
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

        SA_TRY_ASSIGN(sr::Runtime runtime, sa::Detail::create_runtime());
        sb::Engine engine(runtime);

        // Everything below is picked by string, not by type; see applications/programs/README.md.

        SA_TRY_ASSIGN(const sa::Apps::Registries registries, sa::Apps::compose_registries());

        SA_TRY_ASSIGN(const std::unique_ptr<sa::Data::Dataset> dataset,
                      sa::Data::make_dataset(registries.datasets, engine, spec.data.type,
                                             spec.data.attributes, "the data block"));

        const sa::Shape sample_shape = dataset->sample_shape();
        SA_FAIL_IF(sample_shape != model_spec.input_shape,
                   sa::StatusCode::InvalidDescription,
                   "'{}' declares the input shape {}, but the '{}' data supplies {}",
                   spec.model_path.string(), model_spec.input_shape.to_string(),
                   dataset->name(), sample_shape.to_string());

        SA_TRY_ASSIGN(sa::NN::ModelGraph model, sa::NN::build_model(registries.nodes, model_spec));

        SA_TRY_ASSIGN(const sa::Shape batch_shape,
                      sa::Train::batch_shape(dataset->batch_size(), sample_shape));

        // The probe trace states the published shapes an objective factory may read.
        sa::Train::GraphOptions graph_options;
        graph_options.dtype = model_spec.input_dtype;
        SA_TRY_ASSIGN(graph_options.seed, sa::Config::read_run_seed(spec, argv[1]));
        SA_TRY_ASSIGN(const sa::Train::TracedStep probe,
                      sa::Train::build_probe_graph(model, batch_shape, graph_options));

        sa::Train::ObjectiveContext given;
        given.batch = dataset->batch_size();
        given.classes = class_count_of(*dataset);
        given.input_sample = sample_shape;
        given.task = model_spec.task;
        SA_TRY_ASSIGN(given.outputs,
                      sa::Train::list_output_shapes(probe.graph, probe.outputs));

        SA_TRY_ASSIGN(const std::unique_ptr<sa::Train::Objective> objective,
                      sa::Train::make_objective(registries.objectives, spec.objective_type, given,
                                                spec.objective_attributes,
                                                "the objective block"));

        SA_TRY_ASSIGN(const std::optional<sa::Optim::RateSchedule> read_schedule,
                      sa::Optim::read_rate_schedule(spec.schedule, spec.optimizer_attributes));
        const sa::Optim::RateSchedule* const schedule =
            read_schedule ? &*read_schedule : nullptr;

        SA_TRY_ASSIGN(sa::Train::TrainingGraph built,
                      sa::Train::build_training_graph(model, *objective, batch_shape,
                                                      graph_options));
        SA_TRY(sa::Train::match_roles(built.leaves.required(), *dataset));

        // The trace above declared every parameter and buffer the state lists.
        SA_TRY_ASSIGN(sa::NN::ModelState state, sa::NN::ModelState::create(engine, model));

        sa::Graph::TensorPool pool(engine);
        SA_TRY_ASSIGN(std::vector<sa::Optim::ParameterSlot> parameters,
                      sa::Train::bind_parameters(built, state, pool));

        SA_TRY_ASSIGN(const std::unique_ptr<sa::Optim::Optimizer> optimizer,
                      sa::Optim::make_optimizer(engine, spec.optimizer_type, parameters,
                                                spec.optimizer_attributes, "the optimizer block"));

        sa::IO::RunProgress start;
        SA_TRY(sa::Apps::start_state(spec, engine, state, *optimizer, &start));

        SA_TRY_ASSIGN(std::vector<sa::Train::BufferUpdateSlot> updates,
                      sa::Train::bind_buffer_updates(built.updates, state, pool));

        std::cout << "sushiai train\n"
                  << "----------------------------------------------------------------\n"
                  << "SushiAI    : " << SushiAI::version_string()
                  << " on SushiBLAS " << SUSHIBLAS_VERSION_STRING << "\n"
                  << SushiBLAS::Core::format("Config     : {}\n", argv[1])
                  << SushiBLAS::Core::format("Model      : {} ({} layers, {} learnable scalars)\n",
                                             spec.model_path.string(), model.size(),
                                             sa::NN::parameter_elements(model))
                  << SushiBLAS::Core::format("Weights    : {}\n",
                                             sa::Apps::describe_start(spec))
                  << SushiBLAS::Core::format("Objective  : {}\n", objective->name())
                  << SushiBLAS::Core::format("Data       : {} ({} x {})\n",
                                             dataset->name(), dataset->batch_count(),
                                             dataset->batch_size())
                  << SushiBLAS::Core::format("Optimizer  : {}\n", spec.optimizer_type)
                  << SushiBLAS::Core::format("Rate       : {}\n", describe_rate(schedule))
                  << SushiBLAS::Core::format("Graph      : {} IR nodes over {} values\n",
                                             built.graph.node_count(), built.graph.value_count())
                  << SushiBLAS::Core::format("Seed       : {}\n", start.seed)
                  << SushiBLAS::Core::format("Schedule   : {} epochs\n\n", spec.epochs);

        // The run is a row or the fallback run; see applications/programs/README.md.
        sa::Apps::PreparedRun prepared{
            engine, spec, *dataset, given, *objective, built, state, pool, *optimizer,
            std::move(updates), schedule, start,
        };
        SA_TRY_ASSIGN(const sa::Apps::RunSummary summary,
                      sa::Apps::run_training(registries.runs, spec.objective_type, prepared));

        // CTest pins this line's shape, as in `sa demo mlp`; see applications/programs/README.md.
        std::cout << SushiBLAS::Core::format("\ncompiles: {}\n", summary.compile_count);
        if (summary.compile_count != 1)
            std::cerr << "The step must be compiled exactly once.\n";

        SA_FAIL_IF(summary.compile_count != 1, sa::StatusCode::InvalidDescription,
                   "the step was compiled {} times", summary.compile_count);

        if (!spec.checkpoint.empty())
        {
            SA_TRY(sa::IO::save_run_checkpoint(spec.checkpoint, state, *optimizer, engine,
                                               summary.progress));
            std::cout << SushiBLAS::Core::format("wrote {} ({} scalars)\n",
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
