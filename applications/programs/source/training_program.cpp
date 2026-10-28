/**************************************************************************/
/* training_program.cpp                                                   */
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
 * @file training_program.cpp
 * @brief Defines the assembly `sushiai train` runs once it holds a parsed training file.
 * @author Mustafa Garip
 * @see applications/programs/README.md, "training_program.hpp".
 */

#include "training_program.hpp"

#include <cstdint>
#include <filesystem>
#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <variant>
#include <SushiAI/SushiAI.hpp>
#include <SushiAI/io/run_checkpoint.hpp>
#include <SushiAI/optim/cosine_schedule.hpp>
#include <SushiAI/optim/rate_schedule.hpp>
#include <SushiAI/optim/run_schedule.hpp>
#include <SushiAI/optim/step_schedule.hpp>
#include <SushiAI/profile/profile.hpp>
#include <SushiBLAS/core/version.hpp>
#include <SushiBLAS/engine/engine.hpp>
#include <SushiLog/text/text_format.hpp>
#include <SushiLog/text/value_text.hpp>
#include "cluster_training.hpp"
#include "registries.hpp"
#include "run_start.hpp"
#include "training_setup.hpp"

namespace SushiAI
{
    namespace Apps
    {
        namespace
        {
            /** @brief Returns what the header prints for a step schedule. */
            [[nodiscard]] std::string describe_schedule(const Optim::StepSchedule& schedule)
            {
                return SushiLog::format_text("scheduled from {}, {} warm-up iterations, {} steps",
                                               SushiLog::general(schedule.base_rate, 6),
                                               schedule.warmup_iterations,
                                               schedule.steps.size());
            }

            /** @brief Returns what the header prints for a cosine schedule. */
            [[nodiscard]] std::string describe_schedule(const Optim::CosineSchedule& schedule)
            {
                return SushiLog::format_text(
                    "scheduled from {}, {} warm-up iterations, cosine to {} at iteration {}",
                    SushiLog::general(schedule.base_rate, 6), schedule.warmup_iterations,
                    SushiLog::general(schedule.final_rate, 6),
                    schedule.total_iterations);
            }

            /** @brief Returns what the header prints for the rate of the run. */
            [[nodiscard]] std::string describe_rate(const Optim::RateSchedule* schedule)
            {
                if (schedule == nullptr)
                    return "the optimizer's own";

                return std::visit([](const auto& held)
                {
                    return describe_schedule(held);
                }, *schedule);
            }

            /** @brief Returns how many learnable scalars the parameters of @p state hold. */
            [[nodiscard]] std::int64_t learnable_scalars(const NN::ModelState& state)
            {
                std::int64_t total = 0;
                for (const NN::StateEntry& entry : state.entries())
                {
                    if (entry.kind == NN::StateKind::Parameter)
                        total += entry.shape.num_elements();
                }
                return total;
            }

            /**
             * @brief Creates the folder the checkpoint of @p spec is written to, so a run that
             *        cannot write it fails before it trains.
             */
            [[nodiscard]] Status prepare_checkpoint_folder(const Config::TrainSpec& spec)
            {
                const std::filesystem::path folder = spec.checkpoint.parent_path();
                if (spec.checkpoint.empty() || folder.empty())
                    return Status();

                std::error_code failure;
                std::filesystem::create_directories(folder, failure);
                SA_FAIL_IF(failure, StatusCode::InvalidFile,
                           "could not create '{}', the folder of the checkpoint: {}",
                           folder.string(), failure.message());
                return Status();
            }
        } // namespace

        Result<TrainingOutcome> run_training_file(std::string_view origin,
                                                  const Config::TrainSpec& spec,
                                                  SushiBLAS::Engine& engine)
        {
            SA_TRY(Config::check_start(spec, origin));
            SA_TRY(prepare_checkpoint_folder(spec));
            SA_TRY_ASSIGN(const Config::ModelSpec model_spec,
                          Config::load_model(spec.model_path));

            // Everything below is picked by string, not by type; see the programs README.

            SA_TRY_ASSIGN(const Registries registries, compose_registries());

            SA_TRY_ASSIGN(const std::uint64_t seed, Config::read_run_seed(spec, origin));
            // A cluster's master runs no graph, so its pool holds the results the exchange writes.
            const SetupScope scope =
                spec.cluster.is_null() ? SetupScope::Step : SetupScope::Results;
            SA_TRY_ASSIGN(const std::unique_ptr<TrainingSetup> setup,
                          TrainingSetup::make(registries, spec, model_spec, engine, seed,
                                              std::nullopt, scope));
            const Data::Dataset& dataset = setup->dataset();
            const NN::ModelGraph& model = setup->model();
            const Train::TrainingGraph& built = setup->built();

            SA_TRY_ASSIGN(const std::optional<Optim::RateSchedule> read_schedule,
                          Optim::read_rate_schedule(spec.schedule, spec.optimizer_attributes));
            const Optim::RateSchedule* const schedule =
                read_schedule ? &*read_schedule : nullptr;

            SA_TRY_ASSIGN(const std::unique_ptr<Optim::Optimizer> optimizer,
                          Optim::make_optimizer(engine, spec.optimizer_type, setup->parameters(),
                                                spec.optimizer_attributes, "the optimizer block"));

            IO::RunProgress start;
            SA_TRY(start_state(spec, engine, setup->state(), *optimizer, &start));

            std::cout << "sushiai train\n"
                      << "----------------------------------------------------------------\n"
                      << "SushiAI    : " << SushiAI::version_string()
                      << " on SushiBLAS " << SUSHIBLAS_VERSION_STRING << "\n"
                      << SushiLog::format_text("Config     : {}\n", origin)
                      << SushiLog::format_text(
                             "Model      : {} ({} layers, {} learnable scalars)\n",
                             spec.model_path.string(), model.size(),
                             learnable_scalars(setup->state()))
                      << SushiLog::format_text("Weights    : {}\n", describe_start(spec))
                      << SushiLog::format_text("Objective  : {}\n", setup->objective().name())
                      << SushiLog::format_text("Data       : {} ({} x {})\n",
                                                 dataset.name(), dataset.batch_count(),
                                                 dataset.batch_size())
                      << SushiLog::format_text("Optimizer  : {}\n", spec.optimizer_type)
                      << SushiLog::format_text("Rate       : {}\n", describe_rate(schedule))
                      << SushiLog::format_text(
                             "Graph      : {} IR nodes over {} values\n",
                             built.graph.node_count(), built.graph.value_count())
                      << SushiLog::format_text("Seed       : {}\n", start.seed)
                      << SushiLog::format_text("Schedule   : {} epochs\n\n", spec.epochs);

            // The run is a row or the fallback run; see applications/programs/README.md.
            PreparedRun prepared{
                engine, spec, dataset, setup->given(), setup->objective(), built, setup->state(),
                setup->pool(), *optimizer, setup->take_updates(), schedule, start,
            };
            // A file with a cluster section trains over its replicas; see cluster_training.hpp.
            const SushiRuntime::span<const Optim::ParameterSlot> slots(
                setup->parameters().data(), setup->parameters().size());
            SA_TRY_ASSIGN(const RunSummary summary,
                          spec.cluster.is_null()
                              ? run_training(registries.runs, spec.objective_type, prepared)
                              : run_cluster_training(prepared, slots));

            // CTest pins this line's shape, as in `sa demo mlp`; see the programs README.
            std::cout << SushiLog::format_text("\ncompiles: {}\n", summary.compile_count);
            if (summary.compile_count != 1)
                std::cerr << "The step must be compiled exactly once.\n";

            SA_FAIL_IF(summary.compile_count != 1, StatusCode::InvalidDescription,
                       "the step was compiled {} times", summary.compile_count);

            if (!spec.checkpoint.empty())
            {
                {
                    SA_ZONE("io", "checkpoint");
                    SA_TRY(IO::save_run_checkpoint(spec.checkpoint, setup->state(), *optimizer,
                                                   summary.progress));
                }
                std::cout << SushiLog::format_text("wrote {} ({} scalars)\n",
                                                     spec.checkpoint.string(),
                                                     learnable_scalars(setup->state()));
            }

            return TrainingOutcome{summary, optimizer->step_count()};
        }
    } // namespace Apps
} // namespace SushiAI
