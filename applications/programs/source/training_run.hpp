/**************************************************************************/
/* training_run.hpp                                                       */
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
 * @file training_run.hpp
 * @brief Declares the table of the training runs `sushiai train` picks for an objective.
 * @author Mustafa Garip
 * @see applications/programs/README.md, "The run table".
 */

#pragma once

#include <cstddef>
#include <string_view>
#include <vector>
#include <SushiBLAS/engine/engine.hpp>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/config/train_config.hpp>
#include <SushiAI/core/result.hpp>
#include <SushiAI/core/status.hpp>
#include <SushiAI/data/dataset.hpp>
#include <SushiAI/graph/tensor_pool.hpp>
#include <SushiAI/io/run_progress.hpp>
#include <SushiAI/nn/model_state.hpp>
#include <SushiAI/optim/optimizer.hpp>
#include <SushiAI/optim/rate_schedule.hpp>
#include <SushiAI/train/buffer_update_recorder.hpp>
#include <SushiAI/train/objective.hpp>
#include <SushiAI/train/objective_context.hpp>
#include <SushiAI/train/steps.hpp>
#include <SushiAI/train/training_graph.hpp>

namespace SushiAI
{
    namespace Apps
    {
        /**
         * @struct PreparedRun
         * @brief Holds what `sushiai train` built before the run starts; a run owns none of it.
         */
        struct PreparedRun
        {
            /** @brief The engine every step is compiled and run on. */
            SushiBLAS::Engine& engine;

            /** @brief The training file: the epochs and the objective's own keys. */
            const Config::TrainSpec& spec;

            /** @brief The batches of one epoch. */
            const Data::Dataset& dataset;

            /** @brief What the objective was built from; a run may build its typed twin. */
            const Train::ObjectiveContext& given;

            /** @brief The objective the training graph was built with. */
            const Train::Objective& objective;

            /** @brief The training graph, bound into @c pool by bind_parameters(). */
            const Train::TrainingGraph& built;

            /** @brief The parameters and buffers of the model, initialised or loaded. */
            const NN::ModelState& state;

            /** @brief The pool that holds the tensors of the training graph. */
            Graph::TensorPool& pool;

            /** @brief The optimizer over the parameters of @c state. */
            Optim::Optimizer& optimizer;

            /** @brief The running-statistic updates of the training trace; a run moves them. */
            std::vector<Train::BufferUpdateSlot> updates;

            /** @brief The schedule of the rate; null keeps the optimizer's own rate. */
            const Optim::RateSchedule* schedule;

            /** @brief Where the run starts: its seed, and the place a resumed run goes on at. */
            IO::RunProgress start{};
        };

        /**
         * @struct RunSummary
         * @brief Holds what a finished run reports back to the program.
         */
        struct RunSummary
        {
            /** @brief How many times the training step was compiled. */
            std::size_t compile_count = 0;

            /** @brief Where the run stands, as the checkpoint of the run records it. */
            IO::RunProgress progress{};
        };

        /**
         * @brief Returns where @p run stands once every epoch of its file has been trained.
         * @param step_runs The run count of the training step after the last update.
         * @return The seed the run started with and the first batch of the epoch after it.
         */
        [[nodiscard]] IO::RunProgress finished_progress(const PreparedRun& run,
                                                        std::size_t step_runs) noexcept;

        /**
         * @brief Checks that @p run can start at the batch its start names.
         * @return Incompatible, with the batch and the count, for a batch the data lacks.
         */
        [[nodiscard]] Status check_start_batch(const PreparedRun& run);

        /**
         * @brief Trains over every epoch of the run and prints its progress lines.
         * @return The summary, or the first failure of a step, a read or a compilation.
         */
        using RunFunction = Result<RunSummary> (*)(PreparedRun& run);

        /**
         * @struct RunEntry
         * @brief One row of a RunRegistry.
         */
        struct RunEntry
        {
            /** @brief The objective "type" whose own step the row drives. */
            std::string_view objective;

            /** @brief What performs the run. */
            RunFunction run;
        };

        /**
         * @brief Holds the runs `sushiai train` may pick: one row per objective type that
         *        has a step of its own, and the run of every other objective.
         */
        class RunRegistry final
        {
            public:
                /**
                 * @brief Appends a run row.
                 * @param row Its type text must outlive the registry; an empty type or a null
                 *            function is fatal.
                 * @return InvalidDescription when a row holds the type.
                 */
                [[nodiscard]] Status add(const RunEntry& row);

                /** @brief Returns the row of @p objective, or null when none holds it. */
                [[nodiscard]] const RunEntry* find(std::string_view objective) const noexcept;

                /**
                 * @brief Sets the run of every objective type that has no row.
                 * @param run The run; a null function is fatal.
                 * @return InvalidDescription when a fallback is already set.
                 */
                [[nodiscard]] Status set_fallback(RunFunction run);

                /** @brief Returns the run set_fallback() was given, or null when none was. */
                [[nodiscard]] RunFunction fallback() const noexcept;

                /**
                 * @brief Returns the rows in the order they were added.
                 * @return A view the next add() invalidates.
                 */
                [[nodiscard]] SushiRuntime::span<const RunEntry> rows() const noexcept;

            private:
                /** @brief Holds the rows in insertion order. */
                std::vector<RunEntry> rows_;

                /** @brief Holds the run of an objective type with no row, or null. */
                RunFunction fallback_ = nullptr;
        };

        /**
         * @brief Performs the run of the row @p objective names, or the fallback run.
         * @return What the run returns; InvalidDescription, naming every objective type
         *         that has a row, when @p objective is no row and no fallback is set.
         */
        [[nodiscard]] Result<RunSummary> run_training(const RunRegistry& registry,
                                                      std::string_view objective,
                                                      PreparedRun& run);

    } // namespace Apps
} // namespace SushiAI
