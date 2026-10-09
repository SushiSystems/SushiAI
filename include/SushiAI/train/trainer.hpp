/**************************************************************************/
/* trainer.hpp                                                            */
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
 * @file trainer.hpp
 * @brief Drives model training loops and compiled execution plans.
 * @author Mustafa Garip
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>
#include <SushiBLAS/engine.hpp>
#include <SushiRuntime/core/run_report.hpp>
#include <SushiAI/core/common.hpp>
#include <SushiAI/core/result.hpp>
#include <SushiAI/core/status.hpp>
#include <SushiAI/data/dataset.hpp>
#include <SushiAI/graph/compiled_step.hpp>
#include <SushiAI/graph/fusion_selector.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/graph/memory_plan.hpp>
#include <SushiAI/graph/precision.hpp>
#include <SushiAI/graph/tensor_pool.hpp>
#include <SushiAI/optim/optimizer.hpp>
#include <SushiAI/train/buffer_update_recorder.hpp>
#include <SushiAI/train/feed.hpp>
#include <SushiAI/train/steps.hpp>
#include <SushiAI/train/training_graph.hpp>
#include <SushiAI/train/training_step.hpp>

namespace SushiAI
{
    namespace NN
    {
        class ModelState;
    } // namespace NN

    namespace Train
    {
        /**
         * @brief Plans memory, allocates leaves, and pairs parameters with gradients.
         * @param built The traced graph.
         * @param pool Destination tensor pool.
         * @param options Memory packing options.
         * @return Parameter slots paired with gradients, or InvalidDescription for a
         *         parameter no gradient reaches, or the pool's failure.
         */
        [[nodiscard]] Result<std::vector<Optim::ParameterSlot>> bind_parameters(
            const TrainingGraph& built, Graph::TensorPool& pool, Graph::PlanOptions options = {});

        /**
         * @brief Plans memory, binds the state's storage, and pairs parameters with gradients.
         * @param built The traced graph; its pinned values follow those of @p options.
         * @param state The state of the model @p built was traced from; it owns the parameters.
         * @param pool Holds no leaf of the graph yet; a plan it holds must leave them unpacked.
         * @return Parameter slots over the state's tensors, InvalidDescription for a parameter
         *         no gradient reaches, or the failure of the state, the planner or the pool.
         */
        [[nodiscard]] Result<std::vector<Optim::ParameterSlot>> bind_parameters(
            const TrainingGraph& built, const NN::ModelState& state, Graph::TensorPool& pool,
            Graph::PlanOptions options = {});

        /**
         * @struct TrainerOptions
         * @brief How the loop behaves.
         */
        struct TrainerOptions
        {
            /** @brief How many passes over the dataset fit() makes. */
            int64_t epochs = 1;

            /** @brief Score accuracy, at the cost of one host read of the logits per step. */
            bool track_accuracy = true;

            /** @brief Collect the runtime's per-node timings; needs a profiling RuntimeContext. */
            bool profile = false;

            /** @brief Whether to fold the optimizer's update into the compiled step. */
            OptimizerPlacement optimizer_placement = OptimizerPlacement::InPlan;

            /**
             * @brief Specifies the fusion selector for device-specific kernel variants.
             */
            const Graph::FusionSelector* fusion_selector = nullptr;

            /**
             * @brief Precision policy used to validate operand types before lowering.
             */
            const Graph::PrecisionPolicy* precision = nullptr;
        };

        /**
         * @struct StepMetrics
         * @brief What one step produced.
         */
        struct StepMetrics
        {
            /** @brief The batch's mean cross-entropy. */
            double loss = 0.0;

            /** @brief The fraction correct, or 0 when accuracy tracking is off. */
            double accuracy = 0.0;
        };

        /**
         * @struct RunReport
         * @brief What a whole fit() produced.
         */
        struct RunReport
        {
            /** @brief How many times the step was compiled. One, or something is wrong. */
            std::size_t compile_count = 0;

            /** @brief How many steps ran. */
            std::size_t steps = 0;

            /**
             * @brief Total number of operations comprising the compiled step.
             */
            std::size_t operations_per_step = 0;

            /** @brief The first step's loss. */
            double first_loss = 0.0;

            /** @brief The last step's loss. */
            double last_loss = 0.0;

            /** @brief The last step's accuracy. */
            double last_accuracy = 0.0;

            /** @brief Wall-clock time spent inside fit(). */
            double duration_ms = 0.0;

            /** @brief Per-node timings; empty unless the runtime has profiling on. */
            std::vector<SushiRuntime::Core::NodeTiming> node_timings;

            /** @brief A multi-line human-readable summary. */
            std::string to_string() const;
        };

        /**
         * @class Trainer
         * @brief Owns the training step and drives it over a dataset.
         */
        class Trainer final
        {
            public:
                /**
                 * @brief Checks @p spec, lowers @p graph, compiles the step and builds a trainer.
                 * @param engine Must outlive the trainer, as must @p pool and @p optimizer.
                 * @param spec Inputs, targets and loss set; logits set, and of rank 2, when
                 *        accuracy is tracked. The loss holds one float32 or float64 element.
                 * @param updates The running-statistic updates every step applies.
                 * @return The trainer, or InvalidDescription, or the failure of the compilation.
                 */
                [[nodiscard]] static Result<std::unique_ptr<Trainer>> make(
                    SushiBLAS::Engine& engine, const Graph::Graph& graph, Graph::TensorPool& pool,
                    StepSpec spec, Optim::Optimizer& optimizer, TrainerOptions options = {},
                    std::vector<BufferUpdateSlot> updates = {});

                Trainer(const Trainer&) = delete;
                Trainer& operator=(const Trainer&) = delete;
                Trainer(Trainer&&) = delete;
                Trainer& operator=(Trainer&&) = delete;
                ~Trainer() = default;

                /**
                 * @brief Runs one step: feed, replay, read metrics, update.
                 * @param batch The batch to train on.
                 * @return The step's loss and accuracy, or Incompatible when @p batch does not
                 *         match the compiled shapes, or the failure of the run.
                 */
                [[nodiscard]] Result<StepMetrics> run_step(const Data::Batch& batch);

                /**
                 * @brief Runs @c options.epochs passes over @p dataset.
                 * @param dataset The batches to train on.
                 * @return The run's summary, or InvalidDescription for an empty dataset, or
                 *         Incompatible when its shapes do not match the compiled step, or the
                 *         failure of a step.
                 */
                [[nodiscard]] Result<RunReport> fit(const Data::Dataset& dataset);

                /** @brief Returns how many times the step has been compiled. */
                [[nodiscard]] std::size_t compile_count() const noexcept
                {
                    return step_->step().compile_count();
                }

                /** @brief Returns how many steps have run. */
                [[nodiscard]] std::size_t step_count() const noexcept
                {
                    return steps_;
                }

                /** @brief Returns whether the optimizer's update is part of the compiled step. */
                [[nodiscard]] bool optimizer_is_in_plan() const noexcept
                {
                    return step_->optimizer_is_in_plan();
                }

                /** @brief Returns the compiled step. */
                [[nodiscard]] const Graph::CompiledStep& step() const noexcept
                {
                    return step_->step();
                }

                /** @brief Reads the loss tensor's current value from the host. */
                [[nodiscard]] double current_loss() const;

                /** @brief Returns the options in force. */
                [[nodiscard]] TrainerOptions& options() noexcept
                {
                    return options_;
                }

            private:
                /**
                 * @brief Stores the parts make() checked and compiled.
                 * @param step The training step; never null.
                 * @param rows The inputs' first extent.
                 * @param classes The logits' second extent, or 0 when they are not rank 2.
                 */
                Trainer(SushiBLAS::Engine& engine, Graph::TensorPool& pool, StepSpec spec,
                        TrainerOptions options, std::unique_ptr<TrainingStep> step,
                        int64_t rows, int64_t classes) noexcept;

                /**
                 * @brief Returns the fraction of @p batch whose label is the argmax of its logits.
                 * @return The fraction, or InvalidDescription when the logits are not rank 2,
                 *         or neither float32 nor float64.
                 */
                [[nodiscard]] Result<double> score_accuracy(const Data::Batch& batch) const;

                /// @brief The engine the batch copies are recorded on; not owned.
                SushiBLAS::Engine* engine_;

                /// @brief The pool that holds the step's tensors; not owned.
                Graph::TensorPool* pool_;

                /// @brief The values the loop reads and writes.
                StepSpec spec_;

                /// @brief The options in force.
                TrainerOptions options_;

                /// @brief The training step, which owns the compiled replay; never null.
                std::unique_ptr<TrainingStep> step_;

                /// @brief The number of samples per batch.
                int64_t rows_ = 0;

                /// @brief The number of classes per sample; 0 when the logits are not rank 2.
                int64_t classes_ = 0;

                /// @brief The number of steps that have run.
                std::size_t steps_ = 0;
        };

    } // namespace Train
} // namespace SushiAI
