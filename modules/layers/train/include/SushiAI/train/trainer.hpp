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
#include <utility>
#include <vector>
#include <SushiBLAS/engine/engine.hpp>
#include <SushiRuntime/sushiruntime.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/data/dataset.hpp>
#include <SushiAI/data/input.hpp>
#include <SushiAI/graph/compiled_step.hpp>
#include <SushiAI/graph/fusion_selector.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/graph/memory_plan.hpp>
#include <SushiAI/graph/precision.hpp>
#include <SushiAI/graph/tensor_pool.hpp>
#include <SushiAI/optim/optimizer.hpp>
#include <SushiAI/train/buffer_update_recorder.hpp>
#include <SushiAI/train/feed.hpp>
#include <SushiAI/train/metrics.hpp>
#include <SushiAI/train/objective.hpp>
#include <SushiAI/train/step_view.hpp>
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
         * @brief Plans memory and binds the state, the loss, the update sources and gradients.
         * @param built The traced graph; its pinned values follow those of @p options.
         * @param state The state of the model @p built was traced from; it owns the parameters.
         * @param pool Holds no leaf of the graph yet; a plan it holds must leave them unpacked.
         * @return Slots over the state's tensors, decaying by takes_weight_decay() of their
         *         role; InvalidDescription for a parameter no gradient reaches; or the failure.
         */
        [[nodiscard]] Result<std::vector<Optim::ParameterSlot>> bind_parameters(
            const TrainingGraph& built, const NN::ModelState& state, Graph::TensorPool& pool,
            Graph::PlanOptions options = {});

        /**
         * @brief Binds the state's storage, a zeroed gradient per parameter, the loss and the
         *        sources of the buffer updates, and allocates no other value of the graph.
         * @param pool Holds no plan and no leaf of the graph; each value gets its own tensor.
         * @return Slots as bind_parameters() returns them, or the same failures.
         * @see modules/layers/train/README.md
         */
        [[nodiscard]] Result<std::vector<Optim::ParameterSlot>> bind_step_results(
            const TrainingGraph& built, const NN::ModelState& state, Graph::TensorPool& pool);

        /**
         * @struct TrainerOptions
         * @brief How the loop behaves.
         */
        struct TrainerOptions
        {
            /** @brief How many passes over the dataset fit() makes. */
            int64_t epochs = 1;

            /** @brief Asks the objective for its metrics after every step, a host read each. */
            bool track_metrics = true;

            /** @brief Sums the runtime's per-node timings; needs a runtime built with profiling. */
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
            /** @brief The loss the step computed. */
            double loss = 0.0;

            /** @brief What the objective measured; empty when metric tracking is off. */
            Metrics metrics;
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

            /** @brief The last step's metrics; empty when metric tracking is off. */
            Metrics last_metrics;

            /** @brief Wall-clock time spent inside fit(). */
            double duration_ms = 0.0;

            /** @brief Names the per-node timing rows of a SushiRuntime run report. */
            using NodeTimings = std::vector<SushiRuntime::NodeTiming>;

            /**
             * @brief Per-node timings summed over the steps, one row per plan index; empty
             *        unless the runtime profiles.
             */
            NodeTimings node_timings;

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
                 * @brief Lowers the graph of @p built, compiles the step and builds a trainer.
                 * @param engine Must outlive the trainer, as must @p pool and @p optimizer.
                 * @param built Its loss holds one float32 or float64 element.
                 * @param objective What @p built was built with; outlives the trainer.
                 * @param updates The running-statistic updates every step applies.
                 * @return The trainer, or InvalidDescription, or the failure of the compilation.
                 */
                [[nodiscard]] static Result<std::unique_ptr<Trainer>> make(
                    SushiBLAS::Engine& engine, const TrainingGraph& built,
                    Graph::TensorPool& pool, const Objective& objective,
                    Optim::Optimizer& optimizer, TrainerOptions options = {},
                    std::vector<BufferUpdateSlot> updates = {});

                Trainer(const Trainer&) = delete;
                Trainer& operator=(const Trainer&) = delete;
                Trainer(Trainer&&) = delete;
                Trainer& operator=(Trainer&&) = delete;
                ~Trainer() = default;

                /**
                 * @brief Runs one step: feed, replay, read metrics, update.
                 * @param batch The batch to train on; the objective fills its leaves from it.
                 * @return The loss and the metrics of the step, the refusal of the feed() or
                 *         the measure() of the objective, Incompatible when @p batch does
                 *         not match the compiled shapes, or the failure of the run.
                 */
                [[nodiscard]] Result<StepMetrics> run_step(const Data::Batch& batch);

                /**
                 * @brief Runs @c options.epochs passes over @p dataset.
                 * @param dataset The batches to train on.
                 * @return The run's summary, or InvalidDescription for an empty dataset, or
                 *         Incompatible for another batch size or for node timings of another
                 *         plan, or the failure of match_input(), of match_roles() for the roles
                 *         the objective reads, or of a step.
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

                /**
                 * @brief Sets the run count of the compiled step to that of a saved run.
                 * @param count What step().run_count() of the saved run returned; step_count()
                 *              keeps counting the steps of this trainer.
                 * @return What Graph::CompiledStep::restore_run_count() returns.
                 */
                [[nodiscard]] Status restore_run_count(std::size_t count)
                {
                    return step_->step().restore_run_count(count);
                }

                /** @brief Reads the loss tensor's current value from the host. */
                [[nodiscard]] double current_loss() const;

                /** @brief Returns the options in force. */
                [[nodiscard]] TrainerOptions& options() noexcept
                {
                    return options_;
                }

            private:
                /** @brief Holds what the loop reads of a training graph. */
                struct Binding
                {
                    Graph::ValueID inputs = Graph::K_NO_VALUE; /**< The minibatch input leaf. */
                    Graph::ValueID loss = Graph::K_NO_VALUE;   /**< The 1-element loss. */
                    ObjectiveLeaves leaves;                    /**< Leaves the objective fills. */
                    NN::ModelOutputs outputs;                  /**< What the model published. */
                };

                /**
                 * @brief Checks @p binding, compiles the step and builds the trainer.
                 * @param objective Fills the leaves of @p binding; outlives the trainer.
                 * @return The trainer, or InvalidDescription, or the failure of the compilation.
                 */
                [[nodiscard]] static Result<std::unique_ptr<Trainer>> assemble(
                    SushiBLAS::Engine& engine, const Graph::Graph& graph, Graph::TensorPool& pool,
                    Binding binding, const Objective& objective, Optim::Optimizer& optimizer,
                    TrainerOptions options, std::vector<BufferUpdateSlot> updates);

                /**
                 * @brief Stores the parts assemble() checked and compiled.
                 * @param step The training step; never null.
                 * @param rows The first extent of the inputs.
                 * @param input What the input leaf reads from a batch.
                 * @param outputs The tensor of every output the model published.
                 */
                Trainer(SushiBLAS::Engine& engine, Graph::TensorPool& pool, Binding binding,
                        const Objective& objective, TrainerOptions options,
                        std::unique_ptr<TrainingStep> step, int64_t rows,
                        const Data::InputDescription& input,
                        std::vector<StepOutput> outputs) noexcept;

                /// @brief The engine the batch copies are recorded on; not owned.
                SushiBLAS::Engine* engine_;

                /// @brief The pool that holds the step's tensors; not owned.
                Graph::TensorPool* pool_;

                /// @brief The values the loop reads and the leaves the objective fills.
                Binding binding_;

                /// @brief The objective that feeds and measures every step; not owned.
                const Objective* objective_;

                /// @brief The options in force.
                TrainerOptions options_;

                /// @brief The training step, which owns the compiled replay; never null.
                std::unique_ptr<TrainingStep> step_;

                /// @brief The number of samples per batch.
                int64_t rows_ = 0;

                /// @brief The input the step reads, matched against a dataset by fit().
                Data::InputDescription input_;

                /// @brief The tensor of every output the model published, read by measure().
                std::vector<StepOutput> outputs_;

                /// @brief The number of steps that have run.
                std::size_t steps_ = 0;
        };

    } // namespace Train
} // namespace SushiAI
