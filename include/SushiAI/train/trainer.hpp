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
#include <string>
#include <vector>
#include <SushiBLAS/engine.hpp>
#include <SushiRuntime/core/run_report.hpp>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/core/common.hpp>
#include <SushiAI/autograd/backward.hpp>
#include <SushiAI/data/dataset.hpp>
#include <SushiAI/graph/builder.hpp>
#include <SushiAI/graph/cast_insertion.hpp>
#include <SushiAI/graph/compiled_step.hpp>
#include <SushiAI/graph/fusion.hpp>
#include <SushiAI/graph/fusion_selector.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/graph/lowering.hpp>
#include <SushiAI/graph/precision.hpp>
#include <SushiAI/nn/loss.hpp>
#include <SushiAI/nn/module.hpp>
#include <SushiAI/optim/optimizer.hpp>
#include <SushiAI/train/objective.hpp>

namespace SushiAI
{
    namespace Train
    {
        /**
         * @struct StepSpec
         * @brief The four graph values a training loop has to know by name.
         */
        struct StepSpec
        {
            /** @brief The [N, D] minibatch input leaf, rewritten per step. */
            Graph::ValueId inputs = Graph::K_NO_VALUE;

            /** @brief The [N, C] one-hot target leaf, rewritten per step. */
            Graph::ValueId targets = Graph::K_NO_VALUE;

            /** @brief The model's [N, C] logits, read for accuracy. */
            Graph::ValueId logits = Graph::K_NO_VALUE;

            /** @brief The 1-element loss that was differentiated. */
            Graph::ValueId loss = Graph::K_NO_VALUE;
        };

        /**
         * @struct ParameterBinding
         * @brief A parameter's name and the two graph values behind it.
         */
        struct ParameterBinding
        {
            /** @brief The dotted name from the parameter walk, e.g. "0.weight". */
            std::string name;

            /** @brief The parameter's value id. */
            Graph::ValueId value = Graph::K_NO_VALUE;

            /** @brief Its gradient's value id; Graph::NO_VALUE if none flows. */
            Graph::ValueId gradient = Graph::K_NO_VALUE;
        };

        /**
         * @struct TrainingGraph
         * @brief A traced, differentiated model, ready to lower.
         */
        struct TrainingGraph
        {
            /** @brief The forward and backward IR, in one graph. */
            Graph::Graph graph;

            /** @brief The values the loop reads and writes. */
            StepSpec spec;

            /** @brief Every parameter, paired with its gradient. */
            std::vector<ParameterBinding> parameters;

            /**
             * @brief The 1-element leaf the backward seed is multiplied by.
             */
            Graph::ValueId loss_scale = Graph::K_NO_VALUE;

            /**
             * @brief Holds leaves declared by the objective, indexed by role.
             */
            ObjectiveLeaves leaves;

            /** @brief What the cast pass converted, and where. */
            Graph::CastReport casts;

            /** @brief What the fusion pass did to @c graph, and what it refused. */
            Graph::FusionReport fusion;
        };

        /**
         * @brief Traces @p module into a full training graph and differentiates it.
         * @param module The model to trace.
         * @param objective Training loss and metric objective.
         * @param batch Samples per step.
         * @param features Input features per sample.
         * @return The constructed training graph.
         */
        template<typename M>
        TrainingGraph build_graph(M& module, const Objective& objective,
                                  int64_t batch, int64_t features,
                                  DType dtype = DType::Float32,
                                  Graph::FusionOptions fusion = {},
                                  const Graph::PrecisionPolicy* precision = nullptr,
                                  bool loss_scaling = false)
        {
            TrainingGraph built;
            const Graph::GraphBuilder builder(built.graph);

            built.spec.inputs = builder.input(Shape{batch, features}, "inputs", dtype);

            // Objective leaves are declared before model trace to fix value IDs.
            objective.declare(builder, built.leaves);

            built.spec.logits = module.forward(builder, built.spec.inputs);

            // The objective verifies output shape compatibility.
            built.spec.loss = objective.loss(builder, built.graph, built.spec.logits,
                                             built.leaves);

            // Targets leaf binds to StepSpec when present in declared leaves.
            if (built.leaves.has("targets")) built.spec.targets = built.leaves.at("targets");

            std::vector<Graph::ValueId> wrt;
            NN::for_each_parameter(module, [&](const std::string& name, NN::Parameter& parameter)
            {
                built.parameters.push_back(ParameterBinding{name, parameter.id(), Graph::K_NO_VALUE});
                wrt.push_back(parameter.id());
            });

            // Loss scale input is declared before differentiation in master precision.
            if (loss_scaling)
                built.loss_scale = builder.input(Shape{1}, "loss_scale", DType::Float32);

            const std::vector<Graph::ValueId> gradients =
                Autograd::differentiate(built.graph, built.spec.loss,
                                        SushiRuntime::span<const Graph::ValueId>(wrt.data(), wrt.size()),
                                        built.loss_scale);

            for (std::size_t i = 0; i < built.parameters.size(); ++i)
                built.parameters[i].gradient = gradients[i];

            std::vector<Graph::ValueId> observed{
                built.spec.inputs, built.spec.targets, built.spec.logits, built.spec.loss,
            };
            if (built.loss_scale != Graph::K_NO_VALUE) observed.push_back(built.loss_scale);
            observed.insert(observed.end(), fusion.observed.begin(), fusion.observed.end());
            fusion.observed = SushiRuntime::span<const Graph::ValueId>(observed.data(), observed.size());

            Graph::CastOptions casts;
            casts.policy = precision;
            casts.observed = fusion.observed;

            built.casts = Graph::insert_casts(built.graph, casts);
            built.fusion = Graph::fuse(built.graph, fusion);

            return built;
        }

        /**
         * @brief Traces @p module into a cross-entropy classifier training graph.
         * @param module The model to trace.
         * @param batch Samples per step.
         * @param features Input features per sample.
         * @param classes Target class count.
         * @return The constructed training graph.
         */
        template<typename M>
        TrainingGraph build_classifier(M& module, int64_t batch, int64_t features,
                                       int64_t classes, DType dtype = DType::Float32,
                                       Graph::FusionOptions fusion = {},
                                       const Graph::PrecisionPolicy* precision = nullptr,
                                       bool loss_scaling = false)
        {
            const CrossEntropyObjective objective(batch, classes, dtype);
            return build_graph(module, objective, batch, features, dtype, fusion,
                               precision, loss_scaling);
        }

        /**
         * @brief Plans memory, allocates leaves, and pairs parameters with gradients.
         * @param built The traced graph.
         * @param pool Destination tensor pool.
         * @param options Memory packing options.
         * @return Parameter slots paired with gradients.
         */
        std::vector<Optim::ParameterSlot> bind_parameters(const TrainingGraph& built,
                                                          Graph::TensorPool& pool,
                                                          Graph::PlanOptions options = {});

        /**
         * @enum OptimizerPlacement
         * @brief Whether the parameter update is part of the compiled step.
         */
        enum class OptimizerPlacement : std::uint8_t
        {
            /**
             * @brief Recorded into the same plan as forward and backward; the default.
             *
             * One compile, one replay, no host round trip between the loss and
             * the update. Requires @c Optim::Optimizer::is_replay_invariant().
             */
            InPlan,

            /**
             * @brief Stepped on its own after every replay, with its own flush.
             *
             * The behaviour before the update could be compiled, and the only
             * option for an optimizer whose recorded work is not replayable.
             */
            PerStep,
        };

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
         * @brief Owns the compiled step and drives it over a dataset.
         *
         * Non-copyable and non-movable, because it holds a @c CompiledStep,
         * which holds raw node pointers into its own storage.
         */
        class Trainer
        {
            public:
                /**
                 * @brief Lowers and compiles the execution step.
                 * @param engine Execution engine; @p graph Traced graph.
                 * @param pool Tensor pool; @p spec Step boundary values.
                 * @param optimizer Parameter optimizer; @p options Configuration.
                 */
                Trainer(SushiBLAS::Engine& engine,
                        const Graph::Graph& graph,
                        Graph::TensorPool& pool,
                        StepSpec spec,
                        Optim::Optimizer& optimizer,
                        TrainerOptions options = {});

                Trainer(const Trainer&) = delete;
                Trainer& operator=(const Trainer&) = delete;

                /**
                 * @brief Runs one step: feed, replay, read metrics, update.
                 * @param batch The batch to train on.
                 * @return The step's loss and accuracy.
                 * @throws Error If @p batch does not match the compiled shapes.
                 */
                StepMetrics run_step(const Data::Batch& batch);

                /**
                 * @brief Runs @c options.epochs passes over @p dataset.
                 * @param dataset The batches to train on.
                 * @return The run's summary.
                 * @throws Error If the dataset's shapes do not match the compiled step.
                 */
                RunReport fit(const Data::Dataset& dataset);

                /** @brief How many times the step has been compiled; one, always. */
                std::size_t compile_count() const noexcept { return step_.compile_count(); }

                /** @brief How many steps have run. */
                std::size_t step_count() const noexcept { return steps_; }

                /** @brief Whether the optimizer's update is part of the compiled step. */
                bool optimizer_is_in_plan() const noexcept { return step_.has_trailing_work(); }

                /** @brief The compiled step, for a test that wants to inspect it. */
                const Graph::CompiledStep& step() const noexcept { return step_; }

                /** @brief Reads the loss tensor's current value from the host. */
                double current_loss() const;

                /** @brief The options in force. */
                TrainerOptions& options() noexcept { return options_; }

            private:
                double score_accuracy(const Data::Batch& batch) const;

                SushiBLAS::Engine* engine_;
                Graph::TensorPool* pool_;
                StepSpec spec_;
                Optim::Optimizer* optimizer_;
                TrainerOptions options_;
                Graph::CompiledStep step_;
                int64_t rows_ = 0;
                int64_t classes_ = 0;
                std::size_t steps_ = 0;
        };

    } // namespace Train
} // namespace SushiAI
