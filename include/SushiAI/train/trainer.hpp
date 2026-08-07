/**************************************************************************/
/* trainer.hpp                                                            */
/**************************************************************************/
/*                          This file is part of:                         */
/*                                 SushiAI                                */
/*                https://github.com/SushiSystems/SushiAI                 */
/*                         https://sushisystems.io                        */
/**************************************************************************/
/* Copyright (c) 2026-present Mustafa Garip & Sushi Systems               */
/*                                                                        */
/* Licensed under the Apache License, Version 2.0 (the "License");        */
/* you may not use this file except in compliance with the License.       */
/* You may obtain a copy of the License at                                */
/*                                                                        */
/*     http://www.apache.org/licenses/LICENSE-2.0                         */
/*                                                                        */
/* Unless required by applicable law or agreed to in writing, software    */
/* distributed under the License is distributed on an "AS IS" BASIS,      */
/* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or        */
/* implied. See the License for the specific language governing           */
/* permissions and limitations under the License.                         */
/**************************************************************************/

#pragma once

/**
 * @file trainer.hpp
 * @brief The loop: build once, compile once, replay per batch.
 *
 * @section shape What a step actually costs
 *
 * The forward pass, the backward pass, the loss **and the optimizer's update**
 * are one compiled plan, and that plan is compiled exactly once for the life of
 * the trainer - @c compile_count() reports it and the test suite asserts it.
 *
 * The update joined the plan when it became replay-invariant: fused into one
 * kernel per parameter whose per-step numbers are read from a device buffer
 * rather than captured by value (optimizer.hpp and optim/step_scalars.hpp
 * argue it in full). @c TrainerOptions::optimizer_placement decides, and
 * @c IN_PLAN is refused loudly rather than silently downgraded if the optimizer
 * cannot promise replay-invariance.
 *
 * One thing still sits outside the plan, and it is visible rather than hidden:
 *
 * - **Feeding the batch.** The compiled plan reads two fixed tensors, because a
 *   recorded SushiBLAS call captures raw pointers and there is no way to
 *   re-point one between replays. So each step copies the batch into those
 *   tensors with two @c blas().copy() calls and one flush. SushiRuntime's
 *   @c LateBindings does not fix this: it re-points a node's iteration count,
 *   base index and enabled flag between replays, not the addresses its kernel
 *   captured. Fusing does not help either - the two copies read a *different
 *   source buffer* every step, which is a pointer, not a scalar, and a pointer
 *   is exactly what cannot change. What would fix it is a late binding for an
 *   operand address, or a dataset that stages every batch through the two
 *   buffers the plan already owns.
 *
 * Accuracy costs one host read of the logits per step, and is optional for
 * that reason. It needs a per-row argmax; SushiBLAS's argmax is whole-tensor,
 * and a metric does not justify a new device reduction.
 *
 * @section profiling Profiling
 *
 * Per-node timings come from the runtime's own profiler, which only records
 * anything when the @c RuntimeContext was built with @c RuntimeSettings::profiling
 * set. With it off the report carries no rows and says so, rather than
 * printing a table of zeroes that reads like the work took no time.
 */

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
            Graph::ValueId inputs = Graph::NO_VALUE;

            /** @brief The [N, C] one-hot target leaf, rewritten per step. */
            Graph::ValueId targets = Graph::NO_VALUE;

            /** @brief The model's [N, C] logits, read for accuracy. */
            Graph::ValueId logits = Graph::NO_VALUE;

            /** @brief The 1-element loss that was differentiated. */
            Graph::ValueId loss = Graph::NO_VALUE;
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
            Graph::ValueId value = Graph::NO_VALUE;

            /** @brief Its gradient's value id; Graph::NO_VALUE if none flows. */
            Graph::ValueId gradient = Graph::NO_VALUE;
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
             *
             * @c Graph::NO_VALUE unless @c build_classifier was asked for loss
             * scaling. When it is set, the tensor behind it is what
             * @c Optim::LossScaler is constructed over - the pool materialises
             * it like any other input leaf, and the scaler rewrites it from the
             * host between replays.
             */
            Graph::ValueId loss_scale = Graph::NO_VALUE;

            /**
             * @brief The leaves the objective declared, by role.
             *
             * @c StepSpec still names @c targets, because @c Train::Trainer
             * feeds exactly that tensor per step. Wiring the trainer to feed
             * arbitrary declared leaves is a separate change with its own
             * argument; this is the shape it will use. Recorded as debt in
             * docs/ARCHITECTURE.md alongside @c Dataset::num_classes().
             */
            ObjectiveLeaves leaves;

            /** @brief What the cast pass converted, and where. */
            Graph::CastReport casts;

            /** @brief What the fusion pass did to @c graph, and what it refused. */
            Graph::FusionReport fusion;
        };

        /**
         * @brief Traces @p module into a full training graph and differentiates it.
         *
         * Declares the input leaf, lets @p objective declare whatever leaves it
         * needs, runs the model, asks @p objective for the scalar to
         * differentiate, then differentiates with respect to every parameter
         * the generic walk yields - so a parameter cannot be left out of the
         * backward pass by forgetting to mention it.
         *
         * The objective is a parameter rather than a hardcoded loss because a
         * driver written against a hardcoded one is rewritten at the first
         * model that is not a classifier. See train/objective.hpp.
         *
         * Then the two IR-to-IR passes, in this order and not the other one:
         *
         * 1. **Cast insertion**, so that every node's operands carry the
         *    element type that node computes in. After differentiation, because
         *    a backward node needs casting as much as a forward one and there
         *    are none to cast before the transform runs; before fusion, because
         *    a conversion dropped into the middle of a fusable chain would
         *    break it. @c graph/cast_insertion.hpp argues both.
         * 2. **Fusion**, on the joint forward-and-backward graph, because a
         *    forward node is only fusable if the backward pass no longer needs
         *    what the fused form would stop writing - a question that cannot be
         *    asked before the backward pass exists.
         *
         * The four values @c StepSpec names are handed to *both* passes as
         * observed, so an activation the training loop reads by id is neither
         * fused away underneath it nor quietly retyped under it; parameters and
         * gradients are protected by their lifetime class and need no listing.
         * Value ids survive both passes, so everything already recorded above -
         * including the ids inside the model's own @c NN::Parameter members -
         * keeps meaning what it meant.
         *
         * @param module The model to trace; its parameters are declared here.
         * @param objective What the model is being trained to minimise; read
         *        only during this call.
         * @param batch Samples per step.
         * @param features Input features per sample.
         * @param dtype The element type for the whole graph. Under a mixed
         *        policy this is the *master* type: parameters, gradients and
         *        optimizer state keep it, and only derived values narrow.
         * @param fusion What the fusion pass is allowed to do; @c enabled=false
         *        gives the unfused graph, which is the reference a byte-for-byte
         *        comparison needs. @c FusionOptions::observed is filled in here
         *        and any value the caller listed is added to it.
         * @param precision Which element type each operation computes in, or
         *        null for @c Graph::UniformPrecisionPolicy - the default, under
         *        which the cast pass inserts nothing and the graph is exactly
         *        what it was before mixed precision existed. Read only during
         *        this call.
         * @param loss_scaling Whether to give the backward seed a device-resident
         *        factor. False leaves the graph byte-identical to one built
         *        before loss scaling existed - not merely equivalent: no node is
         *        emitted, so nothing about the plan changes. True declares one
         *        1-element input leaf, reports it as
         *        @c TrainingGraph::loss_scale, and multiplies the seeded
         *        gradient by it; see @c optim/loss_scaler.hpp for what then
         *        writes it and why it cannot be a constant.
         * @return The graph, the value ids, the declared leaves, the
         *         parameter/gradient pairs and both passes' reports.
         * @throws Error If the model's output is not the shape @p objective needs.
         */
        template<typename M>
        TrainingGraph build_graph(M& module, const IObjective& objective,
                                  int64_t batch, int64_t features,
                                  DType dtype = DType::FLOAT32,
                                  Graph::FusionOptions fusion = {},
                                  const Graph::IPrecisionPolicy* precision = nullptr,
                                  bool loss_scaling = false)
        {
            TrainingGraph built;
            const Graph::GraphBuilder builder(built.graph);

            built.spec.inputs = builder.input(Shape{batch, features}, "inputs", dtype);

            // Declared here, between the input leaf and the model's own trace,
            // because that is exactly where the target leaf used to be
            // declared. Value ids are positions, so moving this call would
            // renumber the whole graph - and every regression test that pins a
            // node count or a byte-for-byte result would then be comparing two
            // different graphs while appearing to compare one.
            objective.declare(builder, built.leaves);

            built.spec.logits = module.forward(builder, built.spec.inputs);

            // The objective checks the output's shape, because what shape an
            // output must be is the objective's question: a detector's is not
            // a classifier's.
            built.spec.loss = objective.loss(builder, built.graph, built.spec.logits,
                                             built.leaves);

            // StepSpec still names the target leaf, because Trainer feeds
            // exactly that tensor per step. An objective that declares no such
            // role leaves it NO_VALUE, and such an objective cannot yet be
            // driven by Trainer - see TrainingGraph::leaves.
            if (built.leaves.has("targets")) built.spec.targets = built.leaves.at("targets");

            std::vector<Graph::ValueId> wrt;
            NN::for_each_parameter(module, [&](const std::string& name, NN::Parameter& parameter)
            {
                built.parameters.push_back(ParameterBinding{name, parameter.id(), Graph::NO_VALUE});
                wrt.push_back(parameter.id());
            });

            // Declared before differentiation because the seed rule multiplies
            // by it, and always FLOAT32: it lands in the cross-entropy tail,
            // which the precision policy keeps in the master type precisely so
            // there is exponent range left to scale into.
            if (loss_scaling)
                built.loss_scale = builder.input(Shape{1}, "loss_scale", DType::FLOAT32);

            const std::vector<Graph::ValueId> gradients =
                Autograd::differentiate(built.graph, built.spec.loss,
                                        SushiRuntime::span<const Graph::ValueId>(wrt.data(), wrt.size()),
                                        built.loss_scale);

            for (std::size_t i = 0; i < built.parameters.size(); ++i)
                built.parameters[i].gradient = gradients[i];

            std::vector<Graph::ValueId> observed{
                built.spec.inputs, built.spec.targets, built.spec.logits, built.spec.loss,
            };
            if (built.loss_scale != Graph::NO_VALUE) observed.push_back(built.loss_scale);
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
         * @brief Traces @p module into a cross-entropy classifier's training graph.
         *
         * A thin wrapper over @ref build_graph supplying
         * @ref CrossEntropyObjective, kept because every existing caller and
         * test names it and because a classifier is still the common case.
         * The graph it produces is @ref build_graph's, node for node.
         *
         * @param module The model to trace; its parameters are declared here.
         * @param batch Samples per step.
         * @param features Input features per sample.
         * @param classes How many classes.
         * @param dtype The element type for the whole graph; under a mixed
         *        policy this is the master type.
         * @param fusion What the fusion pass is allowed to do.
         * @param precision Which element type each operation computes in, or
         *        null for the uniform policy.
         * @param loss_scaling Whether to give the backward seed a
         *        device-resident factor.
         * @return The graph, the value ids, the parameter/gradient pairs and
         *         both passes' reports.
         * @throws Error If the model's output is not [batch, classes].
         */
        template<typename M>
        TrainingGraph build_classifier(M& module, int64_t batch, int64_t features,
                                       int64_t classes, DType dtype = DType::FLOAT32,
                                       Graph::FusionOptions fusion = {},
                                       const Graph::IPrecisionPolicy* precision = nullptr,
                                       bool loss_scaling = false)
        {
            const CrossEntropyObjective objective(batch, classes, dtype);
            return build_graph(module, objective, batch, features, dtype, fusion,
                               precision, loss_scaling);
        }

        /**
         * @brief Plans the graph's memory, allocates its leaves, and pairs
         *        parameters with gradients.
         *
         * This is bind time, so this is where the memory planner runs: the
         * graph is final by now (traced, differentiated, nothing left to
         * append) and nothing has been lowered yet, which is the one window in
         * which arena offsets can be decided at all. Every arena the plan asks
         * for is allocated here, once, and reused by every replay afterwards.
         *
         * A pool that has already adopted a plan keeps it, so a caller who
         * wants different @c PlanOptions - or no packing at all - can adopt
         * their own first.
         *
         * @param built The traced graph.
         * @param pool Where the tensors are allocated.
         * @param options How to pack; see Graph::PlanOptions.
         * @return One slot per parameter that actually receives a gradient.
         * @throws Error If a parameter received no gradient at all, which
         *         means it is disconnected from the loss and no optimizer
         *         could ever move it.
         */
        std::vector<Optim::ParameterSlot> bind_parameters(const TrainingGraph& built,
                                                          Graph::TensorPool& pool,
                                                          Graph::PlanOptions options = {});

        /**
         * @enum OptimizerPlacement
         * @brief Whether the parameter update is part of the compiled step.
         */
        enum class OptimizerPlacement
        {
            /**
             * @brief Recorded into the same plan as forward and backward; the default.
             *
             * One compile, one replay, no host round trip between the loss and
             * the update. Requires @c Optim::Optimizer::is_replay_invariant().
             */
            IN_PLAN,

            /**
             * @brief Stepped on its own after every replay, with its own flush.
             *
             * The behaviour before the update could be compiled, and the only
             * option for an optimizer whose recorded work is not replayable.
             */
            PER_STEP,
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
            OptimizerPlacement optimizer_placement = OptimizerPlacement::IN_PLAN;

            /**
             * @brief Which spelling to use for each fused node, or null for this
             *        engine's own device.
             *
             * Read once, in the constructor, and must outlive that call. It is
             * here rather than as another constructor parameter because it is a
             * policy knob and not a collaborator: a caller who does not care
             * about fusion never has to name it, and a test that wants to drive
             * the lowering with a device profile this machine does not have
             * sets one field. See graph/fusion_selector.hpp.
             */
            const Graph::IFusionSelector* fusion_selector = nullptr;

            /**
             * @brief The precision policy the graph was built under, for checking.
             *
             * Set it to the same policy handed to @c build_classifier and the
             * trainer walks the finished graph before lowering anything,
             * refusing one whose operands do not agree with the precision each
             * node computes in. Null skips the check.
             *
             * It is a *check* and not the knob that applies mixed precision,
             * and that is forced rather than stylistic: casts have to be
             * inserted between differentiation and fusion, and by the time a
             * @c Trainer exists both have already run. What is left for this
             * layer to do is the one thing worth doing here - being the last
             * place that can refuse, before the graph turns into recorded
             * device work where a misplaced conversion stops being visible and
             * becomes a model that trains slightly worse.
             *
             * Read once, in the constructor, and must outlive that call.
             */
            const Graph::IPrecisionPolicy* precision = nullptr;
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
             * @brief How many SushiBLAS calls the compiled step is made of.
             *
             * Also its node count. A fused GEMM used to add a host-side
             * ordering node beside its kernel, reported apart from this by an
             * @c ordering_nodes_per_step that no longer exists - SushiBLAS's
             * GEMM seam takes the epilogue's own access set now, so there is
             * nothing left to count.
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
                 * @brief Lowers and compiles the step; nothing else compiles after this.
                 *
                 * @param engine The engine everything runs on; must outlive the trainer.
                 * @param graph The traced, differentiated graph.
                 * @param pool The memory behind it, with parameters already initialised.
                 * @param spec The values the loop reads and writes.
                 * @param optimizer The update rule; must outlive the trainer.
                 * @param options How the loop behaves.
                 * @throws Error If @p spec is incomplete, lowering fails, or
                 *         @p options asks for @c IN_PLAN placement of an
                 *         optimizer that is not replay-invariant.
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
