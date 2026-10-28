/**************************************************************************/
/* steps.hpp                                                              */
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
 * @file steps.hpp
 * @brief Declares the builders that trace a model into the graph of each kind of step.
 * @author Mustafa Garip
 */

#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/dtype/dtype.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/seam/seam.hpp>
#include <SushiAI/shape/shape.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/graph/builder.hpp>
#include <SushiAI/graph/cast_insertion.hpp>
#include <SushiAI/graph/fusion.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/graph/memory_plan.hpp>
#include <SushiAI/graph/precision.hpp>
#include <SushiAI/graph/tensor_pool.hpp>
#include <SushiAI/nn/model_outputs.hpp>
#include <SushiAI/nn/module.hpp>
#include <SushiAI/nn/trace.hpp>
#include <SushiAI/train/objective.hpp>
#include <SushiAI/train/training_graph.hpp>

namespace SushiAI
{
    namespace Train
    {
        /** @brief Names the model input among the leaves a run feeds. */
        inline constexpr std::string_view K_MODEL_INPUT_NAME = "inputs";

        /** @brief States how a step's graph is built. */
        struct GraphOptions
        {
            /** @brief Names the element type of the input leaf. */
            DType dtype = DType::Float32;

            /** @brief Configures the fusion pass; its observed list is added to the step's. */
            Graph::FusionOptions fusion{};

            /** @brief Points at the precision policy of the cast pass, or null for uniform. */
            const Graph::PrecisionPolicy* precision = nullptr;

            /** @brief Asks for the loss-scale leaf; read by build_training_graph() only. */
            bool loss_scaling = false;

            /** @brief Seeds the random sites of a Training trace, such as each dropout. */
            std::uint64_t seed = 0;

            /** @brief Whether stored K_IM2COL results are recomputed at their later readers. */
            bool rematerialise = true;
        };

        /** @brief Holds one forward-only graph and what a step reads from it. */
        struct TracedStep
        {
            /** @brief Holds the forward IR. */
            Graph::Graph graph;

            /** @brief Names the minibatch input leaf, rewritten per step. */
            Graph::ValueID input = Graph::K_NO_VALUE;

            /** @brief Holds every value the model published, by name. */
            NN::ModelOutputs outputs;

            /** @brief Holds what the cast pass converted, and where. */
            Graph::CastReport casts;

            /** @brief Holds what the fusion pass did to @c graph, and what it refused. */
            Graph::FusionReport fusion;

            /** @brief Lists the leaves a carrying trace declared beside @c input, in order. */
            NN::FedLeafList fed;

            /** @brief Lists the values the step keeps from a run to the next. */
            NN::CarriedValueList carried;

            /** @brief Lists the leaves the step fills once with numbers of the host. */
            NN::HostTableList tables;
        };

        /**
         * @brief Returns [batch, sample...].
         * @return InvalidDescription for a batch that is not positive or a sample of
         *         K_MAX_RANK extents.
         */
        [[nodiscard]] Result<Shape> batch_shape(int64_t batch, const Shape& sample);

        /**
         * @brief Appends the source of every update to both lists.
         * @return OutOfMemory when a list cannot grow.
         */
        [[nodiscard]] Status retain(const NN::BufferUpdateList& updates,
                                    std::vector<Graph::ValueID>& observed,
                                    std::vector<Graph::ValueID>& pinned);

        /**
         * @brief Writes the values of every table into its leaf, on the host.
         * @pre @p pool holds the leaf of every table and the engine holds no recorded task.
         * @return Success, or Incompatible for a leaf that is not the float32 or float64
         *         tensor of as many elements as its table holds.
         */
        [[nodiscard]] Status fill_host_tables(const NN::HostTableList& tables,
                                              Graph::TensorPool& pool);

        namespace Detail
        {
            /** @brief Detects trace(const Trace&, ValueID) returning Result<ModelOutputs> on M. */
            template<typename M, typename = void>
            struct TracesToOutputs : std::false_type
            {

            };

            /** @brief Holds for an M whose trace publishes its outputs by name. */
            template<typename M>
            struct TracesToOutputs<M, std::enable_if_t<std::is_same<
                decltype(std::declval<M&>().trace(
                    std::declval<const NN::Trace&>(),
                    std::declval<Graph::ValueID>())),
                Result<NN::ModelOutputs>>::value>>
                : std::true_type
            {

            };

            /** @brief Declares the input leaf of a step over @p batch_shape and returns it. */
            [[nodiscard]] Result<Graph::ValueID> declare_input(const Graph::GraphBuilder& builder,
                                                               const Shape& batch_shape,
                                                               DType dtype);

            /** @brief Returns a set that holds @p id under NN::K_SINGLE_PORT_NAME. */
            [[nodiscard]] Result<NN::ModelOutputs> single_output(Graph::ValueID id);

            /** @brief Records the leaves @p objective declares in @c built.leaves. */
            [[nodiscard]] Status declare_objective(TrainingGraph& built,
                                                   const Graph::GraphBuilder& builder,
                                                   const Objective& objective);

            /**
             * @brief Appends the loss over @c built.outputs and fills the rest of @c built.spec.
             * @pre @c built.outputs holds what the model published.
             */
            [[nodiscard]] Status add_loss(TrainingGraph& built,
                                          const Graph::GraphBuilder& builder,
                                          const Objective& objective);

            /**
             * @brief Differentiates the loss, then runs the cast pass and the fusion pass.
             * @pre add_loss() succeeded and @c built.parameters lists the model's parameters.
             * @return InvalidDescription naming a parameter that has no leaf in the graph.
             */
            [[nodiscard]] Status differentiate_and_finish(TrainingGraph& built,
                                                          const Graph::GraphBuilder& builder,
                                                          const GraphOptions& options);

            /**
             * @brief Runs the cast pass and the fusion pass over a forward-only graph.
             * @pre @c step.input, @c step.outputs, @c step.fed and @c step.carried are set.
             */
            [[nodiscard]] Status finish_traced_step(TracedStep& step, const GraphOptions& options);

            /**
             * @brief Adopts a plan of @p graph into @p pool unless the pool holds one.
             * @param options Its pinned span must stay valid for the length of the call.
             * @param lender A pool whose arenas @p pool borrows, or nullptr.
             * @return The failure of the planner or of the pool.
             */
            [[nodiscard]] Status adopt_plan(const Graph::Graph& graph, Graph::TensorPool& pool,
                                            const Graph::PlanOptions& options,
                                            const Graph::TensorPool* lender = nullptr);

            /**
             * @brief Traces @p model once and returns what it published.
             *
             * A model whose trace returns its outputs is called as it is. Any other Traceable
             * goes through NN::trace_layer and publishes one value as NN::K_SINGLE_PORT_NAME.
             * @return The failure of the model, or the one a layer recorded on @p trace.
             */
            template<typename M>
            [[nodiscard]] Result<NN::ModelOutputs> trace_outputs(M& model, const NN::Trace& trace,
                                                                 Graph::ValueID input)
            {
                if constexpr (TracesToOutputs<M>::value)
                {
                    return model.trace(trace, input);
                }
                else
                {
                    const Graph::ValueID traced = NN::trace_layer(model, trace, input);
                    SA_TRY(trace.status());

                    return single_output(traced);
                }
            }

            /**
             * @brief Lists every declared parameter of @p model with its leaf in @p graph.
             * @return The bindings in walk order, each with the role its parameter declared;
             *         a parameter with no leaf holds K_NO_VALUE.
             */
            template<typename M>
            [[nodiscard]] Result<std::vector<ParameterBinding>> list_parameters(
                M& model, const Graph::Graph& graph)
            {
                std::vector<ParameterBinding> parameters;
                SA_TRY(::SushiAI::Detail::guard(::SushiAI::Detail::SeamPhase::Allocate,
                                                [&]() -> Status
                {
                    NN::for_each_parameter(model, [&](const std::string& name,
                                                      NN::Parameter& parameter)
                    {
                        parameters.push_back(ParameterBinding{name, parameter.id_in(graph),
                                                              Graph::K_NO_VALUE,
                                                              parameter.role()});
                    });
                    return Status();
                }));

                return Result<std::vector<ParameterBinding>>(std::move(parameters));
            }

        } // namespace Detail

        /**
         * @brief Traces @p model in Training mode, appends the loss and differentiates it.
         * @param batch_shape The extents of the input leaf, batch first.
         * @return The graph after the cast pass and the fusion pass, or the failure of the
         *         model, the objective or a pass.
         */
        template<typename M>
        [[nodiscard]] Result<TrainingGraph> build_training_graph(M& model,
                                                                 const Objective& objective,
                                                                 const Shape& batch_shape,
                                                                 GraphOptions options = {})
        {
            TrainingGraph built;
            const Graph::GraphBuilder builder(built.graph);

            SA_TRY_ASSIGN(built.spec.inputs,
                          Detail::declare_input(builder, batch_shape, options.dtype));

            // The objective's leaves precede the model's nodes, which fixes their value ids.
            SA_TRY(Detail::declare_objective(built, builder, objective));

            {
                NN::Trace trace = NN::Trace::training(builder, built.updates, options.seed);
                trace.collect_tables(built.tables);
                SA_TRY_ASSIGN(built.outputs,
                              Detail::trace_outputs(model, trace, built.spec.inputs));
            }

            SA_TRY(Detail::add_loss(built, builder, objective));
            SA_TRY_ASSIGN(built.parameters, Detail::list_parameters(model, built.graph));
            SA_TRY(Detail::differentiate_and_finish(built, builder, options));

            return Result<TrainingGraph>(std::move(built));
        }

        /**
         * @brief Traces @p model in Training mode with every buffer update dropped.
         * @param batch_shape The extents of the input leaf, batch first.
         * @return The forward graph after the cast pass and the fusion pass.
         */
        template<typename M>
        [[nodiscard]] Result<TracedStep> build_probe_graph(M& model, const Shape& batch_shape,
                                                           GraphOptions options = {})
        {
            TracedStep step;
            const Graph::GraphBuilder builder(step.graph);

            SA_TRY_ASSIGN(step.input, Detail::declare_input(builder, batch_shape, options.dtype));

            {
                NN::Trace trace = NN::Trace::training(builder, NN::discard_updates(),
                                                      options.seed);
                trace.collect_tables(step.tables);
                SA_TRY_ASSIGN(step.outputs, Detail::trace_outputs(model, trace, step.input));
            }

            SA_TRY(Detail::finish_traced_step(step, options));

            return Result<TracedStep>(std::move(step));
        }

        /**
         * @brief Traces @p model in Evaluation mode.
         * @param batch_shape The extents of the input leaf, batch first.
         * @return The forward graph after the cast pass and the fusion pass.
         */
        template<typename M>
        [[nodiscard]] Result<TracedStep> build_evaluation_graph(M& model, const Shape& batch_shape,
                                                                GraphOptions options = {})
        {
            TracedStep step;
            const Graph::GraphBuilder builder(step.graph);

            SA_TRY_ASSIGN(step.input, Detail::declare_input(builder, batch_shape, options.dtype));

            {
                NN::Trace trace = NN::Trace::evaluation(builder);
                trace.collect_tables(step.tables);
                SA_TRY_ASSIGN(step.outputs, Detail::trace_outputs(model, trace, step.input));
            }

            SA_TRY(Detail::finish_traced_step(step, options));

            return Result<TracedStep>(std::move(step));
        }

        namespace Detail
        {
            /**
             * @brief Builds a forward graph through a carrying trace.
             * @param trace_model Called as trace_model(const NN::Trace&, Graph::ValueID) and
             *        returns Result<NN::ModelOutputs>.
             * @return The graph after the cast pass and the fusion pass, with what the trace
             *         declared in @c fed and @c carried.
             */
            template<typename TraceModel>
            [[nodiscard]] Result<TracedStep> build_carrying(const Shape& batch_shape,
                                                            const GraphOptions& options,
                                                            TraceModel&& trace_model)
            {
                TracedStep step;
                const Graph::GraphBuilder builder(step.graph);

                SA_TRY_ASSIGN(step.input, declare_input(builder, batch_shape, options.dtype));

                {
                    NN::Trace trace = NN::Trace::carrying(builder, step.fed, step.carried);
                    trace.collect_tables(step.tables);
                    SA_TRY_ASSIGN(step.outputs,
                                  std::forward<TraceModel>(trace_model)(trace, step.input));
                }

                SA_TRY(finish_traced_step(step, options));

                return Result<TracedStep>(std::move(step));
            }

        } // namespace Detail

        /**
         * @brief Traces @p model in Evaluation mode for a step that carries state between runs.
         * @param batch_shape The extents of the input leaf, batch first.
         * @return The forward graph after the cast pass and the fusion pass.
         */
        template<typename M>
        [[nodiscard]] Result<TracedStep> build_carrying_graph(M& model, const Shape& batch_shape,
                                                              GraphOptions options = {})
        {
            return Detail::build_carrying(batch_shape, options,
                                          [&model](const NN::Trace& trace, Graph::ValueID input)
            {
                return Detail::trace_outputs(model, trace, input);
            });
        }

        /**
         * @brief Traces the incremental form of @p model for a step that carries state.
         * @tparam M A model with trace_incremental(const NN::Trace&, Graph::ValueID), as
         *         NN::ModelGraph has.
         * @param slice_shape The extents of the input leaf: one slice of the model input.
         * @return The forward graph after the cast pass and the fusion pass.
         */
        template<typename M>
        [[nodiscard]] Result<TracedStep> build_incremental_graph(M& model,
                                                                 const Shape& slice_shape,
                                                                 GraphOptions options = {})
        {
            return Detail::build_carrying(slice_shape, options,
                                          [&model](const NN::Trace& trace, Graph::ValueID input)
            {
                return model.trace_incremental(trace, input);
            });
        }

        /**
         * @brief Traces @p module into a cross-entropy training graph over [batch, features].
         * @param classes How many classes the one-hot target leaf spans.
         * @return What build_training_graph() returns for a CrossEntropyObjective.
         */
        template<typename M>
        [[nodiscard]] Result<TrainingGraph> build_classifier(
            M& module, int64_t batch, int64_t features, int64_t classes,
            DType dtype = DType::Float32, Graph::FusionOptions fusion = {},
            const Graph::PrecisionPolicy* precision = nullptr, bool loss_scaling = false)
        {
            const CrossEntropyObjective objective(batch, classes, dtype);

            GraphOptions options;
            options.dtype = dtype;
            options.fusion = fusion;
            options.precision = precision;
            options.loss_scaling = loss_scaling;

            const std::array<int64_t, 2> input_extents{batch, features};
            SA_TRY_ASSIGN(const Shape inputs,
                          Shape::make(SushiRuntime::span<const int64_t>(input_extents.data(),
                                                                        input_extents.size())));

            return build_training_graph(module, objective, inputs, options);
        }

    } // namespace Train
} // namespace SushiAI
