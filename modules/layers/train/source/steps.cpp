/**************************************************************************/
/* steps.cpp                                                              */
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
 * @file steps.cpp
 * @brief Implements the parts of the step builders that do not depend on the model's type.
 * @author Mustafa Garip
 */

#include <SushiAI/train/steps.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/autograd/backward.hpp>
#include <SushiAI/core/common.hpp>
#include <SushiAI/core/result.hpp>
#include <SushiAI/core/seam.hpp>
#include <SushiAI/core/shape.hpp>
#include <SushiAI/core/status.hpp>
#include <SushiAI/graph/builder.hpp>
#include <SushiAI/graph/cast_insertion.hpp>
#include <SushiAI/graph/fusion.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/graph/memory_plan.hpp>
#include <SushiAI/graph/tensor_pool.hpp>
#include <SushiAI/nn/model_outputs.hpp>
#include <SushiAI/nn/node.hpp>
#include <SushiAI/nn/throwing_bridge.hpp>
#include <SushiAI/nn/trace.hpp>
#include <SushiAI/train/objective.hpp>
#include <SushiAI/train/training_graph.hpp>

namespace SushiAI
{
    namespace Train
    {
        namespace
        {
            /** @brief Views @p values as the span the passes read. */
            SushiRuntime::span<const Graph::ValueID> view_of(
                const std::vector<Graph::ValueID>& values) noexcept
            {
                return SushiRuntime::span<const Graph::ValueID>(values.data(), values.size());
            }

            /**
             * @brief Runs the cast pass, then the fusion pass, over @p graph.
             * @param observed The values the step reads by id; the caller's own are appended.
             * @return The failure of a pass, or OutOfMemory when @p observed cannot grow.
             */
            [[nodiscard]] Status run_passes(Graph::Graph& graph,
                                            std::vector<Graph::ValueID>& observed,
                                            const GraphOptions& options,
                                            Graph::CastReport& casts,
                                            Graph::FusionReport& fusion)
            {
                SA_TRY(::SushiAI::Detail::guard(::SushiAI::Detail::SeamPhase::Allocate,
                                                [&]() -> Status
                {
                    observed.insert(observed.end(), options.fusion.observed.begin(),
                                    options.fusion.observed.end());
                    return Status();
                }));

                Graph::FusionOptions fusion_options = options.fusion;
                fusion_options.observed = view_of(observed);

                Graph::CastOptions cast_options;
                cast_options.policy = options.precision;
                cast_options.observed = fusion_options.observed;

                SA_TRY_ASSIGN(casts, NN::Detail::bridge_throwing([&graph, &cast_options]()
                {
                    return Graph::insert_casts(graph, cast_options);
                }));
                SA_TRY_ASSIGN(fusion, NN::Detail::bridge_throwing([&graph, &fusion_options]()
                {
                    return Graph::fuse(graph, fusion_options);
                }));

                return Status();
            }

            /** @brief Appends the id of every entry of @p outputs to @p observed. */
            void append_outputs(const NN::ModelOutputs& outputs,
                                std::vector<Graph::ValueID>& observed)
            {
                for (const NN::ModelOutputs::Entry& entry : outputs.all())
                    observed.push_back(entry.second);
            }
        } // namespace

        Result<Shape> batch_shape(int64_t batch, const Shape& sample)
        {
            SA_FAIL_IF(batch <= 0, StatusCode::InvalidDescription,
                       "batch_shape: the batch is {}; it must be positive", batch);
            SA_FAIL_IF(sample.rank() >= K_MAX_RANK, StatusCode::InvalidDescription,
                       "batch_shape: a sample of rank {} leaves no extent for the batch; a "
                       "shape holds at most {}",
                       sample.rank(), K_MAX_RANK);

            std::array<int64_t, static_cast<std::size_t>(K_MAX_RANK)> extents{};
            extents[0] = batch;
            for (int32_t axis = 0; axis < sample.rank(); ++axis)
                extents[static_cast<std::size_t>(axis) + 1] = sample[axis];

            return Shape(SushiRuntime::span<const int64_t>(
                extents.data(), static_cast<std::size_t>(sample.rank()) + 1));
        }

        Status retain(const NN::BufferUpdateList& updates,
                      std::vector<Graph::ValueID>& observed,
                      std::vector<Graph::ValueID>& pinned)
        {
            return ::SushiAI::Detail::guard(::SushiAI::Detail::SeamPhase::Allocate,
                                            [&]() -> Status
            {
                const SushiRuntime::span<const Graph::ValueID> sources = updates.sources();
                observed.insert(observed.end(), sources.begin(), sources.end());
                pinned.insert(pinned.end(), sources.begin(), sources.end());
                return Status();
            });
        }

        Status fill_host_tables(const NN::HostTableList& tables, Graph::TensorPool& pool)
        {
            for (const NN::HostTable& table : tables.tables())
            {
                SushiBLAS::Tensor& leaf = pool.tensor(table.leaf);
                const bool is_single = leaf.dtype == DType::Float32;
                SA_FAIL_IF((!is_single && leaf.dtype != DType::Float64) ||
                               static_cast<std::size_t>(leaf.num_elements) != table.values.size(),
                           StatusCode::Incompatible,
                           "the leaf of a host table holds {} {} elements, but the table holds "
                           "{} values for a float32 or a float64 leaf",
                           leaf.num_elements, dtype_name(leaf.dtype), table.values.size());

                for (std::size_t at = 0; at < table.values.size(); ++at)
                {
                    if (is_single)
                        leaf.data_as<float>()[at] = static_cast<float>(table.values[at]);
                    else
                        leaf.data_as<double>()[at] = table.values[at];
                }
            }

            return Status();
        }

        namespace Detail
        {
            Result<Graph::ValueID> declare_input(const Graph::GraphBuilder& builder,
                                                 const Shape& batch_shape, DType dtype)
            {
                return NN::Detail::bridge_throwing([&builder, &batch_shape, dtype]()
                {
                    return builder.input(batch_shape, "inputs", dtype);
                });
            }

            Result<NN::ModelOutputs> single_output(Graph::ValueID id)
            {
                NN::ModelOutputs outputs;
                SA_TRY(::SushiAI::Detail::guard(::SushiAI::Detail::SeamPhase::Allocate,
                                                [&outputs, id]() -> Status
                {
                    return outputs.add(std::string(NN::K_SINGLE_PORT_NAME), id);
                }));

                return Result<NN::ModelOutputs>(std::move(outputs));
            }

            Status declare_objective(TrainingGraph& built, const Graph::GraphBuilder& builder,
                                     const Objective& objective)
            {
                SA_TRY(NN::Detail::bridge_throwing([&built, &builder, &objective]()
                {
                    objective.declare(builder, built.leaves);
                    return true;
                }));

                return Status();
            }

            Status add_loss(TrainingGraph& built, const Graph::GraphBuilder& builder,
                            const Objective& objective)
            {
                SA_TRY_ASSIGN(built.spec.loss,
                              NN::Detail::bridge_throwing([&built, &builder, &objective]()
                {
                    return objective.loss(builder, built.graph, built.outputs, built.leaves);
                }));

                if (built.leaves.has("targets"))
                {
                    SA_TRY_ASSIGN(built.spec.targets, NN::Detail::bridge_throwing([&built]()
                    {
                        return built.leaves.at("targets");
                    }));
                }

                // logits stays unset for a model that publishes more than one value.
                const std::vector<NN::ModelOutputs::Entry>& published = built.outputs.all();
                built.spec.logits =
                    published.size() == 1 ? published.front().second : Graph::K_NO_VALUE;

                return Status();
            }

            Status differentiate_and_finish(TrainingGraph& built,
                                            const Graph::GraphBuilder& builder,
                                            const GraphOptions& options)
            {
                for (const ParameterBinding& parameter : built.parameters)
                {
                    SA_FAIL_IF(parameter.value == Graph::K_NO_VALUE,
                               StatusCode::InvalidDescription,
                               "build_training_graph: the parameter '{}' is declared and has "
                               "no leaf in the traced graph",
                               parameter.name);
                }

                std::vector<Graph::ValueID> wrt;
                SA_TRY(::SushiAI::Detail::guard(::SushiAI::Detail::SeamPhase::Allocate,
                                                [&built, &wrt]() -> Status
                {
                    wrt.reserve(built.parameters.size());
                    for (const ParameterBinding& parameter : built.parameters)
                        wrt.push_back(parameter.value);
                    return Status();
                }));

                // The loss-scale leaf is declared before differentiation, which reads it.
                if (options.loss_scaling)
                {
                    SA_TRY_ASSIGN(built.loss_scale, NN::Detail::bridge_throwing([&builder]()
                    {
                        return builder.input(Shape{1}, "loss_scale", DType::Float32);
                    }));
                }

                SA_TRY_ASSIGN(const std::vector<Graph::ValueID> gradients,
                              NN::Detail::bridge_throwing([&built, &wrt]()
                {
                    return Autograd::differentiate(built.graph, built.spec.loss, view_of(wrt),
                                                   built.loss_scale);
                }));

                for (std::size_t i = 0; i < built.parameters.size(); ++i)
                    built.parameters[i].gradient = gradients[i];

                std::vector<Graph::ValueID> observed;
                SA_TRY(::SushiAI::Detail::guard(::SushiAI::Detail::SeamPhase::Allocate,
                                                [&built, &observed]() -> Status
                {
                    observed.push_back(built.spec.inputs);
                    if (built.spec.targets != Graph::K_NO_VALUE)
                        observed.push_back(built.spec.targets);
                    observed.push_back(built.spec.loss);
                    if (built.loss_scale != Graph::K_NO_VALUE)
                        observed.push_back(built.loss_scale);
                    append_outputs(built.outputs, observed);

                    // The host reads the loss and the outputs after the run.
                    built.pinned.push_back(built.spec.loss);
                    append_outputs(built.outputs, built.pinned);
                    return Status();
                }));
                SA_TRY(retain(built.updates, observed, built.pinned));

                return run_passes(built.graph, observed, options, built.casts, built.fusion);
            }

            Status finish_traced_step(TracedStep& step, const GraphOptions& options)
            {
                std::vector<Graph::ValueID> observed;
                SA_TRY(::SushiAI::Detail::guard(::SushiAI::Detail::SeamPhase::Allocate,
                                                [&step, &observed]() -> Status
                {
                    observed.push_back(step.input);
                    append_outputs(step.outputs, observed);

                    // A fed leaf is written by id, and a carried pair is read by id after the run.
                    for (const NN::FedLeaf& leaf : step.fed.leaves())
                        observed.push_back(leaf.value);

                    for (const NN::CarriedValue& carried : step.carried.values())
                    {
                        observed.push_back(carried.leaf);
                        observed.push_back(carried.next);
                    }

                    return Status();
                }));

                return run_passes(step.graph, observed, options, step.casts, step.fusion);
            }

            Status adopt_plan(const Graph::Graph& graph, Graph::TensorPool& pool,
                              const Graph::PlanOptions& options)
            {
                if (pool.has_plan())
                    return Status();

                SA_TRY_ASSIGN(Graph::MemoryPlan plan,
                              NN::Detail::bridge_throwing([&graph, &options]()
                {
                    return Graph::MemoryPlan::build(graph, options);
                }));

                return pool.adopt(std::move(plan));
            }

        } // namespace Detail
    } // namespace Train
} // namespace SushiAI
