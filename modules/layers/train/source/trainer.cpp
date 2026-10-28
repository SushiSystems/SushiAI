/**************************************************************************/
/* trainer.cpp                                                            */
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
 * @file trainer.cpp
 * @brief Implements training loop driver, step execution, and parameter binding.
 * @author Mustafa Garip
 */

#include <SushiAI/data/epoch.hpp>
#include <SushiAI/train/trainer.hpp>

#include <cstddef>
#include <string>
#include <utility>
#include <vector>

#include <SushiLog/text/text_format.hpp>
#include <SushiLog/text/value_text.hpp>
#include <SushiBLAS/blas/blas.hpp>
#include <SushiBLAS/engine/task_recorder.hpp>
#include <SushiRuntime/core/span.hpp>

#include <SushiAI/seam/seam.hpp>
#include <SushiAI/execution/execution.hpp>
#include <SushiAI/data/input.hpp>
#include <SushiAI/graph/cast_insertion.hpp>
#include <SushiAI/graph/arena.hpp>
#include <SushiAI/graph/memory_plan.hpp>
#include <SushiAI/nn/model_state.hpp>
#include <SushiAI/profile/profile.hpp>
#include <SushiAI/profile/timelines.hpp>
#include <SushiAI/train/input_match.hpp>
#include <SushiAI/train/parameter_groups.hpp>
#include <SushiAI/train/role_match.hpp>

namespace SushiAI
{
    namespace Train
    {
        namespace
        {
            /** @brief Returns the caller's pins of @p options followed by @p pinned. */
            [[nodiscard]] Result<std::vector<Graph::ValueID>> merged_pins(
                const Graph::PlanOptions& options, const std::vector<Graph::ValueID>& pinned)
            {
                return SushiAI::Detail::guard_value(SushiAI::Detail::SeamPhase::Allocate, [&]()
                {
                    std::vector<Graph::ValueID> merged;
                    merged.reserve(options.pinned.size() + pinned.size());
                    merged.insert(merged.end(), options.pinned.begin(), options.pinned.end());
                    merged.insert(merged.end(), pinned.begin(), pinned.end());
                    return merged;
                });
            }

            /** @brief Returns Incompatible when the plan of @p pool packs a value of @p pinned. */
            [[nodiscard]] Status require_unpacked(const Graph::TensorPool& pool,
                                                  const std::vector<Graph::ValueID>& pinned)
            {
                const Graph::MemoryPlan& plan = pool.arenas().plan();
                for (const Graph::ValueID id : pinned)
                {
                    SA_FAIL_IF(plan.is_packed(plan.buffer_of(id)), StatusCode::Incompatible,
                               "The pool's plan packs value {}, which the step must keep past "
                               "its last reader. Plan the pool with TrainingGraph::pinned, or "
                               "pass a pool that holds no plan.",
                               id);
                }

                return Status();
            }

            /**
             * @brief Materialises a zeroed gradient per parameter and pairs the two tensors.
             * @pre @p pool holds the tensor of every parameter leaf of @p built.
             * @return One slot per parameter, decaying by takes_weight_decay() of its role,
             *         or InvalidDescription for a parameter no gradient reaches, or the
             *         pool's failure.
             */
            [[nodiscard]] Result<std::vector<Optim::ParameterSlot>> pair_with_gradients(
                const TrainingGraph& built, Graph::TensorPool& pool)
            {
                std::vector<Optim::ParameterSlot> slots;
                SA_TRY(SushiAI::Detail::guard(SushiAI::Detail::SeamPhase::Allocate, [&]()
                {
                    slots.reserve(built.parameters.size());
                    return Status();
                }));

                for (const ParameterBinding& binding : built.parameters)
                {
                    SA_FAIL_IF(binding.gradient == Graph::K_NO_VALUE,
                               StatusCode::InvalidDescription,
                               "Parameter '{}' receives no gradient. It is not on any path from "
                               "the loss, so no optimizer could ever move it - either wire it "
                               "into the model or stop listing it as a parameter.",
                               binding.name);

                    const Graph::Value& gradient_value = built.graph.value(binding.gradient);

                    SA_TRY_ASSIGN(SushiBLAS::Tensor& gradient,
                                  pool.materialise(binding.gradient, gradient_value));

                    // Gradients are initialised to zero before the first backward execution.
                    SA_TRY(pool.request_constant(gradient, 0.0));
                    SA_TRY(SushiAI::Detail::guard(SushiAI::Detail::SeamPhase::Allocate, [&]()
                    {
                        slots.push_back(Optim::ParameterSlot{
                            binding.name, pool.tensor(binding.value), gradient,
                            takes_weight_decay(binding.role),
                        });
                        return Status();
                    }));
                }

                // Constant buffers must be flushed before optimizer initialisation.
                SA_TRY(pool.flush_constants());

                return std::move(slots);
            }
        } // namespace

        Result<std::vector<Optim::ParameterSlot>> bind_parameters(const TrainingGraph& built,
                                                                  const NN::ModelState& state,
                                                                  Graph::TensorPool& pool,
                                                                  Graph::PlanOptions options) noexcept
        {
            SA_TRY_ASSIGN(const std::vector<Graph::ValueID> pinned,
                          merged_pins(options, built.pinned));
            options.pinned = SushiRuntime::span<const Graph::ValueID>(pinned.data(),
                                                                      pinned.size());

            SA_TRY(Detail::adopt_plan(built.graph, pool, options));
            SA_TRY(require_unpacked(pool, pinned));

            // The state binds its leaves first, so materialise_leaves() allocates the rest.
            SA_TRY(state.attach(built.graph, pool));
            SA_TRY(pool.materialise_leaves(built.graph));
            SA_TRY(fill_host_tables(built.tables, pool));

            // The pinned loss and update sources get their tensors before the step is compiled.
            if (built.spec.loss != Graph::K_NO_VALUE)
                SA_TRY(pool.materialise(built.spec.loss, built.graph.value(built.spec.loss)));
            for (const Graph::ValueID source : built.updates.sources())
                SA_TRY(pool.materialise(source, built.graph.value(source)));

            return pair_with_gradients(built, pool);
        }

        Result<std::vector<Optim::ParameterSlot>> bind_step_results(const TrainingGraph& built,
                                                                    const NN::ModelState& state,
                                                                    Graph::TensorPool& pool) noexcept
        {
            SA_FATAL_IF(pool.has_plan(),
                        "bind_step_results: the pool holds a plan; use bind_parameters()");
            SA_FAIL_IF(built.spec.loss == Graph::K_NO_VALUE, StatusCode::InvalidDescription,
                       "The training graph names no loss to bind.");

            SA_TRY(state.attach(built.graph, pool));
            SA_TRY(pool.materialise(built.spec.loss, built.graph.value(built.spec.loss)));
            for (const Graph::ValueID source : built.updates.sources())
                SA_TRY(pool.materialise(source, built.graph.value(source)));

            return pair_with_gradients(built, pool);
        }

                std::string RunReport::to_string() const noexcept
        {
            std::string text = SushiLog::format_text(
                "compiles       : {}\n"
                "steps          : {}\n"
                "ops per step   : {}\n",
                compile_count, steps, operations_per_step);

            text += SushiLog::format_text("loss           : {} -> {}\n",
                                          SushiLog::fixed(first_loss, 6),
                                            SushiLog::fixed(last_loss, 6));
            for (const Metric& metric : last_metrics.all())
            {
                text += SushiLog::format_text("final {}: {}\n", SushiLog::left(metric.name, 9),
                                                describe_metric_value(metric));
            }
            return text;
        }

        namespace
        {
            /**
             * @brief Checks the operand types of @p graph against the precision policy.
             * @return Success when no policy is set or every node agrees with it, otherwise
             *         InvalidDescription naming the first node that disagrees.
             */
            [[nodiscard]] Status verify_precision(const Graph::Graph& graph,
                                                  const TrainerOptions& options)
            {
                if (options.precision == nullptr)
                    return Status();

                const std::vector<Graph::DTypeMismatch> mismatches =
                    Graph::find_dtype_mismatches(graph, *options.precision);

                SA_FAIL_IF(!mismatches.empty(), StatusCode::InvalidDescription,
                           "This graph does not agree with the precision policy it was given: "
                           "{} node(s) read or write an element type they do not compute in. "
                           "The first is: {}. Build the graph with the same policy - "
                           "build_classifier()'s `precision` argument runs the cast pass "
                           "between differentiation and fusion, which is the only window in "
                           "which the conversions can be placed.",
                           mismatches.size(), Graph::describe(mismatches.front()));

                return Status();
            }
        } // namespace

        Result<std::unique_ptr<Trainer>> Trainer::make(SushiBLAS::Engine& engine,
                                                       const TrainingGraph& built,
                                                       Graph::TensorPool& pool,
                                                       const Objective& objective,
                                                       Optim::Optimizer& optimizer,
                                                       TrainerOptions options,
                                                       std::vector<BufferUpdateSlot> updates) noexcept
        {
            SA_TRY_ASSIGN(Binding binding,
                          SushiAI::Detail::guard_value(SushiAI::Detail::SeamPhase::Allocate, [&]()
            {
                return Binding{built.spec.inputs, built.spec.loss, built.leaves, built.outputs};
            }));

            return assemble(engine, built.graph, pool, std::move(binding), objective, optimizer,
                            options, std::move(updates));
        }

        Result<std::unique_ptr<Trainer>> Trainer::assemble(
            SushiBLAS::Engine& engine, const Graph::Graph& graph, Graph::TensorPool& pool,
            Binding binding, const Objective& objective, Optim::Optimizer& optimizer,
            TrainerOptions options, std::vector<BufferUpdateSlot> updates)
        {
            SA_FAIL_IF(binding.inputs == Graph::K_NO_VALUE || binding.loss == Graph::K_NO_VALUE,
                       StatusCode::InvalidDescription,
                       "The trainer needs both the inputs and the loss of the training graph");

            const Graph::Value& loss = graph.value(binding.loss);

            SA_FAIL_IF(loss.shape.num_elements() != 1, StatusCode::InvalidDescription,
                       "The loss must be a single element; got {}", loss.shape.to_string());
            SA_FAIL_IF(loss.dtype != DType::Float32 && loss.dtype != DType::Float64,
                       StatusCode::InvalidDescription,
                       "The loss is {}; only float32 and float64 can be read back",
                       dtype_name(loss.dtype));

            const Shape& inputs = graph.value(binding.inputs).shape;
            SA_FAIL_IF(inputs.rank() == 0, StatusCode::InvalidDescription,
                       "The inputs have no batch extent; the trainer needs [rows, sample...]");
            const int64_t rows = inputs.dim(0);
            SA_TRY_ASSIGN(Data::InputDescription input,
                          describe_input_leaf(graph.value(binding.inputs)));

            SA_TRY(verify_precision(graph, options));

            TrainingStepOptions step_options;
            step_options.optimizer_placement = options.optimizer_placement;
            step_options.fusion_selector = options.fusion_selector;

            SA_TRY_ASSIGN(std::unique_ptr<TrainingStep> step,
                          TrainingStep::make(engine, graph, pool, optimizer, std::move(updates),
                                             step_options));

            // The compilation above materialised every value the step writes.
            for (const NN::ModelOutputs::Entry& entry : binding.outputs.all())
            {
                SA_FAIL_IF(!pool.is_bound(entry.second), StatusCode::InvalidDescription,
                           "The compiled step holds no tensor for the model output '{}'",
                           entry.first);
            }

            return SushiAI::Detail::guard_value(SushiAI::Detail::SeamPhase::Allocate, [&]()
            {
                std::vector<StepOutput> outputs;
                outputs.reserve(binding.outputs.all().size());
                for (const NN::ModelOutputs::Entry& entry : binding.outputs.all())
                    outputs.push_back(StepOutput{entry.first, &pool.tensor(entry.second)});

                return std::unique_ptr<Trainer>(
                    new Trainer(engine, pool, std::move(binding), objective, options,
                                std::move(step), rows, input, std::move(outputs)));
            });
        }

        Trainer::Trainer(SushiBLAS::Engine& engine, Graph::TensorPool& pool, Binding binding,
                         const Objective& objective, TrainerOptions options,
                         std::unique_ptr<TrainingStep> step, int64_t rows,
                         const Data::InputDescription& input,
                         std::vector<StepOutput> outputs) noexcept
            : engine_(&engine),
              pool_(&pool),
              binding_(std::move(binding)),
              objective_(&objective),
              options_(options),
              step_(std::move(step)),
              rows_(rows),
              input_(input),
              outputs_(std::move(outputs))
        {

        }

        double Trainer::current_loss() const noexcept
        {
            const SushiBLAS::Tensor& loss = pool_->tensor(binding_.loss);

            switch (loss.dtype)
            {
                case DType::Float32:
                    return static_cast<double>(loss.data_as<float>()[0]);
                case DType::Float64:
                    return loss.data_as<double>()[0];
                default:
                    SA_FATAL("The loss is {}; only float32 and float64 can be read back",
                             dtype_name(loss.dtype));
            }
        }

        Result<StepMetrics> Trainer::run_step(const Data::Batch& batch) noexcept
        {
            SushiBLAS::Tensor& inputs = pool_->tensor(binding_.inputs);

            SA_TRY(check_feed(batch.inputs, inputs, "inputs"));

            // No task is recorded here, so the objective may write its leaves on the host.
            {
                SA_ZONE("host", "feed");
                SA_TRY(objective_->feed(batch, binding_.leaves, *pool_));
            }

            // The minibatch inputs are copied into the fixed address buffer the plan reads.
            {
                SA_ZONE("copy", "input");
                SA_TRY(SushiAI::Detail::record_and_run(*engine_,
                                                        [&](SushiBLAS::TaskRecorder& recorder)
                {
                    SushiBLAS::BLASOps blas(recorder);
                    blas.copy(batch.inputs, inputs);
                    return Status();
                }));
            }

            // The step replays the plan and steps a per-step optimizer itself.
            {
                SA_ZONE("device", "run");
                SA_TRY(step_->run());
            }

            StepMetrics result;
            {
                SA_ZONE("host", "loss");
                result.loss = current_loss();
            }
            if (options_.track_metrics)
            {
                SA_ZONE("host", "measure");
                const StepView view{batch, result.loss,
                                    SushiRuntime::span<const StepOutput>(outputs_.data(),
                                                                         outputs_.size())};
                SA_TRY(objective_->measure(view, result.metrics));
            }

            ++steps_;

            return result;
        }

        Result<RunReport> Trainer::fit(const Data::Dataset& dataset) noexcept
        {
            SA_FAIL_IF(dataset.batch_count() == 0, StatusCode::InvalidDescription,
                       "The dataset holds no batches");
            SA_FAIL_IF(dataset.batch_size() != rows_, StatusCode::Incompatible,
                       "The dataset's batch size is {} but the step was compiled for {}",
                       dataset.batch_size(), rows_);
            SA_TRY(match_input(input_, dataset));
            SA_TRY(match_roles(binding_.leaves.required(), dataset));

            RunReport report;
            report.operations_per_step = step_->step().operation_count();

            SA_ZONE("train", "fit");

            // One Batch for the run, so a dataset that builds its batches reuses the tensors.
            Data::Batch batch;
            bool first = true;
            for (int64_t epoch = 0; epoch < options_.epochs; ++epoch)
            {
                Data::begin_epoch(dataset, epoch);
                for (std::size_t index = 0; index < dataset.batch_count(); ++index)
                {
                    Profile::mark_period(Profile::Timeline::TrainStep);
                    {
                        SA_ZONE("data", "read");
                        SA_TRY(dataset.read(index, batch));
                    }
                    SA_TRY_ASSIGN(const StepMetrics metrics, run_step(batch));

                    if (first)
                    {
                        report.first_loss = metrics.loss;
                        first = false;
                    }
                    report.last_loss = metrics.loss;
                    report.last_metrics = metrics.metrics;
                    ++report.steps;
                }
            }

            report.compile_count = step_->step().compile_count();
            return report;
        }

    } // namespace Train
} // namespace SushiAI
