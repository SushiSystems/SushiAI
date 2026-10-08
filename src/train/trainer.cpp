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

#include <SushiAI/train/trainer.hpp>

#include <chrono>
#include <utility>
#include <vector>

#include <SushiBLAS/core/format.hpp>
#include <SushiBLAS/engine/blas.hpp>
#include <SushiRuntime/core/span.hpp>

#include <SushiAI/core/seam.hpp>

namespace SushiAI
{
    namespace Train
    {
        Result<std::vector<Optim::ParameterSlot>> bind_parameters(const TrainingGraph& built,
                                                                  Graph::TensorPool& pool,
                                                                  Graph::PlanOptions options)
        {
            // Memory planning must precede leaf and tensor materialisation.
            if (!pool.has_plan())
            {
                // Lowering constraints determine which values require standalone allocations.
                std::vector<Graph::ValueId> pinned =
                    Graph::Lowering::values_needing_own_allocation(built.graph);
                pinned.insert(pinned.end(), options.pinned.begin(), options.pinned.end());

                Graph::PlanOptions planning = options;
                planning.pinned =
                    SushiRuntime::span<const Graph::ValueId>(pinned.data(), pinned.size());

                SA_TRY(pool.adopt(Graph::MemoryPlan::build(built.graph, planning)));
            }

            SA_TRY(pool.materialise_leaves(built.graph));

            std::vector<Optim::ParameterSlot> slots;
            slots.reserve(built.parameters.size());

            for (const ParameterBinding& binding : built.parameters)
            {
                SA_FAIL_IF(binding.gradient == Graph::K_NO_VALUE, StatusCode::InvalidDescription,
                           "Parameter '{}' receives no gradient. It is not on any path from the "
                           "loss, so no optimizer could ever move it - either wire it into the "
                           "model or stop listing it as a parameter.",
                           binding.name);

                const Graph::Value& gradient_value = built.graph.value(binding.gradient);

                SA_TRY_ASSIGN(SushiBLAS::Tensor& gradient,
                              pool.materialise(binding.gradient, gradient_value));

                // Gradients are initialised to zero before the first backward execution.
                pool.request_constant(gradient, 0.0);

                slots.push_back(Optim::ParameterSlot{
                    binding.name, pool.tensor(binding.value), gradient,
                });
            }

            // Constant buffers must be flushed before optimizer initialisation.
            SA_TRY(pool.flush_constants());

            return std::move(slots);
        }

        std::string RunReport::to_string() const
        {
            std::string text = SushiBLAS::Core::format(
                "compiles       : {}\n"
                "steps          : {}\n"
                "ops per step   : {}\n",
                compile_count, steps, operations_per_step);

            text += SushiBLAS::Core::format(
                "loss           : {:.6f} -> {:.6f}\n"
                "final accuracy : {:.2f}%\n"
                "wall clock     : {:.1f} ms ({:.2f} ms/step)\n",
                first_loss, last_loss, last_accuracy * 100.0,
                duration_ms, steps ? duration_ms / static_cast<double>(steps) : 0.0);

            if (node_timings.empty())
            {
                text += "per-node timing: none recorded (build the RuntimeContext with "
                        "RuntimeSettings::profiling to collect it)\n";
                return text;
            }

            text += "per-node timing:\n";
            for (const SushiRuntime::Core::NodeTiming& timing : node_timings)
                text += SushiBLAS::Core::format("  [{:>3}] {:<24} x{:<6} device {:>8.3f} ms  host {:>8.3f} ms\n",
                                                timing.plan_index, timing.name, timing.invocations,
                                                timing.device_ms, timing.host_ms);

            return text;
        }

        namespace
        {
            /**
             * @brief Returns the recorder whose work is folded into the compiled step.
             * @return The optimizer for InPlan, null for PerStep, or InvalidDescription when
             *         InPlan is asked of an optimizer that is not replay invariant.
             */
            [[nodiscard]] Result<Graph::StepRecorder*> placement_recorder(
                Optim::Optimizer& optimizer, const TrainerOptions& options)
            {
                Graph::StepRecorder* recorder = nullptr;
                if (options.optimizer_placement == OptimizerPlacement::PerStep)
                    return recorder;

                SA_FAIL_IF(!optimizer.is_replay_invariant(), StatusCode::InvalidDescription,
                           "This optimizer's update is recorded as separate operations whose "
                           "scalars are baked in at record time, so it cannot be compiled into "
                           "the step: every replay would reuse the first step's bias "
                           "corrections. Use Optim::UpdateKernel::FUSED, or ask for "
                           "TrainerOptions::optimizer_placement = OptimizerPlacement::PER_STEP.");

                recorder = &optimizer;
                return recorder;
            }

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
                                                       const Graph::Graph& graph,
                                                       Graph::TensorPool& pool, StepSpec spec,
                                                       Optim::Optimizer& optimizer,
                                                       TrainerOptions options)
        {
            SA_FAIL_IF(spec.inputs == Graph::K_NO_VALUE || spec.targets == Graph::K_NO_VALUE
                       || spec.logits == Graph::K_NO_VALUE || spec.loss == Graph::K_NO_VALUE,
                       StatusCode::InvalidDescription,
                       "The trainer needs all four of inputs, targets, logits and loss");

            const Graph::Value& loss = graph.value(spec.loss);

            SA_FAIL_IF(loss.shape.num_elements() != 1, StatusCode::InvalidDescription,
                       "The loss must be a single element; got {}", loss.shape.to_string());
            SA_FAIL_IF(loss.dtype != DType::Float32 && loss.dtype != DType::Float64,
                       StatusCode::InvalidDescription,
                       "The loss is {}; only float32 and float64 can be read back",
                       dtype_name(loss.dtype));

            SA_TRY(verify_precision(graph, options));
            SA_TRY_ASSIGN(Graph::StepRecorder* const trailing,
                          placement_recorder(optimizer, options));
            SA_TRY_ASSIGN(std::unique_ptr<Graph::CompiledStep> step,
                          Graph::CompiledStep::compile(engine, graph, pool, trailing,
                                                       options.fusion_selector));

            const Shape& logits = graph.value(spec.logits).shape;

            std::unique_ptr<Trainer> trainer(new Trainer(engine, pool, spec, optimizer, options,
                                                         std::move(step), logits.dim(0),
                                                         logits.dim(1)));
            return std::move(trainer);
        }

        Trainer::Trainer(SushiBLAS::Engine& engine, Graph::TensorPool& pool, StepSpec spec,
                         Optim::Optimizer& optimizer, TrainerOptions options,
                         std::unique_ptr<Graph::CompiledStep> step, int64_t rows,
                         int64_t classes) noexcept
            : engine_(&engine),
              pool_(&pool),
              spec_(spec),
              optimizer_(&optimizer),
              options_(options),
              step_(std::move(step)),
              rows_(rows),
              classes_(classes)
        {

        }

        double Trainer::current_loss() const
        {
            const SushiBLAS::Tensor& loss = pool_->tensor(spec_.loss);

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

        Result<double> Trainer::score_accuracy(const Data::Batch& batch) const
        {
            const SushiBLAS::Tensor& logits = pool_->tensor(spec_.logits);

            // Accuracy evaluation requires readable host float32 or float64 logits.
            SA_FAIL_IF(logits.dtype != DType::Float32 && logits.dtype != DType::Float64,
                       StatusCode::InvalidDescription,
                       "Accuracy needs a host read of the logits and they are {}; only "
                       "float32 and float64 can be read back. Either switch accuracy "
                       "tracking off, or keep the logits in the master type - "
                       "Train::build_classifier already lists them as observed, so a "
                       "precision policy cannot narrow them.",
                       dtype_name(logits.dtype));

            // Per-row argmax is computed on host to evaluate accuracy per sample.
            const auto at = [&logits](int64_t index) -> double
            {
                return logits.dtype == DType::Float64
                     ? logits.data_as<double>()[index]
                     : static_cast<double>(logits.data_as<float>()[index]);
            };

            std::size_t correct = 0;
            for (int64_t row = 0; row < rows_; ++row)
            {
                int64_t best = 0;
                for (int64_t column = 1; column < classes_; ++column)
                {
                    if (at(row * classes_ + column) > at(row * classes_ + best))
                        best = column;
                }

                if (best == batch.labels[static_cast<std::size_t>(row)])
                    ++correct;
            }

            return static_cast<double>(correct) / static_cast<double>(rows_);
        }

        Result<StepMetrics> Trainer::run_step(const Data::Batch& batch)
        {
            SushiBLAS::Tensor& inputs = pool_->tensor(spec_.inputs);
            SushiBLAS::Tensor& targets = pool_->tensor(spec_.targets);

            SA_FAIL_IF(batch.inputs.num_elements != inputs.num_elements, StatusCode::Incompatible,
                       "The batch holds {} input elements but the compiled step expects {}",
                       batch.inputs.num_elements, inputs.num_elements);
            SA_FAIL_IF(batch.targets.num_elements != targets.num_elements,
                       StatusCode::Incompatible,
                       "The batch holds {} target elements but the compiled step expects {}",
                       batch.targets.num_elements, targets.num_elements);

            // Minibatch tensors are copied into fixed address buffers read by the plan.
            SA_TRY(Detail::guard(Detail::SeamPhase::Record, [&]()
            {
                SushiBLAS::BLASOps blas(*engine_);
                blas.copy(batch.inputs, inputs);
                blas.copy(batch.targets, targets);
                return Status();
            }));
            SA_TRY(Detail::run_recorded(*engine_));

            // The compiled step executes forward, backward, loss, and update in-plan.
            SA_TRY(step_->run());

            StepMetrics metrics;
            metrics.loss = current_loss();
            if (options_.track_accuracy && !batch.labels.empty())
            {
                SA_TRY_ASSIGN(const double accuracy, score_accuracy(batch));
                metrics.accuracy = accuracy;
            }

            if (!step_->has_trailing_work())
                SA_TRY(optimizer_->step());
            ++steps_;

            return metrics;
        }

        Result<RunReport> Trainer::fit(const Data::Dataset& dataset)
        {
            SA_FAIL_IF(dataset.batch_count() == 0, StatusCode::InvalidDescription,
                       "The dataset holds no batches");
            SA_FAIL_IF(dataset.batch_size() != rows_, StatusCode::Incompatible,
                       "The dataset's batch size is {} but the step was compiled for {}",
                       dataset.batch_size(), rows_);
            SA_FAIL_IF(dataset.num_classes() != classes_, StatusCode::Incompatible,
                       "The dataset has {} classes but the step was compiled for {}",
                       dataset.num_classes(), classes_);

            SushiRuntime::Execution::IExecutionEngine& execution =
                engine_->get_context().get_execution_engine();

            if (options_.profile)
                execution.reset_profiling(step_->plan());

            RunReport report;
            report.operations_per_step = step_->operation_count();

            const auto started = std::chrono::steady_clock::now();

            bool first = true;
            for (int64_t epoch = 0; epoch < options_.epochs; ++epoch)
            {
                for (std::size_t index = 0; index < dataset.batch_count(); ++index)
                {
                    SA_TRY_ASSIGN(const StepMetrics metrics, run_step(dataset.batch(index)));

                    if (first)
                    {
                        report.first_loss = metrics.loss;
                        first = false;
                    }
                    report.last_loss = metrics.loss;
                    report.last_accuracy = metrics.accuracy;
                    ++report.steps;
                }
            }

            const auto finished = std::chrono::steady_clock::now();
            report.duration_ms =
                std::chrono::duration<double, std::milli>(finished - started).count();

            if (options_.profile)
            {
                SushiRuntime::Core::RunReport runtime_report;
                execution.collect_profiling(step_->plan(), runtime_report);
                report.node_timings = std::move(runtime_report.node_timings);
            }

            report.compile_count = step_->compile_count();
            return report;
        }

    } // namespace Train
} // namespace SushiAI
