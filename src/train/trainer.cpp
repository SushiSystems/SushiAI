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

#include <chrono>
#include <vector>
#include <SushiBLAS/core/format.hpp>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/train/trainer.hpp>

namespace SushiAI
{
    namespace Train
    {
        std::vector<Optim::ParameterSlot> bind_parameters(const TrainingGraph& built,
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
                planning.pinned = SushiRuntime::span<const Graph::ValueId>(pinned.data(), pinned.size());

                pool.adopt(Graph::MemoryPlan::build(built.graph, planning));
            }

            pool.materialise_leaves(built.graph);

            std::vector<Optim::ParameterSlot> slots;
            slots.reserve(built.parameters.size());

            for (const ParameterBinding& binding : built.parameters)
            {
                SA_THROW_IF(binding.gradient == Graph::NO_VALUE,
                            "Parameter '{}' receives no gradient. It is not on any path from the "
                            "loss, so no optimizer could ever move it - either wire it into the "
                            "model or stop listing it as a parameter.",
                            binding.name);

                const Graph::Value& gradient_value = built.graph.value(binding.gradient);

                SushiBLAS::Tensor& gradient = pool.materialise(binding.gradient, gradient_value);

                // Gradients are initialised to zero before the first backward execution.
                pool.request_constant(gradient, 0.0);

                slots.push_back(Optim::ParameterSlot{
                    binding.name, pool.tensor(binding.value), gradient,
                });
            }

            // Constant buffers must be flushed before optimizer initialisation.
            pool.flush_constants();

            return slots;
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
             * @brief Determines optimizer step recorder to fold into plan.
             * @param optimizer Update rule; @p options Trainer options.
             * @return Step recorder pointer, or nullptr for PER_STEP.
             * @throws Error If IN_PLAN was requested for non-replayable optimizer.
             */
            Graph::IStepRecorder* placement_recorder(Optim::Optimizer& optimizer,
                                                     const TrainerOptions& options)
            {
                if (options.optimizer_placement == OptimizerPlacement::PER_STEP)
                    return nullptr;

                SA_THROW_IF(!optimizer.is_replay_invariant(),
                            "This optimizer's update is recorded as separate operations whose "
                            "scalars are baked in at record time, so it cannot be compiled into "
                            "the step: every replay would reuse the first step's bias "
                            "corrections. Use Optim::UpdateKernel::FUSED, or ask for "
                            "TrainerOptions::optimizer_placement = OptimizerPlacement::PER_STEP.");

                return &optimizer;
            }

            /**
             * @brief Validates graph operand types against the precision policy.
             * @param graph The graph about to be lowered.
             * @param options The trainer configuration options.
             * @return Unchanged graph reference if validation succeeds.
             * @throws Error If any node operand disagrees with precision policy.
             */
            const Graph::Graph& verified(const Graph::Graph& graph, const TrainerOptions& options)
            {
                if (options.precision == nullptr) return graph;

                const std::vector<Graph::DTypeMismatch> mismatches =
                    Graph::find_dtype_mismatches(graph, *options.precision);

                SA_THROW_IF(!mismatches.empty(),
                            "This graph does not agree with the precision policy it was given: "
                            "{} node(s) read or write an element type they do not compute in. "
                            "The first is: {}. Build the graph with the same policy - "
                            "build_classifier()'s `precision` argument runs the cast pass "
                            "between differentiation and fusion, which is the only window in "
                            "which the conversions can be placed.",
                            mismatches.size(), Graph::describe(mismatches.front()));

                return graph;
            }
        }

        Trainer::Trainer(SushiBLAS::Engine& engine,
                         const Graph::Graph& graph,
                         Graph::TensorPool& pool,
                         StepSpec spec,
                         Optim::Optimizer& optimizer,
                         TrainerOptions options)
            : engine_(&engine),
              pool_(&pool),
              spec_(spec),
              optimizer_(&optimizer),
              options_(options),
              step_(engine, verified(graph, options), pool,
                    placement_recorder(optimizer, options),
                    options.fusion_selector)
        {
            SA_THROW_IF(spec_.inputs == Graph::NO_VALUE || spec_.targets == Graph::NO_VALUE
                        || spec_.logits == Graph::NO_VALUE || spec_.loss == Graph::NO_VALUE,
                        "The trainer needs all four of inputs, targets, logits and loss");

            const Shape& logits = graph.value(spec_.logits).shape;
            rows_ = logits.dim(0);
            classes_ = logits.dim(1);

            SA_THROW_IF(graph.value(spec_.loss).shape.num_elements() != 1,
                        "The loss must be a single element; got {}",
                        graph.value(spec_.loss).shape.to_string());
        }

        double Trainer::current_loss() const
        {
            const SushiBLAS::Tensor& loss = pool_->tensor(spec_.loss);

            switch (loss.dtype)
            {
                case DType::FLOAT32: return static_cast<double>(loss.data_as<float>()[0]);
                case DType::FLOAT64: return loss.data_as<double>()[0];
                default:
                    SA_THROW_IF(true, "The loss is {}; only float32 and float64 can be read back",
                                dtype_name(loss.dtype));
            }
            return 0.0;
        }

        double Trainer::score_accuracy(const Data::Batch& batch) const
        {
            const SushiBLAS::Tensor& logits = pool_->tensor(spec_.logits);

            // Accuracy evaluation requires readable host float32 or float64 logits.
            SA_THROW_IF(logits.dtype != DType::FLOAT32 && logits.dtype != DType::FLOAT64,
                        "Accuracy needs a host read of the logits and they are {}; only "
                        "float32 and float64 can be read back. Either switch accuracy "
                        "tracking off, or keep the logits in the master type - "
                        "Train::build_classifier already lists them as observed, so a "
                        "precision policy cannot narrow them.",
                        dtype_name(logits.dtype));

            // Per-row argmax is computed on host to evaluate accuracy per sample.
            const auto at = [&logits](int64_t index) -> double
            {
                return logits.dtype == DType::FLOAT64
                     ? logits.data_as<double>()[index]
                     : static_cast<double>(logits.data_as<float>()[index]);
            };

            std::size_t correct = 0;
            for (int64_t row = 0; row < rows_; ++row)
            {
                int64_t best = 0;
                for (int64_t column = 1; column < classes_; ++column)
                    if (at(row * classes_ + column) > at(row * classes_ + best)) best = column;

                if (best == batch.labels[static_cast<std::size_t>(row)]) ++correct;
            }

            return static_cast<double>(correct) / static_cast<double>(rows_);
        }

        StepMetrics Trainer::run_step(const Data::Batch& batch)
        {
            SushiBLAS::Tensor& inputs = pool_->tensor(spec_.inputs);
            SushiBLAS::Tensor& targets = pool_->tensor(spec_.targets);

            SA_THROW_IF(batch.inputs.num_elements != inputs.num_elements,
                        "The batch holds {} input elements but the compiled step expects {}",
                        batch.inputs.num_elements, inputs.num_elements);
            SA_THROW_IF(batch.targets.num_elements != targets.num_elements,
                        "The batch holds {} target elements but the compiled step expects {}",
                        batch.targets.num_elements, targets.num_elements);

            // Minibatch tensors are copied into fixed address buffers read by the plan.
            engine_->blas().copy(batch.inputs, inputs);
            engine_->blas().copy(batch.targets, targets);
            engine_->execute();
            engine_->get_context().wait_all();

            // The compiled step executes forward, backward, loss, and update in-plan.
            step_.run();

            StepMetrics metrics;
            metrics.loss = current_loss();
            if (options_.track_accuracy && !batch.labels.empty())
                metrics.accuracy = score_accuracy(batch);

            if (!step_.has_trailing_work()) optimizer_->step();
            ++steps_;

            return metrics;
        }

        RunReport Trainer::fit(const Data::Dataset& dataset)
        {
            SA_THROW_IF(dataset.batch_count() == 0, "The dataset holds no batches");
            SA_THROW_IF(dataset.batch_size() != rows_,
                        "The dataset's batch size is {} but the step was compiled for {}",
                        dataset.batch_size(), rows_);
            SA_THROW_IF(dataset.num_classes() != classes_,
                        "The dataset has {} classes but the step was compiled for {}",
                        dataset.num_classes(), classes_);

            SushiRuntime::Execution::IExecutionEngine& execution =
                engine_->get_context().get_execution_engine();

            if (options_.profile) execution.reset_profiling(step_.plan());

            RunReport report;
            report.operations_per_step = step_.operation_count();

            const auto started = std::chrono::steady_clock::now();

            bool first = true;
            for (int64_t epoch = 0; epoch < options_.epochs; ++epoch)
            {
                for (std::size_t index = 0; index < dataset.batch_count(); ++index)
                {
                    const StepMetrics metrics = run_step(dataset.batch(index));

                    if (first) { report.first_loss = metrics.loss; first = false; }
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
                execution.collect_profiling(step_.plan(), runtime_report);
                report.node_timings = std::move(runtime_report.node_timings);
            }

            report.compile_count = step_.compile_count();
            return report;
        }

    } // namespace Train
} // namespace SushiAI
