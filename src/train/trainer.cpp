/**************************************************************************/
/* trainer.cpp                                                            */
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

#include <chrono>
#include <format>
#include <span>
#include <vector>
#include <SushiAI/train/trainer.hpp>

namespace SushiAI
{
    namespace Train
    {
        std::vector<Optim::ParameterSlot> bind_parameters(const TrainingGraph& built,
                                                          Graph::TensorPool& pool,
                                                          Graph::PlanOptions options)
        {
            // Before anything is materialised: adopt() only reaches values that
            // do not have a buffer yet, so planning after the fact would plan
            // nothing and report a saving that never happened.
            if (!pool.has_plan())
            {
                // The planner reasons about the IR; only the lowering knows
                // which values an operation cannot address through an arena, so
                // it gets a veto rather than the planner getting a special case.
                // Added to the caller's list rather than replacing it: this veto
                // is a correctness constraint, not a preference.
                std::vector<Graph::ValueId> pinned =
                    Graph::Lowering::values_needing_own_allocation(built.graph);
                pinned.insert(pinned.end(), options.pinned.begin(), options.pinned.end());

                Graph::PlanOptions planning = options;
                planning.pinned = std::span<const Graph::ValueId>(pinned.data(), pinned.size());

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

                // There is no zero_grad() in this library, because every
                // gradient in a step is written rather than accumulated into -
                // see optimizer.hpp. That leaves exactly one window in which a
                // gradient has never been written at all: between here and the
                // first replay. An optimizer stepping inside it would read a
                // fresh allocation, which is whatever the pool last had in that
                // block, and feed it straight into the weights. Nothing
                // downstream can catch that; the parameters simply become
                // garbage and the loss never recovers. So the buffer is defined
                // from the moment anything can see it, and a step taken before
                // the first backward is a no-op instead of undefined behaviour.
                pool.request_constant(gradient, 0.0);

                slots.push_back(Optim::ParameterSlot{
                    binding.name, pool.tensor(binding.value), gradient,
                });
            }

            // Immediately, not at the next CompiledStep: the caller may build
            // an optimizer over these slots before it builds a trainer.
            pool.flush_constants();

            return slots;
        }

        std::string RunReport::to_string() const
        {
            std::string text = std::format(
                "compiles       : {}\n"
                "steps          : {}\n"
                "ops per step   : {}\n",
                compile_count, steps, operations_per_step);

            // Printed only when there are any, so a run with no fused epilogue
            // reads exactly as it did before rather than carrying a line of
            // zero.
            if (ordering_nodes_per_step != 0)
                text += std::format("ordering nodes : {} (host-side, no kernel)\n",
                                    ordering_nodes_per_step);

            text += std::format(
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
                text += std::format("  [{:>3}] {:<24} x{:<6} device {:>8.3f} ms  host {:>8.3f} ms\n",
                                    timing.plan_index, timing.name, timing.invocations,
                                    timing.device_ms, timing.host_ms);

            return text;
        }

        namespace
        {
            /**
             * @brief The recorder CompiledStep should fold in, or nothing.
             *
             * A free function because the answer is needed inside Trainer's
             * member-initialiser list, before any member exists to hold it -
             * and because refusing loudly is the whole point: an optimizer
             * whose scalars are baked into its recorded tasks would replay step
             * one's bias corrections for ever, converge anyway, and never say
             * a word.
             *
             * @param optimizer The update rule.
             * @param options The trainer's options.
             * @return The optimizer as a recorder, or nullptr for PER_STEP.
             * @throws Error If IN_PLAN was asked for and cannot be given.
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
              step_(engine, graph, pool, placement_recorder(optimizer, options),
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

            // The per-row argmax is done here rather than on the device
            // because SushiBLAS's argmax is whole-tensor and a diagnostic does
            // not justify a new reduction kernel. It is one host read of an
            // [N, C] buffer per step, and it is why accuracy is optional.
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

            // The compiled plan reads these two buffers by address, so the batch
            // is copied into them rather than bound to them. See the header.
            engine_->blas().copy(batch.inputs, inputs);
            engine_->blas().copy(batch.targets, targets);
            engine_->execute();
            engine_->get_context().wait_all();

            // With the optimizer in the plan this one call is forward,
            // backward, the loss and the update. The metrics below are still
            // this step's: the update writes parameters and its own state, and
            // touches neither the loss nor the logits.
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
            report.ordering_nodes_per_step = step_.ordering_node_count();

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
