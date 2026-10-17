/**************************************************************************/
/* training_step.cpp                                                      */
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
 * @file training_step.cpp
 * @brief Implements TrainingStep: the placement check, the compilation of the trailing
 *        recorders and the run that steps a detached optimizer.
 * @author Mustafa Garip
 */

#include <SushiAI/train/training_step.hpp>

#include <algorithm>
#include <memory>
#include <utility>
#include <vector>

#include <SushiAI/seam/seam.hpp>

namespace SushiAI
{
    namespace Train
    {
        namespace
        {
            /**
             * @brief Checks that an optimizer recorded into the step can be replayed unchanged.
             * @return Success, or InvalidDescription when the placement is InPlan and the
             *         optimizer is not replay invariant.
             */
            [[nodiscard]] Status require_replay_invariant(const Optim::Optimizer& optimizer,
                                                          OptimizerPlacement placement)
            {
                SA_FAIL_IF(placement == OptimizerPlacement::InPlan
                               && !optimizer.is_replay_invariant(),
                           StatusCode::InvalidDescription,
                           "This optimizer's update is recorded as separate operations whose "
                           "scalars are baked in at record time, so it cannot be compiled into "
                           "the step: every replay would reuse the first step's bias "
                           "corrections. Use Optim::UpdateKernel::Fused, or ask for "
                           "TrainerOptions::optimizer_placement = OptimizerPlacement::PerStep.");

                return Status();
            }

            /**
             * @brief Returns the node the part behind @p stage starts at.
             * @return The node behind the last producer of a value the stage reads;
             *         InvalidDescription for a read that names no value of the graph or one
             *         no node produces, or for a node before the stage that reads a leaf the
             *         stage writes.
             */
            [[nodiscard]] Result<Graph::NodeID> find_stage_cut(const Graph::Graph& graph,
                                                               const HostStage& stage)
            {
                Graph::NodeID cut = 0;
                for (const Graph::ValueID read : stage.reads)
                {
                    SA_FAIL_IF(read >= graph.value_count(), StatusCode::InvalidDescription,
                               "The host stage reads value {}, but the graph of the step "
                               "holds {} values",
                               read, graph.value_count());

                    const Graph::NodeID producer = graph.value(read).producer;
                    SA_FAIL_IF(producer == Graph::K_NO_NODE, StatusCode::InvalidDescription,
                               "The host stage reads value {}, which no node of the step "
                               "produces",
                               read);
                    cut = std::max(cut, producer + 1);
                }

                for (Graph::NodeID id = 0; id < cut; ++id)
                {
                    for (const Graph::ValueID input : graph.node(id).inputs)
                    {
                        const bool is_written = std::find(stage.writes.begin(),
                                                          stage.writes.end(), input)
                                             != stage.writes.end();
                        SA_FAIL_IF(is_written, StatusCode::InvalidDescription,
                                   "Node {} runs before the host stage but reads value {}, "
                                   "which the stage writes",
                                   id, input);
                    }
                }

                return cut;
            }
        } // namespace

        TrainingStep::TrainingStep(Optim::Optimizer& optimizer,
                                   std::vector<BufferUpdateSlot> updates,
                                   bool optimizer_in_plan) noexcept
            : optimizer_(&optimizer),
              optimizer_in_plan_(optimizer_in_plan),
              updates_(std::move(updates)),
              recorders_(),
              lead_(),
              step_()
        {

        }

        Result<std::unique_ptr<TrainingStep>> TrainingStep::make(
            SushiBLAS::Engine& engine, const Graph::Graph& graph, Graph::TensorPool& pool,
            Optim::Optimizer& optimizer, std::vector<BufferUpdateSlot> updates,
            TrainingStepOptions options)
        {
            SA_TRY(require_replay_invariant(optimizer, options.optimizer_placement));

            const bool in_plan = options.optimizer_placement == OptimizerPlacement::InPlan;
            const bool has_updates = !updates.empty();

            SA_TRY_ASSIGN(std::unique_ptr<TrainingStep> step,
                          ::SushiAI::Detail::guard_value(::SushiAI::Detail::SeamPhase::Allocate,
                                                         [&]()
            {
                return std::unique_ptr<TrainingStep>(
                    new TrainingStep(optimizer, std::move(updates), in_plan));
            }));

            // The optimizer is added first, so its update runs before the statistics blend.
            if (in_plan)
                SA_TRY(step->recorders_.add(optimizer));

            if (has_updates)
                SA_TRY(step->recorders_.add(step->updates_));

            // An empty list means no trailing work, so the step compiles as a plain graph.
            Graph::StepRecorder* const trailing =
                step->recorders_.empty() ? nullptr : &step->recorders_;

            Graph::NodeRange rest;
            if (!options.host_stage.reads.empty())
            {
                SA_TRY_ASSIGN(const Graph::NodeID cut,
                              find_stage_cut(graph, options.host_stage));
                SA_TRY_ASSIGN(std::unique_ptr<Graph::CompiledStep> lead,
                              Graph::CompiledStep::compile(engine, graph, pool, nullptr,
                                                           options.fusion_selector,
                                                           Graph::NodeRange{0, cut}));
                step->lead_ = std::move(lead);
                rest.first = cut;
            }

            SA_TRY_ASSIGN(std::unique_ptr<Graph::CompiledStep> compiled,
                          Graph::CompiledStep::compile(engine, graph, pool, trailing,
                                                       options.fusion_selector, rest));
            step->step_ = std::move(compiled);

            return std::move(step);
        }

        Status TrainingStep::run()
        {
            if (lead_)
                SA_TRY(lead_->run());

            return run_rest();
        }

        Status TrainingStep::restore_run_count(std::size_t count)
        {
            if (lead_)
                SA_TRY(lead_->restore_run_count(count));

            return step_->restore_run_count(count);
        }

        Status TrainingStep::run_lead()
        {
            SA_FAIL_IF(!lead_, StatusCode::InvalidDescription,
                       "This training step was compiled without a host stage; name what the "
                       "host reads in TrainingStepOptions::host_stage");

            return lead_->run();
        }

        Status TrainingStep::run_rest()
        {
            SA_TRY(step_->run());

            // A detached optimizer is stepped here, because the step does not record it.
            if (!optimizer_in_plan_)
                SA_TRY(optimizer_->step());

            return Status();
        }

    } // namespace Train
} // namespace SushiAI
