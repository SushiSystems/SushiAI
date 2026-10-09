/**************************************************************************/
/* evaluation_step.cpp                                                    */
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
 * @file evaluation_step.cpp
 * @brief Implements EvaluationStep.
 * @author Mustafa Garip
 */

#include <SushiAI/train/evaluation_step.hpp>

#include <memory>
#include <string_view>
#include <utility>
#include <SushiBLAS/engine.hpp>
#include <SushiBLAS/engine/blas.hpp>
#include <SushiBLAS/tensor.hpp>
#include <SushiAI/core/result.hpp>
#include <SushiAI/core/seam.hpp>
#include <SushiAI/core/status.hpp>
#include <SushiAI/graph/compiled_step.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/graph/memory_plan.hpp>
#include <SushiAI/graph/tensor_pool.hpp>
#include <SushiAI/nn/model_outputs.hpp>
#include <SushiAI/nn/model_state.hpp>
#include <SushiAI/nn/throwing_bridge.hpp>
#include <SushiAI/train/feed.hpp>
#include <SushiAI/train/steps.hpp>

namespace SushiAI
{
    namespace Train
    {
        Result<std::unique_ptr<EvaluationStep>> EvaluationStep::make(
            SushiBLAS::Engine& engine, const TracedStep& traced, const NN::ModelState& state,
            Graph::TensorPool& pool, const Graph::FusionSelector* selector)
        {
            SA_FAIL_IF(traced.input == Graph::K_NO_VALUE, StatusCode::InvalidDescription,
                       "The evaluation step needs the input leaf of its traced graph");

            // The plan is adopted before any leaf or value of the graph is materialised.
            if (!pool.has_plan())
            {
                // The host reads every output after the run, so none is packed.
                SA_TRY_ASSIGN(Graph::MemoryPlan plan, NN::Detail::bridge_throwing([&traced]()
                {
                    std::vector<Graph::ValueId> pinned;
                    for (const NN::ModelOutputs::Entry& entry : traced.outputs.all())
                        pinned.push_back(entry.second);

                    Graph::PlanOptions options;
                    options.pinned = SushiRuntime::span<const Graph::ValueId>(pinned.data(),
                                                                              pinned.size());
                    return Graph::MemoryPlan::build(traced.graph, options);
                }));
                SA_TRY(pool.adopt(std::move(plan)));
            }

            SA_TRY(state.attach(traced.graph, pool));
            SA_TRY(pool.materialise_leaves(traced.graph));
            SA_TRY_ASSIGN(std::unique_ptr<Graph::CompiledStep> step,
                          Graph::CompiledStep::compile(engine, traced.graph, pool, nullptr,
                                                       selector));

            SushiBLAS::Tensor& input = pool.tensor(traced.input);
            return ::SushiAI::Detail::guard_value(::SushiAI::Detail::SeamPhase::Allocate, [&]()
            {
                return std::unique_ptr<EvaluationStep>(new EvaluationStep(
                    engine, pool, input, traced.outputs, std::move(step)));
            });
        }

        EvaluationStep::EvaluationStep(SushiBLAS::Engine& engine, Graph::TensorPool& pool,
                                       SushiBLAS::Tensor& input, NN::ModelOutputs outputs,
                                       std::unique_ptr<Graph::CompiledStep> step) noexcept
            : engine_(&engine),
              pool_(&pool),
              input_(&input),
              outputs_(std::move(outputs)),
              step_(std::move(step))
        {

        }

        EvaluationStep::~EvaluationStep() = default;

        Status EvaluationStep::run(const SushiBLAS::Tensor& inputs)
        {
            SA_TRY(check_feed(inputs, *input_, "inputs"));
            SA_TRY(::SushiAI::Detail::record_and_run(*engine_, [&]() -> Status
            {
                SushiBLAS::BLASOps blas(*engine_);
                blas.copy(inputs, *input_);
                return Status();
            }));

            return step_->run();
        }

        Result<const SushiBLAS::Tensor*> EvaluationStep::output(std::string_view name) const
        {
            SA_TRY_ASSIGN(const Graph::ValueId id, outputs_.at(name));

            const Graph::TensorPool& pool = *pool_;
            return &pool.tensor(id);
        }

        const Graph::CompiledStep& EvaluationStep::step() const noexcept
        {
            return *step_;
        }

    } // namespace Train
} // namespace SushiAI
