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

#include <cstddef>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>
#include <SushiBLAS/engine/engine.hpp>
#include <SushiBLAS/blas/blas.hpp>
#include <SushiBLAS/core/format.hpp>
#include <SushiBLAS/tensor/tensor.hpp>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/core/check.hpp>
#include <SushiAI/core/result.hpp>
#include <SushiAI/core/seam.hpp>
#include <SushiAI/core/status.hpp>
#include <SushiAI/graph/compiled_step.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/graph/memory_plan.hpp>
#include <SushiAI/graph/step_recorder.hpp>
#include <SushiAI/graph/tensor_pool.hpp>
#include <SushiAI/nn/model_outputs.hpp>
#include <SushiAI/nn/model_state.hpp>
#include <SushiAI/nn/throwing_bridge.hpp>
#include <SushiAI/nn/trace.hpp>
#include <SushiAI/train/buffer_update_recorder.hpp>
#include <SushiAI/train/feed.hpp>
#include <SushiAI/train/steps.hpp>

namespace SushiAI
{
    namespace Train
    {
        namespace
        {
            /** @brief Holds the weight of the next value when a carried leaf takes it whole. */
            constexpr double K_WHOLE = 1.0;

            /**
             * @brief Lists what the planner must not pack: every output the host reads and
             *        the next value of every carried pair, which is read behind the graph.
             */
            [[nodiscard]] Result<std::vector<Graph::ValueID>> pinned_values(
                const TracedStep& traced)
            {
                return NN::Detail::bridge_throwing([&traced]()
                {
                    std::vector<Graph::ValueID> ids;
                    for (const NN::ModelOutputs::Entry& entry : traced.outputs.all())
                        ids.push_back(entry.second);

                    const SushiRuntime::span<const Graph::ValueID> kept =
                        traced.carried.sources();
                    ids.insert(ids.end(), kept.begin(), kept.end());

                    return ids;
                });
            }

            /** @brief Returns Incompatible when the plan of @p pool packs a carried next value. */
            [[nodiscard]] Status require_kept(const Graph::TensorPool& pool,
                                              const NN::CarriedValueList& carried)
            {
                const Graph::MemoryPlan& plan = pool.arenas().plan();
                for (const Graph::ValueID id : carried.sources())
                {
                    SA_FAIL_IF(plan.is_packed(plan.buffer_of(id)), StatusCode::Incompatible,
                               "The pool's plan packs value {}, which a carrying step reads "
                               "after its last reader of the graph. Pass a pool that holds no "
                               "plan.",
                               id);
                }

                return Status();
            }
        } // namespace

        Result<EvaluationStep::Parts> EvaluationStep::gather(const TracedStep& traced,
                                                             Graph::TensorPool& pool)
        {
            for (const NN::FedLeaf& leaf : traced.fed.leaves())
            {
                SA_FAIL_IF(leaf.name == K_MODEL_INPUT_NAME, StatusCode::InvalidDescription,
                           "A layer declared a fed leaf named '{}', which names the model "
                           "input of every step",
                           K_MODEL_INPUT_NAME);
            }

            for (const NN::CarriedValue& carried : traced.carried.values())
            {
                SA_FAIL_IF(pool.tensor(carried.leaf).storage == pool.tensor(carried.next).storage,
                           StatusCode::InvalidDescription,
                           "The carried leaf {} takes value {}, which is a view of the leaf "
                           "itself; a carried value takes a value the graph computes",
                           carried.leaf, carried.next);
            }

            return ::SushiAI::Detail::guard_value(::SushiAI::Detail::SeamPhase::Allocate, [&]()
            {
                Parts parts;
                parts.input_names.emplace_back(K_MODEL_INPUT_NAME);
                parts.input_leaves.push_back(&pool.tensor(traced.input));
                for (const NN::FedLeaf& leaf : traced.fed.leaves())
                {
                    parts.input_names.emplace_back(leaf.name);
                    parts.input_leaves.push_back(&pool.tensor(leaf.value));
                }

                for (const NN::CarriedValue& carried : traced.carried.values())
                {
                    SushiBLAS::Tensor& leaf = pool.tensor(carried.leaf);
                    parts.carried.push_back(CarriedLeaf{&leaf, carried.initial_value});
                    parts.carries.push_back(BufferUpdateSlot{leaf, pool.tensor(carried.next),
                                                             K_WHOLE, K_WHOLE});
                }

                parts.outputs = traced.outputs;
                return parts;
            });
        }

        Result<std::unique_ptr<EvaluationStep>> EvaluationStep::make(
            SushiBLAS::Engine& engine, const TracedStep& traced, const NN::ModelState& state,
            Graph::TensorPool& pool, const Graph::FusionSelector* selector)
        {
            SA_FAIL_IF(traced.input == Graph::K_NO_VALUE, StatusCode::InvalidDescription,
                       "The evaluation step needs the input leaf of its traced graph");

            // The plan is adopted before any leaf or value of the graph is materialised.
            if (!pool.has_plan())
            {
                SA_TRY_ASSIGN(const std::vector<Graph::ValueID> pinned, pinned_values(traced));

                Graph::PlanOptions options;
                options.pinned = SushiRuntime::span<const Graph::ValueID>(pinned.data(),
                                                                          pinned.size());
                SA_TRY(Detail::adopt_plan(traced.graph, pool, options));
            }

            SA_TRY(require_kept(pool, traced.carried));
            SA_TRY(state.attach(traced.graph, pool));
            SA_TRY(pool.materialise_leaves(traced.graph));
            SA_TRY(fill_host_tables(traced.tables, pool));

            // gather() reads the tensor of each next value before the step is compiled.
            for (const Graph::ValueID next : traced.carried.sources())
                SA_TRY(pool.materialise(next, traced.graph.value(next)));

            SA_TRY_ASSIGN(Parts parts, gather(traced, pool));
            SA_TRY_ASSIGN(std::unique_ptr<EvaluationStep> step,
                          ::SushiAI::Detail::guard_value(::SushiAI::Detail::SeamPhase::Allocate,
                                                         [&]()
            {
                return std::unique_ptr<EvaluationStep>(
                    new EvaluationStep(engine, pool, std::move(parts)));
            }));

            // A step that carries nothing has no trailing work and compiles as a plain graph.
            Graph::StepRecorder* const trailing =
                step->carries_state() ? &step->carries_ : nullptr;
            SA_TRY_ASSIGN(step->step_, Graph::CompiledStep::compile(engine, traced.graph, pool,
                                                                    trailing, selector));
            SA_TRY(step->reset());

            return std::move(step);
        }

        EvaluationStep::EvaluationStep(SushiBLAS::Engine& engine, Graph::TensorPool& pool,
                                       Parts parts)
            : engine_(&engine),
              pool_(&pool),
              input_names_(std::move(parts.input_names)),
              input_leaves_(std::move(parts.input_leaves)),
              ordered_(input_leaves_.size(), nullptr),
              carried_(std::move(parts.carried)),
              outputs_(std::move(parts.outputs)),
              carries_(std::move(parts.carries)),
              step_()
        {

        }

        EvaluationStep::~EvaluationStep() = default;

        Status EvaluationStep::run(const SushiBLAS::Tensor& inputs)
        {
            const SushiBLAS::Tensor* const one = &inputs;
            return run_in_order(FeedSpan(&one, 1));
        }

        Status EvaluationStep::run_in_order(FeedSpan feeds)
        {
            SA_FAIL_IF(feeds.size() != input_leaves_.size(), StatusCode::Incompatible,
                       "The run was given {} tensors, but the compiled step is fed {}: {}",
                       feeds.size(), input_leaves_.size(), list_inputs());

            for (std::size_t at = 0; at < feeds.size(); ++at)
            {
                SA_FATAL_IF(feeds[at] == nullptr, "The feed of the input leaf '{}' is null",
                            input_names_[at]);
                SA_TRY(check_feed(*feeds[at], *input_leaves_[at], input_names_[at]));
            }

            SA_TRY(::SushiAI::Detail::record_and_run(*engine_, [&]() -> Status
            {
                SushiBLAS::BLASOps blas(*engine_);
                for (std::size_t at = 0; at < feeds.size(); ++at)
                    blas.copy(*feeds[at], *input_leaves_[at]);

                return Status();
            }));

            return step_->run();
        }

        Status EvaluationStep::run_by_name(SushiRuntime::span<const NamedFeed> feeds)
        {
            SA_FAIL_IF(feeds.size() != input_leaves_.size(), StatusCode::InvalidDescription,
                       "The run was given {} named tensors, but the compiled step is fed {}: {}",
                       feeds.size(), input_leaves_.size(), list_inputs());

            // ordered_ holds one slot per leaf; a slot left null names a leaf the run left out.
            for (const SushiBLAS::Tensor*& slot : ordered_)
                slot = nullptr;

            for (const NamedFeed& feed : feeds)
            {
                std::size_t at = 0;
                while (at < input_names_.size() && input_names_[at] != feed.name)
                    ++at;

                SA_FAIL_IF(at == input_names_.size(), StatusCode::InvalidDescription,
                           "The run feeds '{}', which is no input leaf of the compiled step; "
                           "it is fed: {}",
                           feed.name, list_inputs());
                SA_FAIL_IF(ordered_[at] != nullptr, StatusCode::InvalidDescription,
                           "The run feeds '{}' twice; a run feeds each of {} once", feed.name,
                           list_inputs());
                SA_FATAL_IF(feed.tensor == nullptr, "The feed of the input leaf '{}' is null",
                            feed.name);

                ordered_[at] = feed.tensor;
            }

            return run_in_order(FeedSpan(ordered_.data(), ordered_.size()));
        }

        Status EvaluationStep::reset()
        {
            if (carried_.empty())
                return Status();

            for (const CarriedLeaf& leaf : carried_)
                SA_TRY(pool_->request_constant(*leaf.tensor, leaf.initial_value));

            return pool_->flush_constants();
        }

        bool EvaluationStep::carries_state() const noexcept
        {
            return !carried_.empty();
        }

        SushiRuntime::span<const std::string> EvaluationStep::input_names() const noexcept
        {
            return SushiRuntime::span<const std::string>(input_names_.data(),
                                                         input_names_.size());
        }

        Result<const SushiBLAS::Tensor*> EvaluationStep::input(std::string_view name) const
        {
            for (std::size_t at = 0; at < input_names_.size(); ++at)
            {
                if (input_names_[at] == name)
                    return static_cast<const SushiBLAS::Tensor*>(input_leaves_[at]);
            }

            return Status::from_message(StatusCode::InvalidDescription, [&]()
            {
                return SushiBLAS::Core::format(
                    "The compiled step has no input leaf named '{}'; it is fed: {}", name,
                    list_inputs());
            });
        }

        Result<const SushiBLAS::Tensor*> EvaluationStep::output(std::string_view name) const
        {
            SA_TRY_ASSIGN(const Graph::ValueID id, outputs_.at(name));

            const Graph::TensorPool& pool = *pool_;
            return &pool.tensor(id);
        }

        const Graph::CompiledStep& EvaluationStep::step() const noexcept
        {
            return *step_;
        }

        Status EvaluationStep::restore_run_count(std::size_t count)
        {
            return step_->restore_run_count(count);
        }

        std::string EvaluationStep::list_inputs() const
        {
            std::string listed;
            for (const std::string& name : input_names_)
            {
                if (!listed.empty())
                    listed.append(", ");

                listed.append("'").append(name).append("'");
            }

            return listed;
        }

    } // namespace Train
} // namespace SushiAI
