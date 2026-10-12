/**************************************************************************/
/* session.cpp                                                            */
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
 * @file session.cpp
 * @brief Implements Session, which plans each step against the one before it and runs it.
 * @author Mustafa Garip
 * @see modules/core/eager/README.md
 */

#include <SushiAI/eager/session.hpp>

#include <chrono>
#include <utility>

#include <SushiBLAS/blas/blas.hpp>
#include <SushiBLAS/random/random.hpp>
#include <SushiRuntime/core/span.hpp>

#include <SushiAI/core/allocation.hpp>
#include <SushiAI/core/check.hpp>
#include <SushiAI/core/seam.hpp>
#include <SushiAI/eager/region_boundary.hpp>
#include <SushiAI/eager/step.hpp>

#include "tensor_shape.hpp"

namespace SushiAI
{
    namespace Eager
    {
        namespace
        {
            /** @brief Measures the host time between two points of a flush. */
            class Stopwatch final
            {
                public:
                    /** @brief Returns the nanoseconds since the last lap, or since construction. */
                    [[nodiscard]] std::uint64_t lap() noexcept
                    {
                        const std::chrono::steady_clock::time_point now =
                            std::chrono::steady_clock::now();
                        const std::uint64_t elapsed = static_cast<std::uint64_t>(
                            std::chrono::duration_cast<std::chrono::nanoseconds>(now - last_)
                                .count());
                        last_ = now;
                        return elapsed;
                    }

                private:
                    /// @brief The time of the last lap.
                    std::chrono::steady_clock::time_point last_ = std::chrono::steady_clock::now();
            };

            /** @brief Counts the regions of @p plan by the action taken for each. */
            void count_actions(const StepPlan& plan, StepCost& cost) noexcept
            {
                for (const RegionOrder& order : plan.regions)
                {
                    if (order.action == RegionAction::Keep)
                        ++cost.kept;
                    else if (order.action == RegionAction::Record)
                        ++cost.recorded;
                    else
                        ++cost.built;
                }

                cost.dropped = plan.dropped.size();
            }

        } // namespace

        Session::Session(SushiBLAS::Engine& engine, StepExecutor& executor,
                         const RegionPartitioner& partitioner, SessionOptions options)
            : engine_(&engine), executor_(&executor), partitioner_(&partitioner),
              options_(options), bank_(engine), table_(ids_)
        {

        }

        Status Session::bind_state(Graph::StateID state, const SushiBLAS::Tensor& tensor,
                                   bool trainable)
        {
            SA_FAIL_IF(state == Graph::K_NO_STATE, StatusCode::InvalidDescription,
                       "A state without identity cannot be bound; draw one with "
                       "StateIdentity::get() first");
            SA_FAIL_IF(find_state(state) != nullptr, StatusCode::InvalidDescription,
                       "State {} is already bound to this session",
                       static_cast<std::uint64_t>(state));
            SA_FAIL_IF(!tensor.storage || tensor.num_elements == 0,
                       StatusCode::InvalidDescription,
                       "State {} is bound to a tensor that holds no element",
                       static_cast<std::uint64_t>(state));

            BoundState bound;
            bound.shape = shape_of(tensor);
            bound.dtype = tensor.dtype;
            bound.storage = ids_.next();
            SA_TRY(bank_.adopt(bound.storage, tensor));

            if (trainable)
            {
                SA_TRY_ASSIGN(SushiBLAS::Tensor gradient, allocate_like(*engine_, tensor));
                bound.gradient = ids_.next();
                SA_TRY(bank_.adopt(bound.gradient, std::move(gradient)));

                // An update may read the gradient of a parameter no step has traced yet.
                SushiBLAS::Tensor& zeroed = bank_.tensor(bound.gradient);
                SA_TRY(::SushiAI::Detail::record_and_run(*engine_, [&]()
                {
                    SushiBLAS::RandomOps filler(*engine_);
                    filler.constant(zeroed, 0.0);
                    return Status();
                }));
                bound.gradient_is_zero = true;
            }

            return ::SushiAI::Detail::guard(::SushiAI::Detail::SeamPhase::Allocate, [&]()
            {
                states_.emplace(state, bound);
                return Status();
            });
        }

        Result<SushiBLAS::Tensor> Session::gradient(Graph::StateID state) const
        {
            const BoundState* const bound = find_state(state);
            SA_FAIL_IF(bound == nullptr || bound->gradient == K_NO_STORAGE,
                       StatusCode::InvalidDescription,
                       "State {} is not bound as trainable, so the session keeps no gradient "
                       "for it",
                       static_cast<std::uint64_t>(state));

            return SushiBLAS::Tensor(bank_.tensor(bound->gradient));
        }

        Step Session::begin_step()
        {
            SA_FATAL_IF(open_, "A step of this session is still alive; a session traces one "
                               "step at a time");
            open_ = true;
            return Step(*this);
        }

        const Session::BoundState* Session::find_state(Graph::StateID state) const
        {
            const auto found = states_.find(state);
            return found != states_.end() ? &found->second : nullptr;
        }

        Result<StorageID> Session::input_buffer(std::size_t ordinal, const Graph::Value& leaf)
        {
            SA_TRY(::SushiAI::Detail::guard(::SushiAI::Detail::SeamPhase::Allocate, [&]()
            {
                if (inputs_.size() <= ordinal)
                    inputs_.resize(ordinal + 1);
                return Status();
            }));

            InputBuffer& buffer = inputs_[ordinal];
            if (buffer.storage != K_NO_STORAGE && buffer.shape == leaf.shape
                && buffer.dtype == leaf.dtype)
                return buffer.storage;

            const StorageID fresh = ids_.next();
            SA_TRY(bank_.allocate(fresh, leaf.shape, leaf.dtype));

            bank_.release(buffer.storage);
            buffer.storage = fresh;
            buffer.shape = leaf.shape;
            buffer.dtype = leaf.dtype;
            return fresh;
        }

        Status Session::place_leaves(const StepTrace& trace, std::vector<StorageID>& preset)
        {
            const Graph::Graph& graph = trace.graph;
            SA_TRY(::SushiAI::Detail::guard(::SushiAI::Detail::SeamPhase::Allocate, [&]()
            {
                preset.assign(graph.value_count(), K_NO_STORAGE);
                return Status();
            }));

            for (const std::pair<Graph::StateID, Graph::ValueID>& leaf : graph.state_leaves())
            {
                const BoundState* const bound = find_state(leaf.first);
                SA_FAIL_IF(bound == nullptr, StatusCode::InvalidDescription,
                           "The step traced state {} as value {}, and the session has no "
                           "tensor bound for that state",
                           static_cast<std::uint64_t>(leaf.first), leaf.second);

                const Graph::Value& value = graph.value(leaf.second);
                SA_FAIL_IF(value.shape != bound->shape || value.dtype != bound->dtype,
                           StatusCode::Incompatible,
                           "State {} is bound as {} {}, but the step traced it as {} {}",
                           static_cast<std::uint64_t>(leaf.first), dtype_name(bound->dtype),
                           bound->shape.to_string(), dtype_name(value.dtype),
                           value.shape.to_string());
                preset[leaf.second] = bound->storage;
            }

            for (std::size_t ordinal = 0; ordinal < trace.feeds.size(); ++ordinal)
            {
                const Feed& feed = trace.feeds[ordinal];
                SA_FAIL_IF(feed.leaf >= graph.value_count(), StatusCode::InvalidDescription,
                           "Feed {} names value {}, which the step did not trace", ordinal,
                           feed.leaf);
                SA_FAIL_IF(preset[feed.leaf] != K_NO_STORAGE, StatusCode::InvalidDescription,
                           "Value {} is fed twice, or is fed and stands for a state",
                           feed.leaf);

                const Graph::Value& value = graph.value(feed.leaf);
                SA_FAIL_IF(feed.source.num_elements != value.shape.num_elements()
                               || feed.source.dtype != value.dtype,
                           StatusCode::Incompatible,
                           "Feed {} holds {} {} elements, but its leaf is {} {}", ordinal,
                           feed.source.num_elements, dtype_name(feed.source.dtype),
                           dtype_name(value.dtype), value.shape.to_string());

                SA_TRY_ASSIGN(const StorageID buffer, input_buffer(ordinal, value));
                preset[feed.leaf] = buffer;
            }

            for (const StateGradient& gradient : trace.gradients)
            {
                if (gradient.value == Graph::K_NO_VALUE)
                    continue;

                const BoundState* const bound = find_state(gradient.state);
                SA_FAIL_IF(bound == nullptr || bound->gradient == K_NO_STORAGE,
                           StatusCode::InvalidDescription,
                           "The step differentiated with respect to state {}, which is not "
                           "bound as trainable",
                           static_cast<std::uint64_t>(gradient.state));
                SA_FAIL_IF(preset[gradient.value] != K_NO_STORAGE
                               && preset[gradient.value] != bound->gradient,
                           StatusCode::Unsupported,
                           "Value {} is the gradient of state {} and of another state or "
                           "leaf; the session keeps one gradient tensor per state",
                           gradient.value, static_cast<std::uint64_t>(gradient.state));
                preset[gradient.value] = bound->gradient;
            }

            return Status();
        }

        Status Session::prepare_storage(const StepTrace& trace,
                                        const std::vector<StorageID>& preset)
        {
            for (const StateGradient& gradient : trace.gradients)
            {
                if (gradient.value != Graph::K_NO_VALUE)
                    states_.find(gradient.state)->second.written_in = steps_ + 1;
            }

            std::vector<SushiBLAS::Tensor*> stale;
            SA_TRY(::SushiAI::Detail::guard(::SushiAI::Detail::SeamPhase::Allocate, [&]()
            {
                for (auto& entry : states_)
                {
                    BoundState& bound = entry.second;
                    if (bound.gradient == K_NO_STORAGE)
                        continue;

                    if (bound.written_in == steps_ + 1)
                        bound.gradient_is_zero = false;
                    else if (trace.trailing != nullptr && !bound.gradient_is_zero)
                    {
                        stale.push_back(&bank_.tensor(bound.gradient));
                        bound.gradient_is_zero = true;
                    }
                }

                return Status();
            }));

            if (trace.feeds.empty() && stale.empty())
                return Status();

            // The copies and the fills run on the engine's own graph, before any region does.
            return ::SushiAI::Detail::record_and_run(*engine_, [&]()
            {
                SushiBLAS::BLASOps blas(*engine_);
                for (const Feed& feed : trace.feeds)
                    blas.copy(feed.source, bank_.tensor(preset[feed.leaf]));

                SushiBLAS::RandomOps filler(*engine_);
                for (SushiBLAS::Tensor* gradient : stale)
                    filler.constant(*gradient, 0.0);

                return Status();
            });
        }

        Status Session::recover(const Status& failure)
        {
            for (StorageID id : table_.forget())
                bank_.release(id);
            plan_ = StepPlan{};

            // A fill the failed step asked for may not have run.
            for (auto& entry : states_)
                entry.second.gradient_is_zero = false;

            const Status discarded = executor_->discard();
            if (discarded.is_ok())
                return failure;

            return SA_ANNOTATE(failure, "the regions of the failed step could not be dropped "
                                        "({})",
                               discarded.message());
        }

        Status Session::run(const StepTrace& trace)
        {
            Stopwatch total;
            Stopwatch stage;
            StepCost cost;

            SA_FAIL_IF(trace.graph.has_refusal(), StatusCode::InvalidDescription,
                       "This step holds a refused operation and cannot run; the builder call "
                       "that refused it returned the reason");

            const Graph::Graph& graph = trace.graph;
            SA_TRY_ASSIGN(const Partition partition,
                          ::SushiAI::Detail::guard_value(::SushiAI::Detail::SeamPhase::Record,
                                                         [&]()
            {
                return partitioner_->assign(
                    graph,
                    SushiRuntime::span<const Segment>(trace.segments.data(),
                                                      trace.segments.size()),
                    trace.provenance);
            }));
            cost.partition_ns = stage.lap();

            std::vector<StorageID> preset;
            SA_TRY(place_leaves(trace, preset));

            StepWork work;
            Result<StepPlan> planned = ::SushiAI::Detail::guard_value(
                ::SushiAI::Detail::SeamPhase::Record, [&]()
            {
                std::vector<Graph::ValueID> observed = trace.observed;
                for (const StateGradient& gradient : trace.gradients)
                {
                    if (gradient.value != Graph::K_NO_VALUE)
                        observed.push_back(gradient.value);
                }

                const BoundaryMap boundaries = find_boundaries(
                    graph, partition,
                    SushiRuntime::span<const Graph::ValueID>(observed.data(), observed.size()));

                StepPlan plan = table_.plan(
                    graph, partition, boundaries,
                    SushiRuntime::span<const StorageID>(preset.data(), preset.size()));
                cost.plan_ns = stage.lap();

                work.regions.resize(plan.regions.size());
                for (std::size_t region = 0; region < plan.regions.size(); ++region)
                {
                    work.regions[region].order = plan.regions[region];
                    if (plan.regions[region].action == RegionAction::Build)
                        work.regions[region].program =
                            compile_region(graph, partition, plan.regions[region].key,
                                           boundaries.regions[region], options_.program);
                }
                cost.compile_ns = stage.lap();
                return plan;
            });
            if (!planned.has_value())
                return recover(planned.status());

            StepPlan plan = std::move(planned).value();
            count_actions(plan, cost);

            Status ran;
            for (const SlotRecord& record : plan.created)
            {
                ran = bank_.allocate(record.id, record.shape, record.dtype);
                if (!ran.is_ok())
                    return recover(ran);
            }

            ran = prepare_storage(trace, preset);
            if (!ran.is_ok())
                return recover(ran);

            work.dropped = plan.dropped;
            work.storage_of_value = SushiRuntime::span<const StorageID>(
                plan.storage_of_value.data(), plan.storage_of_value.size());
            work.trailing = trace.trailing;
            work.trailing_key = partition.optimizer_key;
            work.run_index = steps_;

            ran = executor_->execute(work, bank_);
            if (!ran.is_ok())
                return recover(ran);

            for (StorageID id : plan.released)
                bank_.release(id);

            cost.execute_ns = stage.lap();
            cost.total_ns = total.lap();

            plan_ = std::move(plan);
            cost_ = cost;
            ++steps_;
            return Status();
        }

        Result<const SushiBLAS::Tensor*> Session::tensor_of(Graph::ValueID value) const
        {
            const StorageID id = value < plan_.storage_of_value.size()
                                     ? plan_.storage_of_value[value]
                                     : K_NO_STORAGE;
            SA_FAIL_IF(id == K_NO_STORAGE || !bank_.holds(id), StatusCode::InvalidDescription,
                       "Value {} was not kept by the step that ran; pass it to Step::keep() "
                       "or read it before the step runs",
                       value);

            return &bank_.tensor(id);
        }

    } // namespace Eager
} // namespace SushiAI
