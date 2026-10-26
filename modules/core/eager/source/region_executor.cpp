/**************************************************************************/
/* region_executor.cpp                                                    */
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
 * @file region_executor.cpp
 * @brief Implements RegionExecutor, which runs a planned step on a DynamicGraph.
 * @author Mustafa Garip
 * @see modules/core/eager/README.md
 */

#include <SushiAI/eager/region_executor.hpp>

#include <cstddef>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <SushiBLAS/engine/task_recorder.hpp>

#include <SushiAI/failure/result.hpp>
#include <SushiAI/seam/seam.hpp>

namespace SushiAI
{
    namespace Eager
    {
        namespace
        {
            /** @brief Holds the label of the graph that composes the regions of a step. */
            constexpr std::string_view K_GRAPH_LABEL = "eager.step";

            /** @brief Holds the origin of the tasks the trailing recorder adds. */
            constexpr std::string_view K_TRAILING_ORIGIN = "step.trailing";

            /** @brief Returns the refusal @p recorder kept as a Status, or success. */
            [[nodiscard]] Status refusal_of(const SushiRuntime::Recorder& recorder)
            {
                if (const SushiRuntime::Error* refused = recorder.refusal())
                    return ::SushiAI::Detail::status_of_error(
                        *refused, ::SushiAI::Detail::SeamPhase::Record);

                return Status();
            }

            /**
             * @brief Lists the storage of each of @p values in a step, in their order.
             * @return An entry is K_NO_STORAGE for a value the step placed nowhere.
             */
            [[nodiscard]] std::vector<StorageID> storage_of(
                const std::vector<Graph::ValueID>& values,
                SushiRuntime::span<const StorageID> storage_of_value)
            {
                std::vector<StorageID> ids;
                ids.reserve(values.size());
                for (Graph::ValueID value : values)
                    ids.push_back(value < storage_of_value.size() ? storage_of_value[value]
                                                                  : K_NO_STORAGE);

                return ids;
            }

            /**
             * @brief Binds the tensor of ids[i] to values[i] in @p pool.
             * @return InvalidDescription for a value without storage, or the pool's failure.
             */
            [[nodiscard]] Status bind_shared(const std::vector<Graph::ValueID>& values,
                                             const std::vector<StorageID>& ids,
                                             StorageBank& storage, Graph::TensorPool& pool)
            {
                for (std::size_t at = 0; at < values.size(); ++at)
                {
                    const StorageID id = at < ids.size() ? ids[at] : K_NO_STORAGE;
                    SA_FAIL_IF(id == K_NO_STORAGE || !storage.holds(id),
                               StatusCode::InvalidDescription,
                               "Value {} crosses a region boundary and storage {} holds no "
                               "tensor for it",
                               values[at], id);
                    SA_TRY(pool.bind(values[at], storage.tensor(id)));
                }

                return Status();
            }

        } // namespace

        RegionExecutor::LiveRegion::LiveRegion(SushiBLAS::Engine& engine, RegionID region,
                                               RegionProgram compiled, SharedStorage placed,
                                               const Graph::FusionSelector* selector)
            : id(region), program(std::move(compiled)), shared(std::move(placed)), pool(engine),
              emitter(engine, program, pool, selector, region)
        {

        }

        RegionExecutor::RegionExecutor(SushiBLAS::Engine& engine,
                                       const Graph::FusionSelector* selector)
            : engine_(&engine), selector_(selector), trailing_counts_(engine.runtime()),
              graph_(engine.runtime().dynamic_graph())
        {
            graph_.set_label(std::string(K_GRAPH_LABEL));
        }

        RegionExecutor::~RegionExecutor() = default;

        std::size_t RegionExecutor::compose_count() const noexcept
        {
            return graph_.compose_count();
        }

        std::size_t RegionExecutor::emit_count(RegionID id) const noexcept
        {
            const auto found = emits_.find(id);
            return found != emits_.end() ? found->second : 0;
        }

        const RegionProgram* RegionExecutor::program_of(RegionKey key) const noexcept
        {
            const auto found = live_.find(key);
            return found != live_.end() ? &found->second->program : nullptr;
        }

        std::size_t RegionExecutor::allocated_bytes() const noexcept
        {
            std::size_t bytes = 0;
            for (const auto& entry : live_)
                bytes += entry.second->pool.allocated_bytes();

            return bytes;
        }

        std::size_t RegionExecutor::operation_count() const noexcept
        {
            std::size_t operations = trailing_ != nullptr ? trailing_operations_ : 0;
            for (const auto& entry : live_)
                operations += entry.second->emitter.operation_count();

            return operations;
        }

        void RegionExecutor::retire(const StepWork& work, Retired& retired)
        {
            for (RegionKey key : work.dropped)
            {
                live_.erase(key);
                graph_.drop(key);
            }

            for (const RegionWork& region : work.regions)
            {
                const RegionKey previous = region.order.previous_key;
                if (region.order.action == RegionAction::Keep || previous == K_NO_REGION)
                    continue;

                const auto found = live_.find(previous);
                if (found != live_.end())
                {
                    retired.emplace(previous, std::move(found->second));
                    live_.erase(found);
                }
                graph_.drop(previous);
            }

            const bool trailing_moved =
                work.trailing != trailing_ || work.trailing_key != trailing_key_;
            if (trailing_key_ != K_NO_REGION && trailing_moved)
            {
                graph_.drop(trailing_key_);
                trailing_counts_.release();
                trailing_ = nullptr;
                trailing_key_ = K_NO_REGION;
                trailing_operations_ = 0;
            }
        }

        Status RegionExecutor::install(const RegionOrder& order, RegionProgram program,
                                       SharedStorage shared, StorageBank& storage)
        {
            SA_TRY_ASSIGN(std::unique_ptr<LiveRegion> live,
                          ::SushiAI::Detail::guard_value(::SushiAI::Detail::SeamPhase::Allocate,
                                                         [&]()
            {
                return std::make_unique<LiveRegion>(*engine_, order.id, std::move(program),
                                                    std::move(shared), selector_);
            }));

            SA_TRY(live->pool.adopt(live->program.plan));
            SA_TRY(bind_shared(live->program.imports, live->shared.imports, storage, live->pool));
            SA_TRY(bind_shared(live->program.exports, live->shared.exports, storage, live->pool));

            // The region is listed before it is recorded, so discard() drops a failed recording.
            LiveRegion* const listed = live.get();
            SA_TRY_ASSIGN(SushiRuntime::Region* const target,
                          ::SushiAI::Detail::guard_value(::SushiAI::Detail::SeamPhase::Allocate,
                                                         [&]()
            {
                live_[order.key] = std::move(live);
                ++emits_[order.id];
                ++emits_total_;
                return &graph_.region(order.key);
            }));

            SA_TRY(listed->emitter.emit(target->recorder()));
            return listed->pool.flush_constants();
        }

        Status RegionExecutor::record_trailing(const StepWork& work)
        {
            if (work.trailing == nullptr || work.trailing == trailing_)
                return Status();

            SA_FAIL_IF(work.trailing_key == K_NO_REGION, StatusCode::InvalidDescription,
                       "The trailing work of a step needs a key above every region; got none");

            SA_TRY_ASSIGN(SushiRuntime::Region* const target,
                          ::SushiAI::Detail::guard_value(::SushiAI::Detail::SeamPhase::Allocate,
                                                         [&]()
            {
                return &graph_.region(work.trailing_key);
            }));

            // The key is kept before the recording, so discard() drops a failed one.
            trailing_key_ = work.trailing_key;
            ++trailing_records_;

            // A recording reached here replaces the trailing one retire() dropped, or the first.
            trailing_counts_.release();
            SushiBLAS::TaskRecorder tasks(target->recorder(), engine_->scratch_pool(),
                                          trailing_counts_, engine_->mode());
            const Graph::StepRecording recording(tasks, target->recorder());
            const SushiBLAS::TaskRecorder::OriginScope origin = tasks.origin(K_TRAILING_ORIGIN);
            SA_TRY_ASSIGN(const std::size_t recorded, work.trailing->record_into_step(recording));
            SA_TRY(refusal_of(target->recorder()));

            trailing_ = work.trailing;
            trailing_operations_ = recorded;
            return Status();
        }

        Status RegionExecutor::run(std::uint64_t run_index)
        {
            // The runtime refuses a graph with no task, and a step may lower to none.
            if (graph_.node_count() == 0)
                return Status();

            for (const auto& entry : live_)
            {
                for (const Graph::ReplayPrelude& prelude : entry.second->emitter.preludes())
                    prelude(run_index);
            }

            if (trailing_ != nullptr)
                trailing_->before_replay();

            SA_TRY(::SushiAI::Detail::guard(::SushiAI::Detail::SeamPhase::Execute, [&]()
            {
                const SushiRuntime::Result<SushiRuntime::RunReport*> ran = graph_.run(report_);
                if (!ran)
                    return ::SushiAI::Detail::status_of_error(
                        ran.error(), ::SushiAI::Detail::SeamPhase::Execute);

                return Status();
            }));

            if (trailing_ != nullptr)
                trailing_->after_replay();

            return Status();
        }

        Status RegionExecutor::execute(StepWork& work, StorageBank& storage)
        {
            Retired retired;
            SA_TRY(::SushiAI::Detail::guard(::SushiAI::Detail::SeamPhase::Allocate, [&]()
            {
                retire(work, retired);
                return Status();
            }));

            for (RegionWork& region : work.regions)
            {
                const RegionOrder& order = region.order;
                if (order.action == RegionAction::Keep)
                {
                    SA_FAIL_IF(live_.find(order.key) == live_.end(),
                               StatusCode::InvalidDescription,
                               "The step keeps the region under key {}, and the executor holds "
                               "no region there",
                               order.key);
                    continue;
                }

                if (order.action == RegionAction::Record)
                {
                    const auto found = retired.find(order.previous_key);
                    SA_FAIL_IF(found == retired.end(), StatusCode::InvalidDescription,
                               "The step records the region of key {} again under key {}, and "
                               "the executor holds no region under the first",
                               order.previous_key, order.key);
                    // An unchanged region lives in the storage its building step placed.
                    SA_TRY(install(order, std::move(found->second->program),
                                   std::move(found->second->shared), storage));
                    continue;
                }

                SA_TRY_ASSIGN(SharedStorage shared,
                              ::SushiAI::Detail::guard_value(
                                  ::SushiAI::Detail::SeamPhase::Allocate, [&]()
                {
                    return SharedStorage{
                        storage_of(region.program.imports, work.storage_of_value),
                        storage_of(region.program.exports, work.storage_of_value)};
                }));
                SA_TRY(install(order, std::move(region.program), std::move(shared), storage));
            }

            SA_TRY(record_trailing(work));
            return run(work.run_index);
        }

        Status RegionExecutor::discard()
        {
            return ::SushiAI::Detail::guard(::SushiAI::Detail::SeamPhase::Allocate, [&]()
            {
                for (const auto& entry : live_)
                    graph_.drop(entry.first);
                live_.clear();

                if (trailing_key_ != K_NO_REGION)
                    graph_.drop(trailing_key_);
                trailing_counts_.release();
                trailing_ = nullptr;
                trailing_key_ = K_NO_REGION;
                trailing_operations_ = 0;
                return Status();
            });
        }

    } // namespace Eager
} // namespace SushiAI
