/**************************************************************************/
/* replica_step.cpp                                                       */
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
 * @file replica_step.cpp
 * @brief Implements ReplicaStep: its runtime, its packs, its replays and the mean it writes.
 * @author Mustafa Garip
 */

#include <SushiAI/distribution/replica_step.hpp>

#include <array>
#include <cstddef>
#include <cstring>
#include <utility>
#include <SushiBLAS/tensor/tensor.hpp>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/distribution/host_access.hpp>
#include <SushiAI/execution/execution.hpp>
#include <SushiAI/failure/check.hpp>
#include <SushiAI/hash/fnv1a.hpp>
#include <SushiAI/seam/seam.hpp>

#include "replica_accumulator.hpp"

namespace SushiAI
{
    namespace Distribution
    {
        namespace
        {
            /** @brief Returns the FNV-1a hash of the words, each read little-endian. */
            [[nodiscard]] std::uint64_t hash_words(const std::array<std::uint64_t, 3>& words)
            {
                std::uint64_t hash = Hash::K_FNV1A_OFFSET_BASIS;
                for (const std::uint64_t word : words)
                    hash = Hash::fnv1a_word(hash, word);

                return hash;
            }

            /** @brief Returns the layout of @p tensors, or the Status of its refusal. */
            [[nodiscard]] Result<SushiBLAS::PackLayout> layout_of(
                SushiRuntime::span<const SushiBLAS::Tensor> tensors)
            {
                return SushiAI::Detail::guard_value(SushiAI::Detail::SeamPhase::Allocate,
                                                    [tensors]()
                {
                    return SushiBLAS::PackLayout::of(tensors);
                });
            }
        } // namespace

        SushiRuntime::RuntimeConfig replica_runtime_config()
        {
            SushiRuntime::RuntimeConfig config;
            config.worker_count = 1;
            config.pinning = SushiRuntime::PinPolicy::Float;
            return config;
        }

        Result<std::unique_ptr<ReplicaStep>> ReplicaStep::make(
            const ReplicaModelFactory& factory, const ReplicaDraws& draws,
            std::int64_t micro_batch_rows, const SushiRuntime::RuntimeConfig& config)
        {
            SA_FAIL_IF(micro_batch_rows < 1, StatusCode::InvalidDescription,
                       "A replica step's micro-batch holds {} rows; it holds at least one",
                       micro_batch_rows);

            SA_TRY_ASSIGN(SushiRuntime::Runtime created, SushiAI::Detail::create_runtime(config));
            SA_TRY_ASSIGN(std::unique_ptr<SushiRuntime::Runtime> runtime,
                          SushiAI::Detail::guard_value(SushiAI::Detail::SeamPhase::Allocate,
                                                       [&created]()
            {
                return std::make_unique<SushiRuntime::Runtime>(std::move(created));
            }));
            SA_TRY_ASSIGN(std::unique_ptr<SushiBLAS::Engine> engine,
                          SushiAI::Detail::guard_value(SushiAI::Detail::SeamPhase::Allocate,
                                                       [&runtime]()
            {
                return std::make_unique<SushiBLAS::Engine>(*runtime);
            }));

            SA_TRY_ASSIGN(std::unique_ptr<ReplicaModel> model,
                          factory(*engine, draws, micro_batch_rows));
            SA_FAIL_IF(model == nullptr, StatusCode::InvalidDescription,
                       "The replica model factory returned no model for replica {}",
                       draws.replica_index);
            SA_FAIL_IF(model->parameters().empty(), StatusCode::InvalidDescription,
                       "Replica {}'s model has no parameter to pack", draws.replica_index);
            SA_FAIL_IF(model->results().empty(), StatusCode::InvalidDescription,
                       "Replica {}'s model has no result to pack", draws.replica_index);

            SA_TRY_ASSIGN(SushiBLAS::PackLayout parameter_layout, layout_of(model->parameters()));
            SA_TRY_ASSIGN(SushiBLAS::PackLayout result_layout, layout_of(model->results()));

            SA_TRY_ASSIGN(std::unique_ptr<ReplicaStep> step,
                          SushiAI::Detail::guard_value(SushiAI::Detail::SeamPhase::Allocate,
                                                       [&]()
            {
                return std::unique_ptr<ReplicaStep>(
                    new ReplicaStep(std::move(runtime), std::move(engine), std::move(model),
                                    std::move(parameter_layout), std::move(result_layout),
                                    draws));
            }));

            SA_TRY_ASSIGN(step->step_, Graph::CompiledStep::compile(*step->engine_,
                                                                    step->model_->graph(),
                                                                    step->model_->pool()));
            return Result<std::unique_ptr<ReplicaStep>>(std::move(step));
        }

        ReplicaStep::ReplicaStep(std::unique_ptr<SushiRuntime::Runtime> runtime,
                                 std::unique_ptr<SushiBLAS::Engine> engine,
                                 std::unique_ptr<ReplicaModel> model,
                                 SushiBLAS::PackLayout parameter_layout,
                                 SushiBLAS::PackLayout result_layout, const ReplicaDraws& draws)
            : runtime_(std::move(runtime)), engine_(std::move(engine)), model_(std::move(model)),
              parameter_layout_(std::move(parameter_layout)),
              result_layout_(std::move(result_layout)),
              accumulator_(std::make_unique<ReplicaAccumulator>(result_layout_)), draws_(draws),
              fingerprint_(hash_words({model_->fingerprint(), parameter_layout_.fingerprint(),
                                       result_layout_.fingerprint()}))
        {

        }

        ReplicaStep::~ReplicaStep() = default;

        Status ReplicaStep::begin(const SushiBLAS::BoundPack& parameters)
        {
            const SushiRuntime::span<const SushiBLAS::Tensor> targets = model_->parameters();
            for (std::size_t index = 0; index < targets.size(); ++index)
            {
                const SushiBLAS::PackEntry& entry = parameter_layout_.entries()[index];
                SA_FATAL_IF(!SushiBLAS::PackLayout::matches(entry, targets[index]),
                            "ReplicaStep::begin: parameter {} no longer matches its layout entry",
                            index);
                if (entry.size_bytes == 0)
                    continue;

                // A bound pack is host memory: SushiRuntime borrows every view as Shared.
                const SushiBLAS::PackSpan source = parameters.tensor(index);
                SushiBLAS::Tensor target = targets[index];
                require_host_visible(target, "Replica parameter", index);
                std::memcpy(target.data(), source.data, entry.size_bytes);
            }

            accumulator_->reset();
            return Status();
        }

        Status ReplicaStep::run_micro_batch(const ReplicaRows& rows)
        {
            SA_TRY(model_->feed(rows));
            SA_TRY(step_->run());
            return SushiAI::Detail::guard(SushiAI::Detail::SeamPhase::Execute, [this]()
            {
                accumulator_->add(model_->results());
                return Status();
            });
        }

        Status ReplicaStep::finish(const SushiBLAS::BoundPack& results) const
        {
            return accumulator_->write_mean(results);
        }

    } // namespace Distribution
} // namespace SushiAI
