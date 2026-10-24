/**************************************************************************/
/* gradient_exchange.cpp                                                  */
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
 * @file gradient_exchange.cpp
 * @brief Implements GradientExchange: its packs, its recording and the position of each replay.
 * @author Mustafa Garip
 */

#include <SushiAI/distribution/gradient_exchange.hpp>

#include <optional>
#include <string>
#include <utility>
#include <SushiBLAS/cluster/remote_task.hpp>
#include <SushiBLAS/elementwise/elementwise.hpp>
#include <SushiBLAS/engine/task_recorder.hpp>
#include <SushiRuntime/api/graph/node_options.hpp>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/allocation/allocation.hpp>
#include <SushiAI/distribution/replica_position.hpp>
#include <SushiAI/failure/check.hpp>
#include <SushiAI/seam/seam.hpp>

namespace SushiAI
{
    namespace Distribution
    {
        struct GradientExchange::Parts
        {
            /// @brief The run the nodes name.
            ExchangeRun run;

            /// @brief The position tensor.
            SushiBLAS::Tensor position;

            /// @brief The pack the position is gathered into; set once make() allocates it.
            std::optional<SushiBLAS::TensorPack> position_pack;

            /// @brief One result pack per replica.
            std::vector<SushiBLAS::TensorPack> result_packs;

            /// @brief The pack the sums write before one scatter into the targets.
            std::optional<SushiBLAS::TensorPack> sum_pack;

            /// @brief The types whose segment of the result layout holds an element.
            std::vector<SushiBLAS::Core::DataType> segments;

            /// @brief One node's parameters per replica.
            std::vector<std::array<std::uint64_t, K_REPLICA_TASK_WORDS>> words;

            /// @brief One node name per replica.
            std::vector<std::string> names;

            /// @brief One weight per replica.
            std::vector<double> weights;

            /// @brief The tensors the sums write.
            std::vector<SushiBLAS::Tensor> targets;
        };

        namespace
        {
            /** @brief Returns whether a sum of @p dtype refuses every weight other than 1. */
            [[nodiscard]] constexpr bool sums_exactly(SushiBLAS::Core::DataType dtype) noexcept
            {
                return SushiBLAS::Core::is_integer_dtype(dtype);
            }

            /** @brief Returns the types whose segment of @p layout holds an element, in order. */
            [[nodiscard]] std::vector<SushiBLAS::Core::DataType> filled_segments(
                const SushiBLAS::PackLayout& layout)
            {
                std::vector<SushiBLAS::Core::DataType> segments;
                for (std::size_t type = 0; type < SushiBLAS::Core::K_DATA_TYPE_COUNT; ++type)
                {
                    const auto dtype = static_cast<SushiBLAS::Core::DataType>(type);
                    if (layout.segment(dtype).element_count != 0)
                        segments.push_back(dtype);
                }
                return segments;
            }

            /**
             * @brief Checks the arguments of make() that need no allocation.
             * @return Success, or InvalidDescription naming the first rule broken.
             */
            [[nodiscard]] Status require_arguments(const SushiBLAS::ClusterBinding& binding,
                                                   const ReplicaPlan& plan,
                                                   const ExchangeRun& run,
                                                   const SushiBLAS::TensorPack& parameters,
                                                   const SushiBLAS::PackLayout& layout,
                                                   const std::vector<SushiBLAS::Tensor>& targets)
            {
                SA_FAIL_IF(!binding.is_shared(parameters), StatusCode::InvalidDescription,
                           "The parameter pack of a gradient exchange is not shared with the "
                           "cluster");
                SA_FAIL_IF(run.batches_per_epoch == 0, StatusCode::InvalidDescription,
                           "A gradient exchange needs an epoch of at least one batch");
                SA_FAIL_IF(plan.replica_count()
                               > SushiBLAS::ElementwiseOps::K_MAX_ORDERED_SUM_INPUTS,
                           StatusCode::InvalidDescription,
                           "A gradient exchange sums at most {} replicas; the plan holds {}",
                           SushiBLAS::ElementwiseOps::K_MAX_ORDERED_SUM_INPUTS,
                           plan.replica_count());

                const std::vector<SushiBLAS::PackEntry>& entries = layout.entries();
                SA_FAIL_IF(targets.size() != entries.size(), StatusCode::InvalidDescription,
                           "The result pack holds {} tensors and {} targets were given",
                           entries.size(), targets.size());
                for (std::size_t index = 0; index < targets.size(); ++index)
                {
                    SA_FAIL_IF(!SushiBLAS::PackLayout::matches(entries[index], targets[index]),
                               StatusCode::InvalidDescription,
                               "Target {} differs from its result entry in type, layout or "
                               "shape",
                               index);
                    SA_FAIL_IF(!targets[index].storage || !targets[index].is_contiguous(),
                               StatusCode::InvalidDescription,
                               "Target {} has no storage or is not contiguous", index);
                    SA_FAIL_IF(targets[index].dtype == SushiBLAS::Core::DataType::UInt32,
                               StatusCode::Unsupported,
                               "Target {} is uint32, which ordered_sum() does not sum", index);
                }
                return Status();
            }
        } // namespace

        Result<std::unique_ptr<GradientExchange>> GradientExchange::make(
            SushiBLAS::Engine& engine, SushiBLAS::ClusterBinding& binding,
            const ReplicaPlan& plan, const ExchangeRun& run,
            const SushiBLAS::TensorPack& parameters, const SushiBLAS::PackLayout& result_layout,
            std::vector<SushiBLAS::Tensor> targets)
        {
            SA_TRY(require_arguments(binding, plan, run, parameters, result_layout, targets));

            const std::size_t replicas = plan.replica_count();
            Parts parts;
            parts.run = run;
            const std::int64_t position_extent[1] = {2};
            SA_TRY_ASSIGN(parts.position,
                          allocate_tensor(engine,
                                          SushiRuntime::span<const std::int64_t>(position_extent,
                                                                                 1),
                                          SushiBLAS::Core::DataType::Int64));

            SA_TRY(SushiAI::Detail::guard(SushiAI::Detail::SeamPhase::Allocate, [&]()
            {
                const SushiBLAS::PackLayout position_layout = SushiBLAS::PackLayout::of(
                    SushiRuntime::span<const SushiBLAS::Tensor>(&parts.position, 1));
                // ReplicaPosition::read() takes the two words from the pack's first bytes.
                SA_FATAL_IF(position_layout.entries().front().offset_bytes != 0,
                            "GradientExchange: the position pack starts its words at byte {}",
                            position_layout.entries().front().offset_bytes);
                parts.position_pack.emplace(engine.runtime(), position_layout);
                binding.share(*parts.position_pack);

                parts.sum_pack.emplace(engine.runtime(), result_layout);
                parts.segments = filled_segments(result_layout);
                parts.result_packs.reserve(replicas);
                parts.words.reserve(replicas);
                parts.names.reserve(replicas);
                parts.weights.reserve(replicas);
                for (std::size_t replica = 0; replica < replicas; ++replica)
                {
                    parts.result_packs.emplace_back(engine.runtime(), result_layout);
                    binding.share(parts.result_packs.back());

                    ReplicaTask task;
                    task.replica_index = replica;
                    task.replica_count = replicas;
                    task.run_index = run.run_index;
                    task.seed = run.seed;
                    parts.words.push_back(task.encode());
                    parts.names.push_back("sushiai.replica " + std::to_string(replica));
                    parts.weights.push_back(plan.weight_of(replica));
                }
                parts.targets = std::move(targets);
                return Status();
            }));

            return SushiAI::Detail::guard_value(SushiAI::Detail::SeamPhase::Allocate, [&]()
            {
                return std::unique_ptr<GradientExchange>(
                    new GradientExchange(engine, binding, parameters, std::move(parts)));
            });
        }

        GradientExchange::GradientExchange(SushiBLAS::Engine& engine,
                                           SushiBLAS::ClusterBinding& binding,
                                           const SushiBLAS::TensorPack& parameters,
                                           Parts parts) noexcept
            : engine_(&engine), binding_(&binding), parameters_(&parameters), run_(parts.run),
              position_(std::move(parts.position)),
              position_pack_(std::move(*parts.position_pack)),
              sum_pack_(std::move(*parts.sum_pack)), result_packs_(std::move(parts.result_packs)),
              segments_(std::move(parts.segments)),
              words_(std::move(parts.words)), names_(std::move(parts.names)),
              weights_(std::move(parts.weights)), targets_(std::move(parts.targets))
        {

        }

        Result<std::size_t> GradientExchange::record_into_step(
            const Graph::StepRecording& recording)
        {
            SushiBLAS::TaskRecorder& tasks = recording.tasks();
            SA_TRY(SushiAI::Detail::guard(SushiAI::Detail::SeamPhase::Record, [&]()
            {
                position_pack_.record_gather(
                    tasks, SushiRuntime::span<const SushiBLAS::Tensor>(&position_, 1));
                record_replicas(recording.graph());
                record_sums(tasks);
                sum_pack_.record_scatter(
                    tasks, SushiRuntime::span<const SushiBLAS::Tensor>(targets_.data(),
                                                                       targets_.size()));
                return Status();
            }));

            return 1 + result_packs_.size() + segments_.size() + 1;
        }

        void GradientExchange::before_replay()
        {
            const ReplicaPosition position =
                ReplicaPosition::at_step(next_step_, run_.batches_per_epoch);
            std::int64_t* const words = position_.data_as<std::int64_t>();
            words[0] = static_cast<std::int64_t>(position.epoch);
            words[1] = static_cast<std::int64_t>(position.batch_index);
            ++next_step_;
        }

        void GradientExchange::record_replicas(SushiRuntime::Recorder& graph) const
        {
            const SushiBLAS::TensorPack* const inputs[2] = {parameters_, &position_pack_};
            for (std::size_t replica = 0; replica < result_packs_.size(); ++replica)
            {
                SushiRuntime::NodeOptions options;
                options.name = names_[replica];
                options.operation = run_.operation;
                options.parameters = SushiRuntime::span<const std::uint64_t>(
                    words_[replica].data(), words_[replica].size());

                const SushiBLAS::TensorPack* const outputs[1] = {&result_packs_[replica]};
                const SushiRuntime::PlacementScope scope =
                    engine_->runtime().placement_scope(K_REPLICA_PLACEMENT_CLASS,
                                                       static_cast<std::uint64_t>(replica));
                graph.within(scope, [&](SushiRuntime::Recorder& in_scope)
                {
                    SushiBLAS::record_remote(
                        in_scope, *binding_, options,
                        SushiRuntime::span<const SushiBLAS::TensorPack* const>(inputs, 2),
                        SushiRuntime::span<const SushiBLAS::TensorPack* const>(outputs, 1));
                });
            }
        }

        void GradientExchange::record_sums(SushiBLAS::TaskRecorder& tasks)
        {
            SushiBLAS::ElementwiseOps elementwise(tasks);
            const std::vector<double> ones(weights_.size(), 1.0);
            std::vector<SushiBLAS::Tensor> inputs(result_packs_.size());
            for (const SushiBLAS::Core::DataType dtype : segments_)
            {
                for (std::size_t replica = 0; replica < result_packs_.size(); ++replica)
                    inputs[replica] = result_packs_[replica].segment_tensor(dtype);

                SushiBLAS::Tensor sum = sum_pack_.segment_tensor(dtype);
                const std::vector<double>& weights = sums_exactly(dtype) ? ones : weights_;
                elementwise.ordered_sum(
                    SushiRuntime::span<const SushiBLAS::Tensor>(inputs.data(), inputs.size()),
                    SushiRuntime::span<const double>(weights.data(), weights.size()), sum);
            }
        }

    } // namespace Distribution
} // namespace SushiAI
