/**************************************************************************/
/* gradient_exchange.hpp                                                  */
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
 * @file gradient_exchange.hpp
 * @brief Declares GradientExchange, the replica tasks and their ordered sum.
 * @author Mustafa Garip
 * @see modules/core/distribution/README.md
 */

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>
#include <SushiBLAS/cluster/cluster_binding.hpp>
#include <SushiBLAS/cluster/pack_layout.hpp>
#include <SushiBLAS/cluster/tensor_pack.hpp>
#include <SushiBLAS/engine/engine.hpp>
#include <SushiBLAS/tensor/tensor.hpp>
#include <SushiBLAS/engine/task_recorder.hpp>
#include <SushiRuntime/sushiruntime.hpp>
#include <SushiRuntime/core/operation_id.hpp>
#include <SushiAI/distribution/replica_plan.hpp>
#include <SushiAI/distribution/replica_task.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/graph/step_recorder.hpp>
#include <SushiAI/graph/step_recording.hpp>

namespace SushiAI
{
    namespace Distribution
    {
        /** @brief Names the replica operation, the run and the epoch an exchange serves. */
        struct ExchangeRun
        {
            /** @brief The operation the replica kernel registers under. */
            SushiRuntime::OperationID operation{};

            /** @brief The run's seed, word 3 of every replica node. */
            std::uint64_t seed = 0;

            /** @brief The index of the run, word 2 of every replica node. */
            std::uint64_t run_index = 0;

            /** @brief The batches of one epoch; at least 1. */
            std::uint64_t batches_per_epoch = 1;
        };

        /**
         * @brief Records one replica task per replica, each in the placement scope
         *        K_REPLICA_PLACEMENT_CLASS with key k, and their ordered sum into the targets.
         *
         * A floating or complex entry weighs replica k by ReplicaPlan::weight_of(k); an
         * integer entry adds the replicas with weight 1.
         */
        class GradientExchange final : public Graph::StepRecorder
        {
            public:
                /**
                 * @brief Allocates and shares the position pack and the result packs.
                 * @param targets One contiguous tensor per entry of @p result_layout, in order, on
                 *        the device of @p parameters; @p binding and @p parameters outlive it.
                 * @return The exchange; InvalidDescription for an unshared parameter pack, no
                 *         batch, too many replicas for ordered_sum(), a target unlike its entry
                 *         or on another device; Unsupported for a uint32 target.
                 */
                [[nodiscard]] static Result<std::unique_ptr<GradientExchange>> make(
                    SushiBLAS::Engine& engine, SushiBLAS::ClusterBinding& binding,
                    const ReplicaPlan& plan, const ExchangeRun& run,
                    const SushiBLAS::TensorPack& parameters,
                    const SushiBLAS::PackLayout& result_layout,
                    std::vector<SushiBLAS::Tensor> targets) noexcept;

                GradientExchange(const GradientExchange&) = delete;
                GradientExchange& operator=(const GradientExchange&) = delete;
                GradientExchange(GradientExchange&&) = delete;
                GradientExchange& operator=(GradientExchange&&) = delete;
                ~GradientExchange() override = default;

                /**
                 * @brief Records the position gather, the replica tasks, one ordered sum per
                 *        filled type segment and one scatter of the sums into the targets.
                 * @return 2 + replica_count() + the filled segment count, or the failure of a
                 *         recording.
                 */
                [[nodiscard]] Result<std::size_t> record_into_step(
                    const Graph::StepRecording& recording) override;

                /** @brief Writes the position of the coming replay and counts it. */
                void before_replay() override;

                /** @brief Returns how many replicas each replay runs. */
                [[nodiscard]] std::size_t replica_count() const noexcept
                {
                    return result_packs_.size();
                }

                /** @brief Returns the step the next replay serves, counted from 0. */
                [[nodiscard]] std::uint64_t next_step() const noexcept
                {
                    return next_step_;
                }

                /** @brief Sets the step the next replay serves, as a resumed run needs. */
                void set_next_step(std::uint64_t step) noexcept
                {
                    next_step_ = step;
                }

            private:
                /** @brief Holds what make() built; the constructor moves it in. */
                struct Parts;

                /** @brief Takes what make() built. */
                GradientExchange(SushiBLAS::Engine& engine, SushiBLAS::ClusterBinding& binding,
                                 const SushiBLAS::TensorPack& parameters, Parts parts) noexcept;

                /** @brief Records the replica tasks, each inside its placement scope. */
                void record_replicas(SushiRuntime::Recorder& graph) const;

                /** @brief Records one ordered sum per filled segment over the result packs. */
                void record_sums(SushiBLAS::TaskRecorder& tasks);

                /// @brief The engine whose runtime gives the placement scopes.
                SushiBLAS::Engine* engine_;

                /// @brief The binding the packs are shared through.
                SushiBLAS::ClusterBinding* binding_;

                /// @brief The parameter pack, input 0 of every replica node.
                const SushiBLAS::TensorPack* parameters_;

                /// @brief The operation and the run every replica node names.
                ExchangeRun run_;

                /// @brief The epoch, the batch and the step of the coming replay, as three Int64.
                SushiBLAS::Tensor position_;

                /// @brief The pack position_ is gathered into, input 1 of every replica node.
                SushiBLAS::TensorPack position_pack_;

                /// @brief The sums, laid out as a result pack; the master's alone, not shared.
                SushiBLAS::TensorPack sum_pack_;

                /// @brief Entry k is replica k's result pack.
                std::vector<SushiBLAS::TensorPack> result_packs_;

                /// @brief The types whose segment of the result layout holds an element.
                std::vector<SushiBLAS::Core::DataType> segments_;

                /// @brief Entry k holds the parameters of replica k's node.
                std::vector<std::array<std::uint64_t, K_REPLICA_TASK_WORDS>> words_;

                /// @brief Entry k holds the name of replica k's node.
                std::vector<std::string> names_;

                /// @brief Entry k is replica k's weight in a floating or complex sum.
                std::vector<double> weights_;

                /// @brief The tensors the sums write, in result pack order.
                std::vector<SushiBLAS::Tensor> targets_;

                /// @brief The step the next replay serves.
                std::uint64_t next_step_ = 0;
        };

    } // namespace Distribution
} // namespace SushiAI
