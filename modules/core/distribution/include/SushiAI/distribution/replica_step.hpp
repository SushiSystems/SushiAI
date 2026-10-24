/**************************************************************************/
/* replica_step.hpp                                                       */
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
 * @file replica_step.hpp
 * @brief Declares ReplicaStep, a replica's compiled forward and backward step on its own runtime.
 * @author Mustafa Garip
 * @see modules/core/distribution/README.md
 */

#pragma once

#include <cstdint>
#include <memory>
#include <SushiBLAS/cluster/bound_pack.hpp>
#include <SushiBLAS/cluster/pack_layout.hpp>
#include <SushiBLAS/engine/engine.hpp>
#include <SushiRuntime/sushiruntime.hpp>
#include <SushiAI/distribution/replica_draws.hpp>
#include <SushiAI/distribution/replica_model.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/graph/compiled_step.hpp>

namespace SushiAI
{
    namespace Distribution
    {
        class ReplicaAccumulator;

        /**
         * @brief Returns the configuration of the runtime a replica step creates.
         *
         * One pool thread, not pinned, so the replica's runtime takes no core the runtime
         * that serves the kernel pinned; every device the default filter takes.
         */
        [[nodiscard]] SushiRuntime::RuntimeConfig replica_runtime_config();

        /**
         * @brief Owns a replica's runtime, its model, the compiled forward and backward step
         *        with no optimizer, and the sum of its results over its micro-batches.
         *
         * A registered kernel runs inside a graph run of its own runtime, which refuses a
         * nested run, so the step runs on a runtime of its own.
         */
        class ReplicaStep final
        {
            public:
                /**
                 * @brief Creates the runtime from @p config, builds the model through
                 *        @p factory, lays out both packs and compiles the step.
                 * @return The step, or the failure of the runtime, the factory, a layout or the
                 *         compilation; InvalidDescription for a micro-batch below 1, or a
                 *         model with no parameter or no result.
                 */
                [[nodiscard]] static Result<std::unique_ptr<ReplicaStep>> make(
                    const ReplicaModelFactory& factory, const ReplicaDraws& draws,
                    std::int64_t micro_batch_rows,
                    const SushiRuntime::RuntimeConfig& config = replica_runtime_config());

                ReplicaStep(const ReplicaStep&) = delete;
                ReplicaStep& operator=(const ReplicaStep&) = delete;
                ReplicaStep(ReplicaStep&&) = delete;
                ReplicaStep& operator=(ReplicaStep&&) = delete;
                ~ReplicaStep();

                /** @brief Returns where each parameter lies in the parameter pack. */
                [[nodiscard]] const SushiBLAS::PackLayout& parameter_layout() const noexcept
                {
                    return parameter_layout_;
                }

                /** @brief Returns where each gradient, the loss and each statistic lies. */
                [[nodiscard]] const SushiBLAS::PackLayout& result_layout() const noexcept
                {
                    return result_layout_;
                }

                /** @brief Returns the hash of the model and of both pack layouts. */
                [[nodiscard]] std::uint64_t fingerprint() const noexcept
                {
                    return fingerprint_;
                }

                /** @brief Returns the seed, replica and run the model was built for. */
                [[nodiscard]] const ReplicaDraws& draws() const noexcept
                {
                    return draws_;
                }

                /**
                 * @brief Copies the parameter pack into the model's parameters and forgets
                 *        every micro-batch run before.
                 * @pre @p parameters was bound to parameter_layout().
                 */
                [[nodiscard]] Status begin(const SushiBLAS::BoundPack& parameters);

                /**
                 * @brief Feeds one micro-batch, replays the step and adds its results.
                 * @return Success, or the failure of the feed or of the replay.
                 */
                [[nodiscard]] Status run_micro_batch(const ReplicaRows& rows);

                /**
                 * @brief Writes the mean of the results of the micro-batches since begin().
                 * @param results Bound to result_layout().
                 * @return Success, or what ReplicaAccumulator::write_mean() refuses.
                 */
                [[nodiscard]] Status finish(const SushiBLAS::BoundPack& results) const;

            private:
                /**
                 * @brief Takes the runtime, the engine built on it and the model built on the
                 *        engine; make() adds the step.
                 */
                ReplicaStep(std::unique_ptr<SushiRuntime::Runtime> runtime,
                            std::unique_ptr<SushiBLAS::Engine> engine,
                            std::unique_ptr<ReplicaModel> model,
                            SushiBLAS::PackLayout parameter_layout,
                            SushiBLAS::PackLayout result_layout, const ReplicaDraws& draws);

                /// @brief The replica's own runtime, at the address engine_ holds; it dies last.
                std::unique_ptr<SushiRuntime::Runtime> runtime_;

                /// @brief The engine over runtime_ the model and the step record on.
                std::unique_ptr<SushiBLAS::Engine> engine_;

                /// @brief The model, its pool and its feed; dies after step_.
                std::unique_ptr<ReplicaModel> model_;

                /// @brief Where each parameter lies in the parameter pack.
                SushiBLAS::PackLayout parameter_layout_;

                /// @brief Where each result lies in the result pack.
                SushiBLAS::PackLayout result_layout_;

                /// @brief The sum of the results of each micro-batch since begin().
                std::unique_ptr<ReplicaAccumulator> accumulator_;

                /// @brief The seed, replica and run the model was built for.
                ReplicaDraws draws_;

                /// @brief The hash of the model and of both layouts.
                std::uint64_t fingerprint_ = 0;

                /// @brief The compiled forward and backward step; declared last so it dies first.
                std::unique_ptr<Graph::CompiledStep> step_;
        };

    } // namespace Distribution
} // namespace SushiAI
