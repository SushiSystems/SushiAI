/**************************************************************************/
/* share_step.hpp                                                         */
/**************************************************************************/
/*                          This file is part of:                         */
/*                                 SushiAI                                */
/*                 https://github.com/SushiSystems/SushiAI                */
/*                         https://sushisystems.io                        */
/**************************************************************************/
/* Copyright (c) 2026-present Mustafa Garip & Sushi Systems               */
/*                                                                        */
/* Licensed under the Apache License, Version 2.0 (the "License"); you    */
/* may not use this file except in compliance with the License. You may   */
/* obtain a copy of the License at                                        */
/*                                                                        */
/*     https://www.apache.org/licenses/LICENSE-2.0                        */
/*                                                                        */
/* Unless required by applicable law or agreed to in writing, software    */
/* distributed under the License is distributed on an "AS IS" BASIS,      */
/* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or        */
/* implied. See the License for the specific language governing           */
/* permissions and limitations under the License.                         */
/**************************************************************************/

/**
 * @file share_step.hpp
 * @brief Declares ShareStep, the forward and backward of one share of a split step.
 * @author Mustafa Garip
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include <SushiBLAS/engine/engine.hpp>
#include <SushiBLAS/tensor/tensor.hpp>
#include <SushiRuntime/sushiruntime.hpp>
#include <SushiAI/config/model_config.hpp>
#include <SushiAI/config/train_config.hpp>
#include <SushiAI/execution/runtime_execution_backend.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/graph/compiled_step.hpp>
#include <SushiAI/optim/optimizer.hpp>

#include "registries.hpp"
#include "row_feed.hpp"
#include "training_run.hpp"
#include "training_setup.hpp"

namespace SushiAI
{
    namespace Apps
    {
        /**
         * @class ShareStep
         * @brief Runs forward and backward over one share's rows of each batch, on a runtime of
         *        its own that drives one device class.
         *
         * The share holds no optimizer: its parameters are written from the share that runs the
         * update, and its gradients are read by it.
         */
        class ShareStep final
        {
            public:
                /**
                 * @brief Opens the share's runtime and builds its setup and compiled step.
                 * @param share The class and the rows of the share.
                 * @param first_row The first row of each batch the share takes.
                 * @param base The runtime configuration the share's class narrows.
                 * @return The share; Unsupported when the machine has no device of the class,
                 *         or the first refusal of the runtime, the setup or the compile.
                 */
                [[nodiscard]] static Result<std::unique_ptr<ShareStep>> make(
                    const Registries& registries, const Config::TrainSpec& spec,
                    const Config::ModelSpec& model_spec, Config::DeviceShare share,
                    std::int64_t first_row, std::uint64_t seed,
                    const SushiRuntime::RuntimeConfig& base);

                ShareStep(const ShareStep&) = delete;
                ShareStep& operator=(const ShareStep&) = delete;
                ShareStep(ShareStep&&) = delete;
                ShareStep& operator=(ShareStep&&) = delete;
                ~ShareStep() = default;

                /**
                 * @brief Enters epoch @p epoch_index of the share's data.
                 * @pre No other thread reads the share's data.
                 */
                void enter_epoch(std::int64_t epoch_index) noexcept;

                /**
                 * @brief Feeds the share's rows of batch @p batch_index and runs its step once.
                 * @return The refusal of the feed or of the replay.
                 * @pre No other thread runs this share.
                 */
                [[nodiscard]] Status run(std::size_t batch_index);

                /** @brief Returns each parameter with its gradient, in the order of the graph. */
                [[nodiscard]] const std::vector<Optim::ParameterSlot>& parameters()
                    const noexcept;

                /** @brief Returns the loss of the share's rows after the last run. */
                [[nodiscard]] const SushiBLAS::Tensor& loss() noexcept;

                /** @brief Returns what the share ran, for the run's summary. */
                [[nodiscard]] ShareRun report() const;

            private:
                /** @brief Takes the share's runtime; make() builds every other part. */
                ShareStep(SushiRuntime::Runtime runtime, Config::DeviceShare share,
                          std::int64_t first_row) noexcept;

                /// @brief The runtime that drives the share's device class.
                SushiRuntime::Runtime runtime_;

                /// @brief The engine on @c runtime_.
                SushiBLAS::Engine engine_;

                /// @brief The backend over @c runtime_ and @c engine_.
                RuntimeExecutionBackend backend_;

                /// @brief The class and the rows of the share.
                Config::DeviceShare share_;

                /// @brief The first row of each batch the share takes.
                std::int64_t first_row_;

                /// @brief The data, the training graph traced over the share's rows, and the pool.
                std::unique_ptr<TrainingSetup> setup_;

                /// @brief Reads the share's rows into the leaves of its graph.
                std::unique_ptr<RowFeed> feed_;

                /// @brief Forward and backward over the share's rows; declared last, dies first.
                std::unique_ptr<Graph::CompiledStep> step_;
        };
    } // namespace Apps
} // namespace SushiAI
