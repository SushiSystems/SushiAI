/**************************************************************************/
/* run_preparation.hpp                                                    */
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
 * @file run_preparation.hpp
 * @brief Declares the build `sushiai train` runs before a run starts, and what owns it.
 * @author Mustafa Garip
 * @see applications/programs/README.md, "run_preparation.hpp".
 */

#pragma once

#include <memory>
#include <optional>
#include <string_view>
#include <SushiAI/config/model_config.hpp>
#include <SushiAI/config/train_config.hpp>
#include <SushiAI/execution/execution_backend.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/optim/optimizer.hpp>
#include <SushiAI/optim/rate_schedule.hpp>
#include "registries.hpp"
#include "training_run.hpp"
#include "training_setup.hpp"

namespace SushiAI
{
    namespace Apps
    {
        /**
         * @class RunBuild
         * @brief Owns everything `sushiai train` builds before a run starts and the PreparedRun
         *        over it.
         */
        class RunBuild final
        {
            public:
                RunBuild(const RunBuild&) = delete;
                RunBuild& operator=(const RunBuild&) = delete;
                RunBuild(RunBuild&&) = delete;
                RunBuild& operator=(RunBuild&&) = delete;
                ~RunBuild() = default;

                /** @brief Returns the run input over the members of this build. */
                [[nodiscard]] PreparedRun& prepared() noexcept;

                /** @brief Returns the data, the model, the objective, the graph and the state. */
                [[nodiscard]] TrainingSetup& setup() noexcept;

                /** @brief Returns the optimizer over the parameters of the setup. */
                [[nodiscard]] Optim::Optimizer& optimizer() noexcept;

                /** @brief Returns the rate schedule of the file; null when it states none. */
                [[nodiscard]] const Optim::RateSchedule* schedule() const noexcept;

            private:
                friend Result<std::unique_ptr<RunBuild>> prepare_run(
                    const Registries& registries, std::string_view origin,
                    const Config::TrainSpec& spec, const Config::ModelSpec& model_spec,
                    ExecutionBackend& backend, SetupScope scope);

                RunBuild() = default;

                /// @brief The data, the model, the objective, the graph, the state and the pool.
                std::unique_ptr<TrainingSetup> setup_;

                /// @brief The rate schedule of the file; empty when it states none.
                std::optional<Optim::RateSchedule> schedule_;

                /// @brief The optimizer over the parameters of the setup.
                std::unique_ptr<Optim::Optimizer> optimizer_;

                /// @brief The run input over the members above.
                std::optional<PreparedRun> prepared_;
        };

        /**
         * @brief Builds the setup, the schedule and the optimizer of a training file, puts the
         *        state where the file starts the run, and builds the PreparedRun over them.
         * @param spec Outlives the build; with "device_split", the setup traces its first share.
         * @param backend Outlives the build; its engine allocates the tensors.
         * @param scope What the pool of the setup holds memory for.
         * @return The build, or the first failure of a read, a build, a weight load or a bind.
         */
        [[nodiscard]] Result<std::unique_ptr<RunBuild>> prepare_run(
            const Registries& registries, std::string_view origin, const Config::TrainSpec& spec,
            const Config::ModelSpec& model_spec, ExecutionBackend& backend, SetupScope scope);
    } // namespace Apps
} // namespace SushiAI
