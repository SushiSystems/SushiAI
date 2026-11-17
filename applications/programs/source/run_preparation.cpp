/**************************************************************************/
/* run_preparation.cpp                                                    */
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
 * @file run_preparation.cpp
 * @brief Defines the build `sushiai train` runs before a run starts.
 * @author Mustafa Garip
 * @see applications/programs/README.md, "run_preparation.hpp".
 */

#include "run_preparation.hpp"

#include <cstdint>
#include <SushiAI/io/run_progress.hpp>
#include <SushiAI/optim/optimizer_registry.hpp>
#include <SushiAI/optim/run_schedule.hpp>
#include "run_start.hpp"

namespace SushiAI
{
    namespace Apps
    {
        PreparedRun& RunBuild::prepared() noexcept
        {
            return *prepared_;
        }

        TrainingSetup& RunBuild::setup() noexcept
        {
            return *setup_;
        }

        Optim::Optimizer& RunBuild::optimizer() noexcept
        {
            return *optimizer_;
        }

        const Optim::RateSchedule* RunBuild::schedule() const noexcept
        {
            return schedule_ ? &*schedule_ : nullptr;
        }

        Result<std::unique_ptr<RunBuild>> prepare_run(const Registries& registries,
                                                      std::string_view origin,
                                                      const Config::TrainSpec& spec,
                                                      const Config::ModelSpec& model_spec,
                                                      ExecutionBackend& backend,
                                                      SetupScope scope)
        {
            std::unique_ptr<RunBuild> made(new RunBuild());

            SA_TRY_ASSIGN(const std::uint64_t seed, Config::read_run_seed(spec, origin));
            SA_TRY_ASSIGN(made->setup_, TrainingSetup::make(registries, spec, model_spec, backend,
                                                            seed, std::nullopt, scope));
            SA_TRY_ASSIGN(made->schedule_,
                          Optim::read_rate_schedule(spec.schedule, spec.optimizer_attributes));
            SA_TRY_ASSIGN(made->optimizer_,
                          Optim::make_optimizer(backend, spec.optimizer_type,
                                                made->setup_->parameters(),
                                                spec.optimizer_attributes, "the optimizer block"));

            IO::RunProgress start;
            SA_TRY(start_state(spec, backend, made->setup_->state(), *made->optimizer_, &start));

            // The run is a row or the fallback run; see applications/programs/README.md.
            TrainingSetup& setup = *made->setup_;
            made->prepared_.emplace(PreparedRun{
                spec, setup.dataset(), setup.given(), setup.objective(), setup.built(),
                setup.state(), setup.pool(), *made->optimizer_, setup.take_updates(),
                made->schedule(), start,
            });
            return std::move(made);
        }
    } // namespace Apps
} // namespace SushiAI
