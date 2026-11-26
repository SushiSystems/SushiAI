/**************************************************************************/
/* share_step.cpp                                                         */
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
 * @file share_step.cpp
 * @brief Implements ShareStep, which runs one share's rows on a runtime of its own.
 * @author Mustafa Garip
 */

#include "share_step.hpp"

#include <utility>

#include <SushiAI/data/dataset.hpp>
#include <SushiAI/data/epoch.hpp>
#include <SushiAI/execution/execution.hpp>
#include <SushiAI/profile/profile.hpp>
#include <SushiAI/seam/seam.hpp>

#include "share_runtime.hpp"

namespace SushiAI
{
    namespace Apps
    {
        Result<std::unique_ptr<ShareStep>> ShareStep::make(
            const Registries& registries, const Config::TrainSpec& spec,
            const Config::ModelSpec& model_spec, Config::DeviceShare share,
            std::int64_t first_row, std::uint64_t seed, const SushiRuntime::RuntimeConfig& base)
        {
            SA_TRY_ASSIGN(SushiRuntime::Runtime runtime,
                          SushiAI::Detail::create_runtime(narrow_to_class(base, share.device)));
            SA_TRY(check_share_device(runtime, share.device));
            SA_TRY_ASSIGN(std::unique_ptr<ShareStep> made,
                          SushiAI::Detail::guard_value(SushiAI::Detail::SeamPhase::Allocate,
                                                       [&]()
            {
                return std::unique_ptr<ShareStep>(
                    new ShareStep(std::move(runtime), share, first_row));
            }));
            SA_TRY_ASSIGN(made->setup_,
                          TrainingSetup::make(registries, spec, model_spec, made->backend_, seed,
                                              share.rows, SetupScope::Step));
            made->feed_ = std::make_unique<RowFeed>(made->backend_, *made->setup_);
            SA_TRY_ASSIGN(made->step_,
                          Graph::CompiledStep::compile(made->backend_, made->setup_->built().graph,
                                                       made->setup_->pool(), nullptr, nullptr, {},
                                                       "share.step"));
            return std::move(made);
        }

        ShareStep::ShareStep(SushiRuntime::Runtime runtime, Config::DeviceShare share,
                             std::int64_t first_row) noexcept
            : runtime_(std::move(runtime)), engine_(runtime_), backend_(runtime_, engine_),
              share_(share), first_row_(first_row)
        {
        }

        void ShareStep::enter_epoch(std::int64_t epoch_index) noexcept
        {
            Data::begin_epoch(setup_->dataset(), epoch_index);
        }

        Status ShareStep::run(std::size_t batch_index)
        {
            SA_TRY(feed_->feed(batch_index, Data::RowRange{first_row_, share_.rows}));
            SA_ZONE("device", "share");
            return step_->run();
        }

        const std::vector<Optim::ParameterSlot>& ShareStep::parameters() const noexcept
        {
            return setup_->parameters();
        }

        const SushiBLAS::Tensor& ShareStep::loss() noexcept
        {
            return setup_->pool().tensor(setup_->built().spec.loss);
        }

        ShareRun ShareStep::report() const
        {
            ShareRun ran;
            ran.device = share_.device;
            ran.device_name = first_device_name(runtime_);
            ran.rows = share_.rows;
            ran.step_runs = step_->run_count();
            return ran;
        }
    } // namespace Apps
} // namespace SushiAI
