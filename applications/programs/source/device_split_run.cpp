/**************************************************************************/
/* device_split_run.cpp                                                   */
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
 * @file device_split_run.cpp
 * @brief Implements run_device_split() and the step that blends the shares' gradients.
 * @author Mustafa Garip
 */

#include "device_split_run.hpp"

#include <cstddef>
#include <cstdint>
#include <future>
#include <iostream>
#include <memory>
#include <utility>
#include <vector>

#include <SushiBLAS/tensor/tensor.hpp>
#include <SushiLog/text/text_format.hpp>
#include <SushiLog/text/value_text.hpp>
#include <SushiAI/allocation/allocation.hpp>
#include <SushiAI/allocation/host_copy.hpp>
#include <SushiAI/data/epoch.hpp>
#include <SushiAI/dtype/dtype.hpp>
#include <SushiAI/failure/check.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/graph/buffer_update_recorder.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/profile/profile.hpp>
#include <SushiAI/seam/seam.hpp>
#include <SushiAI/train/training_step.hpp>

#include "epoch_loop.hpp"
#include "row_feed.hpp"
#include "share_runtime.hpp"
#include "share_step.hpp"

namespace SushiAI
{
    namespace Apps
    {
        namespace
        {
            /**
             * @brief Copies every element of @p source over @p target through @p staging on
             *        the host, for two tensors on runtimes that share no memory.
             * @return InvalidDescription for tensors of other counts or element types, or the
             *         failure of a copy.
             */
            [[nodiscard]] Status copy_across(const SushiBLAS::Tensor& source,
                                             const SushiBLAS::Tensor& target,
                                             std::vector<std::byte>& staging)
            {
                SA_FAIL_IF(source.num_elements != target.num_elements
                               || source.dtype != target.dtype,
                           StatusCode::InvalidDescription,
                           "a share holds {} {} elements where the first share holds {} {}",
                           source.num_elements, dtype_name(source.dtype), target.num_elements,
                           dtype_name(target.dtype));
                const std::size_t element_bytes = dtype_size(source.dtype);
                const std::size_t bytes =
                    static_cast<std::size_t>(source.num_elements) * element_bytes;
                if (staging.size() < bytes)
                    SA_TRY(SushiAI::Detail::guard(SushiAI::Detail::SeamPhase::Allocate,
                                                  [&staging, bytes]()
                    {
                        staging.resize(bytes);
                        return Status();
                    }));
                SA_TRY(copy_to_host(source, 0, source.num_elements, element_bytes,
                                    staging.data()));
                return copy_from_host(target, 0, target.num_elements, element_bytes,
                                      staging.data());
            }

            /** @brief Returns the one element of @p loss, copied to the host. */
            [[nodiscard]] double read_loss(const SushiBLAS::Tensor& loss) noexcept
            {
                const Result<std::vector<double>> value = read_reals(loss);
                SA_FATAL_IF(!value.has_value(), "The loss of a share could not be read back: {}",
                            value.status().message());
                return value.value().front();
            }

            /**
             * @class SplitStep
             * @brief Holds the shares after the first, the tensors their gradients are staged
             *        into, and the first share's step that blends them before its update.
             */
            class SplitStep final
            {
                public:
                    /**
                     * @brief Builds the shares after the first, the staging and the step.
                     * @return The split step, or the first refusal of a build or a compile.
                     */
                    [[nodiscard]] static Result<std::unique_ptr<SplitStep>> make(
                        const Registries& registries, const Config::ModelSpec& model_spec,
                        RuntimeBackend& backend, TrainingSetup& setup, PreparedRun& run);

                    SplitStep(const SplitStep&) = delete;
                    SplitStep& operator=(const SplitStep&) = delete;
                    SplitStep(SplitStep&&) = delete;
                    SplitStep& operator=(SplitStep&&) = delete;
                    ~SplitStep() = default;

                    /** @brief Enters epoch @p epoch_index of the data of every share. */
                    void enter_epoch(std::int64_t epoch_index) noexcept;

                    /**
                     * @brief Runs every share over its rows of batch @p batch_index and the
                     *        update over the blended gradients, then writes the new parameters
                     *        into the other shares.
                     * @return The first failure of a feed, a replay or a copy.
                     */
                    [[nodiscard]] Status run(std::size_t batch_index);

                    /**
                     * @brief Writes the first share's parameters into every other share.
                     * @return The first failure of a copy.
                     */
                    [[nodiscard]] Status publish_parameters();

                    /** @brief Returns the mean loss of the last batch over all its rows. */
                    [[nodiscard]] double batch_loss() noexcept;

                    /** @brief Returns the first share's step, which applies the update. */
                    [[nodiscard]] Train::TrainingStep& step() noexcept
                    {
                        return *step_;
                    }

                    /** @brief Returns one ShareRun per share, in file order. */
                    [[nodiscard]] std::vector<ShareRun> report() const;

                private:
                    /** @brief Borrows the first share's parts; make() builds every other one. */
                    SplitStep(RuntimeBackend& backend, TrainingSetup& setup,
                              const PreparedRun& run) noexcept;

                    /**
                     * @brief Waits for @p pending and copies every gradient of the other shares
                     *        into the staging tensors the blend reads.
                     * @return The first failure of a share's run or of a copy.
                     */
                    [[nodiscard]] Status gather_gradients(
                        std::vector<std::future<Status>>& pending);

                    /// @brief The backend of the first share; not owned.
                    RuntimeBackend* backend_;

                    /// @brief The setup of the first share; not owned.
                    TrainingSetup* setup_;

                    /// @brief The shares of the file, in file order; the first runs here.
                    std::vector<Config::DeviceShare> shares_;

                    /// @brief Reads the first share's rows into its leaves.
                    std::unique_ptr<RowFeed> feed_;

                    /// @brief The shares after the first, in file order.
                    std::vector<std::unique_ptr<ShareStep>> others_;

                    /// @brief One tensor per gradient of each other share, share by share.
                    std::vector<SushiBLAS::Tensor> staged_;

                    /// @brief The host bytes every cross-runtime copy passes through.
                    std::vector<std::byte> bytes_;

                    /// @brief Blends each staged gradient into the first share's, by rows.
                    std::unique_ptr<Graph::BufferUpdateRecorder> blend_;

                    /// @brief The first share's forward, backward, blend and update; dies first.
                    std::unique_ptr<Train::TrainingStep> step_;
            };

            SplitStep::SplitStep(RuntimeBackend& backend, TrainingSetup& setup,
                                 const PreparedRun& run) noexcept
                : backend_(&backend), setup_(&setup), shares_(run.spec.device_split)
            {
            }

            Result<std::unique_ptr<SplitStep>> SplitStep::make(
                const Registries& registries, const Config::ModelSpec& model_spec,
                RuntimeBackend& backend, TrainingSetup& setup, PreparedRun& run)
            {
                SA_TRY_ASSIGN(std::unique_ptr<SplitStep> made,
                              SushiAI::Detail::guard_value(
                                  SushiAI::Detail::SeamPhase::Allocate, [&]()
                {
                    return std::unique_ptr<SplitStep>(new SplitStep(backend, setup, run));
                }));
                made->feed_ = std::make_unique<RowFeed>(backend, setup);

                const std::vector<Optim::ParameterSlot>& parameters = setup.parameters();
                std::vector<Graph::BufferUpdateSlot> slots;
                std::int64_t first_row = made->shares_.front().rows;
                std::int64_t blended_rows = first_row;
                for (std::size_t share = 1; share < made->shares_.size(); ++share)
                {
                    const Config::DeviceShare& stated = made->shares_[share];
                    SA_TRY_ASSIGN(std::unique_ptr<ShareStep> other,
                                  ShareStep::make(registries, run.spec, model_spec, stated,
                                                  first_row, run.start.seed,
                                                  SushiRuntime::RuntimeConfig{}));
                    SA_FAIL_IF(other->parameters().size() != parameters.size(),
                               StatusCode::InvalidDescription,
                               "the '{}' share holds {} parameters where the first holds {}",
                               Config::device_class_name(stated.device),
                               other->parameters().size(), parameters.size());
                    made->others_.push_back(std::move(other));
                    // Each blend weighs the rows seen so far against the share's own.
                    blended_rows += stated.rows;
                    const double weight =
                        static_cast<double>(stated.rows) / static_cast<double>(blended_rows);
                    for (const Optim::ParameterSlot& slot : parameters)
                    {
                        SA_TRY_ASSIGN(SushiBLAS::Tensor staged,
                                      allocate_like(backend.engine(), slot.gradient));
                        made->staged_.push_back(staged);
                        slots.push_back(Graph::BufferUpdateSlot{slot.gradient, staged, weight, 1.0});
                    }
                    first_row += stated.rows;
                }
                made->blend_ = std::make_unique<Graph::BufferUpdateRecorder>(std::move(slots));

                // The host stage ends where the last gradient and the loss are written.
                const Train::TrainingGraph& built = setup.built();
                Train::TrainingStepOptions options;
                options.optimizer_placement = Train::OptimizerPlacement::InPlan;
                options.before_update = made->blend_.get();
                options.host_stage.reads.push_back(built.spec.loss);
                for (const Train::ParameterBinding& binding : built.parameters)
                {
                    if (binding.gradient != Graph::K_NO_VALUE)
                        options.host_stage.reads.push_back(binding.gradient);
                }
                SA_TRY_ASSIGN(made->step_,
                              Train::TrainingStep::make(backend, built.graph, setup.pool(),
                                                        run.optimizer, std::move(run.updates),
                                                        options));
                return std::move(made);
            }

            void SplitStep::enter_epoch(std::int64_t epoch_index) noexcept
            {
                Data::begin_epoch(setup_->dataset(), epoch_index);
                for (const std::unique_ptr<ShareStep>& other : others_)
                    other->enter_epoch(epoch_index);
            }

            Status SplitStep::run(std::size_t batch_index)
            {
                // Declared before any early return, so every share is joined before it ends.
                std::vector<std::future<Status>> pending;
                for (const std::unique_ptr<ShareStep>& other : others_)
                {
                    ShareStep* const share = other.get();
                    SA_TRY_ASSIGN(std::future<Status> started,
                                  SushiAI::Detail::guard_value(
                                      SushiAI::Detail::SeamPhase::Execute, [share, batch_index]()
                    {
                        return std::async(std::launch::async, [share, batch_index]()
                        {
                            return share->run(batch_index);
                        });
                    }));
                    pending.push_back(std::move(started));
                }
                SA_TRY(feed_->feed(batch_index, Data::RowRange{0, shares_.front().rows}));
                {
                    SA_ZONE("device", "run");
                    SA_TRY(step_->run([this, &pending]()
                    {
                        return gather_gradients(pending);
                    }));
                }
                return publish_parameters();
            }

            Status SplitStep::gather_gradients(std::vector<std::future<Status>>& pending)
            {
                for (std::future<Status>& share : pending)
                    SA_TRY(share.get());
                SA_ZONE("copy", "gradients");
                const std::size_t count = setup_->parameters().size();
                for (std::size_t share = 0; share < others_.size(); ++share)
                {
                    const std::vector<Optim::ParameterSlot>& slots = others_[share]->parameters();
                    for (std::size_t index = 0; index < count; ++index)
                        SA_TRY(copy_across(slots[index].gradient, staged_[share * count + index],
                                           bytes_));
                }
                return Status();
            }

            Status SplitStep::publish_parameters()
            {
                SA_ZONE("copy", "parameters");
                const std::vector<Optim::ParameterSlot>& own = setup_->parameters();
                for (const std::unique_ptr<ShareStep>& other : others_)
                {
                    const std::vector<Optim::ParameterSlot>& slots = other->parameters();
                    for (std::size_t index = 0; index < own.size(); ++index)
                        SA_TRY(copy_across(own[index].parameter, slots[index].parameter, bytes_));
                }
                return Status();
            }

            double SplitStep::batch_loss() noexcept
            {
                const Train::TrainingGraph& built = setup_->built();
                double weighted = static_cast<double>(shares_.front().rows)
                                * read_loss(setup_->pool().tensor(built.spec.loss));
                std::int64_t rows = shares_.front().rows;
                for (std::size_t share = 0; share < others_.size(); ++share)
                {
                    const std::int64_t share_rows = shares_[share + 1].rows;
                    weighted += static_cast<double>(share_rows) * read_loss(others_[share]->loss());
                    rows += share_rows;
                }
                return weighted / static_cast<double>(rows);
            }

            std::vector<ShareRun> SplitStep::report() const
            {
                std::vector<ShareRun> ran;
                ShareRun first;
                first.device = shares_.front().device;
                first.device_name = first_device_name(backend_->runtime());
                first.rows = shares_.front().rows;
                first.step_runs = step_->step().run_count();
                ran.push_back(first);
                for (const std::unique_ptr<ShareStep>& other : others_)
                    ran.push_back(other->report());
                return ran;
            }

            /**
             * @brief Checks what a split run needs before anything is built for it.
             * @return The refusals run_device_split() documents for its inputs.
             */
            [[nodiscard]] Status check_split(const Registries& registries,
                                             RuntimeBackend& backend, const PreparedRun& run)
            {
                const std::vector<Config::DeviceShare>& shares = run.spec.device_split;
                SA_FAIL_IF(shares.size() < 2, StatusCode::InvalidDescription,
                           "a split run needs two shares or more, but the file states {}",
                           shares.size());
                SA_FAIL_IF(registries.runs.find(run.spec.objective_type) != nullptr,
                           StatusCode::Unsupported,
                           "the '{}' objective trains through a run of its own, which a step "
                           "split across devices does not run",
                           run.spec.objective_type);
                std::int64_t rows = 0;
                for (const Config::DeviceShare& share : shares)
                    rows += share.rows;
                SA_FAIL_IF(rows != run.dataset.batch_size(), StatusCode::InvalidDescription,
                           "the shares of 'device_split' take {} rows a batch, but the '{}' data "
                           "reads batches of {}; the rows sum to the data's batch size",
                           rows, run.dataset.name(), run.dataset.batch_size());
                SA_FAIL_IF(run.given.batch != shares.front().rows, StatusCode::InvalidDescription,
                           "the first share takes {} rows, but its graph was traced over {}",
                           shares.front().rows, run.given.batch);
                SA_FAIL_IF(run.built.spec.loss == Graph::K_NO_VALUE,
                           StatusCode::InvalidDescription,
                           "the training graph names no loss, so the shares have none to weigh");
                return check_share_device(backend.runtime(), shares.front().device);
            }
        } // namespace

        Result<RunSummary> run_device_split(const Registries& registries,
                                            const Config::ModelSpec& model_spec,
                                            RuntimeBackend& backend, TrainingSetup& setup,
                                            PreparedRun& run)
        {
            SA_TRY(check_split(registries, backend, run));
            SA_TRY(check_start_batch(run));
            SA_TRY_ASSIGN(const std::unique_ptr<SplitStep> split,
                          SplitStep::make(registries, model_spec, backend, setup, run));
            SA_TRY(split->step().restore_run_count(static_cast<std::size_t>(run.start.step_runs)));
            // Every share starts from the parameters the first share was built or loaded with.
            SA_TRY(split->publish_parameters());

            EpochSteps steps;
            steps.enter_epoch = [&split](std::int64_t epoch_index)
            {
                split->enter_epoch(epoch_index);
            };
            steps.run_step = [&split](std::size_t batch_index)
            {
                return split->run(batch_index);
            };
            steps.report = [&split](std::int64_t epoch, std::int64_t epochs)
            {
                std::cout << SushiLog::format_text("  epoch {}/{} loss {}\n",
                                                   SushiLog::right(epoch, 3),
                                                   SushiLog::left(epochs, 3),
                                                   SushiLog::fixed(split->batch_loss(), 6));
            };
            steps.after_update = [&run, &split](const UpdateAt& at) -> Status
            {
                return checkpoint_after_update(run, at, split->step().step().run_count());
            };
            SA_TRY(run_epochs(run, steps));

            RunSummary summary;
            summary.compile_count = split->step().step().compile_count();
            summary.progress = finished_progress(run, split->step().step().run_count());
            summary.shares = split->report();
            return summary;
        }
    } // namespace Apps
} // namespace SushiAI
