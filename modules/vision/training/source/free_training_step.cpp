/**************************************************************************/
/* free_training_step.cpp                                                 */
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
 * @file free_training_step.cpp
 * @brief Implements FreeTrainingStep: the step up to the ports, the host assignment and the rest.
 * @author Mustafa Garip
 */

#include <SushiAI/vision/training/free_training_step.hpp>

#include <cstdint>
#include <iterator>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <SushiBLAS/blas/blas.hpp>
#include <SushiBLAS/engine/task_recorder.hpp>
#include <SushiRuntime/core/span.hpp>

#include <SushiAI/allocation/host_copy.hpp>
#include <SushiAI/dtype/dtype.hpp>
#include <SushiAI/execution/execution.hpp>
#include <SushiAI/failure/check.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/profile/profile.hpp>
#include <SushiAI/seam/seam.hpp>
#include <SushiAI/shape/tensor_shape.hpp>
#include <SushiAI/train/feed.hpp>
#include <SushiAI/vision/detect/free/distribution_expectation.hpp>
#include <SushiAI/vision/training/detection_truth_check.hpp>

namespace SushiAI::Vision::Training
{
    namespace
    {
        /** @brief Holds the box sides each position predicts a distribution for. */
        constexpr std::int64_t K_BOX_SIDES = 4;

        /**
         * @brief Checks that the host can read or write @p tensor as @p elements floats.
         * @param what Names the tensor in the message.
         * @return Success, or Incompatible naming the dtype or both element counts.
         */
        [[nodiscard]] Status check_host_floats(const SushiBLAS::Tensor& tensor,
                                               std::int64_t elements, std::string_view what)
        {
            SA_FAIL_IF(tensor.dtype != DType::Float32, StatusCode::Incompatible,
                       "'{}' holds {}; the host stage of the anchor-free step reads and writes "
                       "float32",
                       what, dtype_name(tensor.dtype));
            SA_FAIL_IF(static_cast<std::int64_t>(tensor.num_elements) != elements,
                       StatusCode::Incompatible,
                       "'{}' holds {} elements, but the objective describes {}", what,
                       static_cast<std::int64_t>(tensor.num_elements), elements);

            return Status();
        }

        /**
         * @brief Copies the float32 values of @p tensor into @p held, sized to fit.
         * @return A view of @p held, or the failure of the copy.
         */
        [[nodiscard]] Result<SushiRuntime::span<const float>> floats_of(
            const SushiBLAS::Tensor& tensor, std::vector<float>& held)
        {
            held.resize(static_cast<std::size_t>(tensor.num_elements));
            SA_TRY(read_elements(tensor, 0, SushiRuntime::span<float>(held.data(), held.size())));

            return SushiRuntime::span<const float>(held.data(), held.size());
        }

        /** @brief Copies @p value into the first element of @p tensor. */
        [[nodiscard]] Status write_first(const SushiBLAS::Tensor& tensor, float value)
        {
            return write_elements(tensor, 0, SushiRuntime::span<const float>(&value, 1));
        }

        /**
         * @brief Returns the first float32 element of @p tensor, or 0 for a null one.
         * @return The value, or the failure of the copy.
         */
        [[nodiscard]] Result<double> first_of(const SushiBLAS::Tensor* tensor) noexcept
        {
            if (tensor == nullptr)
                return 0.0;

            float value = 0.0f;
            SA_TRY(read_elements(*tensor, 0, SushiRuntime::span<float>(&value, 1)));
            return static_cast<double>(value);
        }
    } // namespace

    Result<std::unique_ptr<FreeTrainingStep>> FreeTrainingStep::make(
        ExecutionBackend& backend, const FreeObjective& objective,
        const Train::TrainingGraph& built, Graph::TensorPool& training_pool,
        Optim::Optimizer& optimizer, std::vector<Train::BufferUpdateSlot> updates,
        Train::TrainingStepOptions options) noexcept
    {
        SA_FAIL_IF(built.spec.inputs == Graph::K_NO_VALUE
                       || !training_pool.is_bound(built.spec.inputs),
                   StatusCode::InvalidDescription,
                   "The anchor-free step needs the input leaf of its training graph bound in "
                   "the training pool; call Train::bind_parameters() on that pool first");

        std::vector<FreeBranch> kinds = {FreeBranch::OneToMany};
        if (objective.facts().has_one_to_one)
            kinds.push_back(FreeBranch::OneToOne);

        std::vector<Branch> branches;
        for (const FreeBranch kind : kinds)
        {
            SA_TRY_ASSIGN(FreeAssignmentStage stage,
                          FreeAssignmentStage::make(objective.facts(), objective.task(),
                                                    objective.batch(), kind));
            branches.push_back(Branch{std::move(stage), {}, {}, {}, {}, {}});
        }
        SushiBLAS::Tensor& input = training_pool.tensor(built.spec.inputs);

        SA_TRY_ASSIGN(std::unique_ptr<FreeTrainingStep> step,
                      ::SushiAI::Detail::guard_value(::SushiAI::Detail::SeamPhase::Allocate,
                                                     [&]()
        {
            return std::unique_ptr<FreeTrainingStep>(new FreeTrainingStep(
                backend, objective, std::move(branches), training_pool, input));
        }));

        SA_TRY_ASSIGN(options.host_stage, step->describe_host_stage(built));
        SA_TRY_ASSIGN(step->training_,
                      Train::TrainingStep::make(backend, built.graph, training_pool, optimizer,
                                                std::move(updates), std::move(options)));
        SA_TRY(step->resolve_tensors(built));

        return std::move(step);
    }

    FreeTrainingStep::FreeTrainingStep(ExecutionBackend& backend, FreeObjective objective,
                                       std::vector<Branch> branches, Graph::TensorPool& pool,
                                       SushiBLAS::Tensor& input) noexcept
        : backend_(&backend),
          objective_(std::move(objective)),
          branches_(std::move(branches)),
          pool_(&pool),
          training_(),
          input_(&input),
          one_to_many_weight_(objective_.branch_weights().start)
    {

    }

    Result<Train::HostStage> FreeTrainingStep::describe_host_stage(
        const Train::TrainingGraph& built)
    {
        Train::HostStage stage;
        SA_TRY(::SushiAI::Detail::guard(::SushiAI::Detail::SeamPhase::Allocate,
                                        [this, &built, &stage]() -> Status
        {
            for (std::size_t copy = 0; copy < branches_.size(); ++copy)
            {
                for (const FreeLevelFacts& level : objective_.facts().levels)
                {
                    const FreeLevelPorts& names = copy == 0 ? level.one_to_many : level.one_to_one;
                    SA_TRY_ASSIGN(const Graph::ValueID boxes, built.outputs.at(names.boxes));
                    SA_TRY_ASSIGN(const Graph::ValueID scores, built.outputs.at(names.scores));
                    branches_[copy].port_ids.push_back(boxes);
                    branches_[copy].port_ids.push_back(scores);
                }
                stage.reads.insert(stage.reads.end(), branches_[copy].port_ids.begin(),
                                   branches_[copy].port_ids.end());
            }

            for (const FreeLeaf& leaf : objective_.stage_leaves())
            {
                if (!objective_.reads_leaf(leaf.kind))
                    continue;
                SA_FAIL_IF(!built.leaves.has(leaf.role), StatusCode::InvalidDescription,
                           "The training graph declares no leaf '{}'; build it with the "
                           "objective the anchor-free step is given",
                           leaf.role);
                stage.writes.push_back(built.leaves.at(leaf.role));
            }

            if (objective_.facts().reg_max > 1)
            {
                SA_FAIL_IF(!built.leaves.has(K_FREE_PROJECTION_ROLE),
                           StatusCode::InvalidDescription,
                           "The training graph declares no leaf '{}'", K_FREE_PROJECTION_ROLE);
                stage.writes.push_back(built.leaves.at(K_FREE_PROJECTION_ROLE));
            }

            if (branches_.size() > 1)
            {
                for (const std::string_view role :
                     {K_FREE_ONE_TO_MANY_WEIGHT_ROLE, K_FREE_ONE_TO_ONE_WEIGHT_ROLE})
                {
                    SA_FAIL_IF(!built.leaves.has(role), StatusCode::InvalidDescription,
                               "The training graph declares no leaf '{}'", role);
                    stage.writes.push_back(built.leaves.at(role));
                }
            }

            return Status();
        }));

        return stage;
    }

    Status FreeTrainingStep::resolve_tensors(const Train::TrainingGraph& built)
    {
        const FreeModelFacts& facts = objective_.facts();
        const std::int64_t batch = objective_.batch();

        return ::SushiAI::Detail::guard(::SushiAI::Detail::SeamPhase::Allocate,
                                        [this, &built, &facts, batch]() -> Status
        {
            for (std::size_t copy = 0; copy < branches_.size(); ++copy)
            {
                Branch& branch = branches_[copy];
                for (std::size_t level = 0; level < facts.levels.size(); ++level)
                {
                    const FreeLevelFacts& described = facts.levels[level];
                    const std::int64_t rows = batch * described.height * described.width;
                    const FreeLevelPorts& names =
                        copy == 0 ? described.one_to_many : described.one_to_one;
                    const std::int64_t counts[] = {rows * K_BOX_SIDES * facts.reg_max,
                                                   rows * facts.classes};
                    const std::string* const labels[] = {&names.boxes, &names.scores};

                    for (std::size_t port = 0; port < 2; ++port)
                    {
                        const Graph::ValueID id = branch.port_ids[level * 2 + port];
                        SA_FAIL_IF(!pool_->is_bound(id), StatusCode::InvalidDescription,
                                   "The port '{}' has no tensor in the training pool",
                                   *labels[port]);
                        const SushiBLAS::Tensor& tensor = pool_->tensor(id);
                        SA_TRY(check_host_floats(tensor, counts[port], *labels[port]));
                        branch.ports.push_back(&tensor);
                    }
                }

                for (const FreeLeaf& leaf : branch.stage.leaves())
                {
                    if (!objective_.reads_leaf(leaf.kind))
                    {
                        branch.leaves.push_back(nullptr);
                        continue;
                    }
                    const Graph::ValueID id = built.leaves.at(leaf.role);
                    SA_FAIL_IF(!pool_->is_bound(id), StatusCode::InvalidDescription,
                               "The leaf '{}' has no tensor in the training pool; call "
                               "Train::bind_parameters() on that pool first",
                               leaf.role);
                    SushiBLAS::Tensor& tensor = pool_->tensor(id);
                    SA_TRY(check_host_floats(tensor, leaf.shape.num_elements(), leaf.role));
                    branch.leaves.push_back(&tensor);
                }
                branch.views.resize(facts.levels.size());
            }

            if (branches_.size() > 1)
            {
                const std::string_view roles[] = {K_FREE_ONE_TO_MANY_WEIGHT_ROLE,
                                                  K_FREE_ONE_TO_ONE_WEIGHT_ROLE};
                for (std::size_t index = 0; index < 2; ++index)
                {
                    const Graph::ValueID id = built.leaves.at(roles[index]);
                    SA_FAIL_IF(!pool_->is_bound(id), StatusCode::InvalidDescription,
                               "The leaf '{}' has no tensor in the training pool",
                               roles[index]);
                    branch_weights_[index] = &pool_->tensor(id);
                    SA_TRY(check_host_floats(*branch_weights_[index], 1, roles[index]));
                }
            }

            if (facts.reg_max > 1)
            {
                const Graph::ValueID id = built.leaves.at(K_FREE_PROJECTION_ROLE);
                SA_FAIL_IF(!pool_->is_bound(id), StatusCode::InvalidDescription,
                           "The leaf '{}' has no tensor in the training pool",
                           K_FREE_PROJECTION_ROLE);
                projection_ = &pool_->tensor(id);
                SA_TRY(check_host_floats(*projection_, facts.reg_max, K_FREE_PROJECTION_ROLE));
            }

            SA_FAIL_IF(!pool_->is_bound(built.spec.loss), StatusCode::InvalidDescription,
                       "The loss of the training graph has no tensor in the training pool");
            loss_ = &pool_->tensor(built.spec.loss);
            SA_TRY(check_host_floats(*loss_, 1, "the loss"));

            const std::string_view names[] = {K_FREE_BOX_VALUE, K_FREE_CLASSES_VALUE,
                                              K_FREE_DISTRIBUTION_VALUE, K_FREE_L1_VALUE};
            for (std::size_t term = 0; term < std::size(names); ++term)
            {
                const Result<Graph::ValueID> id = built.metric_values.at(names[term]);
                if (!id.has_value())
                    continue;

                SA_FAIL_IF(!pool_->is_bound(id.value()), StatusCode::InvalidDescription,
                           "The loss term '{}' has no tensor in the training pool", names[term]);
                terms_[term] = &pool_->tensor(id.value());
                SA_TRY(check_host_floats(*terms_[term], 1, names[term]));
            }

            return Status();
        });
    }

    Status FreeTrainingStep::check_truth(const Detect::GroundTruthBatch& truth) const
    {
        SA_TRY(check_truth_layout(truth, objective_.batch(), "ports were traced for"));

        return check_truth_labels(truth, objective_.facts().classes, "the model");
    }

    Status FreeTrainingStep::feed_leaves(const Detect::GroundTruthBatch& truth)
    {
        return ::SushiAI::Detail::guard(::SushiAI::Detail::SeamPhase::Execute,
                                        [this, &truth]() -> Status
        {
            for (Branch& branch : branches_)
            {
                {
                    SA_ZONE("host", "readback");
                    branch.port_values.resize(branch.ports.size());
                    for (std::size_t level = 0; level < branch.views.size(); ++level)
                    {
                        SA_TRY_ASSIGN(branch.views[level].boxes,
                                      floats_of(*branch.ports[level * 2],
                                                branch.port_values[level * 2]));
                        SA_TRY_ASSIGN(branch.views[level].scores,
                                      floats_of(*branch.ports[level * 2 + 1],
                                                branch.port_values[level * 2 + 1]));
                    }
                }
                SA_TRY(branch.stage.assign(SushiRuntime::span<const FreeLevelViews>(
                                               branch.views.data(), branch.views.size()),
                                           truth));

                SA_ZONE("host", "assign");
                for (std::size_t leaf = 0; leaf < branch.leaves.size(); ++leaf)
                {
                    SushiBLAS::Tensor* const tensor = branch.leaves[leaf];
                    if (tensor == nullptr)
                        continue;
                    staged_.resize(static_cast<std::size_t>(tensor->num_elements));
                    SA_TRY(branch.stage.write_leaf(
                        leaf, SushiRuntime::span<float>(staged_.data(), staged_.size())));
                    SA_TRY(write_elements(*tensor, 0, SushiRuntime::span<const float>(
                                                          staged_.data(), staged_.size())));
                }
            }

            if (branch_weights_[0] != nullptr)
            {
                SA_TRY(write_first(*branch_weights_[0], one_to_many_weight_));
                SA_TRY(write_first(*branch_weights_[1], 1.0f - one_to_many_weight_));
            }

            // The projection is rewritten each run; the pool gives no promise to keep a leaf.
            if (projection_ != nullptr)
            {
                staged_.resize(static_cast<std::size_t>(projection_->num_elements));
                Detect::Free::fill_distribution_projection(
                    SushiRuntime::span<float>(staged_.data(), staged_.size()));
                SA_TRY(write_elements(*projection_, 0, SushiRuntime::span<const float>(
                                                           staged_.data(), staged_.size())));
            }

            return Status();
        });
    }

    Result<FreeLossTerms> FreeTrainingStep::read_terms() const noexcept
    {
        FreeLossTerms terms;
        SA_TRY_ASSIGN(terms.total, first_of(loss_));
        SA_TRY_ASSIGN(terms.box, first_of(terms_[0]));
        SA_TRY_ASSIGN(terms.classes, first_of(terms_[1]));
        SA_TRY_ASSIGN(terms.distribution, first_of(terms_[2]));
        SA_TRY_ASSIGN(terms.l1, first_of(terms_[3]));
        return terms;
    }

    Result<FreeLossTerms> FreeTrainingStep::run(const SushiBLAS::Tensor& inputs,
                                                const Detect::GroundTruthBatch& truth) noexcept
    {
        SA_TRY(Train::check_feed(inputs, *input_, "inputs"));
        SA_TRY(check_truth(truth));

        {
            SA_ZONE("copy", "input");
            SA_TRY(::SushiAI::Detail::record_and_run(
                *backend_, [this, &inputs](SushiBLAS::TaskRecorder& recorder) -> Status
            {
                SushiBLAS::BLASOps blas(recorder);
                blas.copy(inputs, *input_);
                return Status();
            }));
        }

        // The ports are read before the rest of the step may reuse their storage.
        SA_ZONE("device", "run");
        SA_TRY(training_->run([this, &truth]() -> Status
        {
            return feed_leaves(truth);
        }));

        // The loss and its terms are pinned values, so they hold this run's numbers.
        return read_terms();
    }

    const FreeAssignmentStage& FreeTrainingStep::stage() const noexcept
    {
        return branches_.front().stage;
    }

    void FreeTrainingStep::set_one_to_many_weight(float weight) noexcept
    {
        one_to_many_weight_ = weight;
    }

    float FreeTrainingStep::one_to_many_weight() const noexcept
    {
        return one_to_many_weight_;
    }

    const Train::TrainingStep& FreeTrainingStep::training() const noexcept
    {
        return *training_;
    }

    std::size_t FreeTrainingStep::run_count() const noexcept
    {
        return training_->step().run_count();
    }

    Status FreeTrainingStep::restore_run_count(std::size_t count) noexcept
    {
        return training_->restore_run_count(count);
    }

    Result<Data::InputDescription> FreeTrainingStep::input_description() const noexcept
    {
        SA_TRY_ASSIGN(const Shape full, checked_shape_of(*input_));
        SA_FAIL_IF(full.rank() == 0, StatusCode::InvalidDescription,
                   "The input leaf of the anchor-free step has no batch extent");

        const SushiRuntime::span<const std::int64_t> sample(
            full.data() + 1, static_cast<std::size_t>(full.rank()) - 1);

        return Data::InputDescription{Shape(sample), input_->dtype};
    }

    const FreeObjective& FreeTrainingStep::objective() const noexcept
    {
        return objective_;
    }

    Optim::Optimizer& FreeTrainingStep::optimizer() noexcept
    {
        return training_->optimizer();
    }

} // namespace SushiAI::Vision::Training
