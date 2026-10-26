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
 * @file free_training_step.cpp
 * @brief Implements FreeTrainingStep: the step up to the ports, the host assignment and the rest.
 * @author Mustafa Garip
 */

#include <SushiAI/vision/training/free_training_step.hpp>

#include <cstdint>
#include <initializer_list>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <SushiBLAS/blas/blas.hpp>
#include <SushiBLAS/engine/task_recorder.hpp>
#include <SushiRuntime/core/span.hpp>

#include <SushiAI/dtype/dtype.hpp>
#include <SushiAI/execution/execution.hpp>
#include <SushiAI/failure/check.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/profile/profile.hpp>
#include <SushiAI/seam/seam.hpp>
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

        /** @brief Returns the float32 values of @p tensor as a span. */
        [[nodiscard]] SushiRuntime::span<const float> floats_of(const SushiBLAS::Tensor& tensor)
        {
            return SushiRuntime::span<const float>(tensor.data_as<float>(),
                                                   static_cast<std::size_t>(tensor.num_elements));
        }

        /** @brief Overwrites @p values with the float32 values of @p tensor, widened. */
        void widen(const SushiBLAS::Tensor& tensor, std::vector<double>& values)
        {
            const SushiRuntime::span<const float> source = floats_of(tensor);
            values.assign(source.begin(), source.end());
        }

        /** @brief Returns @p values as a span of its elements. */
        [[nodiscard]] SushiRuntime::span<const double> view(const std::vector<double>& values)
        {
            return SushiRuntime::span<const double>(values.data(), values.size());
        }
    } // namespace

    Result<std::unique_ptr<FreeTrainingStep>> FreeTrainingStep::make(
        SushiBLAS::Engine& engine, const FreeObjective& objective,
        const Train::TrainingGraph& built, Graph::TensorPool& training_pool,
        Optim::Optimizer& optimizer, std::vector<Train::BufferUpdateSlot> updates,
        Train::TrainingStepOptions options)
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
            branches.push_back(Branch{std::move(stage), {}, {}, {}, {}});
        }
        SushiBLAS::Tensor& input = training_pool.tensor(built.spec.inputs);

        SA_TRY_ASSIGN(std::unique_ptr<FreeTrainingStep> step,
                      ::SushiAI::Detail::guard_value(::SushiAI::Detail::SeamPhase::Allocate,
                                                     [&]()
        {
            return std::unique_ptr<FreeTrainingStep>(new FreeTrainingStep(
                engine, objective, std::move(branches), training_pool, input));
        }));

        SA_TRY_ASSIGN(options.host_stage, step->describe_host_stage(built));
        SA_TRY_ASSIGN(step->training_,
                      Train::TrainingStep::make(engine, built.graph, training_pool, optimizer,
                                                std::move(updates), std::move(options)));
        SA_TRY(step->resolve_tensors(built));

        return std::move(step);
    }

    FreeTrainingStep::FreeTrainingStep(SushiBLAS::Engine& engine, FreeObjective objective,
                                       std::vector<Branch> branches, Graph::TensorPool& pool,
                                       SushiBLAS::Tensor& input) noexcept
        : engine_(&engine),
          objective_(std::move(objective)),
          branches_(std::move(branches)),
          pool_(&pool),
          training_(),
          input_(&input),
          one_to_many_weight_(objective_.branch_weights().start),
          doubles_()
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
                    for (std::size_t level = 0; level < branch.views.size(); ++level)
                    {
                        branch.views[level].boxes = floats_of(*branch.ports[level * 2]);
                        branch.views[level].scores = floats_of(*branch.ports[level * 2 + 1]);
                    }
                }
                branch.stage.assign(SushiRuntime::span<const FreeLevelViews>(
                                        branch.views.data(), branch.views.size()),
                                    truth);

                SA_ZONE("host", "assign");
                for (std::size_t leaf = 0; leaf < branch.leaves.size(); ++leaf)
                {
                    SushiBLAS::Tensor* const tensor = branch.leaves[leaf];
                    if (tensor == nullptr)
                        continue;
                    branch.stage.write_leaf(leaf, SushiRuntime::span<float>(
                                                      tensor->data_as<float>(),
                                                      static_cast<std::size_t>(
                                                          tensor->num_elements)));
                }
            }

            if (branch_weights_[0] != nullptr)
            {
                branch_weights_[0]->data_as<float>()[0] = one_to_many_weight_;
                branch_weights_[1]->data_as<float>()[0] = 1.0f - one_to_many_weight_;
            }

            // The projection is rewritten each run; the pool gives no promise to keep a leaf.
            if (projection_ != nullptr)
            {
                Detect::Free::fill_distribution_projection(SushiRuntime::span<float>(
                    projection_->data_as<float>(),
                    static_cast<std::size_t>(projection_->num_elements)));
            }

            return Status();
        });
    }

    FreeLossTerms FreeTrainingStep::measure_terms()
    {
        if (branches_.size() == 1)
            return measure_branch(branches_.front());

        const double weights[] = {static_cast<double>(one_to_many_weight_),
                                  1.0 - static_cast<double>(one_to_many_weight_)};
        FreeLossTerms sum;
        for (std::size_t copy = 0; copy < branches_.size(); ++copy)
        {
            const FreeLossTerms terms = measure_branch(branches_[copy]);
            sum.total += weights[copy] * terms.total;
            sum.box += weights[copy] * terms.box;
            sum.classes += weights[copy] * terms.classes;
            sum.distribution += weights[copy] * terms.distribution;
            sum.l1 += weights[copy] * terms.l1;
        }
        return sum;
    }

    FreeLossTerms FreeTrainingStep::measure_branch(const Branch& branch)
    {
        const FreeModelFacts& facts = objective_.facts();
        const std::vector<FreeLeaf>& described = branch.stage.leaves();
        FreeLossTerms sum;

        for (std::size_t level = 0; level < facts.levels.size(); ++level)
        {
            LevelDoubles& host = doubles_;
            {
                SA_ZONE("host", "readback");
                widen(*branch.ports[level * 2], host.boxes);
                widen(*branch.ports[level * 2 + 1], host.scores);
            }
            for (std::vector<double>* field :
                 {&host.target_scores, &host.box_mask, &host.normalized_weight, &host.target_boxes,
                  &host.target_distances, &host.l1_scale, &host.safe_boxes,
                  &host.distribution_weights, &host.anchor_points})
                field->clear();

            for (std::size_t leaf = 0; leaf < described.size(); ++leaf)
            {
                if (branch.leaves[leaf] == nullptr
                    || described[leaf].level != static_cast<std::int64_t>(level))
                    continue;

                std::vector<double>* target = nullptr;
                switch (described[leaf].kind)
                {
                    case FreeLeafKind::TargetScores:
                        target = &host.target_scores;
                        break;
                    case FreeLeafKind::BoxMask:
                        target = &host.box_mask;
                        break;
                    case FreeLeafKind::NormalizedWeight:
                        target = &host.normalized_weight;
                        break;
                    case FreeLeafKind::TargetBoxes:
                        target = &host.target_boxes;
                        break;
                    case FreeLeafKind::TargetDistances:
                        target = &host.target_distances;
                        break;
                    case FreeLeafKind::L1Scale:
                        target = &host.l1_scale;
                        break;
                    case FreeLeafKind::SafeBoxes:
                        target = &host.safe_boxes;
                        break;
                    case FreeLeafKind::DistributionWeights:
                        target = &host.distribution_weights;
                        break;
                    case FreeLeafKind::AnchorPoints:
                        target = &host.anchor_points;
                        break;
                    case FreeLeafKind::AnchorStrides:
                    case FreeLeafKind::TargetScoresSum:
                        break;
                }
                if (target != nullptr)
                {
                    SA_ZONE("host", "readback");
                    widen(*branch.leaves[leaf], *target);
                }
            }

            const FreeLevelLossArrays arrays{
                view(host.boxes),          view(host.scores),           view(host.target_scores),
                view(host.box_mask),       view(host.normalized_weight), view(host.target_boxes),
                view(host.target_distances), view(host.l1_scale),        view(host.safe_boxes),
                view(host.distribution_weights), view(host.anchor_points)};

            // Every term is a sum over rows divided by one score sum, so levels add.
            const FreeLossTerms terms = evaluate_free_detection_loss(
                SushiRuntime::span<const FreeLevelLossArrays>(&arrays, 1), facts.reg_max,
                facts.classes, objective_.batch(),
                static_cast<double>(branch.stage.targets().target_scores_sum[0]),
                objective_.gains());
            sum.total += terms.total;
            sum.box += terms.box;
            sum.classes += terms.classes;
            sum.distribution += terms.distribution;
            sum.l1 += terms.l1;
        }

        return sum;
    }

    Result<FreeLossTerms> FreeTrainingStep::run(const SushiBLAS::Tensor& inputs,
                                                const Detect::GroundTruthBatch& truth)
    {
        SA_TRY(Train::check_feed(inputs, *input_, "inputs"));
        SA_TRY(check_truth(truth));

        {
            SA_ZONE("copy", "input");
            SA_TRY(::SushiAI::Detail::record_and_run(
                *engine_, [this, &inputs](SushiBLAS::TaskRecorder& recorder) -> Status
            {
                SushiBLAS::BLASOps blas(recorder);
                blas.copy(inputs, *input_);
                return Status();
            }));
        }

        // The ports are read before the rest of the step may reuse their storage.
        FreeLossTerms terms;
        SA_ZONE("device", "run");
        SA_TRY(training_->run([this, &truth, &terms]() -> Status
        {
            SA_TRY(feed_leaves(truth));
            return ::SushiAI::Detail::guard(::SushiAI::Detail::SeamPhase::Execute,
                                            [this, &terms]() -> Status
            {
                SA_ZONE("host", "loss");
                terms = measure_terms();
                return Status();
            });
        }));

        return terms;
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

    Status FreeTrainingStep::restore_run_count(std::size_t count)
    {
        return training_->restore_run_count(count);
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
