/**************************************************************************/
/* anchor_training_step.cpp                                               */
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
 * @file anchor_training_step.cpp
 * @brief Implements AnchorTrainingStep: the step up to its heads, the host deltas and the rest.
 * @author Mustafa Garip
 */

#include <SushiAI/vision/training/anchor_training_step.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <utility>

#include <SushiBLAS/blas/blas.hpp>

#include <SushiAI/failure/check.hpp>
#include <SushiAI/dtype/dtype.hpp>
#include <SushiAI/seam/seam.hpp>
#include <SushiAI/execution/execution.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/train/feed.hpp>
#include <SushiAI/vision/detect/anchor/head_options.hpp>

namespace SushiAI::Vision::Training
{
    namespace
    {
        /** @brief Returns the elements of a head's `[B, H, W, A * (5 + C)]` output. */
        [[nodiscard]] std::int64_t count_head_elements(const AnchorObjective& objective,
                                                       std::size_t head)
        {
            const AnchorHead& described = objective.heads()[head];
            const std::int64_t channels = Detect::Anchor::count_head_channels(
                described.options, objective.layout(), static_cast<std::int64_t>(head));

            return described.shape.batch * described.shape.height * described.shape.width
                 * channels;
        }

        /**
         * @brief Checks that the host can read or write @p tensor as @p elements floats.
         * @param what Names the tensor in the message.
         * @return Success, or Incompatible naming the dtype or both element counts.
         */
        [[nodiscard]] Status check_host_floats(const SushiBLAS::Tensor& tensor,
                                               std::int64_t elements, std::string_view what)
        {
            SA_FAIL_IF(tensor.dtype != DType::Float32, StatusCode::Incompatible,
                       "'{}' holds {}; the host stage of the anchor step reads and writes "
                       "float32",
                       what, dtype_name(tensor.dtype));
            SA_FAIL_IF(static_cast<std::int64_t>(tensor.num_elements) != elements,
                       StatusCode::Incompatible,
                       "'{}' holds {} elements, but the objective describes a head of {}", what,
                       static_cast<std::int64_t>(tensor.num_elements), elements);

            return Status();
        }
    } // namespace

    Result<std::unique_ptr<AnchorTrainingStep>> AnchorTrainingStep::make(
        SushiBLAS::Engine& engine, const AnchorObjective& objective,
        const Train::TrainingGraph& built, Graph::TensorPool& training_pool,
        Optim::Optimizer& optimizer, std::vector<Train::BufferUpdateSlot> updates,
        Train::TrainingStepOptions options)
    {
        SA_FAIL_IF(built.spec.inputs == Graph::K_NO_VALUE
                       || !training_pool.is_bound(built.spec.inputs),
                   StatusCode::InvalidDescription,
                   "The anchor step needs the input leaf of its training graph bound in the "
                   "training pool; call Train::bind_parameters() on that pool first");

        SushiBLAS::Tensor& input = training_pool.tensor(built.spec.inputs);
        const std::size_t head_count = objective.heads().size();

        SA_TRY_ASSIGN(std::unique_ptr<AnchorTrainingStep> step,
                      ::SushiAI::Detail::guard_value(::SushiAI::Detail::SeamPhase::Allocate,
                                                     [&]()
        {
            std::unique_ptr<AnchorTrainingStep> made(new AnchorTrainingStep(
                engine, objective, built.outputs, training_pool, input));
            made->heads_.reserve(head_count);
            made->leaves_.reserve(head_count);
            made->deltas_.resize(head_count);
            return made;
        }));

        SA_TRY_ASSIGN(options.host_stage, step->describe_host_stage(built));
        const Train::HostStage stage = options.host_stage;

        SA_TRY_ASSIGN(step->training_,
                      Train::TrainingStep::make(engine, built.graph, training_pool, optimizer,
                                                std::move(updates), std::move(options)));

        for (std::size_t head = 0; head < head_count; ++head)
            SA_TRY(step->resolve_head(head, stage));

        return std::move(step);
    }

    AnchorTrainingStep::AnchorTrainingStep(SushiBLAS::Engine& engine, AnchorObjective objective,
                                           NN::ModelOutputs outputs, Graph::TensorPool& pool,
                                           SushiBLAS::Tensor& input) noexcept
        : engine_(&engine),
          objective_(std::move(objective)),
          outputs_(std::move(outputs)),
          pool_(&pool),
          training_(),
          input_(&input),
          heads_(),
          leaves_(),
          deltas_()
    {

    }

    Result<Train::HostStage> AnchorTrainingStep::describe_host_stage(
        const Train::TrainingGraph& built) const
    {
        Train::HostStage stage;
        SA_TRY(::SushiAI::Detail::guard(::SushiAI::Detail::SeamPhase::Allocate,
                                        [this, &built, &stage]() -> Status
        {
            const std::vector<AnchorHead>& heads = objective_.heads();
            for (std::size_t head = 0; head < heads.size(); ++head)
            {
                SA_TRY_ASSIGN(const Graph::ValueID output, outputs_.at(heads[head].output));

                const std::string role = objective_.leaf_role(head);
                SA_FAIL_IF(!built.leaves.has(role), StatusCode::InvalidDescription,
                           "The training graph declares no leaf '{}'; build it with the "
                           "objective the anchor step is given",
                           role);

                stage.reads.push_back(output);
                stage.writes.push_back(built.leaves.at(role));
            }

            return Status();
        }));

        return stage;
    }

    Status AnchorTrainingStep::resolve_head(std::size_t head, const Train::HostStage& stage)
    {
        const AnchorHead& described = objective_.heads()[head];
        const std::int64_t elements = count_head_elements(objective_, head);

        const Graph::ValueID leaf_id = stage.writes[head];
        SA_FAIL_IF(!pool_->is_bound(leaf_id), StatusCode::InvalidDescription,
                   "The delta leaf of head '{}' has no tensor in the training pool; call "
                   "Train::bind_parameters() on that pool first",
                   described.output);

        const SushiBLAS::Tensor& output = pool_->tensor(stage.reads[head]);
        SA_TRY(check_host_floats(output, elements, described.output));

        SushiBLAS::Tensor& leaf = pool_->tensor(leaf_id);
        SA_TRY(check_host_floats(leaf, elements, described.output));

        // make() reserved both lists, so neither push allocates.
        heads_.push_back(&output);
        leaves_.push_back(&leaf);

        return Status();
    }

    Status AnchorTrainingStep::check_truth(const Detect::GroundTruthBatch& truth) const
    {
        const std::vector<AnchorHead>& heads = objective_.heads();
        const std::int64_t batch = heads.front().shape.batch;

        SA_FAIL_IF(truth.batch != batch, StatusCode::Incompatible,
                   "The truth holds {} images, but the heads were declared for a batch of {}",
                   truth.batch, batch);
        SA_FAIL_IF(truth.max_boxes < 0, StatusCode::Incompatible,
                   "The truth holds {} rows per image; the count cannot be negative",
                   truth.max_boxes);

        const std::size_t rows = static_cast<std::size_t>(batch * truth.max_boxes);
        SA_FAIL_IF(truth.labels.size() != rows || truth.valid.size() != rows
                       || truth.boxes.size() != rows * 4,
                   StatusCode::Incompatible,
                   "A truth of {} images and {} rows per image needs {} labels, {} flags and {} "
                   "box values; it holds {}, {} and {}",
                   batch, truth.max_boxes, rows, rows, rows * 4, truth.labels.size(),
                   truth.valid.size(), truth.boxes.size());

        for (std::size_t row = 0; row < rows; ++row)
        {
            if (truth.valid[row] == 0)
                continue;

            for (const AnchorHead& head : heads)
            {
                SA_FAIL_IF(truth.labels[row] < 0 || truth.labels[row] >= head.options.num_classes,
                           StatusCode::InvalidDescription,
                           "Truth row {} has the label {}, but the head '{}' has {} classes", row,
                           truth.labels[row], head.output, head.options.num_classes);
            }
        }

        return Status();
    }

    Status AnchorTrainingStep::feed_deltas(const Detect::GroundTruthBatch& truth)
    {
        return ::SushiAI::Detail::guard(::SushiAI::Detail::SeamPhase::Allocate,
                                        [this, &truth]() -> Status
        {
            const std::vector<AnchorHead>& heads = objective_.heads();
            for (std::size_t head = 0; head < heads.size(); ++head)
            {
                const AnchorHead& described = heads[head];
                const SushiBLAS::Tensor& output = *heads_[head];
                const std::size_t elements = static_cast<std::size_t>(output.num_elements);
                Detect::Anchor::HeadDelta& delta = deltas_[head];

                SA_TRY(Detect::Anchor::compute_head_delta(
                    SushiRuntime::span<const float>(output.data_as<float>(), elements),
                    described.shape, described.options, objective_.layout(),
                    static_cast<std::int64_t>(head), described.training, truth,
                    objective_.network(), delta));

                SA_FATAL_IF(delta.delta.size() != elements,
                            "the delta of head '{}' holds {} values for a head of {}",
                            described.output, delta.delta.size(), elements);
                std::copy(delta.delta.begin(), delta.delta.end(),
                          leaves_[head]->data_as<float>());
            }

            return Status();
        });
    }

    Result<Detect::Anchor::HeadCost> AnchorTrainingStep::run(
        const SushiBLAS::Tensor& inputs, const Detect::GroundTruthBatch& truth)
    {
        SA_TRY(Train::check_feed(inputs, *input_, "inputs"));
        SA_TRY(check_truth(truth));

        SA_TRY(::SushiAI::Detail::record_and_run(
            *engine_, [this, &inputs](SushiBLAS::TaskRecorder& recorder) -> Status
        {
            SushiBLAS::BLASOps blas(recorder);
            blas.copy(inputs, *input_);
            return Status();
        }));

        // The deltas come from the heads of the pass the rest of the step differentiates.
        SA_TRY(training_->run([this, &truth]() { return feed_deltas(truth); }));

        return objective_.measure_cost(SushiRuntime::span<const Detect::Anchor::HeadDelta>(
            deltas_.data(), deltas_.size()));
    }

    Result<const SushiBLAS::Tensor*> AnchorTrainingStep::output(std::string_view name) const
    {
        SA_TRY_ASSIGN(const Graph::ValueID id, outputs_.at(name));

        return &pool_->tensor(id);
    }

    SushiRuntime::span<const Detect::Anchor::HeadDelta> AnchorTrainingStep::deltas()
        const noexcept
    {
        return SushiRuntime::span<const Detect::Anchor::HeadDelta>(deltas_.data(),
                                                                   deltas_.size());
    }

    const Train::TrainingStep& AnchorTrainingStep::training() const noexcept
    {
        return *training_;
    }

    std::size_t AnchorTrainingStep::run_count() const noexcept
    {
        return training_->step().run_count();
    }

    Status AnchorTrainingStep::restore_run_count(std::size_t count)
    {
        return training_->restore_run_count(count);
    }

    const AnchorObjective& AnchorTrainingStep::objective() const noexcept
    {
        return objective_;
    }

    Optim::Optimizer& AnchorTrainingStep::optimizer() noexcept
    {
        return training_->optimizer();
    }

} // namespace SushiAI::Vision::Training
