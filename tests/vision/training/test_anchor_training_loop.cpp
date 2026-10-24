/**************************************************************************/
/* test_anchor_training_loop.cpp                                          */
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
 * @file test_anchor_training_loop.cpp
 * @brief Tests the loop that drives an anchor training step over a dataset.
 * @author Mustafa Garip
 */

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>
#include <gtest/gtest.h>
#include <SushiBLAS/tensor/tensor.hpp>
#include <SushiAI/SushiAI.hpp>
#include <SushiAI/dtype/dtype.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/shape/shape.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/data/dataset.hpp>
#include <SushiAI/data/epoch.hpp>
#include <SushiAI/data/materialised_dataset.hpp>
#include <SushiAI/data/roles.hpp>
#include <SushiAI/data/target.hpp>
#include <SushiAI/engine_fixture/engine_test.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/graph/tensor_pool.hpp>
#include <SushiAI/nn/conv2d.hpp>
#include <SushiAI/nn/init.hpp>
#include <SushiAI/nn/model_graph.hpp>
#include <SushiAI/nn/model_state.hpp>
#include <SushiAI/optim/cosine_schedule.hpp>
#include <SushiAI/optim/rate_schedule.hpp>
#include <SushiAI/optim/sgd.hpp>
#include <SushiAI/optim/step_schedule.hpp>
#include <SushiAI/seeded_noise/host_noise.hpp>
#include <SushiAI/tensor_values/tensor_values.hpp>
#include <SushiAI/train/buffer_update_recorder.hpp>
#include <SushiAI/train/steps.hpp>
#include <SushiAI/train/trainer.hpp>
#include <SushiAI/vision/detect/anchor/anchor_layout.hpp>
#include <SushiAI/vision/detect/anchor/head_options.hpp>
#include <SushiAI/vision/training/anchor_objective.hpp>
#include <SushiAI/vision/training/anchor_training_loop.hpp>
#include <SushiAI/vision/training/anchor_training_step.hpp>
#include <SushiAI/vision/training/truth_from_roles.hpp>

using namespace SushiAI;
// SaTest stays reachable inside a TEST_F body; see tests/vision/training/README.md.
namespace SaTest = SushiAI::Test;
using Graph::TensorPool;
using Vision::Training::AnchorHead;
using Vision::Training::AnchorLoopOptions;
using Vision::Training::AnchorObjective;
using Vision::Training::AnchorProgress;
using Vision::Training::AnchorRunReport;
using Vision::Training::AnchorTrainingStep;
using Vision::Training::run_anchor_training;

namespace
{
    /** @brief Holds the images per step. */
    constexpr std::int64_t K_BATCH = 2;

    /** @brief Holds the side of the square network input in pixels. */
    constexpr std::int64_t K_NETWORK_SIDE = 8;

    /** @brief Holds the side of the grid the one head publishes. */
    constexpr std::int64_t K_GRID_SIDE = 4;

    /** @brief Holds the classes of the head. */
    constexpr std::int64_t K_CLASSES = 2;

    /** @brief Holds the channels of the head: two anchors of 5 + K_CLASSES entries. */
    constexpr std::int64_t K_HEAD_CHANNELS = 14;

    /** @brief Holds the object slots of one image, M. */
    constexpr std::int64_t K_SLOTS = 2;

    /** @brief Holds the batches of one epoch of the dataset. */
    constexpr std::size_t K_BATCHES = 3;

    /** @brief Holds the rate the optimizer is made with, which no schedule below gives. */
    constexpr double K_MADE_RATE = 0.2;

    /** @brief Seeds the input images. */
    constexpr std::uint64_t K_INPUT_SEED = 20261011ull;

    /** @brief Bounds the error of a rate that is one product of two doubles. */
    constexpr double K_RATE_TOLERANCE = 1e-15;

    /** @brief Bounds the difference of a cost in the report from the callback's copy of it. */
    constexpr double K_COPY_TOLERANCE = 0.0;

    /** @brief Holds the pixel corners of the slots of both images; the last slot is padding. */
    const std::vector<float> K_CORNERS = {1.0f, 1.0f, 3.0f, 3.0f, 1.0f, 1.0f, 8.0f, 7.0f,
                                          4.0f, 4.0f, 6.0f, 7.0f, 0.0f, 0.0f, 0.0f, 0.0f};

    /** @brief Holds the class of each slot; the padding slot holds 0. */
    const std::vector<std::int32_t> K_INDICES = {0, 1, 1, 0};

    /** @brief Holds 1 for each slot in use and 0 for the padding slot. */
    const std::vector<float> K_FLAGS = {1.0f, 1.0f, 1.0f, 0.0f};

    /** @brief Returns a rate of 0.1 that halves from update 2 and again from update 5. */
    Optim::StepSchedule make_halving_schedule()
    {
        Optim::StepSchedule schedule;
        schedule.base_rate = 0.1;
        schedule.steps = {2, 5};
        schedule.scales = {0.5, 0.5};
        return schedule;
    }

    /** @brief Writes the corners, classes and flags above into the roles of @p batch. */
    void write_truth(Data::Batch& batch)
    {
        std::copy(K_CORNERS.begin(), K_CORNERS.end(), batch.targets[0].values.data_as<float>());
        std::copy(K_INDICES.begin(), K_INDICES.end(),
                  batch.targets[1].values.data_as<std::int32_t>());
        std::copy(K_FLAGS.begin(), K_FLAGS.end(), batch.targets[2].values.data_as<float>());
    }

    /** @brief Reads through another dataset and keeps the epoch of every begin_epoch(). */
    class EpochRecorder final : public Data::Dataset, public Data::EpochAware
    {
        public:
            /** @brief Reads through @p inner, which must outlive this. */
            explicit EpochRecorder(const Data::Dataset& inner) : inner_(&inner)
            {

            }

            int64_t batch_size() const noexcept override { return inner_->batch_size(); }
            Shape sample_shape() const override { return inner_->sample_shape(); }
            const std::vector<Data::TargetDescription>& targets() const noexcept override
            {
                return inner_->targets();
            }
            std::size_t batch_count() const noexcept override { return inner_->batch_count(); }
            std::string_view name() const noexcept override { return inner_->name(); }

            [[nodiscard]] Status read(std::size_t index, Data::Batch& batch) const override
            {
                return inner_->read(index, batch);
            }

            [[nodiscard]] Status read_rows(std::size_t index, Data::RowRange rows,
                                           Data::Batch& batch) const override
            {
                return inner_->read_rows(index, rows, batch);
            }

            void begin_epoch(std::int64_t epoch) const noexcept override
            {
                epochs_.push_back(epoch);
            }

            /** @brief Returns the epochs begun, in order. */
            [[nodiscard]] const std::vector<std::int64_t>& epochs() const noexcept
            {
                return epochs_;
            }

        private:
            /** @brief Points at the dataset the batches come from. */
            const Data::Dataset* inner_;

            /** @brief Holds the epoch of each begin_epoch(), in order. */
            mutable std::vector<std::int64_t> epochs_;
    };

    /** @brief Provides an engine, a one-head model with its step, and datasets of truths. */
    class Medium_AnchorTrainingLoop : public SaTest::EngineTest
    {
        protected:
            /** @brief Builds the model, both graphs, the optimizer and the step. */
            void SetUp() override
            {
                SaTest::EngineTest::SetUp();
                const Shape batch_shape{K_BATCH, K_NETWORK_SIDE, K_NETWORK_SIDE, 3};

                model_.emplace(Shape{K_NETWORK_SIDE, K_NETWORK_SIDE, 3});
                NN::Conv2d head = NN::Conv2d::same_padded(3, K_HEAD_CHANNELS, 3, 2);
                head.use_bias = true;
                SA_ASSERT_OK_AND_ASSIGN(const NN::NodeRef published,
                                        model_->add("head", std::move(head), model_->input()));
                SA_EXPECT_OK(model_->output("head", published));

                Vision::Detect::Anchor::AnchorLayout layout;
                layout.anchors = {{0.25f, 0.25f}, {0.75f, 0.75f}};
                layout.head_anchors = {{0, 1}};
                std::vector<AnchorHead> heads(1);
                heads[0].output = "head";
                heads[0].shape = {K_BATCH, K_GRID_SIDE, K_GRID_SIDE};
                heads[0].options.num_classes = K_CLASSES;
                SA_ASSERT_OK_AND_ASSIGN(
                    const AnchorObjective objective,
                    AnchorObjective::make({K_NETWORK_SIDE, K_NETWORK_SIDE}, std::move(layout),
                                          std::move(heads)));

                SA_ASSERT_OK_AND_ASSIGN(
                    built_, Train::build_training_graph(*model_, objective, batch_shape));
                SA_ASSERT_OK_AND_ASSIGN(state_, NN::ModelState::create(*engine, *model_));

                training_pool_ = std::make_unique<TensorPool>(*engine);
                SA_ASSERT_OK_AND_ASSIGN(
                    std::vector<Optim::ParameterSlot> slots,
                    Train::bind_parameters(built_, *state_, *training_pool_));

                // The last weight stays drawn, so the head starts away from zero.
                NN::InitPolicy policy;
                policy.zero_final_layer = false;
                SA_EXPECT_OK(NN::initialize(*engine, *state_, policy));

                SA_ASSERT_OK_AND_ASSIGN(
                    std::vector<Train::BufferUpdateSlot> updates,
                    Train::bind_buffer_updates(built_.updates, *state_, *training_pool_));

                Optim::SGDOptions options;
                options.learning_rate = K_MADE_RATE;
                SA_ASSERT_OK_AND_ASSIGN(optimizer_, Optim::SGD::make(*engine, slots, options));
                SA_ASSERT_OK_AND_ASSIGN(
                    step_, AnchorTrainingStep::make(*engine, objective, built_, *training_pool_,
                                                    *optimizer_, std::move(updates)));
            }

            /** @brief Releases what holds tensors of the engine before the engine goes. */
            void TearDown() override
            {
                step_.reset();
                optimizer_.reset();
                training_pool_.reset();
                state_.reset();
                SaTest::EngineTest::TearDown();
            }

            /**
             * @brief Returns a dataset of @p batches batches of @p rows seeded images each.
             * @param targets The roles every batch carries; the three truth roles get the
             *                values above when they are the roles of describe_truth_roles().
             */
            Data::MaterialisedDataset make_dataset(
                std::size_t batches, std::int64_t rows,
                const std::vector<Data::TargetDescription>& targets)
            {
                const Shape sample{K_NETWORK_SIDE, K_NETWORK_SIDE, 3};
                std::vector<Data::Batch> held;
                SaTest::HostNoise noise(K_INPUT_SEED);

                for (std::size_t index = 0; index < batches; ++index)
                {
                    Result<Data::Batch> batch =
                        Data::allocate_batch(
                            *engine, rows, Data::InputDescription{sample, DType::Float32}, targets);
                    EXPECT_TRUE(batch.status().is_ok());

                    std::vector<double> pixels(
                        static_cast<std::size_t>(batch.value().inputs.num_elements));
                    noise.fill(pixels);
                    SaTest::write_tensor(batch.value().inputs, pixels);

                    if (rows == K_BATCH && targets.size() == 3)
                        write_truth(batch.value());
                    held.push_back(std::move(batch).value());
                }

                return Data::MaterialisedDataset(
                    std::move(held), rows, Data::InputDescription{sample, DType::Float32}, targets,
                    "truths");
            }

            /** @brief Returns the three truth roles of K_SLOTS slots over K_CLASSES classes. */
            static std::vector<Data::TargetDescription> truth_roles()
            {
                Result<std::vector<Data::TargetDescription>> described =
                    Vision::Training::describe_truth_roles(K_SLOTS, K_CLASSES);
                EXPECT_TRUE(described.status().is_ok());
                return std::move(described).value();
            }

            /** @brief Holds the one-head model. */
            std::optional<NN::ModelGraph> model_;

            /** @brief Holds the training graph. */
            Train::TrainingGraph built_;

            /** @brief Holds the one shared state. */
            std::optional<NN::ModelState> state_;

            /** @brief Holds the training values. */
            std::unique_ptr<TensorPool> training_pool_;

            /** @brief Holds the plain SGD the step updates with. */
            std::unique_ptr<Optim::SGD> optimizer_;

            /** @brief Holds the step the loop drives. */
            std::unique_ptr<AnchorTrainingStep> step_;
    };
} // namespace

/** @brief Tests that each update takes the schedule's rate at its number, across epochs. */
TEST_F(Medium_AnchorTrainingLoop, EachUpdateTakesTheRateOfItsIteration)
{
    ASSERT_NE(step_, nullptr);
    const Data::MaterialisedDataset dataset = make_dataset(K_BATCHES, K_BATCH, truth_roles());
    const Optim::RateSchedule schedule = make_halving_schedule();

    std::vector<AnchorProgress> seen;
    AnchorLoopOptions options;
    options.epochs = 2;
    options.schedule = &schedule;
    options.on_update = [&seen](const AnchorProgress& progress) { seen.push_back(progress); };

    SA_ASSERT_OK_AND_ASSIGN(const AnchorRunReport report,
                            run_anchor_training(*step_, dataset, options));

    const double rates[] = {0.1, 0.05, 0.05, 0.05, 0.025, 0.025};
    ASSERT_EQ(seen.size(), 6u);
    for (std::size_t update = 0; update < seen.size(); ++update)
    {
        EXPECT_EQ(seen[update].epoch, static_cast<std::int64_t>(update / K_BATCHES) + 1);
        EXPECT_EQ(seen[update].batch, update % K_BATCHES);
        EXPECT_EQ(seen[update].iteration, static_cast<std::int64_t>(update) + 1);
        EXPECT_NEAR(seen[update].rate, rates[update], K_RATE_TOLERANCE);
    }

    EXPECT_EQ(report.updates, 6u);
    EXPECT_NEAR(report.last_rate, 0.025, K_RATE_TOLERANCE);
    EXPECT_NEAR(optimizer_->learning_rate(), 0.025, K_RATE_TOLERANCE);
    EXPECT_NEAR(report.first.total, seen.front().cost.total, K_COPY_TOLERANCE);
    EXPECT_NEAR(report.last.total, seen.back().cost.total, K_COPY_TOLERANCE);
    EXPECT_EQ(optimizer_->step_count(), 6u);
    EXPECT_EQ(step_->training().step().compile_count(), 1u);
}

/** @brief Tests that a cosine schedule drives the rate of each update of an anchor run. */
TEST_F(Medium_AnchorTrainingLoop, ACosineScheduleSetsTheRateOfEachUpdate)
{
    ASSERT_NE(step_, nullptr);
    const Data::MaterialisedDataset dataset = make_dataset(K_BATCHES, K_BATCH, truth_roles());

    Optim::CosineSchedule cosine;
    cosine.base_rate = 0.1;
    cosine.total_iterations = 6;
    cosine.final_rate = 0.01;
    const Optim::RateSchedule schedule = cosine;

    std::vector<AnchorProgress> seen;
    AnchorLoopOptions options;
    options.epochs = 2;
    options.schedule = &schedule;
    options.on_update = [&seen](const AnchorProgress& progress) { seen.push_back(progress); };

    SA_ASSERT_OK_AND_ASSIGN(const AnchorRunReport report,
                            run_anchor_training(*step_, dataset, options));

    // A double holds the schedule's own rate, so the two are compared exactly.
    ASSERT_EQ(seen.size(), 6u);
    for (std::size_t update = 0; update < seen.size(); ++update)
    {
        const std::int64_t iteration = static_cast<std::int64_t>(update) + 1;
        EXPECT_DOUBLE_EQ(seen[update].rate, cosine.rate_at(iteration)) << "update " << update;
    }
    EXPECT_DOUBLE_EQ(report.last_rate, cosine.rate_at(6));
}

/** @brief Tests that a run without a schedule leaves the optimizer's rate as it was made. */
TEST_F(Medium_AnchorTrainingLoop, ARunWithoutAScheduleKeepsTheOptimizersRate)
{
    ASSERT_NE(step_, nullptr);
    const Data::MaterialisedDataset dataset = make_dataset(K_BATCHES, K_BATCH, truth_roles());

    SA_ASSERT_OK_AND_ASSIGN(const AnchorRunReport report,
                            run_anchor_training(*step_, dataset));

    EXPECT_EQ(report.updates, K_BATCHES);
    EXPECT_NEAR(report.last_rate, K_MADE_RATE, K_RATE_TOLERANCE);
    EXPECT_NEAR(optimizer_->learning_rate(), K_MADE_RATE, K_RATE_TOLERANCE);
}

/** @brief Tests that a dataset the step cannot read is refused before any update. */
TEST_F(Medium_AnchorTrainingLoop, ADatasetTheStepCannotReadIsRefusedBeforeAnyUpdate)
{
    ASSERT_NE(step_, nullptr);

    const Data::MaterialisedDataset other_batch =
        make_dataset(K_BATCHES, K_BATCH + 1, truth_roles());
    SA_EXPECT_FAILURE(run_anchor_training(*step_, other_batch), StatusCode::Incompatible,
                      "'truths'", "batch of 3", "declared for 2");

    const Data::MaterialisedDataset empty = make_dataset(0, K_BATCH, truth_roles());
    SA_EXPECT_FAILURE(run_anchor_training(*step_, empty), StatusCode::InvalidDescription,
                      "'truths'", "no batches");

    const Data::MaterialisedDataset labelled = make_dataset(
        K_BATCHES, K_BATCH,
        {Data::TargetDescription{std::string(Data::K_ROLE_LABELS), Shape{}, DType::Int32,
                                 K_CLASSES}});
    SA_EXPECT_FAILURE(run_anchor_training(*step_, labelled), StatusCode::InvalidDescription,
                      "'truths'", "'boxes'");

    EXPECT_EQ(optimizer_->step_count(), 0u);
}

/** @brief Tests that the options of a run are checked: the epochs and the schedule. */
TEST_F(Medium_AnchorTrainingLoop, OptionsARunCannotUseAreRefused)
{
    ASSERT_NE(step_, nullptr);
    const Data::MaterialisedDataset dataset = make_dataset(K_BATCHES, K_BATCH, truth_roles());

    AnchorLoopOptions no_epoch;
    no_epoch.epochs = 0;
    SA_EXPECT_FAILURE(run_anchor_training(*step_, dataset, no_epoch),
                      StatusCode::InvalidDescription, "at least one epoch", "0");

    Optim::StepSchedule uneven_steps = make_halving_schedule();
    uneven_steps.scales.pop_back();
    const Optim::RateSchedule uneven = uneven_steps;
    AnchorLoopOptions bad_schedule;
    bad_schedule.schedule = &uneven;
    SA_EXPECT_FAILURE(run_anchor_training(*step_, dataset, bad_schedule),
                      StatusCode::InvalidDescription, "2 steps", "1 scales");

    EXPECT_EQ(optimizer_->step_count(), 0u);
}

/** @brief Tests that a run goes on at the epoch index and the batch its options state. */
TEST_F(Medium_AnchorTrainingLoop, ARunStartsAtTheEpochAndTheBatchItsOptionsState)
{
    ASSERT_NE(step_, nullptr);
    const Data::MaterialisedDataset held = make_dataset(K_BATCHES, K_BATCH, truth_roles());
    const EpochRecorder dataset(held);

    std::vector<AnchorProgress> seen;
    AnchorLoopOptions options;
    options.epochs = 2;
    options.first_epoch = 3;
    options.first_batch = 2;
    options.on_update = [&seen](const AnchorProgress& progress) { seen.push_back(progress); };

    SA_ASSERT_OK_AND_ASSIGN(const AnchorRunReport report,
                            run_anchor_training(*step_, dataset, options));

    // The first pass reads its last batch alone; the second reads all three.
    EXPECT_EQ(report.updates, 1u + K_BATCHES);
    EXPECT_EQ(dataset.epochs(), (std::vector<std::int64_t>{3, 4}));
    ASSERT_EQ(seen.size(), 1u + K_BATCHES);
    EXPECT_EQ(seen[0].epoch, 1);
    EXPECT_EQ(seen[0].batch, 2u);
    EXPECT_EQ(seen[1].epoch, 2);
    EXPECT_EQ(seen[1].batch, 0u);
}

/** @brief Tests that a start the run cannot take is refused before any update. */
TEST_F(Medium_AnchorTrainingLoop, AStartTheDatasetLacksIsRefused)
{
    ASSERT_NE(step_, nullptr);
    const Data::MaterialisedDataset dataset = make_dataset(K_BATCHES, K_BATCH, truth_roles());

    AnchorLoopOptions negative;
    negative.first_epoch = -1;
    SA_EXPECT_FAILURE(run_anchor_training(*step_, dataset, negative),
                      StatusCode::InvalidDescription, "epoch index", "-1");

    AnchorLoopOptions past_the_end;
    past_the_end.first_batch = K_BATCHES;
    SA_EXPECT_FAILURE(run_anchor_training(*step_, dataset, past_the_end),
                      StatusCode::Incompatible, "batch 3", "'truths'", "3 batches");

    EXPECT_EQ(optimizer_->step_count(), 0u);
}

/** @brief Tests that a restored count is the count of both parts, and is restored once. */
TEST_F(Medium_AnchorTrainingLoop, ARestoredRunCountIsTheCountOfBothParts)
{
    ASSERT_NE(step_, nullptr);
    const Data::MaterialisedDataset dataset = make_dataset(K_BATCHES, K_BATCH, truth_roles());

    EXPECT_EQ(step_->run_count(), 0u);
    SA_EXPECT_OK(step_->restore_run_count(4));
    EXPECT_EQ(step_->run_count(), 4u);
    EXPECT_EQ(step_->training().lead()->run_count(), 4u);

    SA_EXPECT_OK(run_anchor_training(*step_, dataset));
    EXPECT_EQ(step_->run_count(), 4u + K_BATCHES);
    EXPECT_EQ(step_->training().lead()->run_count(), 4u + K_BATCHES);

    SA_EXPECT_FAILURE(step_->restore_run_count(1), StatusCode::InvalidDescription,
                      "before its first run", "run 7");
}

/** @brief Tests that a callback that throws ends the run as a failure after its update. */
TEST_F(Medium_AnchorTrainingLoop, ACallbackThatThrowsEndsTheRunAsAFailure)
{
    ASSERT_NE(step_, nullptr);
    const Data::MaterialisedDataset dataset = make_dataset(K_BATCHES, K_BATCH, truth_roles());

    AnchorLoopOptions options;
    options.on_update = [](const AnchorProgress&) { throw std::runtime_error("stop here"); };

    const Result<AnchorRunReport> report = run_anchor_training(*step_, dataset, options);

    EXPECT_FALSE(report.status().is_ok());
    EXPECT_NE(report.status().message().find("stop here"), std::string::npos);
    EXPECT_EQ(optimizer_->step_count(), 1u);
}
