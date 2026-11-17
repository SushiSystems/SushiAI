/**************************************************************************/
/* test_metric_values.cpp                                                 */
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
 * @file test_metric_values.cpp
 * @brief Tests the values an objective publishes to its metrics through the step builders.
 * @author Mustafa Garip
 */

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>
#include <gtest/gtest.h>
#include <SushiBLAS/tensor/tensor.hpp>
#include <SushiAI/data/dataset.hpp>
#include <SushiAI/data/input.hpp>
#include <SushiAI/data/materialised_dataset.hpp>
#include <SushiAI/data/roles.hpp>
#include <SushiAI/data/target.hpp>
#include <SushiAI/dtype/dtype.hpp>
#include <SushiAI/engine_fixture/engine_test.hpp>
#include <SushiAI/failure/check.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/shape/shape.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/graph/builder.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/graph/tensor_pool.hpp>
#include <SushiAI/nn/layers.hpp>
#include <SushiAI/nn/model_outputs.hpp>
#include <SushiAI/nn/model_state.hpp>
#include <SushiAI/optim/optimizer.hpp>
#include <SushiAI/optim/sgd.hpp>
#include <SushiAI/tensor_values/tensor_values.hpp>
#include <SushiAI/train/loss_trace.hpp>
#include <SushiAI/train/metrics.hpp>
#include <SushiAI/train/objective.hpp>
#include <SushiAI/train/step_view.hpp>
#include <SushiAI/train/steps.hpp>
#include <SushiAI/train/sum_objective.hpp>
#include <SushiAI/train/trainer.hpp>
#include <SushiAI/train/training_graph.hpp>

using namespace SushiAI;
using Train::CrossEntropyObjective;

// SaTest keeps SushiAI::Test reachable inside TEST_F; see tests/layers/train/README.md.
namespace SaTest = SushiAI::Test;

namespace
{
    /** @brief The model whose graphs the cases below build. */
    using MLP = NN::Sequential<NN::Linear, NN::GELU, NN::Linear>;

    /** @brief Samples per step in every case of this file. */
    constexpr std::int64_t K_BATCH = 4;

    /** @brief Input features of the MLP. */
    constexpr std::int64_t K_FEATURES = 6;

    /** @brief Hidden units of the MLP. */
    constexpr std::int64_t K_HIDDEN = 8;

    /** @brief Classes the MLP ends in. */
    constexpr std::int64_t K_CLASSES = 3;

    /** @brief Names the value the publishing objective declares. */
    constexpr std::string_view K_CLASS_VALUE = "classes";

    /** @brief Names the value the trace-reading objective keeps and publishes. */
    constexpr std::string_view K_KEPT_VALUE = "kept";

    /** @brief Builds the MLP over K_FEATURES, K_HIDDEN and K_CLASSES. */
    MLP make_mlp()
    {
        return MLP(NN::Linear{K_FEATURES, K_HIDDEN}, NN::GELU{}, NN::Linear{K_HIDDEN, K_CLASSES});
    }

    /**
     * @brief Trains like a cross entropy and publishes the argmax of the model's only output, or
     *        refuses to.
     */
    class PublishingObjective final : public Train::Objective
    {
        public:
            /**
             * @brief Builds the objective.
             * @param value The name of the published value.
             * @param refuses Whether declare_metric_values() fails.
             */
            PublishingObjective(std::string value, bool refuses)
                : value_(std::move(value)), refuses_(refuses), inner_(K_BATCH, K_CLASSES)
            {

            }

            /** @brief Declares what the cross entropy declares. */
            void declare(const Graph::GraphBuilder& builder,
                         Train::ObjectiveLeaves& leaves) const override
            {
                inner_.declare(builder, leaves);
            }

            /** @brief Emits what the cross entropy emits. */
            [[nodiscard]] Result<Train::LossTrace> loss(
                const Graph::GraphBuilder& builder, const Graph::Graph& graph,
                const NN::ModelOutputs& outputs,
                const Train::ObjectiveLeaves& leaves) const override
            {
                return inner_.loss(builder, graph, outputs, leaves);
            }

            /** @brief Returns true, as the cross entropy is. */
            [[nodiscard]] bool is_fed_by_batch() const noexcept override { return true; }

            /** @brief Feeds what the cross entropy feeds. */
            [[nodiscard]] Status feed(const Data::Batch& batch,
                                      const Train::ObjectiveLeaves& leaves,
                                      Graph::TensorPool& pool) const override
            {
                return inner_.feed(batch, leaves, pool);
            }

            /** @brief Emits the INT64 [N] argmax of the only output under the given name. */
            [[nodiscard]] Result<NN::ModelOutputs> declare_metric_values(
                const Graph::GraphBuilder& builder, const NN::ModelOutputs& outputs,
                const Train::LossTrace&) const override
            {
                SA_FAIL_IF(refuses_, StatusCode::InvalidDescription,
                           "the test objective refuses to declare '{}'", value_);
                SA_TRY_ASSIGN(const Graph::ValueID only, outputs.only());
                SA_TRY_ASSIGN(const Graph::ValueID classes, builder.argmax_rows(only));

                NN::ModelOutputs values;
                SA_TRY(values.add(value_, classes));

                return Result<NN::ModelOutputs>(std::move(values));
            }

            /** @brief Returns "publishing". */
            std::string_view name() const noexcept override { return "publishing"; }

        private:
            std::string value_;            /**< The name the value is published under. */
            bool refuses_;                 /**< Whether the hook fails. */
            CrossEntropyObjective inner_;  /**< Supplies the loss and its leaves. */
    };

    /**
     * @brief Publishes the argmax like PublishingObjective and, in measure(), records the value
     *        the step holds beside the argmax of the logits it holds, read on the host.
     */
    class RecordingObjective final : public Train::Objective
    {
        public:
            /** @brief Builds the objective over the cross entropy of K_BATCH and K_CLASSES. */
            RecordingObjective() : publisher_(std::string(K_CLASS_VALUE), false) {}

            /** @brief Declares what the publisher declares. */
            void declare(const Graph::GraphBuilder& builder,
                         Train::ObjectiveLeaves& leaves) const override
            {
                publisher_.declare(builder, leaves);
            }

            /** @brief Emits what the publisher emits. */
            [[nodiscard]] Result<Train::LossTrace> loss(
                const Graph::GraphBuilder& builder, const Graph::Graph& graph,
                const NN::ModelOutputs& outputs,
                const Train::ObjectiveLeaves& leaves) const override
            {
                return publisher_.loss(builder, graph, outputs, leaves);
            }

            /** @brief Declares what the publisher declares. */
            [[nodiscard]] Result<NN::ModelOutputs> declare_metric_values(
                const Graph::GraphBuilder& builder, const NN::ModelOutputs& outputs,
                const Train::LossTrace& trace) const override
            {
                return publisher_.declare_metric_values(builder, outputs, trace);
            }

            /** @brief Feeds what the cross entropy feeds. */
            [[nodiscard]] Status feed(const Data::Batch& batch,
                                      const Train::ObjectiveLeaves& leaves,
                                      Graph::TensorPool& pool) const override
            {
                return publisher_.feed(batch, leaves, pool);
            }

            /** @brief Returns true, as the cross entropy is. */
            [[nodiscard]] bool is_fed_by_batch() const noexcept override { return true; }

            /** @brief Records the class of each row from the step's value and from its logits. */
            [[nodiscard]] Status measure(const Train::StepView& step,
                                         Train::Metrics&) const override
            {
                SA_TRY_ASSIGN(const SushiBLAS::Tensor* const classes,
                              Train::named_value(step, K_CLASS_VALUE));
                SA_TRY_ASSIGN(const SushiBLAS::Tensor* const logits, Train::only_output(step));

                for (std::int64_t row = 0; row < K_BATCH; ++row)
                {
                    const float* const scores = logits->data_as<float>() + row * K_CLASSES;
                    device_classes.push_back(classes->data_as<std::int64_t>()[row]);
                    host_classes.push_back(std::max_element(scores, scores + K_CLASSES) - scores);
                }

                return Status();
            }

            /** @brief Returns "recording". */
            std::string_view name() const noexcept override { return "recording"; }

            /** @brief Holds the class the step's value gave each row, over every measure(). */
            mutable std::vector<std::int64_t> device_classes;

            /** @brief Holds the column of the largest logit of each row, over every measure(). */
            mutable std::vector<std::int64_t> host_classes;

        private:
            PublishingObjective publisher_; /**< Supplies the loss, the leaves and the value. */
    };

    /**
     * @brief Keeps the model's only output in its trace and publishes the argmax of what the
     *        trace holds, or nothing when it holds nothing.
     */
    class TraceReadingObjective final : public Train::Objective
    {
        public:
            /** @brief Builds the objective over the cross entropy of K_BATCH and K_CLASSES. */
            TraceReadingObjective() : inner_(K_BATCH, K_CLASSES) {}

            /** @brief Declares what the cross entropy declares. */
            void declare(const Graph::GraphBuilder& builder,
                         Train::ObjectiveLeaves& leaves) const override
            {
                inner_.declare(builder, leaves);
            }

            /** @brief Emits the cross entropy and keeps the only output under K_KEPT_VALUE. */
            [[nodiscard]] Result<Train::LossTrace> loss(
                const Graph::GraphBuilder& builder, const Graph::Graph& graph,
                const NN::ModelOutputs& outputs,
                const Train::ObjectiveLeaves& leaves) const override
            {
                SA_TRY_ASSIGN(Train::LossTrace trace,
                              inner_.loss(builder, graph, outputs, leaves));
                SA_TRY_ASSIGN(const Graph::ValueID only, outputs.only());
                trace.add(K_KEPT_VALUE, only);

                return trace;
            }

            /** @brief Returns true, as the cross entropy is. */
            [[nodiscard]] bool is_fed_by_batch() const noexcept override { return true; }

            /** @brief Emits the argmax of the kept output under K_KEPT_VALUE, when it was kept. */
            [[nodiscard]] Result<NN::ModelOutputs> declare_metric_values(
                const Graph::GraphBuilder& builder, const NN::ModelOutputs&,
                const Train::LossTrace& trace) const override
            {
                NN::ModelOutputs values;
                if (!trace.has(K_KEPT_VALUE))
                    return Result<NN::ModelOutputs>(std::move(values));

                SA_TRY_ASSIGN(const Graph::ValueID classes,
                              builder.argmax_rows(trace.at(K_KEPT_VALUE)));
                SA_TRY(values.add(std::string(K_KEPT_VALUE), classes));

                return Result<NN::ModelOutputs>(std::move(values));
            }

            /** @brief Returns "trace_reading". */
            std::string_view name() const noexcept override { return "trace_reading"; }

        private:
            CrossEntropyObjective inner_; /**< Supplies the loss and its leaves. */
    };

    /** @brief Provides the engine the step is compiled on. */
    class Medium_MetricValuesTrainer : public SaTest::EngineTest
    {

    };

    /** @brief Tells whether @p id is one of the pinned values of @p built. */
    bool is_pinned(const Train::TrainingGraph& built, Graph::ValueID id)
    {
        return std::find(built.pinned.begin(), built.pinned.end(), id) != built.pinned.end();
    }
} // namespace

/** @brief Tests that an objective that overrides nothing publishes no value. */
TEST(Small_MetricValues, TheDefaultObjectiveDeclaresNoValue)
{
    MLP model = make_mlp();
    const CrossEntropyObjective objective(K_BATCH, K_CLASSES);

    SA_ASSERT_OK_AND_ASSIGN(
        const Train::TrainingGraph built,
        Train::build_training_graph(model, objective, Shape{K_BATCH, K_FEATURES}));

    EXPECT_TRUE(built.metric_values.all().empty());
}

/** @brief Tests that a published value is kept in the training graph, pinned, as INT64 [N]. */
TEST(Small_MetricValues, ATrainingGraphHoldsAndPinsThePublishedValue)
{
    MLP model = make_mlp();
    const PublishingObjective objective(std::string(K_CLASS_VALUE), false);

    SA_ASSERT_OK_AND_ASSIGN(
        const Train::TrainingGraph built,
        Train::build_training_graph(model, objective, Shape{K_BATCH, K_FEATURES}));
    SA_ASSERT_OK_AND_ASSIGN(const Graph::ValueID classes, built.metric_values.at(K_CLASS_VALUE));

    EXPECT_EQ(built.metric_values.all().size(), 1u);
    EXPECT_EQ(built.graph.value(classes).dtype, DType::Int64);
    EXPECT_EQ(built.graph.value(classes).shape, Shape{K_BATCH});
    EXPECT_TRUE(is_pinned(built, classes));
}

/** @brief Tests that the evaluation graph built with an objective holds the published value. */
TEST(Small_MetricValues, AnEvaluationGraphHoldsThePublishedValue)
{
    MLP model = make_mlp();
    const PublishingObjective objective(std::string(K_CLASS_VALUE), false);

    SA_ASSERT_OK_AND_ASSIGN(
        const Train::TracedStep step,
        Train::build_evaluation_graph(model, Shape{K_BATCH, K_FEATURES}, objective));
    SA_ASSERT_OK_AND_ASSIGN(const Graph::ValueID classes, step.metric_values.at(K_CLASS_VALUE));

    EXPECT_EQ(step.graph.value(classes).dtype, DType::Int64);
    EXPECT_EQ(step.graph.value(classes).shape, Shape{K_BATCH});
}

/** @brief Tests that the evaluation graph built without an objective holds no value. */
TEST(Small_MetricValues, AnEvaluationGraphWithoutAnObjectiveHoldsNoValue)
{
    MLP model = make_mlp();

    SA_ASSERT_OK_AND_ASSIGN(const Train::TracedStep step,
                            Train::build_evaluation_graph(model, Shape{K_BATCH, K_FEATURES}));

    EXPECT_TRUE(step.metric_values.all().empty());
}

/** @brief Tests that the training graph hands the hook what loss() kept. */
TEST(Small_MetricValues, ATrainingGraphHandsTheHookWhatTheLossKept)
{
    MLP model = make_mlp();
    const TraceReadingObjective objective;

    SA_ASSERT_OK_AND_ASSIGN(
        const Train::TrainingGraph built,
        Train::build_training_graph(model, objective, Shape{K_BATCH, K_FEATURES}));

    ASSERT_EQ(built.metric_values.all().size(), 1u);
    EXPECT_EQ(built.metric_values.all().front().first, K_KEPT_VALUE);
}

/** @brief Tests that the evaluation graph hands the hook a trace that holds nothing. */
TEST(Small_MetricValues, AnEvaluationGraphHandsTheHookAnEmptyTrace)
{
    MLP model = make_mlp();
    const TraceReadingObjective objective;

    SA_ASSERT_OK_AND_ASSIGN(
        const Train::TracedStep step,
        Train::build_evaluation_graph(model, Shape{K_BATCH, K_FEATURES}, objective));

    EXPECT_TRUE(step.metric_values.all().empty());
}

/** @brief Tests that the failure of the hook ends the build of either graph. */
TEST(Small_MetricValues, TheFailureOfTheHookEndsTheBuild)
{
    MLP model = make_mlp();
    const PublishingObjective objective(std::string(K_CLASS_VALUE), true);

    SA_EXPECT_FAILURE(Train::build_training_graph(model, objective, Shape{K_BATCH, K_FEATURES}),
                      StatusCode::InvalidDescription, "refuses to declare 'classes'");
    SA_EXPECT_FAILURE(
        Train::build_evaluation_graph(model, Shape{K_BATCH, K_FEATURES}, objective),
        StatusCode::InvalidDescription, "refuses to declare 'classes'");
}

/** @brief Tests that a sum keeps the values of its terms, the first of a repeated name only. */
TEST(Small_MetricValues, ASumKeepsTheValuesOfItsTermsOnceByName)
{
    MLP model = make_mlp();
    std::vector<Train::SumTerm> terms;
    terms.push_back(Train::SumTerm{
        std::make_unique<PublishingObjective>(std::string(K_CLASS_VALUE), false), 1.0});
    terms.push_back(Train::SumTerm{
        std::make_unique<PublishingObjective>(std::string(K_CLASS_VALUE), false), 1.0});
    terms.push_back(Train::SumTerm{std::make_unique<PublishingObjective>("scores", false), 1.0});
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<Train::SumObjective> sum,
                            Train::SumObjective::make(std::move(terms)));

    SA_ASSERT_OK_AND_ASSIGN(
        const Train::TrainingGraph built,
        Train::build_training_graph(model, *sum, Shape{K_BATCH, K_FEATURES}));

    ASSERT_EQ(built.metric_values.all().size(), 2u);
    EXPECT_EQ(built.metric_values.all()[0].first, "classes");
    EXPECT_EQ(built.metric_values.all()[1].first, "scores");
}

/**
 * @brief Tests that the trainer hands the published value to measure(): the class of each row
 *        equals the column of the largest logit the host reads.
 */
TEST_F(Medium_MetricValuesTrainer, TheTrainerHandsThePublishedValueToMeasure)
{
    constexpr std::int64_t K_INPUTS = 3;
    RecordingObjective objective;
    NN::Linear model{K_INPUTS, K_CLASSES};
    SA_ASSERT_OK_AND_ASSIGN(
        Train::TrainingGraph built,
        Train::build_training_graph(model, objective, Shape{K_BATCH, K_INPUTS}));
    SA_ASSERT_OK_AND_ASSIGN(NN::ModelState state, NN::ModelState::create(*engine, model));
    Graph::TensorPool pool(*engine);
    SA_ASSERT_OK_AND_ASSIGN(std::vector<Optim::ParameterSlot> slots,
                            Train::bind_parameters(backend(), built, state, pool));
    Optim::SGDOptions sgd;
    sgd.learning_rate = 0.0;
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<Optim::SGD> optimizer,
                            Optim::SGD::make(backend(), slots, sgd));
    SA_ASSERT_OK_AND_ASSIGN(
        const std::unique_ptr<Train::Trainer> trainer,
        Train::Trainer::make(backend(), built, pool, objective, *optimizer));
    const std::vector<Data::TargetDescription> roles = {
        Data::TargetDescription{std::string(Data::K_ROLE_LABELS), Shape{}, DType::Int32,
                                K_CLASSES}};
    SA_ASSERT_OK_AND_ASSIGN(
        Data::Batch batch,
        Data::allocate_batch(*engine, K_BATCH,
                             Data::InputDescription{Shape{K_INPUTS}, DType::Float32}, roles));
    SaTest::write_tensor(batch.inputs, std::vector<double>{1.0, 0.0, 0.0, 0.0, 1.0, 0.0,
                                                           0.0, 0.0, 1.0, 1.0, 1.0, 1.0});
    SaTest::write_tensor(batch.targets.front().values, std::vector<double>{0.0, 1.0, 2.0, 1.0});

    SA_EXPECT_OK(trainer->run_step(batch));

    ASSERT_EQ(objective.device_classes.size(), static_cast<std::size_t>(K_BATCH));
    EXPECT_EQ(objective.device_classes, objective.host_classes);
}
