/**************************************************************************/
/* test_row_mask.cpp                                                */
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
 * @file test_row_mask.cpp
 * @brief Tests that the metrics of the four objectives count only the rows a type mask keeps.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>
#include <gtest/gtest.h>
#include <SushiBLAS/tensor/tensor.hpp>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/allocation/allocation.hpp>
#include <SushiAI/config/value.hpp>
#include <SushiAI/data/dataset.hpp>
#include <SushiAI/data/roles.hpp>
#include <SushiAI/data/target.hpp>
#include <SushiAI/dtype/dtype.hpp>
#include <SushiAI/execution/host_execution_backend.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/shape/shape.hpp>
#include <SushiAI/tensor_values/tensor_values.hpp>
#include <SushiAI/train/binary_cross_entropy_objective.hpp>
#include <SushiAI/train/decision_threshold.hpp>
#include <SushiAI/train/mean_squared_error_objective.hpp>
#include <SushiAI/train/metrics.hpp>
#include <SushiAI/train/multi_label_objective.hpp>
#include <SushiAI/train/objective.hpp>
#include <SushiAI/train/objective_output.hpp>
#include <SushiAI/train/step_view.hpp>
#include <support/tolerance.hpp>

using namespace SushiAI;
namespace SaTest = SushiAI::Test;

namespace
{
    /** @brief Holds the samples of every batch the cases build. */
    constexpr std::int64_t K_BATCH = 4;

    /** @brief Holds the classes of the cross-entropy cases. */
    constexpr std::int64_t K_CLASSES = 3;

    /** @brief Holds the columns of the multi-label and regression cases. */
    constexpr std::int64_t K_WIDTH = 2;

    /** @brief Counts the roundings of one ratio of exact integer counts: its quotient. */
    constexpr std::size_t K_RATIO_OPERATIONS = 1;

    /** @brief Counts the roundings of the mean of two F1 ratios: two quotients, a sum, a quotient. */
    constexpr std::size_t K_MEAN_OF_TWO_RATIOS_OPERATIONS = 4;

    /** @brief Provides the host engine the tensors of a step are allocated on. */
    class Small_MaskedMetrics : public ::testing::Test
    {
        protected:
            /** @brief Returns the mask of the cases: row 1 is a row of another question type. */
            [[nodiscard]] static std::vector<double> mask_without_row_one()
            {
                return {1.0, 0.0, 1.0, 1.0};
            }

            /** @brief Returns the selection that turns the type mask on. */
            [[nodiscard]] static Result<Train::ObjectiveSelection> masked()
            {
                SA_TRY_ASSIGN(const Config::Value attributes,
                              Config::parse_json(R"({"type_mask": true})", "<test>"));
                return Train::read_selection(attributes, "objective");
            }

            /** @brief Returns the logits of the cross-entropy cases: the best class is 0, 1, 2, 0. */
            [[nodiscard]] Result<SushiBLAS::Tensor> class_logits()
            {
                return tensor_of(Shape{K_BATCH, K_CLASSES},
                                 {2, 0, 0, 0, 3, 0, 0, 0, 1, 1, 0, 0});
            }

            /** @brief Returns the outputs of the regression cases, two values per sample. */
            [[nodiscard]] Result<SushiBLAS::Tensor> regression_outputs()
            {
                return tensor_of(Shape{K_BATCH, K_WIDTH},
                                 {1.0, 2.0, 10.0, -10.0, 0.0, 3.0, -1.0, 1.0});
            }

            /** @brief Returns a float32 tensor of @p shape that holds @p values. */
            [[nodiscard]] Result<SushiBLAS::Tensor> tensor_of(const Shape& shape,
                                                              const std::vector<double>& values)
            {
                SA_TRY_ASSIGN(SushiBLAS::Tensor tensor, allocate_tensor(backend.engine(), shape));
                SaTest::write_tensor(tensor, values);
                return tensor;
            }

            /** @brief Adds the float32 role @p role of @p shape holding @p values to @p batch. */
            [[nodiscard]] Status add_role(Data::Batch& batch, std::string_view role,
                                          const Shape& shape, const std::vector<double>& values)
            {
                Data::BatchTarget target;
                target.role = std::string(role);
                SA_TRY_ASSIGN(target.values, tensor_of(shape, values));
                batch.targets.push_back(std::move(target));
                return Status();
            }

            /** @brief Adds the int32 labels @p labels to @p batch. */
            [[nodiscard]] Status add_labels(Data::Batch& batch,
                                            const std::vector<std::int32_t>& labels)
            {
                Data::BatchTarget target;
                target.role = std::string(Data::K_ROLE_LABELS);
                SA_TRY_ASSIGN(target.values,
                              allocate_tensor(backend.engine(),
                                              Shape{static_cast<std::int64_t>(labels.size())},
                                              DType::Int32, TensorAccess::Host));
                for (std::size_t index = 0; index < labels.size(); ++index)
                    target.values.data_as<std::int32_t>()[index] = labels[index];

                batch.targets.push_back(std::move(target));
                return Status();
            }

            /** @brief Returns the view of a step over @p batch that published @p outputs. */
            [[nodiscard]] static Train::StepView view_of(
                const Data::Batch& batch, const std::vector<Train::StepOutput>& outputs)
            {
                return Train::StepView{
                    batch, 0.0,
                    SushiRuntime::span<const Train::StepOutput>(outputs.data(), outputs.size())};
            }

            /** @brief Expects the metric @p name of @p metrics to equal @p expected. */
            static void expect_metric(const Train::Metrics& metrics, std::string_view name,
                                      double expected, std::size_t operations)
            {
                ASSERT_NE(metrics.find(name), nullptr);
                EXPECT_TRUE(TestSupport::is_within(TestSupport::measure_relative_error<double>(
                    metrics.find(name)->value, expected, operations)));
            }

            /** @brief The host backend the tensors are allocated from. */
            HostExecutionBackend backend;
    };

} // namespace

/**
 * @brief Tests that accuracy leaves a dropped row out of both counts.
 *
 * The labels 0, 2, 2, 1 against the best logits 0, 1, 2, 0 are right on rows 0 and 2 and wrong on
 * rows 1 and 3. Keeping rows 0, 2 and 3 gives 2 of 3; row 1 counted would give 2 of 4.
 */
TEST_F(Small_MaskedMetrics, CrossEntropyAccuracyCountsTheKeptRowsAlone)
{
    SA_ASSERT_OK_AND_ASSIGN(const Train::ObjectiveSelection selection, masked());
    const Train::CrossEntropyObjective objective(K_BATCH, K_CLASSES, DType::Float32, selection);
    SA_ASSERT_OK_AND_ASSIGN(const SushiBLAS::Tensor logits, class_logits());
    Data::Batch batch;
    SA_EXPECT_OK(add_labels(batch, {0, 2, 2, 1}));
    SA_EXPECT_OK(add_role(batch, Data::K_ROLE_TYPE_MASK, Shape{K_BATCH}, mask_without_row_one()));
    const std::vector<Train::StepOutput> outputs{Train::StepOutput{"output", &logits}};

    Train::Metrics metrics;
    SA_EXPECT_OK(objective.measure(view_of(batch, outputs), metrics));

    expect_metric(metrics, "accuracy", 2.0 / 3.0, K_RATIO_OPERATIONS);
}

/** @brief Tests that a metric over a batch that keeps no row is 0, not a division by 0. */
TEST_F(Small_MaskedMetrics, CrossEntropyAccuracyOfABatchThatKeepsNoRowIsZero)
{
    SA_ASSERT_OK_AND_ASSIGN(const Train::ObjectiveSelection selection, masked());
    const Train::CrossEntropyObjective objective(K_BATCH, K_CLASSES, DType::Float32, selection);
    SA_ASSERT_OK_AND_ASSIGN(const SushiBLAS::Tensor logits, class_logits());
    Data::Batch batch;
    SA_EXPECT_OK(add_labels(batch, {0, 2, 2, 1}));
    SA_EXPECT_OK(add_role(batch, Data::K_ROLE_TYPE_MASK, Shape{K_BATCH}, {0.0, 0.0, 0.0, 0.0}));
    const std::vector<Train::StepOutput> outputs{Train::StepOutput{"output", &logits}};

    Train::Metrics metrics;
    SA_EXPECT_OK(objective.measure(view_of(batch, outputs), metrics));

    ASSERT_NE(metrics.find("accuracy"), nullptr);
    EXPECT_EQ(metrics.find("accuracy")->value, 0.0);
}

/** @brief Tests that a mask of another row count than the objective's batch is refused. */
TEST_F(Small_MaskedMetrics, AMaskOfAnotherRowCountIsRefusedByMeasure)
{
    SA_ASSERT_OK_AND_ASSIGN(const Train::ObjectiveSelection selection, masked());
    const Train::CrossEntropyObjective objective(K_BATCH, K_CLASSES, DType::Float32, selection);
    SA_ASSERT_OK_AND_ASSIGN(const SushiBLAS::Tensor logits, class_logits());
    Data::Batch batch;
    SA_EXPECT_OK(add_labels(batch, {0, 2, 2, 1}));
    SA_EXPECT_OK(add_role(batch, Data::K_ROLE_TYPE_MASK, Shape{5}, {1.0, 1.0, 1.0, 1.0, 1.0}));
    const std::vector<Train::StepOutput> outputs{Train::StepOutput{"output", &logits}};

    Train::Metrics metrics;
    SA_EXPECT_FAILURE(objective.measure(view_of(batch, outputs), metrics),
                      StatusCode::Incompatible, "5 rows", "reads 4");
    EXPECT_TRUE(metrics.all().empty());
}

/**
 * @brief Tests that the binary metrics leave a dropped row out.
 *
 * At the threshold one half the cut on the logit is 0. The logits 2, 1, -1, -3 against the labels
 * 1, 0, 1, 0 give a hit, a false alarm, a miss and a correct rejection. Without row 1, the false
 * alarm: accuracy 2 of 3, precision 1 of 1, recall 1 of 2. With it: 2 of 4, 1 of 2 and 1 of 2.
 */
TEST_F(Small_MaskedMetrics, BinaryMetricsCountTheKeptRowsAlone)
{
    SA_ASSERT_OK_AND_ASSIGN(const Train::ObjectiveSelection selection, masked());
    SA_ASSERT_OK_AND_ASSIGN(
        const std::unique_ptr<Train::BinaryCrossEntropyObjective> objective,
        Train::BinaryCrossEntropyObjective::make(K_BATCH, Train::K_DEFAULT_THRESHOLD,
                                                 DType::Float32, selection));
    SA_ASSERT_OK_AND_ASSIGN(const SushiBLAS::Tensor logits,
                            tensor_of(Shape{K_BATCH, 1}, {2.0, 1.0, -1.0, -3.0}));
    Data::Batch batch;
    SA_EXPECT_OK(add_role(batch, Data::K_ROLE_LABELS, Shape{K_BATCH}, {1.0, 0.0, 1.0, 0.0}));
    SA_EXPECT_OK(add_role(batch, Data::K_ROLE_TYPE_MASK, Shape{K_BATCH}, mask_without_row_one()));
    const std::vector<Train::StepOutput> outputs{Train::StepOutput{"output", &logits}};

    Train::Metrics metrics;
    SA_EXPECT_OK(objective->measure(view_of(batch, outputs), metrics));

    expect_metric(metrics, "accuracy", 2.0 / 3.0, K_RATIO_OPERATIONS);
    expect_metric(metrics, "precision", 1.0, K_RATIO_OPERATIONS);
    expect_metric(metrics, "recall", 0.5, K_RATIO_OPERATIONS);
}

/**
 * @brief Tests that the multi-label F1 leaves a dropped row out of every column.
 *
 * At the cut 0 the rows give, per column (label / logit): row 0 (1,0 / 1,-1), row 1 (0,0 / 1,1),
 * row 2 (1,1 / -1,1), row 3 (1,1 / 1,-1). Without row 1, column 0 holds 2 hits, 1 miss and no
 * false alarm, so its F1 is 4 / 5; column 1 holds 1 hit, 1 miss and a correct rejection, so its F1
 * is 2 / 3. Their mean is 11 / 15. With row 1 the two columns would give 2 / 3 and 1 / 2.
 */
TEST_F(Small_MaskedMetrics, MultiLabelF1CountsTheKeptRowsAlone)
{
    SA_ASSERT_OK_AND_ASSIGN(const Train::ObjectiveSelection selection, masked());
    SA_ASSERT_OK_AND_ASSIGN(
        const std::unique_ptr<Train::MultiLabelObjective> objective,
        Train::MultiLabelObjective::make(K_BATCH, K_WIDTH, Train::K_DEFAULT_THRESHOLD,
                                         DType::Float32, selection));
    SA_ASSERT_OK_AND_ASSIGN(const SushiBLAS::Tensor logits,
                            tensor_of(Shape{K_BATCH, K_WIDTH},
                                      {1.0, -1.0, 1.0, 1.0, -1.0, 1.0, 1.0, -1.0}));
    Data::Batch batch;
    SA_EXPECT_OK(add_role(batch, Data::K_ROLE_LABELS, Shape{K_BATCH, K_WIDTH},
                          {1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0}));
    SA_EXPECT_OK(add_role(batch, Data::K_ROLE_TYPE_MASK, Shape{K_BATCH}, mask_without_row_one()));
    const std::vector<Train::StepOutput> outputs{Train::StepOutput{"output", &logits}};

    Train::Metrics metrics;
    SA_EXPECT_OK(objective->measure(view_of(batch, outputs), metrics));

    expect_metric(metrics, "f1", 11.0 / 15.0, K_MEAN_OF_TWO_RATIOS_OPERATIONS);
}

/**
 * @brief Tests that the mean absolute error leaves the elements of a dropped row out.
 *
 * The absolute differences of the kept rows 0, 2 and 3 are 0.5 0.5, 1 2 and 0.5 1, which sum to
 * 5.5 over 6 elements. Row 1 differs by 10 in each element and would move the mean to 25.5 / 8.
 */
TEST_F(Small_MaskedMetrics, MeanAbsoluteErrorCountsTheKeptRowsAlone)
{
    SA_ASSERT_OK_AND_ASSIGN(const Train::ObjectiveSelection selection, masked());
    SA_ASSERT_OK_AND_ASSIGN(
        const std::unique_ptr<Train::MeanSquaredErrorObjective> objective,
        Train::MeanSquaredErrorObjective::make(K_BATCH, K_WIDTH, DType::Float32, selection));
    SA_ASSERT_OK_AND_ASSIGN(const SushiBLAS::Tensor predicted, regression_outputs());
    Data::Batch batch;
    SA_EXPECT_OK(add_role(batch, Data::K_ROLE_VALUES, Shape{K_BATCH, K_WIDTH},
                          {0.5, 2.5, 0.0, 0.0, 1.0, 1.0, -1.5, 2.0}));
    SA_EXPECT_OK(add_role(batch, Data::K_ROLE_TYPE_MASK, Shape{K_BATCH}, mask_without_row_one()));
    const std::vector<Train::StepOutput> outputs{Train::StepOutput{"output", &predicted}};

    Train::Metrics metrics;
    SA_EXPECT_OK(objective->measure(view_of(batch, outputs), metrics));

    expect_metric(metrics, "mean_absolute_error", 5.5 / 6.0, K_RATIO_OPERATIONS);
}

/** @brief Tests that a regression batch that keeps no row has a mean absolute error of 0. */
TEST_F(Small_MaskedMetrics, MeanAbsoluteErrorOfABatchThatKeepsNoRowIsZero)
{
    SA_ASSERT_OK_AND_ASSIGN(const Train::ObjectiveSelection selection, masked());
    SA_ASSERT_OK_AND_ASSIGN(
        const std::unique_ptr<Train::MeanSquaredErrorObjective> objective,
        Train::MeanSquaredErrorObjective::make(K_BATCH, K_WIDTH, DType::Float32, selection));
    SA_ASSERT_OK_AND_ASSIGN(const SushiBLAS::Tensor predicted, regression_outputs());
    Data::Batch batch;
    SA_EXPECT_OK(add_role(batch, Data::K_ROLE_VALUES, Shape{K_BATCH, K_WIDTH},
                          {0.5, 2.5, 0.0, 0.0, 1.0, 1.0, -1.5, 2.0}));
    SA_EXPECT_OK(add_role(batch, Data::K_ROLE_TYPE_MASK, Shape{K_BATCH}, {0.0, 0.0, 0.0, 0.0}));
    const std::vector<Train::StepOutput> outputs{Train::StepOutput{"output", &predicted}};

    Train::Metrics metrics;
    SA_EXPECT_OK(objective->measure(view_of(batch, outputs), metrics));

    ASSERT_NE(metrics.find("mean_absolute_error"), nullptr);
    EXPECT_EQ(metrics.find("mean_absolute_error")->value, 0.0);
}
