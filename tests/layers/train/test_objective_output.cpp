/**************************************************************************/
/* test_objective_output.cpp                                              */
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
 * @file test_objective_output.cpp
 * @brief Tests the output an objective reads and the row mask it applies: keys, leaves, feed, loss.
 * @author Mustafa Garip
 */

#include <algorithm>
#include <cmath>
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
#include <SushiAI/config/value.hpp>
#include <SushiAI/allocation/allocation.hpp>
#include <SushiAI/dtype/dtype.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/shape/shape.hpp>
#include <SushiAI/data/dataset.hpp>
#include <SushiAI/data/roles.hpp>
#include <SushiAI/data/target.hpp>
#include <SushiAI/execution/host_execution_backend.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/graph/builder.hpp>
#include <SushiAI/graph/host_evaluator.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/graph/tensor_pool.hpp>
#include <SushiAI/nn/model_outputs.hpp>
#include <SushiAI/tensor_values/tensor_values.hpp>
#include <SushiAI/train/binary_cross_entropy_objective.hpp>
#include <SushiAI/train/mean_squared_error_objective.hpp>
#include <SushiAI/train/multi_label_objective.hpp>
#include <SushiAI/train/objective.hpp>
#include <SushiAI/train/objective_context.hpp>
#include <SushiAI/train/objective_output.hpp>
#include <SushiAI/train/row_mask.hpp>
#include <SushiAI/train/step_view.hpp>
#include <support/tolerance.hpp>

using namespace SushiAI;
namespace SaTest = SushiAI::Test;

namespace
{
    /** @brief Holds the samples of the batch every case builds. */
    constexpr std::int64_t K_BATCH = 4;

    /** @brief Holds the features of a sample the factories are told of. */
    constexpr std::int64_t K_FEATURES = 6;

    /** @brief Holds the rows of the weights the mask cases feed. */
    constexpr std::int64_t K_MASK_WIDTH = 3;

    /** @brief Names the first port of the two-output models. */
    constexpr const char* K_FIRST = "first";

    /** @brief Names the second port of the two-output models. */
    constexpr const char* K_SECOND = "second";

    /** @brief Holds the type of an objective factory a training file reaches. */
    using Factory = Result<std::unique_ptr<Train::Objective>> (*)(
        const Train::ObjectiveContext&, const Config::Value&, std::string_view);

    /** @brief Names the closed form each kind of objective is checked against. */
    enum class Kind
    {
        CrossEntropy, /**< Minus the log of the softmax at the labelled class. */
        Logistic,     /**< The cross entropy of sigmoid(x) against 0 or 1, per element. */
        Squared       /**< The squared difference, per element. */
    };

    /** @brief Holds one model output: its name, its extents and its values. */
    struct Port
    {
        std::string name;           /**< The name the model publishes it under. */
        Shape shape;                /**< The extents. */
        std::vector<double> values; /**< The elements, row-major. */
    };

    /** @brief Holds one objective under test with the data its loss is checked on. */
    struct KindRow
    {
        std::string name;                /**< The row's name. */
        Factory make = nullptr;          /**< The factory of the objective. */
        Kind kind = Kind::Logistic;      /**< The closed form of its loss. */
        std::int64_t width = 0;          /**< The columns of its output. */
        std::int64_t classes = 0;        /**< The classes the data states, 0 for none. */
        std::vector<double> targets;     /**< The [N, width] values of its targets leaf. */
        std::size_t term_operations = 0; /**< The rounded operations of one term of its loss. */
    };

    /** @brief Returns the attributes @p text spells. */
    Config::Value keys(const char* text)
    {
        return SaTest::require_value(Config::parse_json(text, "<test>"));
    }

    /** @brief Returns element @p index of a pattern of 11 values from -1.25 to 1.25. */
    double pattern(std::size_t index, std::size_t stride)
    {
        return (static_cast<double>((stride * index) % 11) - 5.0) / 4.0;
    }

    /** @brief Returns the [K_BATCH, @p width] port @p name; @p stride picks another pattern. */
    Port make_port(const char* name, std::int64_t width, std::size_t stride)
    {
        Port port;
        port.name = name;
        port.shape = Shape{K_BATCH, width};
        for (std::int64_t index = 0; index < K_BATCH * width; ++index)
            port.values.push_back(pattern(static_cast<std::size_t>(index), stride));

        return port;
    }

    /** @brief Returns the two ports of a model that publishes K_FIRST and K_SECOND. */
    std::vector<Port> two_ports(const KindRow& row)
    {
        return {make_port(K_FIRST, row.width, 3), make_port(K_SECOND, row.width, 7)};
    }

    /** @brief Returns what a factory is given for a model that publishes @p ports. */
    Train::ObjectiveContext context_of(const KindRow& row, const std::vector<Port>& ports)
    {
        Train::ObjectiveContext given;
        given.batch = K_BATCH;
        given.classes = row.classes;
        given.input_sample = Shape{K_FEATURES};
        for (const Port& port : ports)
            given.outputs.push_back(Train::OutputShape{port.name, port.shape});

        return given;
    }

    /** @brief Returns the objective @p row builds over @p ports for the attributes @p text. */
    Result<std::unique_ptr<Train::Objective>> objective_of(const KindRow& row,
                                                           const std::vector<Port>& ports,
                                                           const char* text)
    {
        return row.make(context_of(row, ports), keys(text), "objective");
    }

    /** @brief Returns one weight per element: a kept row 1 / kept, a dropped row 0. */
    std::vector<double> weights_of(const std::vector<double>& mask, std::int64_t width)
    {
        const double kept = std::max(1.0, std::count(mask.begin(), mask.end(), 1.0) * 1.0);
        std::vector<double> weights;
        for (const double flag : mask)
        {
            for (std::int64_t column = 0; column < width; ++column)
                weights.push_back(flag == 1.0 ? 1.0 / kept : 0.0);
        }

        return weights;
    }

    /**
     * @brief Returns the loss of @p objective over a model that publishes @p ports.
     * @param targets The values of the targets leaf.
     * @param weights The values of the weights leaf; read only when the objective declares one.
     */
    Result<double> loss_of(const Train::Objective& objective, const std::vector<Port>& ports,
                           const std::vector<double>& targets, const std::vector<double>& weights)
    {
        Graph::Graph graph;
        const Graph::GraphBuilder builder(graph);
        Train::ObjectiveLeaves leaves;
        objective.declare(builder, leaves);

        NN::ModelOutputs outputs;
        std::vector<Graph::ValueID> ids;
        for (const Port& port : ports)
        {
            ids.push_back(builder.input(port.shape, port.name));
            SA_TRY(outputs.add(port.name, ids.back()));
        }

        SA_TRY_ASSIGN(const Train::LossTrace loss_trace, objective.loss(builder, graph, outputs, leaves));
        const Graph::ValueID loss = loss_trace.loss();

        Graph::HostEvaluator evaluator(graph);
        for (std::size_t index = 0; index < ports.size(); ++index)
            SA_TRY(evaluator.write(ids[index], ports[index].values));
        SA_TRY(evaluator.write(leaves.at("targets"), targets));
        if (leaves.has(Train::K_TYPE_MASK_WEIGHTS_LEAF))
            SA_TRY(evaluator.write(leaves.at(Train::K_TYPE_MASK_WEIGHTS_LEAF), weights));
        SA_TRY(evaluator.run());

        SA_TRY_ASSIGN(const std::vector<double> read, evaluator.read(loss));
        SA_FAIL_IF(read.size() != 1, StatusCode::InvalidDescription, "The loss holds {} elements",
                   read.size());

        return read.front();
    }

    /** @brief Returns the loss of one row of @p row's closed form. */
    double row_loss_of(const KindRow& row, const std::vector<double>& logits, std::int64_t at)
    {
        const std::size_t begin = static_cast<std::size_t>(at * row.width);
        double total = 0.0;

        if (row.kind == Kind::CrossEntropy)
        {
            double normaliser = 0.0;
            for (std::int64_t column = 0; column < row.width; ++column)
                normaliser += std::exp(logits[begin + static_cast<std::size_t>(column)]);
            for (std::int64_t column = 0; column < row.width; ++column)
            {
                const std::size_t index = begin + static_cast<std::size_t>(column);
                total -= row.targets[index] * (logits[index] - std::log(normaliser));
            }

            return total;
        }

        for (std::int64_t column = 0; column < row.width; ++column)
        {
            const std::size_t index = begin + static_cast<std::size_t>(column);
            const double x = logits[index];
            const double t = row.targets[index];
            if (row.kind == Kind::Logistic)
                total += std::max(x, 0.0) - x * t + std::log1p(std::exp(-std::abs(x)));
            else
                total += (x - t) * (x - t);
        }

        return total / static_cast<double>(row.width);
    }

    /** @brief Returns the mean of the row losses of the rows @p mask keeps; 0 for none. */
    double masked_mean_of(const KindRow& row, const std::vector<double>& logits,
                          const std::vector<double>& mask)
    {
        double total = 0.0;
        double kept = 0.0;
        for (std::int64_t at = 0; at < K_BATCH; ++at)
        {
            if (mask[static_cast<std::size_t>(at)] != 1.0)
                continue;
            total += row_loss_of(row, logits, at);
            kept += 1.0;
        }

        return kept == 0.0 ? 0.0 : total / kept;
    }

    /**
     * @brief Returns the rounded operations on the longest path to a masked loss, and as many
     *        again for the closed form it is compared with.
     *
     * One term takes row.term_operations; the product with its weight is one more, the sum of the
     * K_BATCH * width elements adds one fewer than their count, and the scale is the last.
     */
    std::size_t loss_operations(const KindRow& row)
    {
        const std::size_t elements = static_cast<std::size_t>(K_BATCH * row.width);
        return 2 * (row.term_operations + 1 + (elements - 1) + 1);
    }

    /** @brief Returns the four objectives the cases run over, each with data of its own. */
    std::vector<KindRow> kind_rows()
    {
        return {
            KindRow{"CrossEntropy", &Train::make_cross_entropy, Kind::CrossEntropy, 3, 3,
                    {0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0}, 3 + 4},
            KindRow{"Binary", &Train::make_binary_cross_entropy, Kind::Logistic, 1, 0,
                    {1, 0, 1, 0}, 3},
            KindRow{"MultiLabel", &Train::make_multi_label, Kind::Logistic, 3, 0,
                    {1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0}, 3},
            KindRow{"MeanSquaredError", &Train::make_mean_squared_error, Kind::Squared, 2, 0,
                    {0.5, -1.0, 2.0, 0.25, -0.5, 1.5, 1.0, 1.0}, 2}};
    }

    /** @brief Runs the cases that hold for each of the four objectives. */
    class Small_ObjectiveOutputKinds : public ::testing::TestWithParam<KindRow>
    {
    };

    /** @brief Provides the host engine the cases allocate their tensors and batches on. */
    class HostTensors : public ::testing::Test
    {
        protected:
            /** @brief Returns a float32 tensor of @p shape that holds @p values. */
            [[nodiscard]] Result<SushiBLAS::Tensor> tensor_of(const Shape& shape,
                                                              const std::vector<double>& values)
            {
                SA_TRY_ASSIGN(SushiBLAS::Tensor tensor, allocate_tensor(backend.engine(), shape));
                SaTest::write_tensor(tensor, values);
                return tensor;
            }

            /** @brief Returns a batch that holds @p values of @p shape under @p role. */
            [[nodiscard]] Result<Data::Batch> batch_of(std::string_view role, const Shape& shape,
                                                       const std::vector<double>& values)
            {
                Data::BatchTarget target;
                target.role = std::string(role);
                SA_TRY_ASSIGN(target.values, tensor_of(shape, values));

                Data::Batch batch;
                batch.targets.push_back(std::move(target));
                return batch;
            }

            /** @brief The host backend the tensors and the pool are allocated from. */
            HostExecutionBackend backend;
    };

    /** @brief Returns the role an objective of @p kind reads when no target is named. */
    std::string_view default_role_of(Kind kind)
    {
        return kind == Kind::Squared ? Data::K_ROLE_VALUES : Data::K_ROLE_LABELS;
    }

    /** @brief Returns the class of the largest value of each row of the [K_BATCH, @p width] values. */
    std::vector<double> classes_of(const std::vector<double>& one_hot, std::int64_t width)
    {
        std::vector<double> classes;
        for (std::int64_t row = 0; row < K_BATCH; ++row)
        {
            std::int64_t best = 0;
            for (std::int64_t column = 1; column < width; ++column)
            {
                if (one_hot[static_cast<std::size_t>(row * width + column)]
                    > one_hot[static_cast<std::size_t>(row * width + best)])
                    best = column;
            }

            classes.push_back(static_cast<double>(best));
        }

        return classes;
    }

    /** @brief Runs the cases of the mask's feed on tensors of the host engine. */
    class Small_RowMask : public HostTensors
    {
        protected:
            /**
             * @brief Declares the weights leaf of @p mask over [@p rows, K_MASK_WIDTH] and binds a
             *        tensor of 7s to it.
             * @return The id of the leaf.
             */
            [[nodiscard]] Result<Graph::ValueID> bind_weights(const Train::RowMask& mask,
                                                              std::int64_t rows,
                                                              Graph::Graph& graph,
                                                              Train::ObjectiveLeaves& leaves,
                                                              Graph::TensorPool& pool)
            {
                const Graph::GraphBuilder builder(graph);
                mask.declare(builder, leaves, Shape{rows, K_MASK_WIDTH}, DType::Float32);
                const Graph::ValueID id = leaves.at(Train::K_TYPE_MASK_WEIGHTS_LEAF);

                SA_TRY_ASSIGN(SushiBLAS::Tensor stale,
                              tensor_of(Shape{rows, K_MASK_WIDTH},
                                        std::vector<double>(
                                            static_cast<std::size_t>(rows * K_MASK_WIDTH), 7.0)));
                SA_TRY(pool.bind(id, stale));
                return id;
            }
    };

    /** @brief Runs the cases of the target role of each of the four objectives, on the host engine. */
    class Small_ObjectiveTarget : public HostTensors,
                                  public ::testing::WithParamInterface<KindRow>
    {
        protected:
            /**
             * @brief Returns a batch whose one role @p role holds the targets of @p row: the class
             *        of each sample as int32 for the cross entropy, the values as float32 else.
             */
            [[nodiscard]] Result<Data::Batch> batch_of_targets(std::string_view role,
                                                               const KindRow& row)
            {
                const bool is_class = row.kind == Kind::CrossEntropy;
                const Shape shape = is_class ? Shape{K_BATCH}
                                    : row.width == 1 && row.kind == Kind::Logistic
                                        ? Shape{K_BATCH}
                                        : Shape{K_BATCH, row.width};
                Data::BatchTarget target;
                target.role = std::string(role);
                SA_TRY_ASSIGN(target.values,
                              allocate_tensor(backend.engine(), shape,
                                              is_class ? DType::Int32 : DType::Float32));
                SaTest::write_tensor(target.values,
                                     is_class ? classes_of(row.targets, row.width) : row.targets);

                Data::Batch batch;
                batch.targets.push_back(std::move(target));
                return batch;
            }
    };

    /** @brief Runs the cases of the selector over the outputs of a step on the host engine. */
    class Small_ObjectiveOutputStep : public HostTensors
    {
    };

} // namespace

/** @brief Tests that the default selector reads the only output and refuses a model of two. */
TEST(Small_ObjectiveOutput, TheDefaultSelectorReadsTheOnlyOutput)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    NN::ModelOutputs one;
    const Graph::ValueID alone = builder.input(Shape{K_BATCH, 2}, K_FIRST);
    SA_EXPECT_OK(one.add(K_FIRST, alone));
    NN::ModelOutputs two;
    SA_EXPECT_OK(two.add(K_FIRST, alone));
    SA_EXPECT_OK(two.add(K_SECOND, builder.input(Shape{K_BATCH, 2}, K_SECOND)));

    const Train::ObjectiveOutput selector;

    EXPECT_FALSE(selector.is_named());
    SA_ASSERT_OK_AND_ASSIGN(const Graph::ValueID found, selector.resolve(one));
    EXPECT_EQ(found, alone);
    SA_EXPECT_FAILURE(selector.resolve(two), StatusCode::InvalidDescription,
                      "exactly one model output", K_FIRST, K_SECOND);
}

/** @brief Tests that a named selector reads its port among several and refuses an absent one. */
TEST(Small_ObjectiveOutput, ANamedSelectorReadsItsPortAmongSeveral)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    NN::ModelOutputs outputs;
    SA_EXPECT_OK(outputs.add(K_FIRST, builder.input(Shape{K_BATCH, 2}, K_FIRST)));
    const Graph::ValueID second = builder.input(Shape{K_BATCH, 2}, K_SECOND);
    SA_EXPECT_OK(outputs.add(K_SECOND, second));

    SA_ASSERT_OK_AND_ASSIGN(const Train::ObjectiveOutput chosen,
                            Train::ObjectiveOutput::named(K_SECOND));
    SA_ASSERT_OK_AND_ASSIGN(const Train::ObjectiveOutput absent,
                            Train::ObjectiveOutput::named("third"));

    EXPECT_TRUE(chosen.is_named());
    EXPECT_EQ(chosen.name(), K_SECOND);
    SA_ASSERT_OK_AND_ASSIGN(const Graph::ValueID found, chosen.resolve(outputs));
    EXPECT_EQ(found, second);
    SA_EXPECT_FAILURE(absent.resolve(outputs), StatusCode::InvalidDescription,
                      "no output named 'third'", K_FIRST, K_SECOND);
}

/** @brief Tests that a named selector picks its tensor among the outputs of a step. */
TEST_F(Small_ObjectiveOutputStep, ANamedSelectorReadsItsTensorAmongTheOutputsOfAStep)
{
    SA_ASSERT_OK_AND_ASSIGN(const SushiBLAS::Tensor first,
                            tensor_of(Shape{2, 1}, {1.0, 2.0}));
    SA_ASSERT_OK_AND_ASSIGN(const SushiBLAS::Tensor second,
                            tensor_of(Shape{2, 1}, {3.0, 4.0}));
    SA_ASSERT_OK_AND_ASSIGN(const Data::Batch batch, batch_of("labels", Shape{2}, {0.0, 1.0}));
    const std::vector<Train::StepOutput> outputs{Train::StepOutput{K_FIRST, &first},
                                                 Train::StepOutput{K_SECOND, &second}};
    const Train::StepView step{
        batch, 0.0, SushiRuntime::span<const Train::StepOutput>(outputs.data(), outputs.size())};

    SA_ASSERT_OK_AND_ASSIGN(const Train::ObjectiveOutput chosen,
                            Train::ObjectiveOutput::named(K_SECOND));
    SA_ASSERT_OK_AND_ASSIGN(const Train::ObjectiveOutput absent,
                            Train::ObjectiveOutput::named("third"));

    SA_ASSERT_OK_AND_ASSIGN(const SushiBLAS::Tensor* const found, chosen.resolve(step));
    EXPECT_EQ(found, &second);
    SA_EXPECT_FAILURE(absent.resolve(step), StatusCode::InvalidDescription, "2 outputs",
                      "'third'");
    SA_EXPECT_FAILURE(Train::ObjectiveOutput().resolve(step), StatusCode::InvalidDescription,
                      "one model output", "2");
}

/** @brief Tests that an empty name and a name that is no string are refused. */
TEST(Small_ObjectiveOutput, AnEmptyNameAndANumberAreRefused)
{
    SA_EXPECT_FAILURE(Train::ObjectiveOutput::named(""), StatusCode::InvalidDescription,
                      "at least 1 character");
    SA_EXPECT_FAILURE(Train::ObjectiveOutput::read(keys(R"({"output": 3})"), "objective"),
                      StatusCode::InvalidDescription, "objective", "'output'");
    SA_EXPECT_FAILURE(Train::ObjectiveOutput::read(keys(R"({"output": ""})"), "objective"),
                      StatusCode::InvalidDescription, "at least 1 character");
}

/** @brief Tests that the selection keys are read and any other key names the keys it knows. */
TEST(Small_ObjectiveOutput, TheSelectionKeysAreReadAndAStrayKeyIsRefused)
{
    SA_ASSERT_OK_AND_ASSIGN(
        const Train::ObjectiveSelection selection,
        Train::read_selection(keys(R"({"output": "second", "type_mask": true})"), "objective"));
    SA_ASSERT_OK_AND_ASSIGN(
        const Train::ObjectiveSelection plain,
        Train::read_selection(keys(R"({"threshold": 0.9})"), "objective", "threshold"));

    EXPECT_EQ(selection.output.name(), K_SECOND);
    EXPECT_TRUE(selection.mask.is_active());
    EXPECT_FALSE(plain.output.is_named());
    EXPECT_FALSE(plain.mask.is_active());
    EXPECT_TRUE(Train::is_selection_key("output"));
    EXPECT_TRUE(Train::is_selection_key("type_mask"));
    EXPECT_FALSE(Train::is_selection_key("threshold"));
    SA_EXPECT_FAILURE(Train::read_selection(keys(R"({"huber": 1})"), "objective"),
                      StatusCode::InvalidDescription, "objective", "'huber'", "'output'",
                      "'type_mask'");
    SA_EXPECT_FAILURE(
        Train::read_selection(keys(R"({"huber": 1})"), "objective", "threshold"),
        StatusCode::InvalidDescription, "'huber'", "'threshold'", "'output'", "'type_mask'");
}

/** @brief Tests the mask key: absent and false are off, true is [N], an object is [N, k]. */
TEST(Small_RowMaskKey, TheKeyReadsOffOneValueAndColumns)
{
    SA_ASSERT_OK_AND_ASSIGN(const Train::RowMask absent,
                            Train::RowMask::read(keys("{}"), "objective"));
    SA_ASSERT_OK_AND_ASSIGN(const Train::RowMask off,
                            Train::RowMask::read(keys(R"({"type_mask": false})"), "objective"));
    SA_ASSERT_OK_AND_ASSIGN(const Train::RowMask on,
                            Train::RowMask::read(keys(R"({"type_mask": true})"), "objective"));
    SA_ASSERT_OK_AND_ASSIGN(
        const Train::RowMask picked,
        Train::RowMask::read(keys(R"({"type_mask": {"columns": 4, "column": 2}})"), "objective"));
    SA_ASSERT_OK_AND_ASSIGN(
        const Train::RowMask first,
        Train::RowMask::read(keys(R"({"type_mask": {"columns": 4}})"), "objective"));

    EXPECT_FALSE(absent.is_active());
    EXPECT_FALSE(off.is_active());
    EXPECT_TRUE(on.is_active());
    EXPECT_EQ(on.columns(), 0);
    EXPECT_EQ(on.column(), 0);
    EXPECT_TRUE(picked.is_active());
    EXPECT_EQ(picked.columns(), 4);
    EXPECT_EQ(picked.column(), 2);
    EXPECT_EQ(first.column(), 0);
}

/** @brief Tests what the mask key refuses. */
TEST(Small_RowMaskKey, TheKeyRefusesWhatNoMaskCanBe)
{
    SA_EXPECT_FAILURE(Train::RowMask::read(keys(R"({"type_mask": "yes"})"), "objective"),
                      StatusCode::InvalidDescription, "objective", "'type_mask'", "boolean");
    SA_EXPECT_FAILURE(
        Train::RowMask::read(keys(R"({"type_mask": {"column": 1}})"), "objective"),
        StatusCode::InvalidDescription, "'columns'");
    SA_EXPECT_FAILURE(
        Train::RowMask::read(keys(R"({"type_mask": {"columns": 2, "column": 2}})"), "objective"),
        StatusCode::InvalidDescription, "no column 2");
    SA_EXPECT_FAILURE(
        Train::RowMask::read(keys(R"({"type_mask": {"columns": 0}})"), "objective"),
        StatusCode::InvalidDescription, "at least 1");
    SA_EXPECT_FAILURE(
        Train::RowMask::read(keys(R"({"type_mask": {"columns": 2, "rows": 1}})"), "objective"),
        StatusCode::InvalidDescription, "rows");
    SA_EXPECT_FAILURE(Train::RowMask::make(-1, 0), StatusCode::InvalidDescription, "got -1");
}

/** @brief Tests the target key: absent is the default role, a string names the role to read. */
TEST(Small_ObjectiveOutput, TheTargetKeyNamesTheRoleAnObjectiveReads)
{
    SA_ASSERT_OK_AND_ASSIGN(
        const Train::ObjectiveSelection named,
        Train::read_selection(keys(R"({"output": "second", "target": "choice"})"), "objective"));
    SA_ASSERT_OK_AND_ASSIGN(const Train::ObjectiveSelection plain,
                            Train::read_selection(keys("{}"), "objective"));
    SA_ASSERT_OK_AND_ASSIGN(const Train::ObjectiveTarget direct,
                            Train::ObjectiveTarget::named("number"));

    EXPECT_TRUE(named.target.is_named());
    EXPECT_EQ(named.target.role_or(Data::K_ROLE_LABELS), "choice");
    EXPECT_FALSE(plain.target.is_named());
    EXPECT_EQ(plain.target.role_or(Data::K_ROLE_LABELS), Data::K_ROLE_LABELS);
    EXPECT_EQ(direct.role_or(Data::K_ROLE_VALUES), "number");
    EXPECT_TRUE(Train::is_selection_key("target"));
}

/** @brief Tests that a target that is empty or no string is refused naming the key. */
TEST(Small_ObjectiveOutput, ATargetThatIsEmptyOrNoStringIsRefused)
{
    SA_EXPECT_FAILURE(Train::ObjectiveTarget::named(""), StatusCode::InvalidDescription,
                      "at least 1 character");
    SA_EXPECT_FAILURE(Train::ObjectiveTarget::read(keys(R"({"target": 3})"), "objective"),
                      StatusCode::InvalidDescription, "objective", "'target'", "batch role");
    SA_EXPECT_FAILURE(Train::ObjectiveTarget::read(keys(R"({"target": ""})"), "objective"),
                      StatusCode::InvalidDescription, "at least 1 character");
    SA_EXPECT_FAILURE(Train::read_selection(keys(R"({"huber": 1})"), "objective"),
                      StatusCode::InvalidDescription, "'huber'", "'target'");
}

/** @brief Tests that the factories refuse an output the model does not publish or a choice of none. */
TEST_P(Small_ObjectiveOutputKinds, TheFactoryRefusesAMissingOutputAndAnUnnamedChoice)
{
    const KindRow& row = GetParam();
    const std::vector<Port> ports = two_ports(row);

    SA_EXPECT_FAILURE(objective_of(row, ports, R"({"output": "third"})"),
                      StatusCode::InvalidDescription, "objective", "'third'", "2 outputs");

    // The cross-entropy factory reads no output shape of its own, so its refusal is the loss's.
    if (row.kind != Kind::CrossEntropy)
    {
        SA_EXPECT_FAILURE(objective_of(row, ports, "{}"), StatusCode::InvalidDescription,
                          "objective", "publishes 2", "'output'");
    }
}

/** @brief Tests that the loss of a named port equals the loss of a model that publishes only it. */
TEST_P(Small_ObjectiveOutputKinds, TheNamedPortGivesTheLossOfAModelPublishingOnlyIt)
{
    const KindRow& row = GetParam();
    const std::vector<Port> ports = two_ports(row);
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<Train::Objective> named,
                            objective_of(row, ports, R"({"output": "second"})"));
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<Train::Objective> alone,
                            objective_of(row, {ports[1]}, "{}"));
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<Train::Objective> other,
                            objective_of(row, ports, R"({"output": "first"})"));

    SA_ASSERT_OK_AND_ASSIGN(const double of_named, loss_of(*named, ports, row.targets, {}));
    SA_ASSERT_OK_AND_ASSIGN(const double of_alone,
                            loss_of(*alone, {ports[1]}, row.targets, {}));
    SA_ASSERT_OK_AND_ASSIGN(const double of_other, loss_of(*other, ports, row.targets, {}));

    // The two graphs run the same operations on the same numbers, so the results are one bit pattern.
    EXPECT_EQ(of_named, of_alone);
    EXPECT_NE(of_named, of_other);
}

/** @brief Tests that an objective with no new key declares no weights leaf and no mask role. */
TEST_P(Small_ObjectiveOutputKinds, WithoutTheKeysNoMaskLeafOrRoleIsDeclared)
{
    const KindRow& row = GetParam();
    const std::vector<Port> ports = {make_port(K_SECOND, row.width, 7)};
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<Train::Objective> plain,
                            objective_of(row, ports, "{}"));
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<Train::Objective> masked,
                            objective_of(row, ports, R"({"type_mask": true})"));
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    Train::ObjectiveLeaves without;
    Train::ObjectiveLeaves with;

    plain->declare(builder, without);
    masked->declare(builder, with);

    EXPECT_FALSE(without.has(Train::K_TYPE_MASK_WEIGHTS_LEAF));
    ASSERT_TRUE(with.has(Train::K_TYPE_MASK_WEIGHTS_LEAF));
    EXPECT_EQ(graph.value(with.at(Train::K_TYPE_MASK_WEIGHTS_LEAF)).shape,
              (Shape{K_BATCH, row.width}));
    ASSERT_EQ(with.required().size(), without.required().size() + 1);
    EXPECT_EQ(with.required().back().role, Data::K_ROLE_TYPE_MASK);
    EXPECT_EQ(with.required().back().sample_shape, Shape{});
    EXPECT_EQ(with.required().back().dtype, DType::Float32);
}

/** @brief Tests that a mask of k columns requires a role of the sample shape [k]. */
TEST_P(Small_ObjectiveOutputKinds, AMaskOfColumnsRequiresARoleOfThatSampleShape)
{
    const KindRow& row = GetParam();
    SA_ASSERT_OK_AND_ASSIGN(
        const std::unique_ptr<Train::Objective> masked,
        objective_of(row, {make_port(K_SECOND, row.width, 7)},
                     R"({"type_mask": {"columns": 5, "column": 3}})"));
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    Train::ObjectiveLeaves leaves;

    masked->declare(builder, leaves);

    EXPECT_EQ(leaves.required().back().role, Data::K_ROLE_TYPE_MASK);
    EXPECT_EQ(leaves.required().back().sample_shape, Shape{5});
}

/** @brief Tests that a mask turns the loss into the mean of the closed form over the kept rows. */
TEST_P(Small_ObjectiveOutputKinds, AMaskKeepsTheMeanOverTheKeptRows)
{
    const KindRow& row = GetParam();
    const std::vector<Port> ports = two_ports(row);
    const std::vector<double> mask{1.0, 0.0, 1.0, 1.0};
    SA_ASSERT_OK_AND_ASSIGN(
        const std::unique_ptr<Train::Objective> objective,
        objective_of(row, ports, R"({"output": "second", "type_mask": true})"));

    SA_ASSERT_OK_AND_ASSIGN(
        const double loss,
        loss_of(*objective, ports, row.targets, weights_of(mask, row.width)));

    EXPECT_TRUE(TestSupport::is_within(TestSupport::measure_relative_error<double>(
        loss, masked_mean_of(row, ports[1].values, mask), loss_operations(row))));
}

/** @brief Tests that a mask that keeps every row gives the loss of an objective with no mask. */
TEST_P(Small_ObjectiveOutputKinds, AMaskThatKeepsEveryRowGivesTheUnmaskedLoss)
{
    const KindRow& row = GetParam();
    const std::vector<Port> ports = {make_port(K_SECOND, row.width, 7)};
    const std::vector<double> mask{1.0, 1.0, 1.0, 1.0};
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<Train::Objective> plain,
                            objective_of(row, ports, "{}"));
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<Train::Objective> masked,
                            objective_of(row, ports, R"({"type_mask": true})"));

    SA_ASSERT_OK_AND_ASSIGN(const double unmasked, loss_of(*plain, ports, row.targets, {}));
    SA_ASSERT_OK_AND_ASSIGN(
        const double kept,
        loss_of(*masked, ports, row.targets, weights_of(mask, row.width)));

    // The weights are 1/4 per row, so the two sums are the same terms in the same order.
    EXPECT_TRUE(TestSupport::is_within(TestSupport::measure_relative_error<double>(
        kept, unmasked, loss_operations(row))));
}

/**
 * @brief Tests that a batch that drops every row has the loss 0 for any port values, which is
 *        also a gradient of 0.
 */
TEST_P(Small_ObjectiveOutputKinds, AMaskThatDropsEveryRowGivesLossZeroForAnyOutput)
{
    const KindRow& row = GetParam();
    std::vector<Port> ports = {make_port(K_SECOND, row.width, 7)};
    const std::vector<double> mask{0.0, 0.0, 0.0, 0.0};
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<Train::Objective> objective,
                            objective_of(row, ports, R"({"type_mask": true})"));
    const std::vector<double> weights = weights_of(mask, row.width);

    SA_ASSERT_OK_AND_ASSIGN(const double at_rest, loss_of(*objective, ports, row.targets, weights));
    for (double& value : ports[0].values)
        value += 0.5;
    SA_ASSERT_OK_AND_ASSIGN(const double shifted, loss_of(*objective, ports, row.targets, weights));
    for (double& value : ports[0].values)
        value *= 40.0;
    SA_ASSERT_OK_AND_ASSIGN(const double scaled, loss_of(*objective, ports, row.targets, weights));

    // The loss is a sum of terms each multiplied by 0, so it is 0 and no output moves it.
    EXPECT_EQ(at_rest, 0.0);
    EXPECT_EQ(shifted, 0.0);
    EXPECT_EQ(scaled, 0.0);
}

/** @brief Tests that the outputs of a dropped row do not move the loss and those of a kept row do. */
TEST_P(Small_ObjectiveOutputKinds, ADroppedRowDoesNotMoveTheLossAndAKeptRowDoes)
{
    const KindRow& row = GetParam();
    std::vector<Port> ports = {make_port(K_SECOND, row.width, 7)};
    const std::vector<double> mask{1.0, 0.0, 1.0, 1.0};
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<Train::Objective> objective,
                            objective_of(row, ports, R"({"type_mask": true})"));
    const std::vector<double> weights = weights_of(mask, row.width);
    SA_ASSERT_OK_AND_ASSIGN(const double before,
                            loss_of(*objective, ports, row.targets, weights));

    for (std::int64_t column = 0; column < row.width; ++column)
        ports[0].values[static_cast<std::size_t>(row.width + column)] +=
            3.0 * static_cast<double>(column + 1);
    SA_ASSERT_OK_AND_ASSIGN(const double dropped_moved,
                            loss_of(*objective, ports, row.targets, weights));
    for (std::int64_t column = 0; column < row.width; ++column)
        ports[0].values[static_cast<std::size_t>(column)] += 3.0 * static_cast<double>(column + 1);
    SA_ASSERT_OK_AND_ASSIGN(const double kept_moved,
                            loss_of(*objective, ports, row.targets, weights));

    EXPECT_EQ(dropped_moved, before);
    EXPECT_NE(kept_moved, before);
}

INSTANTIATE_TEST_SUITE_P(Kinds, Small_ObjectiveOutputKinds, ::testing::ValuesIn(kind_rows()),
                         [](const ::testing::TestParamInfo<KindRow>& info)
                         {
                             return info.param.name;
                         });

/** @brief Tests that an objective with no target requires the role it is written for. */
TEST_P(Small_ObjectiveTarget, WithoutATargetTheObjectiveRequiresItsOwnRole)
{
    const KindRow& row = GetParam();
    SA_ASSERT_OK_AND_ASSIGN(
        const std::unique_ptr<Train::Objective> objective,
        objective_of(row, {make_port(K_SECOND, row.width, 7)}, "{}"));
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    Train::ObjectiveLeaves leaves;

    objective->declare(builder, leaves);

    ASSERT_EQ(leaves.required().size(), 1u);
    EXPECT_EQ(leaves.required().front().role, default_role_of(row.kind));
}

/** @brief Tests that a named target is the role an objective requires, in place of its own. */
TEST_P(Small_ObjectiveTarget, ANamedTargetIsTheRoleTheObjectiveRequires)
{
    const KindRow& row = GetParam();
    SA_ASSERT_OK_AND_ASSIGN(
        const std::unique_ptr<Train::Objective> objective,
        objective_of(row, {make_port(K_SECOND, row.width, 7)}, R"({"target": "mine"})"));
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    Train::ObjectiveLeaves leaves;

    objective->declare(builder, leaves);

    ASSERT_EQ(leaves.required().size(), 1u);
    EXPECT_EQ(leaves.required().front().role, "mine");
}

/** @brief Tests that feed writes the targets leaf from the named role and ignores the default one. */
TEST_P(Small_ObjectiveTarget, FeedReadsTheNamedRoleAndNotTheDefaultOne)
{
    const KindRow& row = GetParam();
    SA_ASSERT_OK_AND_ASSIGN(
        const std::unique_ptr<Train::Objective> objective,
        objective_of(row, {make_port(K_SECOND, row.width, 7)}, R"({"target": "mine"})"));
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    Train::ObjectiveLeaves leaves;
    objective->declare(builder, leaves);
    Graph::TensorPool pool(backend.engine());
    const Graph::ValueID id = leaves.at("targets");
    SA_ASSERT_OK_AND_ASSIGN(
        SushiBLAS::Tensor stale,
        tensor_of(graph.value(id).shape,
                  std::vector<double>(
                      static_cast<std::size_t>(graph.value(id).shape.num_elements()), 7.0)));
    SA_EXPECT_OK(pool.bind(id, stale));
    SA_ASSERT_OK_AND_ASSIGN(const Data::Batch named, batch_of_targets("mine", row));
    SA_ASSERT_OK_AND_ASSIGN(const Data::Batch own,
                            batch_of_targets(default_role_of(row.kind), row));

    SA_EXPECT_FAILURE(objective->feed(own, leaves, pool), StatusCode::InvalidDescription,
                      "'mine'");
    SA_EXPECT_OK(objective->feed(named, leaves, pool));

    EXPECT_EQ(SaTest::read_tensor(pool.tensor(id)), row.targets);
}

INSTANTIATE_TEST_SUITE_P(Kinds, Small_ObjectiveTarget, ::testing::ValuesIn(kind_rows()),
                         [](const ::testing::TestParamInfo<KindRow>& info)
                         {
                             return info.param.name;
                         });

/** @brief Tests that a mask of one value per sample writes 1 over the kept count to a kept row. */
TEST_F(Small_RowMask, AKeptRowGetsOneOverTheKeptCountAndADroppedRowZero)
{
    SA_ASSERT_OK_AND_ASSIGN(const Train::RowMask mask, Train::RowMask::make(0, 0));
    Graph::Graph graph;
    Train::ObjectiveLeaves leaves;
    Graph::TensorPool pool(backend.engine());
    SA_ASSERT_OK_AND_ASSIGN(const Graph::ValueID weights,
                            bind_weights(mask, K_BATCH, graph, leaves, pool));
    SA_ASSERT_OK_AND_ASSIGN(
        const Data::Batch batch,
        batch_of(Data::K_ROLE_TYPE_MASK, Shape{K_BATCH}, {1.0, 0.0, 1.0, 1.0}));

    SA_EXPECT_OK(mask.feed(batch, leaves, pool));

    // 1/3 is written as float32, so the expected value is the float that 1/3 rounds to.
    const double third = static_cast<double>(static_cast<float>(1.0 / 3.0));
    std::vector<double> expected;
    for (const double weight : {third, 0.0, third, third})
        expected.insert(expected.end(), static_cast<std::size_t>(K_MASK_WIDTH), weight);
    EXPECT_EQ(SaTest::read_tensor(pool.tensor(weights)), expected);
}

/** @brief Tests that a mask of k columns reads the column it was given. */
TEST_F(Small_RowMask, AMaskOfColumnsReadsItsColumn)
{
    SA_ASSERT_OK_AND_ASSIGN(const Train::RowMask mask, Train::RowMask::make(2, 1));
    Graph::Graph graph;
    Train::ObjectiveLeaves leaves;
    Graph::TensorPool pool(backend.engine());
    SA_ASSERT_OK_AND_ASSIGN(const Graph::ValueID weights,
                            bind_weights(mask, 2, graph, leaves, pool));

    // Row 0 holds (1, 0) and row 1 holds (0, 1): column 1 drops row 0 and keeps row 1.
    SA_ASSERT_OK_AND_ASSIGN(
        const Data::Batch batch,
        batch_of(Data::K_ROLE_TYPE_MASK, Shape{2, 2}, {1.0, 0.0, 0.0, 1.0}));

    SA_EXPECT_OK(mask.feed(batch, leaves, pool));

    EXPECT_EQ(SaTest::read_tensor(pool.tensor(weights)),
              (std::vector<double>{0.0, 0.0, 0.0, 1.0, 1.0, 1.0}));
}

/** @brief Tests that a batch that drops every row writes weights of 0, not a division by 0. */
TEST_F(Small_RowMask, ABatchThatDropsEveryRowWritesZeroWeights)
{
    SA_ASSERT_OK_AND_ASSIGN(const Train::RowMask mask, Train::RowMask::make(0, 0));
    Graph::Graph graph;
    Train::ObjectiveLeaves leaves;
    Graph::TensorPool pool(backend.engine());
    SA_ASSERT_OK_AND_ASSIGN(const Graph::ValueID weights,
                            bind_weights(mask, K_BATCH, graph, leaves, pool));
    SA_ASSERT_OK_AND_ASSIGN(
        const Data::Batch batch,
        batch_of(Data::K_ROLE_TYPE_MASK, Shape{K_BATCH}, {0.0, 0.0, 0.0, 0.0}));

    SA_EXPECT_OK(mask.feed(batch, leaves, pool));

    EXPECT_EQ(SaTest::read_tensor(pool.tensor(weights)),
              std::vector<double>(static_cast<std::size_t>(K_BATCH * K_MASK_WIDTH), 0.0));
}

/** @brief Tests that a mask value that is not 0 or 1 is refused and the leaf keeps its values. */
TEST_F(Small_RowMask, AMaskValueThatIsNotZeroOrOneIsRefused)
{
    SA_ASSERT_OK_AND_ASSIGN(const Train::RowMask mask, Train::RowMask::make(0, 0));
    Graph::Graph graph;
    Train::ObjectiveLeaves leaves;
    Graph::TensorPool pool(backend.engine());
    SA_ASSERT_OK_AND_ASSIGN(const Graph::ValueID weights,
                            bind_weights(mask, K_BATCH, graph, leaves, pool));
    SA_ASSERT_OK_AND_ASSIGN(
        const Data::Batch batch,
        batch_of(Data::K_ROLE_TYPE_MASK, Shape{K_BATCH}, {1.0, 0.5, 1.0, 0.0}));

    SA_EXPECT_FAILURE(mask.feed(batch, leaves, pool), StatusCode::Incompatible, "row 1",
                      "0 or 1");
    EXPECT_EQ(SaTest::read_tensor(pool.tensor(weights)),
              std::vector<double>(static_cast<std::size_t>(K_BATCH * K_MASK_WIDTH), 7.0));
}

/** @brief Tests that a batch with no mask role, or a mask of another row count, is refused. */
TEST_F(Small_RowMask, ABatchWithoutTheRoleOrWithAnotherRowCountIsRefused)
{
    SA_ASSERT_OK_AND_ASSIGN(const Train::RowMask mask, Train::RowMask::make(0, 0));
    Graph::Graph graph;
    Train::ObjectiveLeaves leaves;
    Graph::TensorPool pool(backend.engine());
    SA_ASSERT_OK_AND_ASSIGN(const Graph::ValueID weights,
                            bind_weights(mask, K_BATCH, graph, leaves, pool));
    SA_ASSERT_OK_AND_ASSIGN(
        const Data::Batch other,
        batch_of(Data::K_ROLE_LABELS, Shape{K_BATCH}, {1.0, 1.0, 1.0, 1.0}));
    SA_ASSERT_OK_AND_ASSIGN(const Data::Batch five,
                            batch_of(Data::K_ROLE_TYPE_MASK, Shape{5},
                                     {1.0, 1.0, 1.0, 1.0, 1.0}));

    SA_EXPECT_FAILURE(mask.feed(other, leaves, pool), StatusCode::InvalidDescription,
                      "'type_mask'");
    SA_EXPECT_FAILURE(mask.feed(five, leaves, pool), StatusCode::Incompatible, "5 rows",
                      "holds 12");
    EXPECT_EQ(SaTest::read_tensor(pool.tensor(weights)),
              std::vector<double>(static_cast<std::size_t>(K_BATCH * K_MASK_WIDTH), 7.0));
}

/** @brief Tests that an objective's feed writes its targets and then the weights of its mask. */
TEST_F(Small_RowMask, AnObjectiveFeedWritesTheTargetsAndTheWeights)
{
    SA_ASSERT_OK_AND_ASSIGN(
        const Train::ObjectiveSelection selection,
        Train::read_selection(keys(R"({"type_mask": true})"), "objective"));
    SA_ASSERT_OK_AND_ASSIGN(
        const std::unique_ptr<Train::BinaryCrossEntropyObjective> objective,
        Train::BinaryCrossEntropyObjective::make(K_BATCH, Train::K_DEFAULT_THRESHOLD,
                                                 DType::Float32, selection));
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    Train::ObjectiveLeaves leaves;
    objective->declare(builder, leaves);
    Graph::TensorPool pool(backend.engine());
    for (const std::string_view leaf : {std::string_view("targets"),
                                        Train::K_TYPE_MASK_WEIGHTS_LEAF})
    {
        const Graph::ValueID id = leaves.at(leaf);
        SA_ASSERT_OK_AND_ASSIGN(
            SushiBLAS::Tensor stale,
            tensor_of(graph.value(id).shape,
                      std::vector<double>(
                          static_cast<std::size_t>(graph.value(id).shape.num_elements()), 7.0)));
        SA_EXPECT_OK(pool.bind(id, stale));
    }

    SA_ASSERT_OK_AND_ASSIGN(
        Data::Batch batch,
        batch_of(Data::K_ROLE_LABELS, Shape{K_BATCH}, {1.0, 0.0, 0.0, 1.0}));
    Data::BatchTarget mask;
    mask.role = std::string(Data::K_ROLE_TYPE_MASK);
    SA_ASSERT_OK_AND_ASSIGN(mask.values, tensor_of(Shape{K_BATCH}, {0.0, 1.0, 1.0, 0.0}));
    batch.targets.push_back(std::move(mask));

    SA_EXPECT_OK(objective->feed(batch, leaves, pool));

    EXPECT_EQ(SaTest::read_tensor(pool.tensor(leaves.at("targets"))),
              (std::vector<double>{1.0, 0.0, 0.0, 1.0}));
    EXPECT_EQ(SaTest::read_tensor(pool.tensor(leaves.at(Train::K_TYPE_MASK_WEIGHTS_LEAF))),
              (std::vector<double>{0.0, 0.5, 0.5, 0.0}));
}
