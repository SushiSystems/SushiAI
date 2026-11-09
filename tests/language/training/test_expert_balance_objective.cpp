/**************************************************************************/
/* test_expert_balance_objective.cpp                                      */
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
 * @file test_expert_balance_objective.cpp
 * @brief Tests the loss, the metrics, the refusals and the factory of ExpertBalanceObjective.
 * @author Mustafa Garip
 */

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <gtest/gtest.h>
#include <SushiBLAS/tensor/tensor.hpp>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/config/value.hpp>
#include <SushiAI/allocation/allocation.hpp>
#include <SushiAI/dtype/dtype.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/shape/shape.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/data/dataset.hpp>
#include <SushiAI/engine_fixture/engine_test.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/graph/builder.hpp>
#include <SushiAI/graph/host_evaluator.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/language/training/expert_balance_objective.hpp>
#include <SushiAI/nn/model_outputs.hpp>
#include <SushiAI/tensor_values/tensor_values.hpp>
#include <SushiAI/train/metrics.hpp>
#include <SushiAI/train/objective.hpp>
#include <SushiAI/train/objective_context.hpp>
#include <SushiAI/train/step_view.hpp>

using namespace SushiAI;
using Language::Training::ExpertBalanceObjective;
using Language::Training::ExpertUsageOutput;
using Language::Training::make_expert_balance_objective;
// SaTest stays reachable inside a TEST_F body; see tests/language/training/README.md.
namespace SaTest = SushiAI::Test;

namespace
{
    /** @brief Holds the experts of each block the cases describe. */
    constexpr std::int64_t K_EXPERTS = 4;

    /** @brief Holds the substrings a refusal names; an empty one is satisfied by any message. */
    using Needles = std::array<std::string, 4>;

    /** @brief Returns the attributes of an objective that states no key. */
    Config::Value no_keys()
    {
        return SaTest::require_value(Config::parse_json("{}", "<test>"));
    }

    /** @brief Returns what a factory is given for a decoder of two blocks with experts. */
    Train::ObjectiveContext make_context()
    {
        Train::ObjectiveContext given;
        given.batch = 2;
        given.input_sample = Shape{3};
        given.outputs.push_back(Train::OutputShape{"output", Shape{2, 3, 5}});
        given.outputs.push_back(Train::OutputShape{"balance2", Shape{1, 1}});
        given.outputs.push_back(Train::OutputShape{"usage2", Shape{1, K_EXPERTS}});
        given.outputs.push_back(Train::OutputShape{"balance3", Shape{1, 1}});
        given.outputs.push_back(Train::OutputShape{"usage3", Shape{1, K_EXPERTS}});
        return given;
    }

    /** @brief Returns the context of a model that publishes only its logits. */
    Train::ObjectiveContext logits_only_context()
    {
        Train::ObjectiveContext plain = make_context();
        plain.outputs.resize(1);
        return plain;
    }

    /** @brief Returns the context whose "balance2" output holds two elements. */
    Train::ObjectiveContext wide_balance_context()
    {
        Train::ObjectiveContext wide = make_context();
        wide.outputs[1].shape = Shape{1, 2};
        return wide;
    }

    /** @brief Names the arguments make() refuses and what its message names. */
    struct RefusedMake
    {
        std::vector<std::string> balances;     /**< The balance outputs given. */
        std::vector<ExpertUsageOutput> usages; /**< The usage outputs given. */
        Needles needles;                       /**< What the message names. */
    };

    /** @brief Names the balance outputs of a loss and the value it takes over fixed inputs. */
    struct LossCase
    {
        std::vector<std::string> balances; /**< The balance outputs the objective averages. */
        double loss;                       /**< Their mean when "balance2" is 2 and "balance3" is 4. */
    };

    /** @brief Names the context and the attributes the factory refuses and what it names. */
    struct RefusedFactory
    {
        Train::ObjectiveContext given; /**< What the factory is given. */
        std::string attributes;        /**< The objective's own keys, as JSON. */
        Needles needles;               /**< What the message names. */
    };

    /** @brief Returns the descriptions the factory refuses. */
    std::vector<RefusedFactory> refused_factories()
    {
        return {{make_context(), R"({"weight": 2})", {"objective states the key 'weight'", "", "", ""}},
                {logits_only_context(), "{}",
                 {"begin with 'balance'", "'usage'", "0 and 0", "among its 1"}},
                {wide_balance_context(), "{}", {"'balance2'", "one element", "[1, 2]", ""}}};
    }

    /** @brief Names a usage the objective reads from a step that publishes "usage2" alone. */
    struct RefusedUsage
    {
        std::string name;     /**< The usage output the objective reads. */
        std::int64_t experts; /**< The experts it expects in that output. */
        StatusCode code;      /**< The refusal. */
        Needles needles;      /**< What the message names. */
    };

    /** @brief Provides the arguments make() refuses. */
    class Small_LanguageTrainingExpertBalanceMake : public ::testing::TestWithParam<RefusedMake>
    {

    };

    /** @brief Provides the balance outputs of the loss. */
    class Small_LanguageTrainingExpertBalanceLoss : public ::testing::TestWithParam<LossCase>
    {

    };

    /** @brief Provides the descriptions the factory refuses. */
    class Small_LanguageTrainingExpertBalanceFactory
        : public ::testing::TestWithParam<RefusedFactory>
    {

    };

    /** @brief Provides the engine the usage tensors of a step are allocated on. */
    class Medium_LanguageTrainingExpertBalanceMetrics : public SaTest::EngineTest
    {
        protected:
            /** @brief Returns a FLOAT32 [1, K_EXPERTS] tensor holding @p shares. */
            SushiBLAS::Tensor make_usage(const std::vector<double>& shares)
            {
                Result<SushiBLAS::Tensor> made =
                    allocate_tensor(*engine, Shape{1, K_EXPERTS}, DType::Float32);
                EXPECT_TRUE(made.status().is_ok());
                SaTest::write_tensor(made.value(), shares);

                return std::move(made).value();
            }
    };

    /** @brief Provides the usages measure() cannot read. */
    class Medium_LanguageTrainingExpertBalanceUsage
        : public Medium_LanguageTrainingExpertBalanceMetrics,
          public ::testing::WithParamInterface<RefusedUsage>
    {

    };
} // namespace

/** @brief Tests that make() refuses no balance, no usage, no name and a usage of no expert. */
TEST_P(Small_LanguageTrainingExpertBalanceMake, MakeRefusesWhatCannotBeBalanced)
{
    const RefusedMake& row = GetParam();

    SA_EXPECT_FAILURE(ExpertBalanceObjective::make(row.balances, row.usages),
                      StatusCode::InvalidDescription, row.needles[0], row.needles[1]);
}

INSTANTIATE_TEST_SUITE_P(
    Rows, Small_LanguageTrainingExpertBalanceMake,
    ::testing::Values(
        RefusedMake{{}, {ExpertUsageOutput{"usage", 4}},
                    {"at least one balance output", "0 and 1", "", ""}},
        RefusedMake{{"balance"}, {}, {"1 and 0", "", "", ""}},
        RefusedMake{{""}, {ExpertUsageOutput{"usage", 4}}, {"no name", "", "", ""}},
        RefusedMake{{"balance"}, {ExpertUsageOutput{"usage", 0}}, {"'usage' of 0", "", "", ""}}));

/** @brief Tests that the objective declares no leaf, requires no role and feeds nothing. */
TEST(Small_LanguageTrainingExpertBalanceObjective, ItDeclaresNoLeafAndReadsNoRole)
{
    SA_ASSERT_OK_AND_ASSIGN(
        const std::unique_ptr<ExpertBalanceObjective> objective,
        ExpertBalanceObjective::make({"balance"}, {ExpertUsageOutput{"usage", K_EXPERTS}}));
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    Train::ObjectiveLeaves leaves;

    objective->declare(builder, leaves);

    EXPECT_TRUE(leaves.all().empty());
    EXPECT_TRUE(leaves.required().empty());
    EXPECT_EQ(graph.node_count(), 0u);
}

/** @brief Tests that the loss of one balance is that output, and of two is half their sum. */
TEST_P(Small_LanguageTrainingExpertBalanceLoss, TheLossIsTheMeanOfTheBalanceOutputs)
{
    const LossCase& row = GetParam();
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueID first = builder.input(Shape{1, 1}, "first");
    const Graph::ValueID second = builder.input(Shape{1, 1}, "second");
    NN::ModelOutputs outputs;
    SA_EXPECT_OK(outputs.add("balance2", first));
    SA_EXPECT_OK(outputs.add("balance3", second));
    const Train::ObjectiveLeaves leaves;
    SA_ASSERT_OK_AND_ASSIGN(
        const std::unique_ptr<ExpertBalanceObjective> objective,
        ExpertBalanceObjective::make(row.balances, {ExpertUsageOutput{"usage", K_EXPERTS}}));
    SA_ASSERT_OK_AND_ASSIGN(const Graph::ValueID loss,
                            objective->loss(builder, graph, outputs, leaves));

    Graph::HostEvaluator evaluator(graph);
    SA_EXPECT_OK(evaluator.write(first, {2.0}));
    SA_EXPECT_OK(evaluator.write(second, {4.0}));
    SA_EXPECT_OK(evaluator.run());

    EXPECT_EQ(graph.value(loss).shape, (Shape{1}));
    SA_ASSERT_OK_AND_ASSIGN(const std::vector<double> value, evaluator.read(loss));
    EXPECT_EQ(value, (std::vector<double>{row.loss}));
}

INSTANTIATE_TEST_SUITE_P(Rows, Small_LanguageTrainingExpertBalanceLoss,
                         ::testing::Values(LossCase{{"balance3"}, 4.0},
                                           LossCase{{"balance2", "balance3"}, 3.0}));

/** @brief Tests that the factory pairs the outputs by the start of their names. */
TEST(Small_LanguageTrainingExpertBalanceObjective, TheFactoryReadsTheOutputsByTheirNames)
{
    SA_ASSERT_OK_AND_ASSIGN(
        const std::unique_ptr<Train::Objective> objective,
        make_expert_balance_objective(make_context(), no_keys(), "objective"));

    const auto* made = dynamic_cast<const ExpertBalanceObjective*>(objective.get());
    ASSERT_NE(made, nullptr);
    EXPECT_EQ(made->balances(), (std::vector<std::string>{"balance2", "balance3"}));
    ASSERT_EQ(made->usages().size(), 2u);
    EXPECT_EQ(made->usages()[0].name, "usage2");
    EXPECT_EQ(made->usages()[1].name, "usage3");
    EXPECT_EQ(made->usages()[1].experts, K_EXPERTS);
}

/** @brief Tests each description the factory refuses. */
TEST_P(Small_LanguageTrainingExpertBalanceFactory, TheFactoryRefusesWhatDoesNotFit)
{
    const RefusedFactory& row = GetParam();

    SA_EXPECT_FAILURE(
        make_expert_balance_objective(
            row.given, SaTest::require_value(Config::parse_json(row.attributes, "<test>")),
            "objective"),
        StatusCode::InvalidDescription, row.needles[0], row.needles[1], row.needles[2],
        row.needles[3]);
}

INSTANTIATE_TEST_SUITE_P(Rows, Small_LanguageTrainingExpertBalanceFactory,
                         ::testing::ValuesIn(refused_factories()));

/** @brief Tests that measure reports the extremes over every usage output of the step. */
TEST_F(Medium_LanguageTrainingExpertBalanceMetrics, MeasureReportsTheSmallestAndLargestShare)
{
    SA_ASSERT_OK_AND_ASSIGN(
        const std::unique_ptr<ExpertBalanceObjective> objective,
        ExpertBalanceObjective::make({"balance"}, {ExpertUsageOutput{"usage2", K_EXPERTS},
                                                   ExpertUsageOutput{"usage3", K_EXPERTS}}));
    const SushiBLAS::Tensor even = make_usage({0.25, 0.25, 0.25, 0.25});
    const SushiBLAS::Tensor skewed = make_usage({0.125, 0.5, 0.0625, 0.3125});
    const std::vector<Train::StepOutput> published = {
        Train::StepOutput{"usage2", &even}, Train::StepOutput{"usage3", &skewed}};
    const Data::Batch batch{};
    const Train::StepView step{
        batch, 0.0,
        SushiRuntime::span<const Train::StepOutput>(published.data(), published.size())};

    Train::Metrics metrics;
    SA_EXPECT_OK(objective->measure(step, metrics));

    ASSERT_EQ(metrics.all().size(), 2u);
    const Train::Metric* const smallest = metrics.find("expert_usage_min");
    const Train::Metric* const largest = metrics.find("expert_usage_max");
    ASSERT_NE(smallest, nullptr);
    ASSERT_NE(largest, nullptr);
    // Every share is a sum of powers of two, so float32 holds it exactly and the metric is exact.
    EXPECT_EQ(smallest->value, 0.0625);
    EXPECT_EQ(largest->value, 0.5);
    EXPECT_TRUE(smallest->is_fraction);
    EXPECT_TRUE(largest->is_fraction);
}

/** @brief Tests that measure refuses a usage the step lacks and one of another size. */
TEST_P(Medium_LanguageTrainingExpertBalanceUsage, MeasureRefusesAUsageItCannotRead)
{
    const RefusedUsage& row = GetParam();
    const SushiBLAS::Tensor even = make_usage({0.25, 0.25, 0.25, 0.25});
    const std::vector<Train::StepOutput> published = {Train::StepOutput{"usage2", &even}};
    const Data::Batch batch{};
    const Train::StepView step{
        batch, 0.0,
        SushiRuntime::span<const Train::StepOutput>(published.data(), published.size())};
    SA_ASSERT_OK_AND_ASSIGN(
        const std::unique_ptr<ExpertBalanceObjective> objective,
        ExpertBalanceObjective::make({"balance"}, {ExpertUsageOutput{row.name, row.experts}}));
    Train::Metrics metrics;

    SA_EXPECT_FAILURE(objective->measure(step, metrics), row.code, row.needles[0],
                      row.needles[1], row.needles[2]);

    EXPECT_TRUE(metrics.all().empty());
}

INSTANTIATE_TEST_SUITE_P(
    Rows, Medium_LanguageTrainingExpertBalanceUsage,
    ::testing::Values(
        RefusedUsage{"usage9", K_EXPERTS, StatusCode::InvalidDescription,
                     {"'usage9'", "did not publish", "", ""}},
        RefusedUsage{"usage2", 8, StatusCode::Incompatible,
                     {"the usage of the experts", "8 values", "holds 4", ""}}));
