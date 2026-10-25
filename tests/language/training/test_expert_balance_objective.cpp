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
 * @file test_expert_balance_objective.cpp
 * @brief Tests the loss, the metrics, the refusals and the factory of ExpertBalanceObjective.
 * @author Mustafa Garip
 */

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
#include <SushiAI/ops/ops.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/shape/shape.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/data/dataset.hpp>
#include <SushiAI/engine_fixture/engine_test.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/graph/builder.hpp>
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

    /** @brief Bounds the difference between a float32 share and the metric read from it. */
    constexpr double K_TOLERANCE = 1e-7;

    /** @brief Returns the attributes of an objective that states no key. */
    Config::Value no_keys()
    {
        return Config::parse_json("{}", "<test>");
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

    /** @brief Counts the nodes of @p graph whose operation is @p op. */
    std::size_t count_nodes(const Graph::Graph& graph, OpID op)
    {
        std::size_t count = 0;
        for (const Graph::Node& node : graph.nodes())
        {
            if (node.op == op)
                ++count;
        }

        return count;
    }

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
} // namespace

/** @brief Tests that make() keeps the names and refuses none, no name and no expert. */
TEST(Small_LanguageTrainingExpertBalanceObjective, MakeKeepsItsOutputsAndRefusesNone)
{
    SA_ASSERT_OK_AND_ASSIGN(
        const std::unique_ptr<ExpertBalanceObjective> objective,
        ExpertBalanceObjective::make({"balance"}, {ExpertUsageOutput{"usage", K_EXPERTS}}));

    EXPECT_EQ(objective->balances(), (std::vector<std::string>{"balance"}));
    ASSERT_EQ(objective->usages().size(), 1u);
    EXPECT_EQ(objective->usages().front().experts, K_EXPERTS);
    EXPECT_EQ(objective->name(), "expert_balance");
    EXPECT_TRUE(objective->is_fed_by_batch());

    SA_EXPECT_FAILURE(ExpertBalanceObjective::make({}, {ExpertUsageOutput{"usage", 4}}),
                      StatusCode::InvalidDescription, "at least one balance output",
                      "0 and 1");
    SA_EXPECT_FAILURE(ExpertBalanceObjective::make({"balance"}, {}),
                      StatusCode::InvalidDescription, "1 and 0");
    SA_EXPECT_FAILURE(ExpertBalanceObjective::make({""}, {ExpertUsageOutput{"usage", 4}}),
                      StatusCode::InvalidDescription, "no name");
    SA_EXPECT_FAILURE(ExpertBalanceObjective::make({"balance"}, {ExpertUsageOutput{"usage", 0}}),
                      StatusCode::InvalidDescription, "'usage' of 0");
}

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
TEST(Small_LanguageTrainingExpertBalanceObjective, TheLossIsTheMeanOfTheBalanceOutputs)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueID first = builder.input(Shape{1, 1}, "first");
    const Graph::ValueID second = builder.input(Shape{1, 1}, "second");
    NN::ModelOutputs outputs;
    SA_EXPECT_OK(outputs.add("balance2", first));
    SA_EXPECT_OK(outputs.add("balance3", second));
    const Train::ObjectiveLeaves leaves;

    SA_ASSERT_OK_AND_ASSIGN(
        const std::unique_ptr<ExpertBalanceObjective> single,
        ExpertBalanceObjective::make({"balance3"}, {ExpertUsageOutput{"usage", K_EXPERTS}}));
    const Graph::ValueID one = single->loss(builder, graph, outputs, leaves);

    EXPECT_EQ(graph.value(one).shape, (Shape{1}));
    const Graph::Node& viewed = graph.node(graph.value(one).producer);
    ASSERT_EQ(viewed.op, Ops::K_RESHAPE.id);
    EXPECT_EQ(viewed.inputs[0], second);
    EXPECT_EQ(count_nodes(graph, Ops::K_SCALE.id), 0u);

    SA_ASSERT_OK_AND_ASSIGN(
        const std::unique_ptr<ExpertBalanceObjective> both,
        ExpertBalanceObjective::make({"balance2", "balance3"},
                                     {ExpertUsageOutput{"usage", K_EXPERTS}}));
    const Graph::ValueID mean = both->loss(builder, graph, outputs, leaves);

    EXPECT_EQ(graph.value(mean).shape, (Shape{1}));
    const Graph::Node& halved = graph.node(graph.value(mean).producer);
    ASSERT_EQ(halved.op, Ops::K_SCALE.id);
    EXPECT_EQ(halved.attributes.get(0), 0.5);
    EXPECT_EQ(graph.node(graph.value(halved.inputs[0]).producer).op, Ops::K_ADD.id);
}

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
TEST(Small_LanguageTrainingExpertBalanceObjective, TheFactoryRefusesWhatDoesNotFit)
{
    SA_EXPECT_FAILURE(
        make_expert_balance_objective(make_context(),
                                      Config::parse_json(R"({"weight": 2})", "<test>"),
                                      "objective"),
        StatusCode::InvalidDescription, "objective states the key 'weight'");

    Train::ObjectiveContext plain = make_context();
    plain.outputs.resize(1);
    SA_EXPECT_FAILURE(make_expert_balance_objective(plain, no_keys(), "objective"),
                      StatusCode::InvalidDescription, "begin with 'balance'", "'usage'",
                      "0 and 0", "among its 1");

    Train::ObjectiveContext wide = make_context();
    wide.outputs[1].shape = Shape{1, 2};
    SA_EXPECT_FAILURE(make_expert_balance_objective(wide, no_keys(), "objective"),
                      StatusCode::InvalidDescription, "'balance2'", "one element", "[1, 2]");
}

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
    EXPECT_NEAR(smallest->value, 0.0625, K_TOLERANCE);
    EXPECT_NEAR(largest->value, 0.5, K_TOLERANCE);
    EXPECT_TRUE(smallest->is_fraction);
    EXPECT_TRUE(largest->is_fraction);
}

/** @brief Tests that measure refuses a usage the step lacks and one of another size. */
TEST_F(Medium_LanguageTrainingExpertBalanceMetrics, MeasureRefusesAUsageItCannotRead)
{
    const SushiBLAS::Tensor even = make_usage({0.25, 0.25, 0.25, 0.25});
    const std::vector<Train::StepOutput> published = {Train::StepOutput{"usage2", &even}};
    const Data::Batch batch{};
    const Train::StepView step{
        batch, 0.0,
        SushiRuntime::span<const Train::StepOutput>(published.data(), published.size())};

    SA_ASSERT_OK_AND_ASSIGN(
        const std::unique_ptr<ExpertBalanceObjective> absent,
        ExpertBalanceObjective::make({"balance"}, {ExpertUsageOutput{"usage9", K_EXPERTS}}));
    Train::Metrics metrics;
    SA_EXPECT_FAILURE(absent->measure(step, metrics), StatusCode::InvalidDescription,
                      "'usage9'", "did not publish");

    SA_ASSERT_OK_AND_ASSIGN(
        const std::unique_ptr<ExpertBalanceObjective> longer,
        ExpertBalanceObjective::make({"balance"}, {ExpertUsageOutput{"usage2", 8}}));
    SA_EXPECT_FAILURE(longer->measure(step, metrics), StatusCode::Incompatible,
                      "the usage of the experts", "8 values", "holds 4");
    EXPECT_TRUE(metrics.all().empty());
}
