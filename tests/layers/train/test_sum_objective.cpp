/**************************************************************************/
/* test_sum_objective.cpp                                                 */
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
 * @file test_sum_objective.cpp
 * @brief Tests the composite objective: its terms, its prefixed leaves, its loss and its factory.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>
#include <gtest/gtest.h>
#include <SushiAI/config/value.hpp>
#include <SushiAI/core/common.hpp>
#include <SushiAI/core/ops.hpp>
#include <SushiAI/core/result.hpp>
#include <SushiAI/core/shape.hpp>
#include <SushiAI/core/status.hpp>
#include <SushiAI/data/dataset.hpp>
#include <SushiAI/data/roles.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/graph/builder.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/nn/layers.hpp>
#include <SushiAI/nn/model_outputs.hpp>
#include <SushiAI/train/metrics.hpp>
#include <SushiAI/train/objective.hpp>
#include <SushiAI/train/objective_context.hpp>
#include <SushiAI/train/objective_registry.hpp>
#include <SushiAI/train/step_view.hpp>
#include <SushiAI/train/steps.hpp>
#include <SushiAI/train/sum_objective.hpp>
#include <SushiAI/train/training_graph.hpp>

using namespace SushiAI;
using Train::CrossEntropyObjective;
using Train::ObjectiveLeaves;
using Train::SumObjective;
using Train::SumTerm;

namespace
{
    /** @brief Holds the samples of a batch. */
    constexpr std::int64_t K_BATCH = 4;

    /** @brief Holds the features one sample enters the model with. */
    constexpr std::int64_t K_FEATURES = 6;

    /** @brief Holds the classes the model scores. */
    constexpr std::int64_t K_CLASSES = 3;

    /** @brief Names the objective block in a diagnostic. */
    constexpr std::string_view K_CONTEXT = "the objective block";

    /** @brief An objective that keeps the feed of the base, as one fed by its own step does. */
    class StepFedObjective final : public Train::Objective
    {
        public:
            /** @brief Declares no leaf. */
            void declare(const Graph::GraphBuilder&, ObjectiveLeaves&) const override
            {

            }

            /** @brief Returns the absent value; the tests never build a graph with it. */
            Graph::ValueID loss(const Graph::GraphBuilder&, const Graph::Graph&,
                                const NN::ModelOutputs&, const ObjectiveLeaves&) const override
            {
                return Graph::K_NO_VALUE;
            }

            /** @brief Returns "step_fed". */
            std::string_view name() const noexcept override { return "step_fed"; }
    };

    /** @brief An objective that reports the two metrics it was made with and nothing else. */
    class ReportingObjective final : public Train::Objective
    {
        public:
            /** @brief Stores the metrics measure() appends, in order. */
            ReportingObjective(Train::Metric first, Train::Metric second) noexcept
                : first_(first), second_(second)
            {

            }

            /** @brief Declares no leaf. */
            void declare(const Graph::GraphBuilder&, ObjectiveLeaves&) const override
            {

            }

            /** @brief Returns the absent value; the tests never build a graph with it. */
            Graph::ValueID loss(const Graph::GraphBuilder&, const Graph::Graph&,
                                const NN::ModelOutputs&, const ObjectiveLeaves&) const override
            {
                return Graph::K_NO_VALUE;
            }

            /** @brief Returns true, so a sum takes the objective. */
            [[nodiscard]] bool is_fed_by_batch() const noexcept override { return true; }

            /** @brief Appends the two metrics. */
            [[nodiscard]] Status measure(const Train::StepView&,
                                         Train::Metrics& metrics) const override
            {
                SA_TRY(metrics.add(first_));

                return metrics.add(second_);
            }

            /** @brief Returns "reporting". */
            std::string_view name() const noexcept override { return "reporting"; }

        private:
            /// @brief The first metric measure() appends.
            Train::Metric first_;

            /// @brief The second metric measure() appends.
            Train::Metric second_;
    };

    /** @brief Returns a cross-entropy term of @p weight over K_BATCH rows of K_CLASSES. */
    SumTerm cross_entropy_term(double weight)
    {
        return SumTerm{std::make_unique<CrossEntropyObjective>(K_BATCH, K_CLASSES), weight};
    }

    /** @brief Returns the sum of two cross-entropy terms of the weights given. */
    Result<std::unique_ptr<SumObjective>> make_pair(double first, double second)
    {
        std::vector<SumTerm> terms;
        terms.push_back(cross_entropy_term(first));
        terms.push_back(cross_entropy_term(second));

        return SumObjective::make(std::move(terms));
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

    /** @brief Returns what a factory is given for a [K_BATCH, K_CLASSES] classifier. */
    Train::ObjectiveContext make_context()
    {
        Train::ObjectiveContext given;
        given.batch = K_BATCH;
        given.classes = K_CLASSES;
        given.input_sample = Shape{K_FEATURES};
        given.outputs.push_back(Train::OutputShape{"output", Shape{K_BATCH, K_CLASSES}});

        return given;
    }

    /** @brief Returns the registry of the layers tier, which holds the sum row. */
    Train::ObjectiveRegistry make_registry()
    {
        Train::ObjectiveRegistry registry;
        EXPECT_TRUE(Train::register_objectives(registry).is_ok());

        return registry;
    }

    /** @brief Builds the objective @p text describes through the "sum" row of the registry. */
    Result<std::unique_ptr<Train::Objective>> make_from(const Train::ObjectiveRegistry& registry,
                                                        std::string_view text)
    {
        return Train::make_objective(registry, "sum", make_context(),
                                     Config::parse_json(text, "<test>"), K_CONTEXT);
    }
} // namespace

/** @brief Tests that make() keeps its terms in order and names itself "sum". */
TEST(Small_SumObjective, MakeKeepsItsTermsInOrder)
{
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<SumObjective> objective, make_pair(1.0, 0.5));

    ASSERT_EQ(objective->terms().size(), 2u);
    EXPECT_EQ(objective->terms()[0].weight, 1.0);
    EXPECT_EQ(objective->terms()[1].weight, 0.5);
    EXPECT_EQ(objective->terms()[1].objective->name(), "cross_entropy");
    EXPECT_EQ(objective->name(), "sum");
    EXPECT_TRUE(objective->is_fed_by_batch());
    EXPECT_EQ(SumObjective::leaf_prefix(1), "term1.");
}

/** @brief Tests that no term and a weight that is not finite and above zero are refused. */
TEST(Small_SumObjective, MakeRefusesNoTermAndAWeightItCannotUse)
{
    SA_EXPECT_FAILURE(SumObjective::make({}), StatusCode::InvalidDescription,
                      "at least one term");
    SA_EXPECT_FAILURE(make_pair(1.0, 0.0), StatusCode::InvalidDescription, "Term 1",
                      "'cross_entropy'", "above zero");
    SA_EXPECT_FAILURE(make_pair(-0.5, 1.0), StatusCode::InvalidDescription, "Term 0");
    SA_EXPECT_FAILURE(make_pair(1.0, std::numeric_limits<double>::infinity()),
                      StatusCode::InvalidDescription, "Term 1", "finite");
    SA_EXPECT_FAILURE(make_pair(std::numeric_limits<double>::quiet_NaN(), 1.0),
                      StatusCode::InvalidDescription, "Term 0");
}

/** @brief Tests that a term its own step feeds is refused when the sum is made, by name. */
TEST(Small_SumObjective, MakeRefusesATermItsOwnStepFeeds)
{
    EXPECT_FALSE(StepFedObjective().is_fed_by_batch());
    EXPECT_TRUE(CrossEntropyObjective(K_BATCH, K_CLASSES).is_fed_by_batch());

    std::vector<SumTerm> terms;
    terms.push_back(cross_entropy_term(1.0));
    terms.push_back(SumTerm{std::make_unique<StepFedObjective>(), 1.0});

    SA_EXPECT_FAILURE(SumObjective::make(std::move(terms)), StatusCode::InvalidDescription,
                      "Term 1", "'step_fed'", "fed by its own step");
}

/** @brief Tests that two terms of one role name declare two leaves, each under its prefix. */
TEST(Small_SumObjective, EachTermDeclaresItsLeavesUnderItsOwnPrefix)
{
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<SumObjective> objective, make_pair(1.0, 0.5));
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    ObjectiveLeaves leaves;

    objective->declare(builder, leaves);

    ASSERT_EQ(leaves.all().size(), 2u);
    EXPECT_FALSE(leaves.has("targets"));
    ASSERT_TRUE(leaves.has("term0.targets"));
    ASSERT_TRUE(leaves.has("term1.targets"));
    EXPECT_NE(leaves.at("term0.targets"), leaves.at("term1.targets"));
    EXPECT_EQ(graph.value(leaves.at("term1.targets")).shape, (Shape{K_BATCH, K_CLASSES}));

    // Each term requires its own role, and a role required twice is matched twice.
    ASSERT_EQ(leaves.required().size(), 2u);
    EXPECT_EQ(leaves.required()[0].role, Data::K_ROLE_LABELS);
    EXPECT_EQ(leaves.required()[1].role, Data::K_ROLE_LABELS);
}

/** @brief Tests that the loss is one element: a SCALE per weight that is not 1 and one ADD. */
TEST(Small_SumObjective, TheLossAddsTheWeightedLossesInTheGraph)
{
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<SumObjective> objective, make_pair(1.0, 0.5));
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    ObjectiveLeaves leaves;
    objective->declare(builder, leaves);

    const Graph::ValueID logits = builder.input(Shape{K_BATCH, K_CLASSES}, "logits");
    NN::ModelOutputs outputs;
    SA_EXPECT_OK(outputs.add("output", logits));

    const Graph::ValueID loss = objective->loss(builder, graph, outputs, leaves);

    EXPECT_EQ(graph.value(loss).shape, (Shape{1}));
    EXPECT_EQ(count_nodes(graph, Ops::K_NLL_LOSS.id), 2u);
    EXPECT_EQ(count_nodes(graph, Ops::K_ADD.id), 1u);
    ASSERT_EQ(count_nodes(graph, Ops::K_SCALE.id), 1u);

    const Graph::Node& sum = graph.nodes().back();
    ASSERT_EQ(sum.op, Ops::K_ADD.id);
    ASSERT_EQ(sum.outputs.size(), 1u);
    EXPECT_EQ(sum.outputs[0], loss);

    for (const Graph::Node& node : graph.nodes())
    {
        if (node.op == Ops::K_SCALE.id)
            EXPECT_EQ(node.attributes.values[0], 0.5);
    }
}

/** @brief Tests that a gradient reaches the model through both terms of a training graph. */
TEST(Small_SumObjective, ATrainingGraphDifferentiatesTheSum)
{
    NN::Sequential<NN::Linear> model(NN::Linear{K_FEATURES, K_CLASSES});
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<SumObjective> objective, make_pair(2.0, 0.5));

    SA_ASSERT_OK_AND_ASSIGN(
        const Train::TrainingGraph built,
        Train::build_training_graph(model, *objective, Shape{K_BATCH, K_FEATURES}));

    EXPECT_EQ(built.graph.value(built.spec.loss).shape.num_elements(), 1);
    EXPECT_EQ(built.spec.targets, Graph::K_NO_VALUE);
    EXPECT_TRUE(built.leaves.has("term0.targets"));
    EXPECT_TRUE(built.leaves.has("term1.targets"));
    ASSERT_EQ(built.parameters.size(), 2u);
    EXPECT_NE(built.parameters[0].gradient, Graph::K_NO_VALUE);
    EXPECT_NE(built.parameters[1].gradient, Graph::K_NO_VALUE);
}

/** @brief Tests that of two metrics of one name the first term's is the one reported. */
TEST(Small_SumObjective, MeasureKeepsTheFirstOfTwoMetricsOfOneName)
{
    std::vector<SumTerm> terms;
    terms.push_back(SumTerm{std::make_unique<ReportingObjective>(
                                Train::Metric{"accuracy", 0.25, true},
                                Train::Metric{"precision", 0.5, true}),
                            1.0});
    terms.push_back(SumTerm{std::make_unique<ReportingObjective>(
                                Train::Metric{"recall", 0.75, true},
                                Train::Metric{"accuracy", 1.0, true}),
                            1.0});
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<SumObjective> objective,
                            SumObjective::make(std::move(terms)));
    const Data::Batch batch{};
    const Train::StepView step{batch, 0.0, {}};

    Train::Metrics metrics;
    SA_EXPECT_OK(objective->measure(step, metrics));

    // The values are the literals above, so they are compared exactly.
    ASSERT_EQ(metrics.all().size(), 3u);
    EXPECT_EQ(metrics.all()[0].name, "accuracy");
    EXPECT_EQ(metrics.all()[0].value, 0.25);
    EXPECT_EQ(metrics.all()[1].name, "precision");
    EXPECT_EQ(metrics.all()[2].name, "recall");
    EXPECT_EQ(metrics.all()[2].value, 0.75);
}

/** @brief Tests that the "sum" row builds each term through the registry it was found in. */
TEST(Small_SumObjective, TheFactoryBuildsEachTermThroughTheRegistry)
{
    const Train::ObjectiveRegistry registry = make_registry();

    SA_ASSERT_OK_AND_ASSIGN(
        const std::unique_ptr<Train::Objective> objective,
        make_from(registry, R"({"terms": [{"type": "cross_entropy"},
                                          {"type": "cross_entropy", "weight": 0.5}]})"));

    const auto* const sum = dynamic_cast<const SumObjective*>(objective.get());
    ASSERT_NE(sum, nullptr);
    ASSERT_EQ(sum->terms().size(), 2u);
    EXPECT_EQ(sum->terms()[0].weight, 1.0);
    EXPECT_EQ(sum->terms()[1].weight, 0.5);
    EXPECT_EQ(sum->terms()[0].objective->name(), "cross_entropy");
}

/** @brief Tests that a term may itself be a sum, and that its leaves carry both prefixes. */
TEST(Small_SumObjective, ATermMayBeASum)
{
    const Train::ObjectiveRegistry registry = make_registry();

    SA_ASSERT_OK_AND_ASSIGN(
        const std::unique_ptr<Train::Objective> objective,
        make_from(registry,
                  R"({"terms": [{"type": "cross_entropy"},
                                {"type": "sum", "weight": 2,
                                 "terms": [{"type": "cross_entropy"}]}]})"));
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    ObjectiveLeaves leaves;

    objective->declare(builder, leaves);

    ASSERT_EQ(leaves.all().size(), 2u);
    EXPECT_TRUE(leaves.has("term0.targets"));
    EXPECT_TRUE(leaves.has("term1.term0.targets"));
}

/** @brief Tests each section the factory refuses, and that a refusal names the entry. */
TEST(Small_SumObjective, TheFactoryRefusesASectionItCannotRead)
{
    const Train::ObjectiveRegistry registry = make_registry();

    SA_EXPECT_FAILURE(make_from(registry, "{}"), StatusCode::InvalidDescription,
                      "the objective block", "'terms'");
    SA_EXPECT_FAILURE(make_from(registry, R"({"terms": [], "gamma": 2})"),
                      StatusCode::InvalidDescription, "'gamma'", "'terms'");
    SA_EXPECT_FAILURE(make_from(registry, R"({"terms": {}})"), StatusCode::InvalidDescription,
                      "'terms'", "an array");
    SA_EXPECT_FAILURE(make_from(registry, R"({"terms": []})"), StatusCode::InvalidDescription,
                      "at least one term");
    SA_EXPECT_FAILURE(make_from(registry, R"({"terms": [3]})"), StatusCode::InvalidDescription,
                      "'terms' entry 0", "an object");
    SA_EXPECT_FAILURE(make_from(registry, R"({"terms": [{"weight": 1}]})"),
                      StatusCode::InvalidDescription, "'terms' entry 0", "'type'");
    SA_EXPECT_FAILURE(
        make_from(registry, R"({"terms": [{"type": "cross_entropy", "weight": "half"}]})"),
        StatusCode::InvalidDescription, "entry 0 'weight'", "a number");
    SA_EXPECT_FAILURE(make_from(registry, R"({"terms": [{"type": "cross_entropy"},
                                                        {"type": "yolo"}]})"),
                      StatusCode::InvalidDescription, "'yolo'", "'cross_entropy'", "'sum'");
    SA_EXPECT_FAILURE(
        make_from(registry, R"({"terms": [{"type": "cross_entropy", "smoothing": 0.1}]})"),
        StatusCode::InvalidDescription, "'terms' entry 0", "'smoothing'");
}

/** @brief Tests that the factory, called with a context that names no registry, refuses. */
TEST(Small_SumObjective, TheFactoryNeedsTheRegistryOfItsContext)
{
    SA_EXPECT_FAILURE(
        Train::make_sum_objective(make_context(),
                                  Config::parse_json(R"({"terms": []})", "<test>"), K_CONTEXT),
        StatusCode::InvalidDescription, "the objective block", "registry");
}
