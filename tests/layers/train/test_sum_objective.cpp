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
 * @file test_sum_objective.cpp
 * @brief Tests the composite objective: its terms, its prefixed leaves, its loss and its factory.
 * @author Mustafa Garip
 */

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
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
#include <SushiAI/failure/result.hpp>
#include <SushiAI/shape/shape.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/data/dataset.hpp>
#include <SushiAI/data/roles.hpp>
#include <SushiAI/execution/host_execution_backend.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/graph/builder.hpp>
#include <SushiAI/graph/host_evaluator.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/nn/layers.hpp>
#include <SushiAI/nn/model_outputs.hpp>
#include <SushiAI/tensor_values/tensor_values.hpp>
#include <SushiAI/train/in_memory_objective.hpp>
#include <SushiAI/train/metrics.hpp>
#include <SushiAI/train/objective.hpp>
#include <SushiAI/train/objective_context.hpp>
#include <SushiAI/train/objective_registry.hpp>
#include <SushiAI/train/step_view.hpp>
#include <SushiAI/train/steps.hpp>
#include <SushiAI/train/sum_objective.hpp>
#include <SushiAI/train/training_graph.hpp>
#include <support/tolerance.hpp>

using namespace SushiAI;
using Train::CrossEntropyObjective;
using Train::ObjectiveLeaves;
using Train::SumObjective;
using Train::SumTerm;
namespace SaTest = SushiAI::Test;

namespace
{
    using SushiAI::Test::require_value;
    /** @brief Holds the samples of a batch. */
    constexpr std::int64_t K_BATCH = 4;

    /** @brief Holds the features one sample enters the model with. */
    constexpr std::int64_t K_FEATURES = 6;

    /** @brief Holds the classes the model scores. */
    constexpr std::int64_t K_CLASSES = 3;

    /** @brief Names the objective block in a diagnostic. */
    constexpr std::string_view K_CONTEXT = "the objective block";

    /** @brief The rounded operations from the counts to the accuracy: the one quotient rounds. */
    constexpr std::size_t K_RATIO_OPERATIONS = 1;

    /**
     * @brief Counts the roundings on the longest path to the weighted loss in double: 20 in a
     *        term (7 in the log-softmax, 1 in the product, 11 in the sum of 12, 1 in the mean),
     *        1 in the sum of the terms and 1 in the literal of the expected value; the scale
     *        by 0.5 is exact.
     */
    constexpr std::size_t K_WEIGHTED_LOSS_OPERATIONS = 22;

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
                                     require_value(Config::parse_json(text, "<test>")), K_CONTEXT);
    }

    /** @brief Traces a sum of the weights 2 and 0.5 over one Linear layer into a training graph. */
    Result<Train::TrainingGraph> build_sum_training_graph()
    {
        NN::Sequential<NN::Linear> model(NN::Linear{K_FEATURES, K_CLASSES});
        SA_TRY_ASSIGN(const std::unique_ptr<SumObjective> objective, make_pair(2.0, 0.5));

        return Train::build_training_graph(model, *objective, Shape{K_BATCH, K_FEATURES});
    }

    /** @brief Holds a weight pair a sum refuses and the words of the refusal. */
    struct WeightRow
    {
        std::string name;    /**< The row's name. */
        double first;        /**< The weight of term 0. */
        double second;       /**< The weight of term 1. */
        std::string where;   /**< The term the message names. */
        std::string reason;  /**< Another text the message holds. */
    };

    /** @brief Runs one refused weight pair. */
    class Small_SumObjectiveWeight : public ::testing::TestWithParam<WeightRow>
    {

    };

    /** @brief Holds a section the factory refuses and the words of the refusal. */
    struct SectionRow
    {
        std::string name;    /**< The row's name. */
        std::string text;    /**< The JSON of the section. */
        std::string first;   /**< A text the message holds. */
        std::string second;  /**< Another text the message holds; empty when one is enough. */
        std::string third;   /**< A third text the message holds; empty when two are enough. */
    };

    /** @brief Runs one refused section. */
    class Small_SumObjectiveSection : public ::testing::TestWithParam<SectionRow>
    {

    };

    /** @brief Runs one parameter index of the gradient cases. */
    class Small_SumObjectiveGradient : public ::testing::TestWithParam<std::size_t>
    {

    };

    /** @brief Provides the host engine the scored tensors and the labels live on. */
    class Small_SumObjectiveScore : public ::testing::Test
    {
        protected:
            /** @brief Returns [3, 2] logits whose rows have their largest value at 1, 0, 1. */
            [[nodiscard]] Result<SushiBLAS::Tensor> logits()
            {
                SA_TRY_ASSIGN(SushiBLAS::Tensor output,
                              allocate_tensor(backend.engine(), Shape{3, 2}));
                SaTest::write_tensor(output,
                                     std::vector<double>{0.1, 0.9, 0.8, 0.2, -1.0, 0.5});
                return output;
            }

            /** @brief Returns a batch whose labels role holds @p values as int32. */
            [[nodiscard]] Result<Data::Batch> batch_of(const std::vector<int32_t>& values)
            {
                Data::BatchTarget labels;
                labels.role = std::string(Data::K_ROLE_LABELS);
                SA_TRY_ASSIGN(labels.values,
                              allocate_tensor(backend.engine(),
                                              Shape{static_cast<int64_t>(values.size())},
                                              DType::Int32));
                for (std::size_t i = 0; i < values.size(); ++i)
                    labels.values.data_as<int32_t>()[i] = values[i];

                Data::Batch batch;
                batch.targets.push_back(std::move(labels));
                return batch;
            }

            /** @brief The host backend the tensors are allocated from. */
            HostExecutionBackend backend;
    };

    /** @brief Names a weight row by its name field. */
    std::string weight_row_name(const ::testing::TestParamInfo<WeightRow>& info)
    {
        return info.param.name;
    }

    /** @brief Names a section row by its name field. */
    std::string section_row_name(const ::testing::TestParamInfo<SectionRow>& info)
    {
        return info.param.name;
    }

    /** @brief Names a parameter row by its index. */
    std::string index_row_name(const ::testing::TestParamInfo<std::size_t>& info)
    {
        return "Index" + std::to_string(info.param);
    }
} // namespace

/** @brief Tests that make() keeps its terms in order, each with its weight. */
TEST(Small_SumObjective, MakeKeepsItsTermsInOrder)
{
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<SumObjective> objective, make_pair(1.0, 0.5));

    ASSERT_EQ(objective->terms().size(), 2u);
    EXPECT_EQ(objective->terms()[0].weight, 1.0);
    EXPECT_EQ(objective->terms()[1].weight, 0.5);
    EXPECT_EQ(objective->terms()[1].objective->name(), "cross_entropy");
}

/** @brief Tests that a sum names itself "sum". */
TEST(Small_SumObjective, ASumIsNamedSum)
{
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<SumObjective> objective, make_pair(1.0, 0.5));

    EXPECT_EQ(objective->name(), "sum");
}

/** @brief Tests that a sum is fed from a batch. */
TEST(Small_SumObjective, ASumIsFedByABatch)
{
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<SumObjective> objective, make_pair(1.0, 0.5));

    EXPECT_TRUE(objective->is_fed_by_batch());
}

/** @brief Tests that the leaves of term 1 hang under the prefix "term1.". */
TEST(Small_SumObjective, TheLeafPrefixOfATermIsItsIndex)
{
    EXPECT_EQ(SumObjective::leaf_prefix(1), "term1.");
}

/** @brief Tests that a sum of no term is refused. */
TEST(Small_SumObjective, MakeRefusesNoTerm)
{
    SA_EXPECT_FAILURE(SumObjective::make({}), StatusCode::InvalidDescription,
                      "at least one term");
}

/** @brief Tests that a weight that is not finite and above zero is refused, naming its term. */
TEST_P(Small_SumObjectiveWeight, MakeRefusesAWeightItCannotUse)
{
    const WeightRow& row = GetParam();

    SA_EXPECT_FAILURE(make_pair(row.first, row.second), StatusCode::InvalidDescription,
                      row.where, row.reason);
}

INSTANTIATE_TEST_SUITE_P(
    Rows, Small_SumObjectiveWeight,
    ::testing::Values(
        WeightRow{"ZeroWeight", 1.0, 0.0, "Term 1", "above zero"},
        WeightRow{"NegativeWeight", -0.5, 1.0, "Term 0", "above zero"},
        WeightRow{"InfiniteWeight", 1.0, std::numeric_limits<double>::infinity(), "Term 1",
                  "finite"},
        WeightRow{"NaNWeight", std::numeric_limits<double>::quiet_NaN(), 1.0, "Term 0",
                  "finite"}),
    weight_row_name);

/** @brief Tests that a term its own step feeds is refused when the sum is made, by name. */
TEST(Small_SumObjective, MakeRefusesATermItsOwnStepFeeds)
{
    std::vector<SumTerm> terms;
    terms.push_back(cross_entropy_term(1.0));
    terms.push_back(SumTerm{std::make_unique<Train::InMemoryObjective>(), 1.0});

    SA_EXPECT_FAILURE(SumObjective::make(std::move(terms)), StatusCode::InvalidDescription,
                      "Term 1", "'in-memory'", "fed by its own step");
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
    ASSERT_TRUE(leaves.has("term0.targets"));
    ASSERT_TRUE(leaves.has("term1.targets"));
    EXPECT_NE(leaves.at("term0.targets"), leaves.at("term1.targets"));
    EXPECT_EQ(graph.value(leaves.at("term1.targets")).shape, (Shape{K_BATCH, K_CLASSES}));
}

/** @brief Tests that no leaf is declared under the bare name a term would give it. */
TEST(Small_SumObjective, NoLeafIsDeclaredWithoutAPrefix)
{
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<SumObjective> objective, make_pair(1.0, 0.5));
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    ObjectiveLeaves leaves;

    objective->declare(builder, leaves);

    EXPECT_FALSE(leaves.has("targets"));
}

/** @brief Tests that each term requires its own role, and a role required twice is matched twice. */
TEST(Small_SumObjective, EachTermRequiresItsRole)
{
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<SumObjective> objective, make_pair(1.0, 0.5));
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    ObjectiveLeaves leaves;

    objective->declare(builder, leaves);

    ASSERT_EQ(leaves.required().size(), 2u);
    EXPECT_EQ(leaves.required()[0].role, Data::K_ROLE_LABELS);
    EXPECT_EQ(leaves.required()[1].role, Data::K_ROLE_LABELS);
}

/**
 * @brief Tests the value of the weighted loss on the host.
 *
 * Every row has the logits {0, ln 3, 0}, so a row's log-probabilities are -ln 5, ln 3 - ln 5 and
 * -ln 5. Term 0 targets class 1 in every row and has the loss ln 5 - ln 3; term 1 targets class 0
 * and has the loss ln 5. With the weights 1 and 0.5 the sum is ln 5 - ln 3 + 0.5 ln 5 =
 * 1.5 ln 5 - ln 3 = 1.3155445799830407.
 */
TEST(Small_SumObjective, TheLossAddsTheWeightedLossesOfItsTerms)
{
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<SumObjective> objective, make_pair(1.0, 0.5));
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    ObjectiveLeaves leaves;
    objective->declare(builder, leaves);
    const Graph::ValueID logits = builder.input(Shape{K_BATCH, K_CLASSES}, "logits");
    NN::ModelOutputs outputs;
    SA_EXPECT_OK(outputs.add("output", logits));
    SA_ASSERT_OK_AND_ASSIGN(const Graph::ValueID loss,
                            objective->loss(builder, graph, outputs, leaves));
    Graph::HostEvaluator evaluator(graph);
    SA_EXPECT_OK(evaluator.write(logits, {0, 1.0986122886681098, 0, 0, 1.0986122886681098, 0,
                                          0, 1.0986122886681098, 0, 0, 1.0986122886681098, 0}));
    SA_EXPECT_OK(evaluator.write(leaves.at("term0.targets"),
                                 {0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0}));
    SA_EXPECT_OK(evaluator.write(leaves.at("term1.targets"),
                                 {1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0}));

    SA_EXPECT_OK(evaluator.run());

    SA_ASSERT_OK_AND_ASSIGN(const std::vector<double> result, evaluator.read(loss));
    ASSERT_EQ(result.size(), 1u);
    EXPECT_TRUE(TestSupport::is_within(TestSupport::measure_relative_error<double>(
        result.front(), 1.3155445799830407, K_WEIGHTED_LOSS_OPERATIONS)));
}

/** @brief Tests that the loss of a sum is one element. */
TEST(Small_SumObjective, ATrainingGraphOfTheSumHasAScalarLoss)
{
    SA_ASSERT_OK_AND_ASSIGN(const Train::TrainingGraph built, build_sum_training_graph());

    EXPECT_EQ(built.graph.value(built.spec.loss).shape.num_elements(), 1);
}

/** @brief Tests that a sum has no targets value of its own. */
TEST(Small_SumObjective, ATrainingGraphOfTheSumHasNoTargetsValue)
{
    SA_ASSERT_OK_AND_ASSIGN(const Train::TrainingGraph built, build_sum_training_graph());

    EXPECT_EQ(built.spec.targets, Graph::K_NO_VALUE);
}

/** @brief Tests that the training graph holds the prefixed leaf of each term. */
TEST(Small_SumObjective, ATrainingGraphOfTheSumHoldsTheLeavesOfEachTerm)
{
    SA_ASSERT_OK_AND_ASSIGN(const Train::TrainingGraph built, build_sum_training_graph());

    EXPECT_TRUE(built.leaves.has("term0.targets"));
    EXPECT_TRUE(built.leaves.has("term1.targets"));
}

/** @brief Tests that a gradient reaches each parameter through both terms. */
TEST_P(Small_SumObjectiveGradient, ATrainingGraphDifferentiatesTheSum)
{
    SA_ASSERT_OK_AND_ASSIGN(const Train::TrainingGraph built, build_sum_training_graph());

    ASSERT_EQ(built.parameters.size(), 2u);
    EXPECT_NE(built.parameters[GetParam()].gradient, Graph::K_NO_VALUE);
}

INSTANTIATE_TEST_SUITE_P(Rows, Small_SumObjectiveGradient, ::testing::Values(0u, 1u),
                         index_row_name);

/**
 * @brief Tests that two cross-entropy terms, which both measure accuracy, report it once.
 *
 * The logits of the fixture put the largest value at 1, 0, 1 and the labels are 1, 0, 0, so the
 * accuracy is 2/3.
 */
TEST_F(Small_SumObjectiveScore, MeasureReportsTheAccuracyOfTwoTermsOnce)
{
    std::vector<SumTerm> terms;
    terms.push_back(SumTerm{std::make_unique<CrossEntropyObjective>(3, 2), 1.0});
    terms.push_back(SumTerm{std::make_unique<CrossEntropyObjective>(3, 2), 0.5});
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<SumObjective> objective,
                            SumObjective::make(std::move(terms)));
    SA_ASSERT_OK_AND_ASSIGN(const SushiBLAS::Tensor output, logits());
    SA_ASSERT_OK_AND_ASSIGN(const Data::Batch batch, batch_of({1, 0, 0}));
    const std::vector<Train::StepOutput> outputs{Train::StepOutput{"output", &output}};
    const Train::StepView step{
        batch, 0.0, SushiRuntime::span<const Train::StepOutput>(outputs.data(), outputs.size())};

    Train::Metrics metrics;
    SA_EXPECT_OK(objective->measure(step, metrics));

    ASSERT_EQ(metrics.all().size(), 1u);
    ASSERT_EQ(metrics.all()[0].name, "accuracy");
    EXPECT_TRUE(TestSupport::is_within(TestSupport::measure_relative_error<double>(
        metrics.all()[0].value, 2.0 / 3.0, K_RATIO_OPERATIONS)));
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
TEST_P(Small_SumObjectiveSection, TheFactoryRefusesASectionItCannotRead)
{
    const SectionRow& row = GetParam();
    const Train::ObjectiveRegistry registry = make_registry();

    SA_EXPECT_FAILURE(make_from(registry, row.text), StatusCode::InvalidDescription, row.first,
                      row.second, row.third);
}

INSTANTIATE_TEST_SUITE_P(
    Rows, Small_SumObjectiveSection,
    ::testing::Values(
        SectionRow{"NoTerms", "{}", "the objective block", "'terms'", ""},
        SectionRow{"AnUnknownKey", R"({"terms": [], "gamma": 2})", "'gamma'", "'terms'", ""},
        SectionRow{"TermsThatAreAnObject", R"({"terms": {}})", "'terms'", "an array", ""},
        SectionRow{"AnEmptyList", R"({"terms": []})", "at least one term", "", ""},
        SectionRow{"AnEntryThatIsANumber", R"({"terms": [3]})", "'terms' entry 0", "an object",
                   ""},
        SectionRow{"AnEntryWithoutAType", R"({"terms": [{"weight": 1}]})", "'terms' entry 0",
                   "'type'", ""},
        SectionRow{"AWeightThatIsNoNumber",
                   R"({"terms": [{"type": "cross_entropy", "weight": "half"}]})",
                   "entry 0 'weight'", "a number", ""},
        SectionRow{"AnUnknownTermType",
                   R"({"terms": [{"type": "cross_entropy"}, {"type": "yolo"}]})", "'yolo'",
                   "'cross_entropy'", "'sum'"},
        SectionRow{"AnUnknownTermKey",
                   R"({"terms": [{"type": "cross_entropy", "smoothing": 0.1}]})",
                   "'terms' entry 0", "'smoothing'", ""}),
    section_row_name);

/** @brief Tests that the factory, called with a context that names no registry, refuses. */
TEST(Small_SumObjective, TheFactoryNeedsTheRegistryOfItsContext)
{
    SA_EXPECT_FAILURE(
        Train::make_sum_objective(make_context(),
                                  require_value(Config::parse_json(R"({"terms": []})", "<test>")),
                                  K_CONTEXT),
        StatusCode::InvalidDescription, "the objective block", "registry");
}
