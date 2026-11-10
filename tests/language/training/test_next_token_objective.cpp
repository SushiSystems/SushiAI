/**************************************************************************/
/* test_next_token_objective.cpp                                          */
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
 * @file test_next_token_objective.cpp
 * @brief Tests the next-token objective: its graph, its loss and gradient on an engine, and
 *        decoders that overfit through it.
 * @author Mustafa Garip
 */

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <tuple>
#include <utility>
#include <vector>
#include <gtest/gtest.h>
#include <SushiBLAS/engine/engine.hpp>
#include <SushiBLAS/tensor/tensor.hpp>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/config/value.hpp>
#include <SushiAI/allocation/allocation.hpp>
#include <SushiAI/dtype/dtype.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/shape/shape.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/data/dataset.hpp>
#include <SushiAI/data/input.hpp>
#include <SushiAI/data/materialised_dataset.hpp>
#include <SushiAI/data/roles.hpp>
#include <SushiAI/data/target.hpp>
#include <SushiAI/engine_fixture/engine_test.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/graph/builder.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/graph/tensor_pool.hpp>
#include <SushiAI/graph/tensor_source.hpp>
#include <SushiAI/language/blocks/causal_self_attention.hpp>
#include <SushiAI/language/blocks/language_head.hpp>
#include <SushiAI/language/blocks/position_embedding.hpp>
#include <SushiAI/language/blocks/transformer_block.hpp>
#include <SushiAI/language/datasets/next_token_collator.hpp>
#include <SushiAI/language/datasets/token_dataset.hpp>
#include <SushiAI/language/datasets/token_file.hpp>
#include <SushiAI/language/text/token.hpp>
#include <SushiAI/language/training/next_token_objective.hpp>
#include <SushiAI/language/training/next_token_targets.hpp>
#include <SushiAI/language/training/perplexity.hpp>
#include <SushiAI/nn/embedding.hpp>
#include <SushiAI/nn/init.hpp>
#include <SushiAI/nn/layer_norm.hpp>
#include <SushiAI/nn/model_outputs.hpp>
#include <SushiAI/nn/model_state.hpp>
#include <SushiAI/nn/module.hpp>
#include <SushiAI/nn/trace.hpp>
#include <support/scratch_names.hpp>
#include <support/tolerance.hpp>
#include <SushiAI/optim/adamw.hpp>
#include <SushiAI/optim/apply_step_schedule.hpp>
#include <SushiAI/optim/cosine_schedule.hpp>
#include <SushiAI/optim/optimizer.hpp>
#include <SushiAI/optim/sgd.hpp>
#include <SushiAI/scratch_files/scratch_file.hpp>
#include <SushiAI/tensor_values/tensor_values.hpp>
#include <SushiAI/train/feed.hpp>
#include <SushiAI/train/metrics.hpp>
#include <SushiAI/train/objective.hpp>
#include <SushiAI/train/objective_context.hpp>
#include <SushiAI/train/step_view.hpp>
#include <SushiAI/train/steps.hpp>
#include <SushiAI/train/trainer.hpp>
#include <SushiAI/train/training_graph.hpp>
#include <SushiAI/train/training_step.hpp>

using namespace SushiAI;
using Language::Training::make_next_token_objective;
using Language::Training::NextTokenObjective;
// SaTest stays reachable inside a TEST_F body; see tests/language/training/README.md.
namespace SaTest = SushiAI::Test;

namespace
{
    /** @brief Holds the rows of a batch, N. */
    constexpr std::int64_t K_BATCH = 2;

    /** @brief Holds the positions of a row, T. */
    constexpr std::int64_t K_LENGTH = 3;

    /** @brief Holds the vocabulary size, V. */
    constexpr std::int64_t K_VOCABULARY = 5;

    /** @brief Holds the substrings a refusal names; an empty one is satisfied by any message. */
    using Needles = std::array<std::string, 4>;

    /** @brief Returns a model whose [N, T] ids give [N, T, V] logits: one table row each. */
    NN::Embedding make_bigram_model()
    {
        NN::Embedding model{};
        model.num_embeddings = K_VOCABULARY;
        model.embedding_dim = K_VOCABULARY;
        return model;
    }

    /** @brief Returns what a factory is given for the bigram model over K_VOCABULARY ids. */
    Train::ObjectiveContext make_context()
    {
        Train::ObjectiveContext given;
        given.batch = K_BATCH;
        given.classes = K_VOCABULARY;
        given.input_sample = Shape{K_LENGTH};
        given.outputs.push_back(
            Train::OutputShape{"out", Shape{K_BATCH, K_LENGTH, K_VOCABULARY}});
        return given;
    }

    /** @brief Returns the context of a model that publishes "balance" beside its logits. */
    Train::ObjectiveContext make_several_context()
    {
        Train::ObjectiveContext several = make_context();
        several.outputs.push_back(Train::OutputShape{"balance", Shape{1, 1}});
        return several;
    }

    /** @brief Returns the context of a model that publishes a second output "aux". */
    Train::ObjectiveContext two_outputs_context()
    {
        Train::ObjectiveContext two_outputs = make_context();
        two_outputs.outputs.push_back(Train::OutputShape{"aux", Shape{K_BATCH, 1}});
        return two_outputs;
    }

    /** @brief Returns the context whose logits are [N, V]. */
    Train::ObjectiveContext flat_context()
    {
        Train::ObjectiveContext flat = make_context();
        flat.outputs.front().shape = Shape{K_BATCH, K_VOCABULARY};
        return flat;
    }

    /** @brief Returns the context whose input sample is an image. */
    Train::ObjectiveContext images_context()
    {
        Train::ObjectiveContext images = make_context();
        images.input_sample = Shape{4, 4, 3};
        return images;
    }

    /** @brief Returns the context whose data holds more token ids than the model scores. */
    Train::ObjectiveContext wide_data_context()
    {
        Train::ObjectiveContext wide_data = make_context();
        wide_data.classes = K_VOCABULARY + 1;
        return wide_data;
    }

    /** @brief Parses @p text, the attributes of an objective. */
    Config::Value attributes_of(const std::string& text)
    {
        return SaTest::require_value(Config::parse_json(text, "<test>"));
    }

    /** @brief Names the three extents of an objective. */
    struct Extents
    {
        std::int64_t batch;      /**< N. */
        std::int64_t length;     /**< T. */
        std::int64_t vocabulary; /**< V. */
    };

    /** @brief Names extents make() refuses and what its message names. */
    struct RefusedExtents
    {
        Extents extents; /**< The extents given. */
        Needles needles; /**< What the message names. */
    };

    /** @brief Names the context and the attributes the factory refuses and what it names. */
    struct RefusedFactory
    {
        Train::ObjectiveContext given; /**< What the factory is given. */
        std::string attributes;        /**< The objective's own keys, as JSON. */
        Needles needles;               /**< What the message names. */
    };

    /** @brief Names the attributes the factory refuses over two published outputs. */
    struct RefusedOutputKey
    {
        std::string attributes; /**< The objective's own keys, as JSON. */
        Needles needles;        /**< What the message names. */
    };

    /** @brief Returns the descriptions the factory refuses. */
    std::vector<RefusedFactory> refused_factories()
    {
        return {
            {make_context(), R"({"smoothing": 0.1})",
             {"objective states the key 'smoothing'", "", "", ""}},
            {two_outputs_context(), "{}", {"one [N, T, V] output", "publishes 2", "", ""}},
            {flat_context(), "{}", {"[2, 3, V] logits", "'out'", "", ""}},
            {images_context(), "{}", {"[T] token ids per sample", "", "", ""}},
            {wide_data_context(), "{}", {"6 token ids", "scores 5", "", ""}}};
    }

    /** @brief Provides the extents below 1. */
    class Small_LanguageTrainingNextTokenObjectiveExtents
        : public ::testing::TestWithParam<RefusedExtents>
    {

    };

    /** @brief Provides the products INT32 cannot index. */
    class Small_LanguageTrainingNextTokenObjectiveIndex
        : public ::testing::TestWithParam<RefusedExtents>
    {

    };

    /** @brief Provides the largest products INT32 indexes. */
    class Small_LanguageTrainingNextTokenObjectiveLargest
        : public ::testing::TestWithParam<Extents>
    {

    };

    /** @brief Provides the descriptions the factory refuses. */
    class Small_LanguageTrainingNextTokenObjectiveFactory
        : public ::testing::TestWithParam<RefusedFactory>
    {

    };

    /** @brief Provides the forms of "output" the factory refuses. */
    class Small_LanguageTrainingNextTokenObjectiveOutputKey
        : public ::testing::TestWithParam<RefusedOutputKey>
    {

    };
} // namespace

/** @brief Tests that make() refuses an extent below 1, naming all three. */
TEST_P(Small_LanguageTrainingNextTokenObjectiveExtents, MakeRefusesAnExtentBelowOne)
{
    const RefusedExtents& row = GetParam();

    SA_EXPECT_FAILURE(NextTokenObjective::make(row.extents.batch, row.extents.length,
                                               row.extents.vocabulary),
                      StatusCode::InvalidDescription, row.needles[0], row.needles[1]);
}

INSTANTIATE_TEST_SUITE_P(
    Rows, Small_LanguageTrainingNextTokenObjectiveExtents,
    ::testing::Values(RefusedExtents{{0, 3, 5}, {"at least 1", "got 0, 3 and 5", "", ""}},
                      RefusedExtents{{2, 0, 5}, {"at least 1", "got 2, 0 and 5", "", ""}},
                      RefusedExtents{{2, 3, 0}, {"at least 1", "got 2, 3 and 0", "", ""}}));

/** @brief Tests that a product INT32 cannot index is refused, naming it. */
TEST_P(Small_LanguageTrainingNextTokenObjectiveIndex, MakeRefusesWhatInt32CannotIndex)
{
    const RefusedExtents& row = GetParam();

    SA_EXPECT_FAILURE(NextTokenObjective::make(row.extents.batch, row.extents.length,
                                               row.extents.vocabulary),
                      StatusCode::InvalidDescription, row.needles[0], row.needles[1]);
}

// 64 * 1024 * 50257 is 3 293 642 752, above 2 147 483 647.
INSTANTIATE_TEST_SUITE_P(
    Rows, Small_LanguageTrainingNextTokenObjectiveIndex,
    ::testing::Values(
        RefusedExtents{{64, 1024, 50257}, {"int32 index", "64 * 1024 * 50257", "", ""}},
        RefusedExtents{{1, 2, 2147483647}, {"int32 index", "1 * 2 * 2147483647", "", ""}}));

/** @brief Tests that the largest products INT32 indexes are accepted. */
TEST_P(Small_LanguageTrainingNextTokenObjectiveLargest, MakeAcceptsWhatInt32CanIndex)
{
    const Extents& row = GetParam();

    SA_EXPECT_OK(NextTokenObjective::make(row.batch, row.length, row.vocabulary));
}

// 32 * 256 * 8192 is 67 108 864.
INSTANTIATE_TEST_SUITE_P(Rows, Small_LanguageTrainingNextTokenObjectiveLargest,
                         ::testing::Values(Extents{32, 256, 8192}, Extents{1, 1, 2147483647}));

/** @brief Tests that the one leaf is INT32 [N * T] and that labels are read as [T] ids. */
TEST(Small_LanguageTrainingNextTokenObjective, ItDeclaresAnIndexLeafAndReadsLabels)
{
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<NextTokenObjective> objective,
                            NextTokenObjective::make(K_BATCH, K_LENGTH, K_VOCABULARY));
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    Train::ObjectiveLeaves leaves;
    objective->declare(builder, leaves);

    ASSERT_EQ(leaves.all().size(), 1u);
    const Graph::Value& targets = graph.value(leaves.at("targets"));
    EXPECT_EQ(targets.shape, (Shape{K_BATCH * K_LENGTH}));
    EXPECT_EQ(targets.dtype, DType::Int32);

    ASSERT_EQ(leaves.required().size(), 1u);
    const Data::TargetDescription& labels = leaves.required().front();
    EXPECT_EQ(labels.role, Data::K_ROLE_LABELS);
    EXPECT_EQ(labels.sample_shape, (Shape{K_LENGTH}));
    EXPECT_EQ(labels.dtype, DType::Int32);
    EXPECT_EQ(labels.classes, 0);
}

/** @brief Tests that the loss is one element and the parameter of the model has a gradient. */
TEST(Small_LanguageTrainingNextTokenObjective, TheLossGathersOneLogProbabilityPerPosition)
{
    NN::Embedding model = make_bigram_model();
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<NextTokenObjective> objective,
                            NextTokenObjective::make(K_BATCH, K_LENGTH, K_VOCABULARY));
    Train::GraphOptions options;
    options.dtype = DType::Int32;
    SA_ASSERT_OK_AND_ASSIGN(
        const Train::TrainingGraph built,
        Train::build_training_graph(model, *objective, Shape{K_BATCH, K_LENGTH}, options));

    EXPECT_EQ(built.graph.value(built.spec.loss).shape.num_elements(), 1);
    EXPECT_EQ(built.graph.value(built.spec.targets).dtype, DType::Int32);

    ASSERT_EQ(built.parameters.size(), 1u);
    EXPECT_NE(built.parameters.front().gradient, Graph::K_NO_VALUE);
    EXPECT_EQ(built.parameters.front().role, NN::ParameterRole::Embedding);
}

/** @brief Tests that the factory reads N, T and V from the model's one output. */
TEST(Small_LanguageTrainingNextTokenObjective, TheFactoryReadsTheModelsOutput)
{
    SA_ASSERT_OK_AND_ASSIGN(
        const std::unique_ptr<Train::Objective> objective,
        make_next_token_objective(make_context(), attributes_of("{}"), "objective"));

    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    Train::ObjectiveLeaves leaves;
    objective->declare(builder, leaves);
    EXPECT_EQ(graph.value(leaves.at("targets")).shape, (Shape{K_BATCH * K_LENGTH}));
    EXPECT_EQ(objective->name(), "next_token");
}

/** @brief Tests each description the factory refuses. */
TEST_P(Small_LanguageTrainingNextTokenObjectiveFactory, TheFactoryRefusesWhatDoesNotFit)
{
    const RefusedFactory& row = GetParam();

    SA_EXPECT_FAILURE(make_next_token_objective(row.given, attributes_of(row.attributes),
                                                "objective"),
                      StatusCode::InvalidDescription, row.needles[0], row.needles[1]);
}

INSTANTIATE_TEST_SUITE_P(Rows, Small_LanguageTrainingNextTokenObjectiveFactory,
                         ::testing::ValuesIn(refused_factories()));

/** @brief Tests that the objective measures the perplexity of the loss a step computed. */
TEST(Small_LanguageTrainingNextTokenObjective, MeasureReportsThePerplexityOfTheLoss)
{
    // A uniform guess over 8 tokens costs ln(8) nats and has a perplexity of 8.
    constexpr std::size_t K_OPERATIONS = 6;
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<NextTokenObjective> objective,
                            NextTokenObjective::make(K_BATCH, K_LENGTH, K_VOCABULARY));
    const Data::Batch batch{};
    const Train::StepView step{batch, std::log(8.0), {}};

    Train::Metrics metrics;
    SA_EXPECT_OK(objective->measure(step, metrics));

    ASSERT_EQ(metrics.all().size(), 1u);
    EXPECT_EQ(metrics.all()[0].name, "perplexity");
    EXPECT_FALSE(metrics.all()[0].is_fraction);
    EXPECT_TRUE(TestSupport::is_within(
        TestSupport::measure_relative_error<double>(metrics.all()[0].value, 8.0, K_OPERATIONS)));
}

/** @brief Tests that a batch with no labels role is refused, naming the role. */
TEST(Small_LanguageTrainingNextTokenObjective, ABatchWithNoLabelsRoleIsRefused)
{
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<NextTokenObjective> objective,
                            NextTokenObjective::make(K_BATCH, K_LENGTH, K_VOCABULARY));
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    Train::ObjectiveLeaves leaves;
    objective->declare(builder, leaves);

    // The refusal is decided on the roles of the batch, before the pool is asked for a tensor.
    Graph::InMemoryTensorSource source;
    Graph::TensorPool pool(source);
    const Data::Batch batch{};

    SA_EXPECT_FAILURE(objective->feed(batch, leaves, pool), StatusCode::InvalidDescription,
                      "holds no 'labels' role");
    EXPECT_TRUE(source.requests().empty());
}

/** @brief Tests that "output" names the logits of a model that publishes several values. */
TEST(Small_LanguageTrainingNextTokenObjective, TheOutputKeyNamesTheLogitsAmongSeveral)
{
    SA_ASSERT_OK_AND_ASSIGN(
        const std::unique_ptr<Train::Objective> objective,
        make_next_token_objective(make_several_context(), attributes_of(R"({"output": "out"})"),
                                  "objective"));

    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    NN::ModelOutputs outputs;
    SA_EXPECT_OK(outputs.add("balance", builder.input(Shape{1, 1}, "balance")));
    SA_EXPECT_OK(outputs.add(
        "out", builder.input(Shape{K_BATCH, K_LENGTH, K_VOCABULARY}, "logits")));
    Train::ObjectiveLeaves leaves;
    objective->declare(builder, leaves);

    SA_ASSERT_OK_AND_ASSIGN(const Graph::ValueID loss,
                            objective->loss(builder, graph, outputs, leaves));

    EXPECT_EQ(graph.value(loss).shape.num_elements(), 1);
}

/** @brief Tests each form of "output" the factory refuses. */
TEST_P(Small_LanguageTrainingNextTokenObjectiveOutputKey, TheFactoryRefusesAnOutputItCannotFind)
{
    const RefusedOutputKey& row = GetParam();

    SA_EXPECT_FAILURE(make_next_token_objective(make_several_context(),
                                                attributes_of(row.attributes), "objective"),
                      StatusCode::InvalidDescription, row.needles[0], row.needles[1],
                      row.needles[2]);
}

INSTANTIATE_TEST_SUITE_P(
    Rows, Small_LanguageTrainingNextTokenObjectiveOutputKey,
    ::testing::Values(
        RefusedOutputKey{"{}", {"publishes 2", "name the logits with 'output'", "", ""}},
        RefusedOutputKey{R"({"output": "logits"})",
                         {"the output 'logits'", "none of that name among its 2", "", ""}},
        RefusedOutputKey{R"({"output": 3})", {"'output' names a published output", "", "", ""}},
        RefusedOutputKey{R"({"output": ""})", {"'output'", "is empty", "", ""}},
        RefusedOutputKey{R"({"output": "balance"})",
                         {"[2, 3, V] logits", "'balance'", "[1, 1]", ""}}));

namespace
{

    /**
     * @brief Holds the Bounds the absolute error of a float32 loss or gradient of order one: 256
     *        times the float unit roundoff, the smallest power of two not tighter than the earlier
     *        1e-5.
     */
    constexpr double K_TOLERANCE = 256.0 * TestSupport::unit_roundoff<float>();

    /** @brief Returns the token of each of the N * T positions, row after row. */
    std::vector<std::int32_t> token_ids()
    {
        return {0, 3, 3, 1, 4, 2};
    }

    /** @brief Returns the label of each position; token 3 occurs twice with two labels. */
    std::vector<std::int32_t> label_ids()
    {
        return {3, 3, 1, 4, 2, 0};
    }

    /** @brief Returns the [V, V] table whose row r holds 0.3 r - 0.2 c + 0.1 r c. */
    std::vector<double> make_table()
    {
        std::vector<double> table;
        table.reserve(static_cast<std::size_t>(K_VOCABULARY * K_VOCABULARY));
        for (std::int64_t row = 0; row < K_VOCABULARY; ++row)
        {
            for (std::int64_t column = 0; column < K_VOCABULARY; ++column)
            {
                table.push_back(0.3 * static_cast<double>(row)
                                - 0.2 * static_cast<double>(column)
                                + 0.1 * static_cast<double>(row * column));
            }
        }
        return table;
    }

    /** @brief Holds the loss and its gradient with respect to the table, worked on the host. */
    struct Expected
    {
        double loss = 0.0;            /**< Mean of -log p(label) over the positions. */
        std::vector<double> gradient; /**< [V, V]: (softmax - one_hot) / (N T), summed by row. */
    };

    /** @brief Returns the loss and the gradient of a model whose logits are a table row. */
    Expected work_on_host(const std::vector<double>& table)
    {
        const std::size_t width = static_cast<std::size_t>(K_VOCABULARY);
        const std::vector<std::int32_t> tokens = token_ids();
        const std::vector<std::int32_t> labels = label_ids();
        const double positions = static_cast<double>(tokens.size());

        Expected expected;
        expected.gradient.assign(width * width, 0.0);
        for (std::size_t position = 0; position < tokens.size(); ++position)
        {
            const std::size_t row = static_cast<std::size_t>(tokens[position]);
            const std::size_t label = static_cast<std::size_t>(labels[position]);

            double total = 0.0;
            for (std::size_t column = 0; column < width; ++column)
                total += std::exp(table[row * width + column]);

            expected.loss -= (table[row * width + label] - std::log(total)) / positions;
            for (std::size_t column = 0; column < width; ++column)
            {
                const double probability = std::exp(table[row * width + column]) / total;
                const double hit = column == label ? 1.0 : 0.0;
                expected.gradient[row * width + column] += (probability - hit) / positions;
            }
        }
        return expected;
    }

    /** @brief Writes @p values into the INT32 tensor @p tensor, element by element. */
    void write_int32(SushiBLAS::Tensor& tensor, const std::vector<std::int32_t>& values)
    {
        ASSERT_EQ(static_cast<std::size_t>(tensor.num_elements), values.size());
        std::copy(values.begin(), values.end(), tensor.data_as<std::int32_t>());
    }

    /** @brief Expects @p found to equal @p expected element by element within @p tolerance. */
    void expect_all_near(const std::vector<double>& found, const std::vector<double>& expected,
                         double tolerance)
    {
        ASSERT_EQ(found.size(), expected.size());
        for (std::size_t i = 0; i < found.size(); ++i)
            EXPECT_NEAR(found[i], expected[i], tolerance) << "element " << i;
    }

    /** @brief Fills the INT32 leaf @p targets from the labels @p labels. */
    Status fill_targets(const std::vector<std::int32_t>& labels, SushiBLAS::Tensor& targets)
    {
        return Language::Training::fill_next_token_targets(
            SushiRuntime::span<const std::int32_t>(labels.data(), labels.size()), K_VOCABULARY,
            SushiRuntime::span<std::int32_t>(targets.data_as<std::int32_t>(),
                                             static_cast<std::size_t>(targets.num_elements)));
    }

    /** @brief Provides the engine the state and the step are built on. */
    class Medium_NextTokenObjective : public SaTest::EngineTest
    {

    };
} // namespace

/** @brief Tests the loss and the table's gradient of one step against the host's numbers. */
TEST_F(Medium_NextTokenObjective, TheLossAndItsGradientEqualTheHostComputation)
{
    NN::Embedding model = make_bigram_model();

    SA_ASSERT_OK_AND_ASSIGN(
        const std::unique_ptr<Language::Training::NextTokenObjective> objective,
        Language::Training::NextTokenObjective::make(K_BATCH, K_LENGTH, K_VOCABULARY));
    Train::GraphOptions options;
    options.dtype = DType::Int32;
    SA_ASSERT_OK_AND_ASSIGN(Train::TrainingGraph built,
                            Train::build_training_graph(model, *objective,
                                                        Shape{K_BATCH, K_LENGTH}, options));

    SA_ASSERT_OK_AND_ASSIGN(NN::ModelState state, NN::ModelState::create(*engine, model));
    Graph::TensorPool pool(*engine);
    SA_ASSERT_OK_AND_ASSIGN(std::vector<Optim::ParameterSlot> slots,
                            Train::bind_parameters(backend(), built, state, pool));
    ASSERT_EQ(slots.size(), 1u);
    const SushiBLAS::Tensor gradient = slots.front().gradient;

    const std::vector<double> table = make_table();
    SA_ASSERT_OK_AND_ASSIGN(SushiBLAS::Tensor* const weight, state.tensor("weight"));
    SaTest::write_tensor(*weight, SushiRuntime::span<const double>(table.data(), table.size()));

    // A rate of zero leaves the table as written, so the step can be read after it ran.
    Optim::SGDOptions sgd;
    sgd.learning_rate = 0.0;
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<Optim::SGD> optimizer,
                            Optim::SGD::make(backend(), slots, sgd));
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<Train::TrainingStep> step,
                            Train::TrainingStep::make(backend(), built.graph, pool, *optimizer,
                                                      {}));

    write_int32(pool.tensor(built.spec.inputs), token_ids());
    SA_EXPECT_OK(fill_targets(label_ids(), pool.tensor(built.spec.targets)));

    SA_EXPECT_OK(step->run());

    const Expected expected = work_on_host(table);
    EXPECT_NEAR(static_cast<double>(pool.tensor(built.spec.loss).data_as<float>()[0]),
                expected.loss, K_TOLERANCE);
    expect_all_near(SaTest::read_tensor(gradient), expected.gradient, K_TOLERANCE);
}

namespace
{

    /** @brief Holds the passes fit() makes over the one batch. */
    constexpr std::int64_t K_EPOCHS = 30;

    /**
     * @brief Holds the absolute Bounds the difference between exp of a float32 loss and the
     *        reported perplexity: 16777216 times the double unit roundoff, the smallest power of
     *        two not tighter than the earlier 1e-9.
     */
    constexpr double K_TRAINER_TOLERANCE = 16777216.0 * TestSupport::unit_roundoff<double>();

    /** @brief Returns the description of [T] int32 ids per sample. */
    Data::InputDescription token_input()
    {
        return Data::InputDescription{Shape{K_LENGTH}, DType::Int32};
    }

    /** @brief Returns the labels role as [T] int32 ids per sample over @p vocabulary tokens. */
    std::vector<Data::TargetDescription> label_roles(std::int64_t vocabulary)
    {
        return {Data::TargetDescription{std::string(Data::K_ROLE_LABELS), Shape{K_LENGTH},
                                        DType::Int32, vocabulary}};
    }

    /** @brief Returns a dataset of one batch in which each token is followed by one label. */
    Result<Data::MaterialisedDataset> make_dataset(SushiBLAS::Engine& engine,
                                                   const std::vector<std::int32_t>& labels)
    {
        SA_TRY_ASSIGN(Data::Batch batch,
                      Data::allocate_batch(engine, K_BATCH, token_input(),
                                           label_roles(K_VOCABULARY)));

        const std::vector<std::int32_t> tokens = {0, 3, 1, 1, 4, 2};
        for (std::size_t i = 0; i < tokens.size(); ++i)
        {
            batch.inputs.data_as<std::int32_t>()[i] = tokens[i];
            batch.targets.front().values.data_as<std::int32_t>()[i] = labels[i];
        }

        std::vector<Data::Batch> batches;
        batches.push_back(std::move(batch));
        return Data::MaterialisedDataset(std::move(batches), K_BATCH, token_input(),
                                         label_roles(K_VOCABULARY), "tokens");
    }

    /** @brief Provides the engine the state, the dataset and the step are built on. */
    class Medium_NextTokenTrainer : public SaTest::EngineTest
    {
        protected:
            /**
             * @brief Fits a bigram table to @p labels for K_EPOCHS passes.
             * @return The report of the run, or the first failure.
             */
            [[nodiscard]] Result<Train::RunReport> fit_bigram(
                const std::vector<std::int32_t>& labels)
            {
                NN::Embedding model = make_bigram_model();

                SA_TRY_ASSIGN(
                    const std::unique_ptr<Language::Training::NextTokenObjective> objective,
                    Language::Training::NextTokenObjective::make(K_BATCH, K_LENGTH,
                                                                 K_VOCABULARY));
                Train::GraphOptions graph_options;
                graph_options.dtype = DType::Int32;
                SA_TRY_ASSIGN(Train::TrainingGraph built,
                              Train::build_training_graph(model, *objective,
                                                          Shape{K_BATCH, K_LENGTH},
                                                          graph_options));

                SA_TRY_ASSIGN(NN::ModelState state, NN::ModelState::create(*engine, model));
                Graph::TensorPool pool(*engine);
                SA_TRY_ASSIGN(std::vector<Optim::ParameterSlot> slots,
                              Train::bind_parameters(backend(), built, state, pool));
                SA_TRY(NN::initialize(backend(), state));

                Optim::SGDOptions sgd;
                sgd.learning_rate = 0.5;
                SA_TRY_ASSIGN(const std::unique_ptr<Optim::SGD> optimizer,
                              Optim::SGD::make(backend(), slots, sgd));

                Train::TrainerOptions options;
                options.epochs = K_EPOCHS;
                SA_TRY_ASSIGN(const std::unique_ptr<Train::Trainer> trainer,
                              Train::Trainer::make(backend(), built, pool, *objective, *optimizer,
                                                   options));

                SA_TRY_ASSIGN(const Data::MaterialisedDataset dataset,
                              make_dataset(*engine, labels));
                return trainer->fit(dataset);
            }
    };
} // namespace

/** @brief Tests that fit lowers the next-token loss and reports its perplexity. */
TEST_F(Medium_NextTokenTrainer, FitLowersTheLossAndReportsPerplexity)
{
    SA_ASSERT_OK_AND_ASSIGN(const Train::RunReport report, fit_bigram({3, 1, 1, 4, 2, 0}));

    EXPECT_EQ(report.steps, static_cast<std::size_t>(K_EPOCHS));
    EXPECT_LT(report.last_loss, report.first_loss) << report.to_string();

    ASSERT_EQ(report.last_metrics.all().size(), 1u);
    const Train::Metric* const perplexity = report.last_metrics.find("perplexity");
    ASSERT_NE(perplexity, nullptr);
    EXPECT_FALSE(perplexity->is_fraction);
    EXPECT_NEAR(perplexity->value, std::exp(report.last_loss), K_TRAINER_TOLERANCE);
}

/** @brief Tests that a label outside the vocabulary stops the first step and names its row. */
TEST_F(Medium_NextTokenTrainer, ALabelOutsideTheVocabularyStopsTheFirstStep)
{
    SA_EXPECT_FAILURE(fit_bigram({3, 1, 1, 9, 2, 0}), StatusCode::Incompatible, "position 3",
                      "is 9", "holds 5 tokens");
}

namespace
{
    /** @brief Holds the tokens the model scores, V. */
    constexpr std::int64_t K_OVERFIT_VOCABULARY = 64;

    /** @brief Holds the rows of the one batch, N. */
    constexpr std::int64_t K_OVERFIT_BATCH = 4;

    /** @brief Holds the positions of a row, T. */
    constexpr std::int64_t K_OVERFIT_LENGTH = 16;

    /** @brief Holds the width of the model. */
    constexpr std::int64_t K_DIM = 32;

    /** @brief Holds the heads of each attention. */
    constexpr std::int64_t K_HEADS = 2;

    /** @brief Holds the width between the two linear layers of a feed-forward. */
    constexpr std::int64_t K_HIDDEN = 64;

    /** @brief Holds how long the one batch is trained and how far the loss must have fallen. */
    struct OverfitRun
    {
        std::int64_t steps;          /**< The updates the one batch is trained for. */
        double remaining_fraction;   /**< The fraction of the first loss the last one must be below. */
    };

    /** @brief Holds the run that overfits: 400 updates, the last loss below a tenth of the first. */
    constexpr OverfitRun K_LONG_RUN{400, 0.1};

    /**
     * @brief Holds the run that checks the direction of the overfit in a `medium` test: 40 updates,
     *        the last loss below nine tenths of the first.
     *
     * An update takes about 75 ms on the debug build, so 5 s holds 66 at most; 40 leaves room for
     * the graph build. The first loss is near ln(64), 4.2, so 0.9 of it is a fall of 0.4.
     */
    constexpr OverfitRun K_SHORT_RUN{40, 0.9};

    /**
     * @brief Bounds how far the first loss may lie from ln(V), the loss of a uniform guess;
     *        absolute, a loss margin and not a rounding bound.
     */
    constexpr double K_FIRST_LOSS_TOLERANCE = 0.5;

    /** @brief Holds the norm every update's gradients are clipped to. */
    constexpr double K_CLIP_NORM = 1.0;

    /** @brief Holds a decoder of two blocks whose head scores with the token table. */
    struct Decoder
    {
        /** @brief Names the decoder in a test name. */
        static constexpr const char* K_NAME = "Learned";

        /** @brief Names the token file the decoder's test writes. */
        static constexpr const char* K_FILE = "sushiai_decoder_overfit.satk";

        /** @brief Holds the [V, dim] token table, which the head is tied to. */
        NN::Embedding tokens{};

        /** @brief Holds the learned position table. */
        Language::Blocks::PositionEmbedding positions;

        /** @brief Holds the first block. */
        Language::Blocks::TransformerBlock first;

        /** @brief Holds the second block. */
        Language::Blocks::TransformerBlock second;

        /** @brief Holds the final normalisation and the tied scores. */
        Language::Blocks::LanguageHead head;

        /** @brief Builds the five children and ties the head to the token table. */
        Decoder()
            : positions(K_OVERFIT_LENGTH, K_DIM),
              first(K_DIM, K_HEADS, K_HIDDEN, Language::Blocks::AttentionMask::Causal,
                    NN::K_LAYER_NORM_EPSILON),
              second(K_DIM, K_HEADS, K_HIDDEN, Language::Blocks::AttentionMask::Causal,
                     NN::K_LAYER_NORM_EPSILON),
              head(K_DIM, K_OVERFIT_VOCABULARY, NN::K_LAYER_NORM_EPSILON)
        {
            tokens.num_embeddings = K_OVERFIT_VOCABULARY;
            tokens.embedding_dim = K_DIM;
            head.tie_to(tokens.weight);
        }

        /** @brief Forbids a copy, whose head would stay tied to this object's table. */
        Decoder(const Decoder&) = delete;

        /** @brief Forbids copy assignment, for the same reason. */
        Decoder& operator=(const Decoder&) = delete;

        /** @brief Lists the five children in the order they are traced. */
        static constexpr auto fields()
        {
            return std::tuple{
                NN::child("tokens", &Decoder::tokens),
                NN::child("positions", &Decoder::positions),
                NN::child("first", &Decoder::first),
                NN::child("second", &Decoder::second),
                NN::child("head", &Decoder::head),
            };
        }

        /** @brief Traces [N, T] ids to [N, T, V] logits. */
        Graph::ValueID forward(const NN::Trace& trace, Graph::ValueID ids)
        {
            Graph::ValueID x = NN::trace_layer(tokens, trace, ids);
            x = NN::trace_layer(positions, trace, x);
            x = NN::trace_layer(first, trace, x);
            x = NN::trace_layer(second, trace, x);
            return NN::trace_layer(head, trace, x);
        }
    };

    /** @brief Holds a decoder of two rotary blocks whose head scores with the token table. */
    struct RotaryDecoder
    {
        /** @brief Names the decoder in a test name. */
        static constexpr const char* K_NAME = "Rotary";

        /** @brief Names the token file the decoder's test writes. */
        static constexpr const char* K_FILE = "sushiai_rotary_decoder_overfit.satk";

        /** @brief Holds the [V, dim] token table, which the head is tied to. */
        NN::Embedding tokens{};

        /** @brief Holds the first block. */
        Language::Blocks::TransformerBlock first;

        /** @brief Holds the second block. */
        Language::Blocks::TransformerBlock second;

        /** @brief Holds the final normalisation and the tied scores. */
        Language::Blocks::LanguageHead head;

        /** @brief Builds the four children with rotary attentions and ties the head. */
        RotaryDecoder()
            : first(K_DIM, K_HEADS, K_HIDDEN, Language::Blocks::AttentionMask::Causal,
                    NN::K_LAYER_NORM_EPSILON, Language::Blocks::AttentionPositions::Rotary),
              second(K_DIM, K_HEADS, K_HIDDEN, Language::Blocks::AttentionMask::Causal,
                     NN::K_LAYER_NORM_EPSILON, Language::Blocks::AttentionPositions::Rotary),
              head(K_DIM, K_OVERFIT_VOCABULARY, NN::K_LAYER_NORM_EPSILON)
        {
            tokens.num_embeddings = K_OVERFIT_VOCABULARY;
            tokens.embedding_dim = K_DIM;
            head.tie_to(tokens.weight);
        }

        /** @brief Forbids a copy, whose head would stay tied to this object's table. */
        RotaryDecoder(const RotaryDecoder&) = delete;

        /** @brief Forbids copy assignment, for the same reason. */
        RotaryDecoder& operator=(const RotaryDecoder&) = delete;

        /** @brief Lists the four children in the order they are traced. */
        static constexpr auto fields()
        {
            return std::tuple{
                NN::child("tokens", &RotaryDecoder::tokens),
                NN::child("first", &RotaryDecoder::first),
                NN::child("second", &RotaryDecoder::second),
                NN::child("head", &RotaryDecoder::head),
            };
        }

        /** @brief Traces [N, T] ids to [N, T, V] logits. */
        Graph::ValueID forward(const NN::Trace& trace, Graph::ValueID ids)
        {
            Graph::ValueID x = NN::trace_layer(tokens, trace, ids);
            x = NN::trace_layer(first, trace, x);
            x = NN::trace_layer(second, trace, x);
            return NN::trace_layer(head, trace, x);
        }
    };

    /** @brief Names a decoder in the name of its typed case. */
    struct DecoderName
    {
        /** @brief Returns the name of the decoder type @p Model. */
        template <typename Model>
        static std::string GetName(int)
        {
            return Model::K_NAME;
        }
    };

    /** @brief Holds what training one batch measured. */
    struct Overfit
    {
        double first_loss = 0.0;             /**< The loss of the first update. */
        double last_loss = 0.0;              /**< The loss of the last update. */
        std::size_t compile_count = 0;       /**< The compilations of the step. */
        std::uint64_t optimizer_steps = 0;   /**< The updates the optimizer made. */
    };

    /**
     * @brief Returns K_OVERFIT_BATCH windows of K_OVERFIT_LENGTH + 1 ids, id i being 7 i^2 + 3 i +
     *        1 mod V.
     */
    std::vector<Language::Text::TokenID> make_ids()
    {
        const std::int64_t count = K_OVERFIT_BATCH * (K_OVERFIT_LENGTH + 1);
        std::vector<Language::Text::TokenID> ids;
        ids.reserve(static_cast<std::size_t>(count));
        for (std::int64_t i = 0; i < count; ++i)
            ids.push_back(static_cast<Language::Text::TokenID>((7 * i * i + 3 * i + 1)
                                                               % K_OVERFIT_VOCABULARY));
        return ids;
    }

    /**
     * @brief Returns the cosine schedule of the overfit: a warm-up of a twentieth of the run, 20
     *        of the 400 updates of the long run, then a decay.
     */
    Optim::CosineSchedule make_schedule(std::int64_t steps)
    {
        Optim::CosineSchedule schedule;
        schedule.base_rate = 3e-3;
        schedule.warmup_iterations = steps / 20;
        schedule.total_iterations = steps;
        schedule.final_rate = 3e-4;
        return schedule;
    }

    /**
     * @brief Writes the token file at @p path and reads it back as a dataset of one batch.
     * @return The dataset, or the first failure.
     */
    Result<std::unique_ptr<Language::Datasets::TokenDataset>> make_token_dataset(
        SushiBLAS::Engine& engine, const std::filesystem::path& path)
    {
        const std::vector<Language::Text::TokenID> ids = make_ids();
        SA_TRY(Language::Datasets::write_token_file(
            path, SushiRuntime::span<const Language::Text::TokenID>(ids.data(), ids.size()),
            static_cast<std::int32_t>(K_OVERFIT_VOCABULARY)));

        SA_TRY_ASSIGN(
            std::unique_ptr<Language::Datasets::NextTokenCollator> collator,
            Language::Datasets::NextTokenCollator::make(
                K_OVERFIT_LENGTH, static_cast<std::int32_t>(K_OVERFIT_VOCABULARY)));
        Language::Datasets::TokenDatasetOptions dataset_options;
        dataset_options.batch_size = K_OVERFIT_BATCH;
        dataset_options.stride = K_OVERFIT_LENGTH + 1;
        return Language::Datasets::TokenDataset::make(engine, path, std::move(collator),
                                                      dataset_options);
    }

    /**
     * @brief Feeds the one batch into the leaves of @p built once.
     * @pre No task is recorded yet, so the host may write the leaves.
     */
    Status feed_once(const Data::Batch& batch, const Train::TrainingGraph& built,
                     Graph::TensorPool& pool)
    {
        SushiBLAS::Tensor& inputs = pool.tensor(built.spec.inputs);
        SA_TRY(Train::check_feed(batch.inputs, inputs, "inputs"));
        const std::int32_t* const fed = batch.inputs.data_as<std::int32_t>();
        std::copy(fed, fed + inputs.num_elements, inputs.data_as<std::int32_t>());

        const Data::BatchTarget* const labels =
            Data::find_target(batch.targets, Data::K_ROLE_LABELS);
        SA_FAIL_IF(labels == nullptr, StatusCode::InvalidDescription,
                   "The batch holds no labels role");
        return Language::Training::fill_next_token_targets(labels->values, K_OVERFIT_VOCABULARY,
                                                           pool.tensor(built.spec.targets));
    }

    /**
     * @brief Runs the @p steps updates of the schedule and returns the first and last loss.
     * @return The losses, or DeviceFailed for an update whose loss is not finite.
     */
    Result<Overfit> run_updates(Train::TrainingStep& step, Optim::AdamW& optimizer,
                                const Optim::CosineSchedule& schedule, std::int64_t steps,
                                const Train::TrainingGraph& built, Graph::TensorPool& pool)
    {
        Overfit result;
        for (std::int64_t update = 0; update < steps; ++update)
        {
            Optim::apply_step_schedule(schedule, optimizer);
            SA_TRY(step.run());

            result.last_loss =
                static_cast<double>(pool.tensor(built.spec.loss).data_as<float>()[0]);
            SA_FAIL_IF(!std::isfinite(result.last_loss), StatusCode::DeviceFailed,
                       "The loss of update {} is not finite", update);
            if (update == 0)
                result.first_loss = result.last_loss;
        }
        return result;
    }

    /** @brief Provides the engine the dataset, the state and the step of a decoder are built on. */
    template <typename Model>
    class DecoderOverfit : public SaTest::EngineTest
    {
        protected:
            /**
             * @brief Trains a decoder of type Model on one fixed batch for the updates of @p run.
             * @return What the run measured, or the first failure.
             */
            [[nodiscard]] Result<Overfit> train_one_batch(const OverfitRun& run)
            {
                const SaTest::ScratchFile file(SushiAI::Test::unique_scratch_name(Model::K_FILE).c_str());
                SA_TRY_ASSIGN(const std::unique_ptr<Language::Datasets::TokenDataset> dataset,
                              make_token_dataset(*this->engine, file.path()));
                SA_FAIL_IF(dataset->batch_count() != 1u, StatusCode::Incompatible,
                           "The token file holds {} batches, not one", dataset->batch_count());

                Model model;
                SA_TRY_ASSIGN(
                    const std::unique_ptr<Language::Training::NextTokenObjective> objective,
                    Language::Training::NextTokenObjective::make(
                        K_OVERFIT_BATCH, K_OVERFIT_LENGTH, K_OVERFIT_VOCABULARY));
                Train::GraphOptions graph_options;
                graph_options.dtype = DType::Int32;
                SA_TRY_ASSIGN(Train::TrainingGraph built,
                              Train::build_training_graph(
                                  model, *objective, Shape{K_OVERFIT_BATCH, K_OVERFIT_LENGTH},
                                  graph_options));

                SA_TRY_ASSIGN(NN::ModelState state,
                              NN::ModelState::create(*this->engine, model));
                Graph::TensorPool pool(*this->engine);
                SA_TRY_ASSIGN(std::vector<Optim::ParameterSlot> slots,
                              Train::bind_parameters(this->backend(), built, state, pool));

                NN::InitPolicy policy;
                policy.family = NN::InitFamily::Xavier;
                policy.nonlinearity = NN::Nonlinearity::Linear;
                policy.zero_final_layer = false;
                SA_TRY(NN::initialize(this->backend(), state, policy));

                Optim::AdamWOptions optimizer_options;
                optimizer_options.weight_decay = 0.01;
                SA_TRY_ASSIGN(const std::unique_ptr<Optim::AdamW> optimizer,
                              Optim::AdamW::make(this->backend(), slots, optimizer_options));
                SA_TRY(optimizer->enable_gradient_clip(K_CLIP_NORM));

                SA_TRY_ASSIGN(const std::unique_ptr<Train::TrainingStep> step,
                              Train::TrainingStep::make(this->backend(), built.graph, pool,
                                                        *optimizer, {}));

                Data::Batch batch;
                SA_TRY(dataset->read(0, batch));
                SA_TRY(feed_once(batch, built, pool));

                SA_TRY_ASSIGN(Overfit result, run_updates(*step, *optimizer, make_schedule(run.steps),
                                                          run.steps, built, pool));
                result.compile_count = step->step().compile_count();
                result.optimizer_steps = optimizer->step_count();
                return result;
            }
    };

    /** @brief Provides the short run of each decoder. */
    template <typename Model>
    class Medium_DecoderOverfit : public DecoderOverfit<Model>
    {

    };

    /** @brief Provides the long run of each decoder. */
    template <typename Model>
    class Large_DecoderOverfit : public DecoderOverfit<Model>
    {

    };

    using Decoders = ::testing::Types<Decoder, RotaryDecoder>;

    /** @brief Expects the losses of @p result to hold what a run of @p run promises. */
    void expect_overfit(const Overfit& result, const OverfitRun& run)
    {
        EXPECT_NEAR(result.first_loss, std::log(static_cast<double>(K_OVERFIT_VOCABULARY)),
                    K_FIRST_LOSS_TOLERANCE);
        EXPECT_LT(result.last_loss, run.remaining_fraction * result.first_loss);
        EXPECT_LT(Language::Training::perplexity(result.last_loss),
                  Language::Training::perplexity(result.first_loss));
        EXPECT_EQ(result.compile_count, 1u);
        EXPECT_EQ(result.optimizer_steps, static_cast<std::uint64_t>(run.steps));
    }
} // namespace

TYPED_TEST_SUITE(Medium_DecoderOverfit, Decoders, DecoderName);
TYPED_TEST_SUITE(Large_DecoderOverfit, Decoders, DecoderName);

/** @brief Tests that the loss of one fixed batch falls below nine tenths of its start in 40 updates. */
TYPED_TEST(Medium_DecoderOverfit, ATwoBlockDecoderLowersTheLossOfOneBatch)
{
    SA_ASSERT_OK_AND_ASSIGN(const Overfit result, this->train_one_batch(K_SHORT_RUN));

    expect_overfit(result, K_SHORT_RUN);
}

/** @brief Tests that the loss of one fixed batch falls below a tenth of its start in 400 updates. */
TYPED_TEST(Large_DecoderOverfit, ATwoBlockDecoderOverfitsOneBatch)
{
    SA_ASSERT_OK_AND_ASSIGN(const Overfit result, this->train_one_batch(K_LONG_RUN));

    expect_overfit(result, K_LONG_RUN);
}

namespace
{
    /** @brief Names the suite of the cases that expect the process to stop. */
    class Medium_NextTokenObjectiveDeathTest : public ::testing::Test {};
}

/** @brief Tests that logits other than the declared [N, T, V] stop the process, naming both. */
TEST_F(Medium_NextTokenObjectiveDeathTest, OutputsOtherThanTheDeclaredShapeStopTheProcess)
{
    NN::Embedding model = make_bigram_model();
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<NextTokenObjective> objective,
                            NextTokenObjective::make(K_BATCH, K_LENGTH, K_VOCABULARY));
    Train::GraphOptions options;
    options.dtype = DType::Int32;

    // One position more than the objective was made for: the logits are [2, 4, 5], not [2, 3, 5].
    SA_EXPECT_FATAL(
        (void)Train::build_training_graph(model, *objective, Shape{K_BATCH, K_LENGTH + 1}, options),
        "The model produced", "the next-token objective was made for");
}
