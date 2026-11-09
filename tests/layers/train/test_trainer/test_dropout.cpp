/**************************************************************************/
/* test_dropout.cpp                                                       */
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
 * @file test_dropout.cpp
 * @brief Checks DROPOUT on an engine: the mask and its output, the seeds, the loss over the masked
 *        value, and a model with dropout trained through the Trainer.
 * @author Mustafa Garip
 */

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <numeric>
#include <utility>
#include <vector>
#include <gtest/gtest.h>
#include <SushiBLAS/engine/engine.hpp>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/dtype/dtype.hpp>
#include <SushiAI/ops/ops.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/shape/shape.hpp>
#include <SushiAI/data/dataset.hpp>
#include <SushiAI/engine_fixture/engine_test.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/graph/builder.hpp>
#include <SushiAI/graph/compiled_step.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/graph/tensor_pool.hpp>
#include <SushiAI/nn/dropout.hpp>
#include <SushiAI/nn/init.hpp>
#include <SushiAI/nn/layers.hpp>
#include <SushiAI/nn/model_state.hpp>
#include <SushiAI/optim/adamw.hpp>
#include <SushiAI/optim/optimizer.hpp>
#include <SushiAI/seeded_noise/host_noise.hpp>
#include <SushiAI/tensor_values/tensor_values.hpp>
#include <SushiAI/train/objective.hpp>
#include <SushiAI/train/steps.hpp>
#include <SushiAI/train/trainer.hpp>
#include <SushiAI/train/training_graph.hpp>
#include <support/tolerance.hpp>

using namespace SushiAI;
// SaTest stays reachable inside a TEST_F body; see tests/layers/train/README.md.
namespace SaTest = SushiAI::Test;
using Graph::CompiledStep;
using Graph::GraphBuilder;
using Graph::TensorPool;
using Graph::ValueID;
using SaTest::read_tensor;
using SaTest::read_tensor_bytes;
using SaTest::write_tensor;

namespace
{
    /** @brief Holds the row count of the masked value; a power of two, so 1 / K_ROWS is exact. */
    constexpr int64_t K_ROWS = 64;

    /** @brief Holds the row length of the masked value. */
    constexpr int64_t K_WIDTH = 64;

    /** @brief Holds the element count of the masked value. */
    constexpr std::size_t K_ELEMENTS = static_cast<std::size_t>(K_ROWS * K_WIDTH);

    /** @brief Holds the rate of every probe; its keep of 0.75 and scale of 4/3 follow. */
    constexpr double K_RATE = 0.25;

    /** @brief Holds the probability that an element is kept. */
    constexpr double K_KEEP = 1.0 - K_RATE;

    /** @brief Holds the value a kept element of the mask carries, as the device stores it. */
    constexpr double K_SCALE = static_cast<double>(static_cast<float>(1.0 / K_KEEP));

    /**
     * @brief Holds the number of standard deviations of the kept fraction that the kept fraction
     *        may stray from K_KEEP; a draw outside four deviations fails one run in 15,000.
     */
    constexpr double K_SPREADS = 4.0;

    /** @brief Holds the site seed of every probe that does not state another. */
    constexpr std::uint64_t K_SEED = 0xE220A8397B1DCDAFull;

    /** @brief Holds the seed of the values written into the masked leaf. */
    constexpr std::uint64_t K_DATA_SEED = 20261011ull;

    /**
     * @brief Holds a compiled graph that masks a [K_ROWS, K_WIDTH] parameter and takes the sum of
     *        the masked elements as its loss.
     *
     * The loss is NLL_LOSS over the masked value itself with every target at -K_ROWS, which is
     * the sum of the masked elements.
     */
    class Probe
    {
        public:
            /** @brief Builds and compiles the graph with the site seed @p seed. */
            Probe(SushiAI::ExecutionBackend& backend, std::uint64_t seed) : pool_(backend.engine())
            {
                const GraphBuilder builder(graph_);
                const Shape shape{K_ROWS, K_WIDTH};
                x_ = builder.parameter(shape, "x");

                const Result<Graph::DropoutResult> dropped = builder.dropout(x_, K_RATE, seed);
                SA_EXPECT_OK(dropped);
                if (!dropped.has_value())
                    return;

                y_ = dropped.value().output;
                mask_ = dropped.value().saved_mask;

                const ValueID targets = builder.input(shape, "targets");
                // The extents are fixed here, so a refusal is a defect: value() stops the test.
                loss_ = builder.nll_loss(y_, targets).value();

                SA_EXPECT_OK(pool_.materialise_leaves(graph_));

                Result<std::unique_ptr<CompiledStep>> compiled =
                    CompiledStep::compile(backend, graph_, pool_);
                SA_EXPECT_OK(compiled);
                if (!compiled.has_value())
                    return;

                step_ = std::move(compiled).value();

                // Every element lies in [1, 3], so a kept one is never zero.
                std::vector<double> values(K_ELEMENTS);
                SaTest::HostNoise(K_DATA_SEED).fill(values, 1.0);
                std::transform(values.begin(), values.end(), values.begin(),
                               [](double value) { return value + 2.0; });

                write_tensor(pool_.tensor(x_), values);
                write_tensor(pool_.tensor(targets),
                             std::vector<double>(K_ELEMENTS, -static_cast<double>(K_ROWS)));
            }

            /** @brief Returns whether the graph compiled. */
            [[nodiscard]] bool ready() const noexcept
            {
                return step_ != nullptr;
            }

            /** @brief Runs the step once. */
            void run()
            {
                SA_EXPECT_OK(step_->run());
            }

            /** @brief Returns the values of the masked leaf. */
            [[nodiscard]] std::vector<double> x() const
            {
                return read_tensor(pool_.tensor(x_));
            }

            /** @brief Returns the masked value of the last run. */
            [[nodiscard]] std::vector<double> y() const
            {
                return read_tensor(pool_.tensor(y_));
            }

            /** @brief Returns the bytes of the masked value of the last run. */
            [[nodiscard]] std::vector<unsigned char> y_bytes() const
            {
                return read_tensor_bytes(pool_.tensor(y_));
            }

            /** @brief Returns the mask the last run saved. */
            [[nodiscard]] std::vector<double> mask() const
            {
                return read_tensor(pool_.tensor(mask_));
            }

            /** @brief Returns the loss of the last run. */
            [[nodiscard]] double loss() const
            {
                return read_tensor(pool_.tensor(loss_))[0];
            }

        private:
            /** @brief Holds the forward nodes. */
            Graph::Graph graph_;

            /** @brief Holds the tensor of each value. */
            TensorPool pool_;

            /** @brief Holds the compiled step, or null when a stage failed. */
            std::unique_ptr<CompiledStep> step_;

            /** @brief Names the masked leaf. */
            ValueID x_ = Graph::K_NO_VALUE;

            /** @brief Names the masked value. */
            ValueID y_ = Graph::K_NO_VALUE;

            /** @brief Names the mask the forward node saves. */
            ValueID mask_ = Graph::K_NO_VALUE;

            /** @brief Names the one-element loss. */
            ValueID loss_ = Graph::K_NO_VALUE;
    };

    /** @brief Returns how many elements of @p values are not zero. */
    std::size_t nonzero_count(const std::vector<double>& values)
    {
        return static_cast<std::size_t>(
            std::count_if(values.begin(), values.end(), [](double value) { return value != 0.0; }));
    }

    /** @brief Returns whether every element of @p mask is 0 or K_SCALE. */
    bool is_zero_or_scale(const std::vector<double>& mask)
    {
        return std::all_of(mask.begin(), mask.end(),
                           [](double value) { return value == 0.0 || value == K_SCALE; });
    }

    /** @brief Returns the largest relative distance of @p y from x times @p mask, over its kept elements. */
    double worst_product_error(const std::vector<double>& x, const std::vector<double>& mask,
                               const std::vector<double>& y)
    {
        double worst = 0.0;
        for (std::size_t element = 0; element < x.size(); ++element)
        {
            const double product = x[element] * mask[element];
            const double error = std::abs(y[element] - product);
            worst = std::max(worst, product == 0.0 ? error : error / std::abs(product));
        }

        return worst;
    }

    /** @brief Returns how many nodes of @p graph carry DROPOUT. */
    std::size_t dropout_count(const Graph::Graph& graph)
    {
        return static_cast<std::size_t>(std::count_if(
            graph.nodes().begin(), graph.nodes().end(),
            [](const Graph::Node& node) { return node.op == Ops::K_DROPOUT.id; }));
    }

    /** @brief Holds the samples per step of the model cases. */
    constexpr int64_t K_BATCH = 16;

    /** @brief Holds the features per sample of the model cases. */
    constexpr int64_t K_FEATURES = 8;

    /** @brief Holds the hidden width of the model cases. */
    constexpr int64_t K_HIDDEN = 32;

    /** @brief Holds the class count of the model cases. */
    constexpr int64_t K_CLASSES = 4;

    /** @brief Holds the drop rate of the model cases. */
    constexpr double K_MODEL_RATE = 0.2;

    /** @brief Holds the seed the training graph of the model cases states. */
    constexpr std::uint64_t K_GRAPH_SEED = 7;

    /** @brief Names a classifier with one dropout between its two linear layers. */
    using DropoutMLP = NN::Sequential<NN::Linear, NN::ReLU, NN::Dropout, NN::Linear>;

    /** @brief Builds the classifier of the model cases. */
    DropoutMLP make_model()
    {
        return DropoutMLP(NN::Linear{K_FEATURES, K_HIDDEN}, NN::ReLU{},
                          NN::Dropout{K_MODEL_RATE}, NN::Linear{K_HIDDEN, K_CLASSES});
    }

    /** @brief Provides the Engine every dropout case records on. */
    class Medium_Dropout : public SaTest::EngineTest {};

    /** @brief Provides two probes of one seed and a third of the next, each compiled once. */
    class Medium_DropoutSeeds : public SaTest::EngineTest
    {
        protected:
            /** @brief Compiles the three probes. */
            void SetUp() override
            {
                SaTest::EngineTest::SetUp();
                first = std::make_unique<Probe>(backend(), K_SEED);
                second = std::make_unique<Probe>(backend(), K_SEED);
                other = std::make_unique<Probe>(backend(), K_SEED + 1);
            }

            /** @brief Runs the two probes of one seed once and expects the same output bytes. */
            void expect_same_seed_agrees()
            {
                first->run();
                second->run();
                EXPECT_EQ(first->y_bytes(), second->y_bytes());
            }

            /** @brief Runs the probes of two seeds once and expects different output bytes. */
            void expect_other_seed_differs()
            {
                first->run();
                other->run();
                EXPECT_NE(first->y_bytes(), other->y_bytes());
            }

            std::unique_ptr<Probe> first;   /**< @brief Holds the seed K_SEED. */
            std::unique_ptr<Probe> second;  /**< @brief Holds the seed K_SEED again. */
            std::unique_ptr<Probe> other;   /**< @brief Holds the seed K_SEED + 1. */
    };
}

/** @brief Tests that the mask holds 0 or the scale and each output is the input times it. */
TEST_F(Medium_Dropout, TheOutputIsTheInputTimesAMaskOfZeroOrTheScale)
{
    Probe probe(backend(), K_SEED);
    ASSERT_TRUE(probe.ready());

    probe.run();

    const std::vector<double> x = probe.x();
    const std::vector<double> y = probe.y();
    const std::vector<double> mask = probe.mask();
    ASSERT_EQ(y.size(), K_ELEMENTS);
    ASSERT_EQ(mask.size(), K_ELEMENTS);

    EXPECT_TRUE(is_zero_or_scale(mask));

    // The product of one float input and one float mask rounds once.
    EXPECT_LE(worst_product_error(x, mask, y), TestSupport::unit_roundoff<float>());
}

/** @brief Tests that the kept fraction of the mask is within four spreads of 3/4. */
TEST_F(Medium_Dropout, TheKeptFractionIsAboutTheKeepProbability)
{
    Probe probe(backend(), K_SEED);
    ASSERT_TRUE(probe.ready());

    probe.run();

    const double kept = static_cast<double>(nonzero_count(probe.mask()))
                      / static_cast<double>(K_ELEMENTS);
    const double spread = std::sqrt(K_KEEP * K_RATE / static_cast<double>(K_ELEMENTS));

    EXPECT_NEAR(kept, K_KEEP, K_SPREADS * spread);
}

/** @brief Tests that two fresh steps with one seed agree byte for byte over three runs. */
TEST_F(Medium_DropoutSeeds, TwoFreshStepsWithTheSameSeedAgreeStepForStep)
{
    ASSERT_TRUE(first->ready() && second->ready());

    expect_same_seed_agrees();
    expect_same_seed_agrees();
    expect_same_seed_agrees();
}

/** @brief Tests that a step of the next seed differs from the first at each of three runs. */
TEST_F(Medium_DropoutSeeds, AStepWithAnotherSeedDiffersAtEveryRun)
{
    ASSERT_TRUE(first->ready() && other->ready());

    expect_other_seed_differs();
    expect_other_seed_differs();
    expect_other_seed_differs();
}

/** @brief Tests that the loss is the sum of the masked elements the step produced. */
TEST_F(Medium_Dropout, TheLossIsTheSumOfTheMaskedOutput)
{
    Probe probe(backend(), K_SEED);
    ASSERT_TRUE(probe.ready());

    probe.run();

    const std::vector<double> y = probe.y();
    const double sum = std::accumulate(y.begin(), y.end(), 0.0);

    // The loss is the sum of about 3000 values near 2.7, read back from FLOAT32.
    EXPECT_TRUE(TestSupport::is_within(
        TestSupport::measure_sum_error<float>(probe.loss(), sum, sum, K_ELEMENTS)));
}

/** @brief Tests that a step whose graph holds no dropout registers no prelude. */
TEST_F(Medium_Dropout, AStepWithoutDropoutRegistersNoPrelude)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);
    const ValueID x = builder.input(Shape{2, 3}, "x");
    SA_ASSERT_OK_AND_ASSIGN(const ValueID doubled, builder.scale(x, 2.0));
    static_cast<void>(doubled);

    TensorPool pool(*engine);
    ASSERT_TRUE(SaTest::status_is_ok(pool.materialise_leaves(graph)));
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<CompiledStep> step,
                            CompiledStep::compile(backend(), graph, pool));

    EXPECT_EQ(step->replay_prelude_count(), 0u);
}

/** @brief Tests that a classifier with dropout lowers its loss over 80 steps of one compile. */
TEST_F(Medium_Dropout, AModelWithDropoutTrainsThroughTheTrainer)
{
    DropoutMLP model = make_model();
    const Train::CrossEntropyObjective objective(K_BATCH, K_CLASSES, DType::Float32);
    Train::GraphOptions graph_options;
    graph_options.seed = K_GRAPH_SEED;

    SA_ASSERT_OK_AND_ASSIGN(Train::TrainingGraph built,
                            Train::build_training_graph(model, objective,
                                                        Shape{K_BATCH, K_FEATURES},
                                                        graph_options));
    ASSERT_GE(dropout_count(built.graph), 1u);

    SA_ASSERT_OK_AND_ASSIGN(NN::ModelState state, NN::ModelState::create(*engine, model));
    TensorPool pool(*engine);
    SA_ASSERT_OK_AND_ASSIGN(std::vector<Optim::ParameterSlot> slots,
                            Train::bind_parameters(backend(), built, state, pool));
    SA_EXPECT_OK(NN::initialize(backend(), state));

    Optim::AdamWOptions optimizer_options;
    optimizer_options.learning_rate = 1e-2;
    SA_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Optim::AdamW> optimizer,
                            Optim::AdamW::make(backend(), slots, optimizer_options));
    SA_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Train::Trainer> trainer,
                            Train::Trainer::make(backend(), built, pool, objective, *optimizer));

    Data::SyntheticOptions data_options;
    data_options.batch_size = K_BATCH;
    data_options.batches = 8;
    data_options.features = K_FEATURES;
    data_options.classes = K_CLASSES;
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<Data::SyntheticClassification> dataset,
                            Data::SyntheticClassification::make(backend(), data_options));

    Train::TrainerOptions options;
    options.epochs = 10;
    trainer->options() = options;

    SA_ASSERT_OK_AND_ASSIGN(const Train::RunReport report, trainer->fit(*dataset));

    EXPECT_EQ(report.steps, 80u);
    EXPECT_EQ(report.compile_count, 1u);
    EXPECT_TRUE(std::isfinite(report.last_loss)) << report.to_string();
    EXPECT_LT(report.last_loss, report.first_loss) << report.to_string();
}
