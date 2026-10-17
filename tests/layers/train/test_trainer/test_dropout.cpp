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
 * @file test_dropout.cpp
 * @brief Checks DROPOUT on an engine: the mask of each replay, its gradient, the two trace modes of
 *        the Dropout layer, and a model with dropout trained through the Trainer.
 * @author Mustafa Garip
 */

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>
#include <gtest/gtest.h>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/SushiAI.hpp>
#include <SushiAI/autograd/backward.hpp>
#include <SushiAI/dtype/dtype.hpp>
#include <SushiAI/ops/ops.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/shape/shape.hpp>
#include <SushiAI/data/dataset.hpp>
#include <SushiAI/engine_fixture/engine_test.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/graph/builder.hpp>
#include <SushiAI/graph/compiled_step.hpp>
#include <SushiAI/graph/dropout_mask.hpp>
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
    /** @brief Provides the Engine every dropout case records on. */
    class Medium_Dropout : public SaTest::EngineTest
    {

    };

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

    /** @brief Holds the site seed of every probe that does not state another. */
    constexpr std::uint64_t K_SEED = 0xE220A8397B1DCDAFull;

    /** @brief Bounds a FLOAT32 product computed on the device against a double host value. */
    constexpr double K_TOLERANCE = 1e-5;

    /** @brief Bounds the kept fraction of K_ELEMENTS draws, four deviations of 0.0068. */
    constexpr double K_FRACTION_TOLERANCE = 0.03;

    /** @brief Holds the seed of the values written into the masked leaf. */
    constexpr std::uint64_t K_DATA_SEED = 20261011ull;

    /**
     * @brief Holds a compiled graph that masks a [K_ROWS, K_WIDTH] parameter and differentiates
     *        the sum of the masked elements at it.
     *
     * The loss is NLL_LOSS over the masked value itself with every target at -K_ROWS, which is
     * the sum of the masked elements, so the gradient at the parameter is the mask.
     */
    class Probe
    {
        public:
            /** @brief Builds, differentiates and compiles the graph with the site seed @p seed. */
            Probe(SushiBLAS::Engine& engine, std::uint64_t seed) : pool_(engine)
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
                loss_ = builder.nll_loss(y_, targets);

                const std::vector<ValueID> wrt = {x_};
                gradient_ = Autograd::differentiate(
                    graph_, loss_,
                    SushiRuntime::span<const ValueID>(wrt.data(), wrt.size())).at(0);
                EXPECT_NE(gradient_, Graph::K_NO_VALUE);

                SA_EXPECT_OK(pool_.materialise_leaves(graph_));

                Result<std::unique_ptr<CompiledStep>> compiled =
                    CompiledStep::compile(engine, graph_, pool_);
                SA_EXPECT_OK(compiled);
                if (!compiled.has_value())
                    return;

                step_ = std::move(compiled).value();

                // Every element lies in [1, 3], so a kept one is never zero.
                std::vector<double> values(K_ELEMENTS);
                SaTest::HostNoise(K_DATA_SEED).fill(values, 1.0);
                for (double& value : values)
                    value += 2.0;

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

            /** @brief Returns the compiled step. */
            [[nodiscard]] const CompiledStep& step() const noexcept
            {
                return *step_;
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

            /** @brief Returns the gradient of the loss at the masked leaf, from the last run. */
            [[nodiscard]] std::vector<double> gradient() const
            {
                return read_tensor(pool_.tensor(gradient_));
            }

            /** @brief Returns the loss of the last run. */
            [[nodiscard]] double loss() const
            {
                return read_tensor(pool_.tensor(loss_))[0];
            }

        private:
            /** @brief Holds the forward and the gradient nodes. */
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

            /** @brief Names the gradient of the loss at the masked leaf. */
            ValueID gradient_ = Graph::K_NO_VALUE;
    };

    /** @brief Expects @p mask to hold what the generator draws for @p seed in run @p run. */
    void expect_mask_of_run(const std::vector<double>& mask, std::uint64_t seed,
                            std::uint64_t run)
    {
        ASSERT_EQ(mask.size(), K_ELEMENTS);
        for (std::size_t element = 0; element < K_ELEMENTS; ++element)
        {
            const double expected =
                Graph::dropout_keeps(seed, run, element, K_KEEP) ? 1.0 / K_KEEP : 0.0;
            EXPECT_NEAR(mask[element], expected, K_TOLERANCE) << "at element " << element;
        }
    }

    /** @brief Returns how many elements of @p values are not zero. */
    std::size_t nonzero_count(const std::vector<double>& values)
    {
        std::size_t found = 0;
        for (const double value : values)
            found += value != 0.0 ? 1u : 0u;

        return found;
    }

    /** @brief Returns how many nodes of @p graph carry DROPOUT. */
    std::size_t dropout_count(const Graph::Graph& graph)
    {
        std::size_t found = 0;
        for (const Graph::Node& node : graph.nodes())
            found += node.op == Ops::K_DROPOUT.id ? 1u : 0u;

        return found;
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
}

/** @brief Tests that each output is 0 or x / (1 - rate), and that 3/4 of them are kept. */
TEST_F(Medium_Dropout, TheOutputIsTheInputTimesAMaskOfZeroOrTheScale)
{
    Probe probe(*engine, K_SEED);
    ASSERT_TRUE(probe.ready());

    probe.run();

    const std::vector<double> x = probe.x();
    const std::vector<double> y = probe.y();
    const std::vector<double> mask = probe.mask();
    ASSERT_EQ(y.size(), K_ELEMENTS);
    ASSERT_EQ(mask.size(), K_ELEMENTS);

    for (std::size_t element = 0; element < K_ELEMENTS; ++element)
    {
        const bool dropped = mask[element] == 0.0;
        if (!dropped)
            EXPECT_NEAR(mask[element], 1.0 / K_KEEP, K_TOLERANCE) << "at element " << element;

        EXPECT_NEAR(y[element], dropped ? 0.0 : x[element] / K_KEEP, K_TOLERANCE)
            << "at element " << element;
    }

    const double kept = static_cast<double>(nonzero_count(mask))
                      / static_cast<double>(K_ELEMENTS);
    EXPECT_NEAR(kept, K_KEEP, K_FRACTION_TOLERANCE);
    expect_mask_of_run(mask, K_SEED, 0);
}

/** @brief Tests that two runs of one step draw the masks of runs 0 and 1 from one compile. */
TEST_F(Medium_Dropout, TwoStepsOfOneCompiledStepDrawDifferentMasks)
{
    Probe probe(*engine, K_SEED);
    ASSERT_TRUE(probe.ready());
    EXPECT_EQ(probe.step().replay_prelude_count(), 1u);

    probe.run();
    const std::vector<double> first = probe.mask();
    probe.run();
    const std::vector<double> second = probe.mask();

    EXPECT_NE(first, second);
    expect_mask_of_run(first, K_SEED, 0);
    expect_mask_of_run(second, K_SEED, 1);
    EXPECT_EQ(probe.step().compile_count(), 1u);
    EXPECT_EQ(probe.step().run_count(), 2u);
}

/** @brief Tests that two fresh steps with one seed agree byte for byte over three runs. */
TEST_F(Medium_Dropout, TwoFreshStepsWithTheSameSeedAgreeStepForStep)
{
    Probe first(*engine, K_SEED);
    Probe second(*engine, K_SEED);
    Probe other(*engine, K_SEED + 1);
    ASSERT_TRUE(first.ready() && second.ready() && other.ready());

    for (int run = 0; run < 3; ++run)
    {
        first.run();
        second.run();
        other.run();

        EXPECT_EQ(first.y_bytes(), second.y_bytes()) << "at run " << run;
        EXPECT_NE(first.y_bytes(), other.y_bytes()) << "at run " << run;
    }
}

/** @brief Tests that the gradient at the input is the mask of the same run, on two runs. */
TEST_F(Medium_Dropout, TheGradientAtTheInputIsTheMaskOfThatStep)
{
    Probe probe(*engine, K_SEED);
    ASSERT_TRUE(probe.ready());

    for (std::uint64_t run = 0; run < 2; ++run)
    {
        probe.run();

        const std::vector<double> mask = probe.mask();
        const std::vector<double> gradient = probe.gradient();
        const std::vector<double> y = probe.y();
        ASSERT_EQ(gradient.size(), mask.size());
        expect_mask_of_run(mask, K_SEED, run);

        double sum = 0.0;
        for (std::size_t element = 0; element < mask.size(); ++element)
        {
            EXPECT_NEAR(gradient[element], mask[element], K_TOLERANCE)
                << "at run " << run << ", element " << element;
            sum += y[element];
        }

        // The loss is the sum of about 3000 values near 2.7, read back from FLOAT32.
        EXPECT_NEAR(probe.loss(), sum, 1e-4 * sum) << "at run " << run;
    }
}

/** @brief Tests that a step whose graph holds no dropout registers no prelude and runs. */
TEST_F(Medium_Dropout, AStepWithoutDropoutRegistersNoPrelude)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);
    const ValueID x = builder.input(Shape{2, 3}, "x");
    const ValueID doubled = builder.scale(x, 2.0);

    TensorPool pool(*engine);
    ASSERT_TRUE(SaTest::status_is_ok(pool.materialise_leaves(graph)));
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<CompiledStep> step,
                            CompiledStep::compile(*engine, graph, pool));
    write_tensor(pool.tensor(x), std::vector<double>{1.0, 2.0, 3.0, 4.0, 5.0, 6.0});

    EXPECT_EQ(step->replay_prelude_count(), 0u);
    SA_EXPECT_OK(step->run());
    EXPECT_EQ(read_tensor(pool.tensor(doubled)),
              (std::vector<double>{2.0, 4.0, 6.0, 8.0, 10.0, 12.0}));
}

/** @brief Tests that a model's evaluation graph holds no DROPOUT node and its probe graph one. */
TEST_F(Medium_Dropout, AnEvaluationTraceOfAModelEmitsNoDropoutNode)
{
    DropoutMLP model = make_model();
    const Shape batch{K_BATCH, K_FEATURES};
    Train::GraphOptions options;
    options.seed = K_GRAPH_SEED;

    SA_ASSERT_OK_AND_ASSIGN(const Train::TracedStep evaluation,
                            Train::build_evaluation_graph(model, batch, options));
    SA_ASSERT_OK_AND_ASSIGN(const Train::TracedStep training,
                            Train::build_probe_graph(model, batch, options));

    EXPECT_EQ(dropout_count(evaluation.graph), 0u);
    ASSERT_EQ(dropout_count(training.graph), 1u);

    for (const Graph::Node& node : training.graph.nodes())
    {
        if (node.op != Ops::K_DROPOUT.id)
            continue;

        EXPECT_EQ(node.attributes.values[Graph::K_DROPOUT_RATE_SLOT], K_MODEL_RATE);
        EXPECT_EQ(Graph::dropout_seed(node.attributes),
                  Graph::dropout_site_seed(K_GRAPH_SEED, 0));
    }
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
                            Train::bind_parameters(built, state, pool));
    SA_EXPECT_OK(NN::initialize(*engine, state));

    Optim::AdamWOptions optimizer_options;
    optimizer_options.learning_rate = 1e-2;
    SA_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Optim::AdamW> optimizer,
                            Optim::AdamW::make(*engine, slots, optimizer_options));
    SA_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Train::Trainer> trainer,
                            Train::Trainer::make(*engine, built, pool, objective, *optimizer));

    Data::SyntheticOptions data_options;
    data_options.batch_size = K_BATCH;
    data_options.batches = 8;
    data_options.features = K_FEATURES;
    data_options.classes = K_CLASSES;
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<Data::SyntheticClassification> dataset,
                            Data::SyntheticClassification::make(*engine, data_options));

    Train::TrainerOptions options;
    options.epochs = 10;
    trainer->options() = options;

    SA_ASSERT_OK_AND_ASSIGN(const Train::RunReport report, trainer->fit(*dataset));

    EXPECT_EQ(report.steps, 80u);
    EXPECT_EQ(report.compile_count, 1u);
    EXPECT_TRUE(std::isfinite(report.last_loss)) << report.to_string();
    EXPECT_LT(report.last_loss, report.first_loss) << report.to_string();
}
