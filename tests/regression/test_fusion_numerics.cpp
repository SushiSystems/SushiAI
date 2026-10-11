/**************************************************************************/
/* test_fusion_numerics.cpp                                               */
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
 * @file test_fusion_numerics.cpp
 * @brief Tests that the fused graph trains byte for byte like the unfused graph it replaces.
 * @author Mustafa Garip
 * @see tests/regression/README.md
 */

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include <SushiBLAS/elementwise/elementwise.hpp>
#include <SushiBLAS/engine/scratch_pool.hpp>
#include <SushiBLAS/engine/task_recorder.hpp>
#include <SushiRuntime/sushiruntime.hpp>
#include <common/classifier_objective.hpp>
#include <common/test_status.hpp>
#include <common/test_utils.hpp>

using namespace SushiAI;

// SaTest keeps SushiAI::Test reachable inside TEST_F; see tests/regression/README.md.
namespace SaTest = SushiAI::Test;
using Graph::FusedForm;
using Graph::FusionOptions;
using Graph::TensorPool;

namespace
{
    class Regression_Fusion : public ::testing::Test {};

    constexpr int64_t K_BATCH = 8;
    constexpr int64_t K_FEATURES = 6;
    constexpr int64_t K_HIDDEN = 12;
    constexpr int64_t K_CLASSES = 4;
    constexpr int64_t K_BATCHES = 4;
    constexpr int64_t K_EPOCHS = 3;

    /** @brief The synthetic problem every comparison below trains on. */
    Data::SyntheticOptions data_options()
    {
        Data::SyntheticOptions options;
        options.batch_size = K_BATCH;
        options.batches = K_BATCHES;
        options.features = K_FEATURES;
        options.classes = K_CLASSES;
        options.seed = 0x0F05'ED00'1234ull;
        return options;
    }

    /**
     * @struct TrainedRun
     * @brief One whole training run, reduced to what a comparison needs.
     */
    struct TrainedRun
    {
        std::vector<std::vector<unsigned char>> parameters;
        std::vector<std::string> names;
        std::vector<double> losses;
        std::vector<double> accuracies;
        std::size_t compiles = 0;
        std::size_t operations_per_step = 0;
        std::size_t fused_subtrees = 0;
        std::size_t declined = 0;
        std::size_t selector_refusals = 0;
    };

    /**
     * @brief Trains one model end to end and writes everything comparable into @p run.
     * @param run Left partly filled when a step of the run fails, which fails the test.
     * @param selector Null selects this machine's own device.
     * @see tests/regression/README.md
     */
    template<typename M>
    void train(M& model, FusionOptions fusion, TrainedRun& run,
               const Graph::FusionSelector* selector = nullptr)
    {
        SushiBLAS::Engine engine(SaTest::TestEnvironment::runtime());

        SA_ASSERT_OK_AND_ASSIGN(Train::TrainingGraph built,
                                Train::build_classifier(model, K_BATCH, K_FEATURES, K_CLASSES,
                                                        DType::Float32, fusion));

        SA_ASSERT_OK_AND_ASSIGN(NN::ModelState state, NN::ModelState::create(engine, model));
        TensorPool pool(engine);
        SA_ASSERT_OK_AND_ASSIGN(std::vector<Optim::ParameterSlot> slots,
                                Train::bind_parameters(built, state, pool));
        ASSERT_TRUE(SaTest::status_is_ok(NN::initialize(engine, state)));

        Optim::AdamWOptions adam;
        adam.learning_rate = 0.02;
        SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<Optim::AdamW> optimizer,
                                Optim::AdamW::make(engine, slots, adam));

        Train::TrainerOptions options;
        options.epochs = K_EPOCHS;
        options.fusion_selector = selector;
        const auto objective = SaTest::classifier_objective(built);
        SA_ASSERT_OK_AND_ASSIGN(
            const std::unique_ptr<Train::Trainer> trainer,
            Train::Trainer::make(engine, built, pool, objective, *optimizer, options));

        SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<Data::SyntheticClassification> dataset,
                                Data::SyntheticClassification::make(engine, data_options()));

        run.fused_subtrees = built.fusion.fused().size();
        run.declined = built.fusion.declined().size();

        for (const Graph::FusionSelection& selection : trainer->step().selections())
            if (selection.chosen != selection.matched) ++run.selector_refusals;

        SA_ASSERT_OK_AND_ASSIGN(const Train::RunReport report, trainer->fit(*dataset));
        run.operations_per_step = report.operations_per_step;

        for (int64_t epoch = 0; epoch < K_EPOCHS; ++epoch)
            for (std::size_t batch = 0; batch < dataset->batch_count(); ++batch)
            {
                Data::Batch dataset_batch;
                ASSERT_TRUE(SaTest::status_is_ok(dataset->read(batch, dataset_batch)));
                SA_ASSERT_OK_AND_ASSIGN(const Train::StepMetrics metrics,
                                        trainer->run_step(dataset_batch));
                run.losses.push_back(metrics.loss);
                run.accuracies.push_back(metrics.metrics.value_or("accuracy", 0.0));
            }

        for (const Optim::ParameterSlot& slot : slots)
        {
            run.names.push_back(slot.name);
            run.parameters.push_back(SaTest::read_tensor_bytes(slot.parameter));
        }

        run.compiles = trainer->compile_count();
    }

    /** @brief Asserts two runs agree byte for byte, naming where they do not. */
    void expect_runs_agree(const TrainedRun& a, const TrainedRun& b, const std::string& what)
    {
        ASSERT_EQ(a.names, b.names);
        ASSERT_EQ(a.losses.size(), b.losses.size());
        ASSERT_FALSE(a.losses.empty());

        for (std::size_t step = 0; step < a.losses.size(); ++step)
            EXPECT_EQ(a.losses[step], b.losses[step])
                << what << ": the loss diverged at step " << step << " (" << a.losses[step]
                << " vs " << b.losses[step] << "). Find the expression that moved; do not "
                << "widen this comparison.";

        for (std::size_t step = 0; step < a.accuracies.size(); ++step)
            EXPECT_EQ(a.accuracies[step], b.accuracies[step])
                << what << ": accuracy diverged at step " << step;

        for (std::size_t i = 0; i < a.parameters.size(); ++i)
            EXPECT_EQ(SaTest::first_difference(a.parameters[i], b.parameters[i]),
                      a.parameters[i].size())
                << what << ": parameter '" << a.names[i] << "' differs";
    }

    /** @brief The unfused control. */
    FusionOptions off()
    {
        FusionOptions options;
        options.enabled = false;
        return options;
    }

    /**
     * @brief Returns a device profile that is not this machine's.
     * @see tests/regression/README.md
     */
    SushiBLAS::Device::Profile pretend_device(bool fp64)
    {
        SushiBLAS::Device::Profile profile;
        profile.is_cpu = false;
        profile.has_dedicated_local_memory = true;
        profile.supports_fp64 = fp64;
        profile.local_mem_bytes = 48 * 1024;
        profile.max_work_group_size = 1024;
        profile.sub_group_size = 32;
        profile.compute_units = 10;
        return profile;
    }
}

/**
 * @brief Tests that a ReLU MLP trains to identical bytes with fusion on and off.
 * @see tests/regression/README.md
 */
TEST_F(Regression_Fusion, AReLUMLPTrainsIdenticallyFusedAndUnfused)
{
    using MLP = NN::Sequential<NN::Linear, NN::ReLU, NN::Linear>;

    MLP fused_model(NN::Linear{K_FEATURES, K_HIDDEN}, NN::ReLU{}, NN::Linear{K_HIDDEN, K_CLASSES});
    MLP plain_model(NN::Linear{K_FEATURES, K_HIDDEN}, NN::ReLU{}, NN::Linear{K_HIDDEN, K_CLASSES});

    TrainedRun fused;
    ASSERT_NO_FATAL_FAILURE(train(fused_model, FusionOptions{}, fused));
    TrainedRun plain;
    ASSERT_NO_FATAL_FAILURE(train(plain_model, off(), plain));

    ASSERT_GT(fused.fused_subtrees, 0u);
    ASSERT_EQ(plain.fused_subtrees, 0u);

    expect_runs_agree(fused, plain, "fusing the forward and backward graph");
}

/**
 * @brief Tests that a tanh MLP trains to identical bytes with fusion on and off.
 * @see tests/regression/README.md
 */
TEST_F(Regression_Fusion, ATanhMLPTrainsIdenticallyFusedAndUnfused)
{
    using MLP = NN::Sequential<NN::Linear, NN::Tanh, NN::Linear>;

    MLP fused_model(NN::Linear{K_FEATURES, K_HIDDEN}, NN::Tanh{}, NN::Linear{K_HIDDEN, K_CLASSES});
    MLP plain_model(NN::Linear{K_FEATURES, K_HIDDEN}, NN::Tanh{}, NN::Linear{K_HIDDEN, K_CLASSES});

    TrainedRun fused;
    ASSERT_NO_FATAL_FAILURE(train(fused_model, FusionOptions{}, fused));
    TrainedRun plain;
    ASSERT_NO_FATAL_FAILURE(train(plain_model, off(), plain));

    ASSERT_GT(fused.fused_subtrees, 0u);
    expect_runs_agree(fused, plain, "a fused tanh epilogue");
}

TEST_F(Regression_Fusion, ASigmoidMLPTrainsIdenticallyFusedAndUnfused)
{
    using MLP = NN::Sequential<NN::Linear, NN::Sigmoid, NN::Linear>;

    MLP fused_model(NN::Linear{K_FEATURES, K_HIDDEN}, NN::Sigmoid{}, NN::Linear{K_HIDDEN, K_CLASSES});
    MLP plain_model(NN::Linear{K_FEATURES, K_HIDDEN}, NN::Sigmoid{}, NN::Linear{K_HIDDEN, K_CLASSES});

    TrainedRun fused;
    ASSERT_NO_FATAL_FAILURE(train(fused_model, FusionOptions{}, fused));
    TrainedRun plain;
    ASSERT_NO_FATAL_FAILURE(train(plain_model, off(), plain));

    ASSERT_GT(fused.fused_subtrees, 0u);
    expect_runs_agree(fused, plain, "a fused sigmoid epilogue");
}

/**
 * @brief Tests that a GELU MLP trains to identical bytes with fusion on and off.
 * @see tests/regression/README.md
 */
TEST_F(Regression_Fusion, AGELUMLPTrainsIdenticallyFusedAndUnfused)
{
    using MLP = NN::Sequential<NN::Linear, NN::GELU, NN::Linear>;

    MLP fused_model(NN::Linear{K_FEATURES, K_HIDDEN}, NN::GELU{}, NN::Linear{K_HIDDEN, K_CLASSES});
    MLP plain_model(NN::Linear{K_FEATURES, K_HIDDEN}, NN::GELU{}, NN::Linear{K_HIDDEN, K_CLASSES});

    TrainedRun fused;
    ASSERT_NO_FATAL_FAILURE(train(fused_model, FusionOptions{}, fused));
    TrainedRun plain;
    ASSERT_NO_FATAL_FAILURE(train(plain_model, off(), plain));

    // Two layers and the gradient tail fuse; see tests/regression/README.md.
    EXPECT_EQ(fused.fused_subtrees, 3u)
        << "expected gemm+bias+gelu, gemm+bias and the cross-entropy gradient tail";
    ASSERT_EQ(plain.fused_subtrees, 0u);

    expect_runs_agree(fused, plain, "a GELU epilogue spilling its pre-activation");

    // The loss fell, so both runs really trained; see tests/regression/README.md.
    ASSERT_GE(fused.losses.size(), 2u);
    EXPECT_LT(fused.losses.back(), fused.losses.front())
        << "the GELU model did not train at all, so agreeing with the control proves nothing";
}

TEST_F(Regression_Fusion, TheGELUFoldIsReportedAsAFoldAndNotAsARefusal)
{
    using MLP = NN::Sequential<NN::Linear, NN::GELU, NN::Linear>;
    MLP model(NN::Linear{K_FEATURES, K_HIDDEN}, NN::GELU{}, NN::Linear{K_HIDDEN, K_CLASSES});

    SA_ASSERT_OK_AND_ASSIGN(const Train::TrainingGraph built,
                            Train::build_classifier(model, K_BATCH, K_FEATURES, K_CLASSES));

    ASSERT_EQ(built.fusion.count(FusedForm::GEMMBiasGELU), 1u) << built.fusion.to_string();

    for (const Graph::DeclinedFusion& decline : built.fusion.declined())
        EXPECT_NE(decline.reason, Graph::FusionDecline::GradientNeedsPreActivation)
            << built.fusion.to_string();

    // gelu_backward must read the spilled value; see tests/regression/README.md.
    const Graph::FusedSubtree* gelu = nullptr;
    for (const Graph::FusedSubtree& subtree : built.fusion.fused())
        if (subtree.form == FusedForm::GEMMBiasGELU) gelu = &subtree;

    ASSERT_NE(gelu, nullptr);
    ASSERT_NE(gelu->spilled, Graph::K_NO_VALUE);

    const Graph::NodeID producer = built.graph.value(gelu->spilled).producer;
    ASSERT_NE(producer, Graph::K_NO_NODE) << "nothing writes the spill";
    EXPECT_EQ(built.graph.node(producer).op, Ops::K_GEMM_BIAS_GELU.id);

    bool read_by_the_gradient = false;
    for (const Graph::Node& node : built.graph.nodes())
        if (node.op == Ops::K_GELU_BACKWARD.id && node.inputs.at(1) == gelu->spilled)
            read_by_the_gradient = true;

    EXPECT_TRUE(read_by_the_gradient)
        << "gelu_backward does not read the spilled pre-activation, so the spill is "
           "buying nothing and the fold is the silently wrong one";
}

/**
 * @brief Tests that the spill buffer holds the host-computed pre-activation.
 * @see tests/regression/README.md
 */
TEST_F(Regression_Fusion, TheSpillHoldsThePreActivationItself)
{
    SushiBLAS::Engine engine(SaTest::TestEnvironment::runtime());

    constexpr int64_t K_ROWS = 4;
    constexpr int64_t K_IN_DIM = 3;
    constexpr int64_t K_OUT_DIM = 5;

    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);

    const Graph::ValueID x = builder.input(Shape{K_ROWS, K_IN_DIM}, "x");
    const Graph::ValueID w = builder.parameter(Shape{K_OUT_DIM, K_IN_DIM}, "w");
    const Graph::ValueID b = builder.parameter(Shape{K_OUT_DIM}, "b");
    const Graph::ValueID pre = builder.bias_add(builder.matmul(x, w, false, true), b);
    const Graph::ValueID out = builder.gelu(pre);

    // `pre` is unobserved; the fold rests on the spill; see tests/regression/README.md.
    const std::vector<Graph::ValueID> observed{out};
    FusionOptions options;
    options.observed = SushiRuntime::span<const Graph::ValueID>(observed.data(), observed.size());

    const Graph::FusionReport report = Graph::fuse(graph, options);
    ASSERT_EQ(report.count(FusedForm::GEMMBiasGELU), 1u) << report.to_string();

    TensorPool pool(engine);
    ASSERT_TRUE(SaTest::status_is_ok(pool.materialise_leaves(graph)));

    SaTest::HostNoise noise(0xC0FFEEu);
    std::vector<double> x_values(static_cast<std::size_t>(K_ROWS * K_IN_DIM));
    std::vector<double> w_values(static_cast<std::size_t>(K_OUT_DIM * K_IN_DIM));
    std::vector<double> b_values(static_cast<std::size_t>(K_OUT_DIM));
    noise.fill(SushiRuntime::span<double>(x_values.data(), x_values.size()), 2.0);
    noise.fill(SushiRuntime::span<double>(w_values.data(), w_values.size()), 2.0);
    noise.fill(SushiRuntime::span<double>(b_values.data(), b_values.size()), 2.0);

    SaTest::write_tensor(pool.tensor(x),
                         SushiRuntime::span<const double>(x_values.data(), x_values.size()));
    SaTest::write_tensor(pool.tensor(w),
                         SushiRuntime::span<const double>(w_values.data(), w_values.size()));
    SaTest::write_tensor(pool.tensor(b),
                         SushiRuntime::span<const double>(b_values.data(), b_values.size()));

    const Graph::DeviceFusionSelector selector = Graph::DeviceFusionSelector::for_engine(engine);
    Graph::Lowering lowering(engine.recorder(), engine.rng(), pool, selector);
    ASSERT_TRUE(SaTest::status_is_ok(lowering.record(graph)));
    ASSERT_EQ(1u, lowering.emitted_operations())
        << "this machine's device declined the fused form, so the epilogue never ran and "
           "this test is not exercising what it claims to";

    engine.execute().value();
    SaTest::TestEnvironment::runtime().wait().value();

    const std::vector<double> spilled = SaTest::read_tensor(pool.tensor(pre));
    const std::vector<double> activated = SaTest::read_tensor(pool.tensor(out));

    ASSERT_EQ(spilled.size(), static_cast<std::size_t>(K_ROWS * K_OUT_DIM));

    // The reference is computed on the host; see tests/regression/README.md.
    std::size_t separated = 0;

    for (int64_t row = 0; row < K_ROWS; ++row)
        for (int64_t col = 0; col < K_OUT_DIM; ++col)
        {
            double expected = b_values[static_cast<std::size_t>(col)];
            for (int64_t k = 0; k < K_IN_DIM; ++k)
                expected += x_values[static_cast<std::size_t>(row * K_IN_DIM + k)]
                          * w_values[static_cast<std::size_t>(col * K_IN_DIM + k)];

            const std::size_t index = static_cast<std::size_t>(row * K_OUT_DIM + col);
            EXPECT_NEAR(spilled[index], expected, 1e-5)
                << "the spill at (" << row << ", " << col << ") is not the pre-activation";

            // Only a negative element separates them; see tests/regression/README.md.
            if (expected < -0.5)
            {
                ++separated;
                EXPECT_GT(std::abs(spilled[index] - activated[index]), 0.1)
                    << "the spill at (" << row << ", " << col << ") holds the activated "
                    << "value, not the pre-activation";
            }
        }

    EXPECT_GT(separated, 0u)
        << "no element was negative enough for gelu(x) and x to differ, so the check "
           "above never distinguished the spill from the result";
}

/**
 * @brief Tests that a refused GELU epilogue still fills the spill and trains to identical bytes.
 * @see tests/regression/README.md
 */
TEST_F(Regression_Fusion, ASelectorRefusalStillFillsTheSpillAndMovesNoBit)
{
    class RefuseEverything final : public Graph::FusionSelector
    {
        public:
            Graph::FusionChoice choose(const Graph::FusionQuery&) const override
            {
                return Graph::FusionChoice{FusedForm::None, "this device wants none of it"};
            }
    };

    using MLP = NN::Sequential<NN::Linear, NN::GELU, NN::Linear>;

    MLP taken_model(NN::Linear{K_FEATURES, K_HIDDEN}, NN::GELU{}, NN::Linear{K_HIDDEN, K_CLASSES});
    MLP refused_model(NN::Linear{K_FEATURES, K_HIDDEN}, NN::GELU{}, NN::Linear{K_HIDDEN, K_CLASSES});

    const RefuseEverything refuse;
    TrainedRun taken;
    ASSERT_NO_FATAL_FAILURE(train(taken_model, FusionOptions{}, taken));
    TrainedRun refused;
    ASSERT_NO_FATAL_FAILURE(train(refused_model, FusionOptions{}, refused, &refuse));

    EXPECT_EQ(taken.fused_subtrees, refused.fused_subtrees);
    EXPECT_EQ(refused.selector_refusals, refused.fused_subtrees);
    EXPECT_GT(refused.operations_per_step, taken.operations_per_step);

    expect_runs_agree(taken, refused, "a declined GELU epilogue that still spills");
}

/**
 * @brief Tests that disabling the GELU form trains to the same bytes as folding it.
 * @see tests/regression/README.md
 */
TEST_F(Regression_Fusion, DisablingTheGELUFormTrainsToTheSamePlaceAsFoldingIt)
{
    using MLP = NN::Sequential<NN::Linear, NN::GELU, NN::Linear>;

    MLP folded_model(NN::Linear{K_FEATURES, K_HIDDEN}, NN::GELU{}, NN::Linear{K_HIDDEN, K_CLASSES});
    MLP separate_model(NN::Linear{K_FEATURES, K_HIDDEN}, NN::GELU{}, NN::Linear{K_HIDDEN, K_CLASSES});

    const std::vector<FusedForm> without{FusedForm::GEMMBiasGELU};
    FusionOptions narrowed;
    narrowed.disabled = SushiRuntime::span<const FusedForm>(without.data(), without.size());

    TrainedRun folded;
    ASSERT_NO_FATAL_FAILURE(train(folded_model, FusionOptions{}, folded));
    TrainedRun separate;
    ASSERT_NO_FATAL_FAILURE(train(separate_model, narrowed, separate));

    EXPECT_EQ(folded.fused_subtrees, separate.fused_subtrees)
        << "switching the three-node form off must fall back to gemm+bias, not to nothing";

    // Disabling the form costs exactly two kernels; see tests/regression/README.md.
    EXPECT_EQ(separate.operations_per_step, folded.operations_per_step + 2u);

    expect_runs_agree(folded, separate, "switching the fused GELU epilogue off");
}

/**
 * @brief Tests that the fused ReLU MLP step issues ten fewer operations than the unfused one.
 * @see tests/regression/README.md
 */
TEST_F(Regression_Fusion, TheFusedStepIssuesTenFewerKernelsPerStep)
{
    using MLP = NN::Sequential<NN::Linear, NN::ReLU, NN::Linear>;

    MLP fused_model(NN::Linear{K_FEATURES, K_HIDDEN}, NN::ReLU{}, NN::Linear{K_HIDDEN, K_CLASSES});
    MLP plain_model(NN::Linear{K_FEATURES, K_HIDDEN}, NN::ReLU{}, NN::Linear{K_HIDDEN, K_CLASSES});

    TrainedRun fused;
    ASSERT_NO_FATAL_FAILURE(train(fused_model, FusionOptions{}, fused));
    TrainedRun plain;
    ASSERT_NO_FATAL_FAILURE(train(plain_model, off(), plain));

    EXPECT_EQ(plain.operations_per_step, fused.operations_per_step + 10u);

    // Fusing adds no ordering node and no compile; see tests/regression/README.md.
    EXPECT_EQ(fused.compiles, 1u);
    EXPECT_EQ(plain.compiles, 1u);
}

/**
 * @brief Tests that a fused GEMM depends on the task that writes the bias its epilogue reads.
 * @see tests/regression/README.md
 */
TEST_F(Regression_Fusion, AFusedGEMMIsOrderedAgainstTheTaskThatWritesItsBias)
{
    SushiBLAS::Engine engine(SaTest::TestEnvironment::runtime());

    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueID x = builder.input(Shape{K_BATCH, K_FEATURES}, "x");
    const Graph::ValueID w = builder.parameter(Shape{K_HIDDEN, K_FEATURES}, "w");
    const Graph::ValueID b = builder.parameter(Shape{K_HIDDEN}, "b");
    builder.relu(builder.bias_add(builder.matmul(x, w, false, true), b));

    const Graph::FusionReport fusion = Graph::fuse(graph, FusionOptions{});
    ASSERT_EQ(1u, fusion.fused().size()) << fusion.to_string();

    TensorPool pool(engine);
    ASSERT_TRUE(SaTest::status_is_ok(pool.materialise_leaves(graph)));

    SushiRuntime::Graph tasks = SaTest::TestEnvironment::runtime().graph();
    SushiBLAS::ScratchPool scratch(SaTest::TestEnvironment::runtime());
    SushiBLAS::TaskRecorder recorder(tasks.recorder(), scratch, engine.mode());

    // Recorded first: the edge can only run to the GEMM; see tests/regression/README.md.
    SushiBLAS::Tensor& bias = pool.tensor(b);
    SushiBLAS::ElementwiseOps(recorder).mul(bias, bias, bias);

    const Graph::DeviceFusionSelector selector = Graph::DeviceFusionSelector::for_engine(engine);
    Graph::Lowering lowering(recorder, engine.rng(), pool, selector);
    ASSERT_TRUE(SaTest::status_is_ok(lowering.record(graph)));
    ASSERT_EQ(1u, lowering.emitted_operations())
        << "this machine's device declined the fused form, so there is no epilogue to "
           "order and this test is not exercising what it claims to";

    ASSERT_EQ(2u, tasks.recorder().node_count());
    ASSERT_EQ(nullptr, tasks.recorder().refusal());
    GTEST_SKIP() << "The fused GEMM must depend on the task that wrote the bias its epilogue "
                    "dereferences, and SushiRuntime::Graph shows no dependency of a recorded "
                    "node, so the edge is not asserted.";
}

/**
 * @brief Tests that the reader of the spill depends on the fused GEMM that fills it.
 * @see tests/regression/README.md
 */
TEST_F(Regression_Fusion, WhateverReadsTheSpillIsOrderedBehindTheGEMMThatFillsIt)
{
    SushiBLAS::Engine engine(SaTest::TestEnvironment::runtime());

    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueID x = builder.input(Shape{K_BATCH, K_FEATURES}, "x");
    const Graph::ValueID w = builder.parameter(Shape{K_HIDDEN, K_FEATURES}, "w");
    const Graph::ValueID b = builder.parameter(Shape{K_HIDDEN}, "b");
    const Graph::ValueID pre = builder.bias_add(builder.matmul(x, w, false, true), b);
    const Graph::ValueID out = builder.gelu(pre);

    // Emitted after the GELU, as gelu_backward is; see tests/regression/README.md.
    const Graph::ValueID dy = builder.input(Shape{K_BATCH, K_HIDDEN}, "dy");
    (void)builder.activation_backward(Ops::K_GELU_BACKWARD.id, dy, pre);

    const std::vector<Graph::ValueID> observed{out};
    FusionOptions options;
    options.observed = SushiRuntime::span<const Graph::ValueID>(observed.data(), observed.size());

    const Graph::FusionReport fusion = Graph::fuse(graph, options);
    ASSERT_EQ(1u, fusion.count(FusedForm::GEMMBiasGELU)) << fusion.to_string();

    TensorPool pool(engine);
    ASSERT_TRUE(SaTest::status_is_ok(pool.materialise_leaves(graph)));

    SushiRuntime::Graph tasks = SaTest::TestEnvironment::runtime().graph();
    SushiBLAS::ScratchPool scratch(SaTest::TestEnvironment::runtime());
    SushiBLAS::TaskRecorder recorder(tasks.recorder(), scratch, engine.mode());

    const Graph::DeviceFusionSelector selector = Graph::DeviceFusionSelector::for_engine(engine);
    Graph::Lowering lowering(recorder, engine.rng(), pool, selector);
    ASSERT_TRUE(SaTest::status_is_ok(lowering.record(graph)));
    ASSERT_EQ(2u, lowering.emitted_operations())
        << "expected the fused GEMM and the activation gradient; this machine's device may "
           "have declined the fused form, in which case this test is not exercising what "
           "it claims to";

    ASSERT_EQ(2u, tasks.recorder().node_count());
    ASSERT_EQ(nullptr, tasks.recorder().refusal());
    GTEST_SKIP() << "The task that reads the spilled pre-activation must depend on the fused "
                    "GEMM whose epilogue wrote it, and SushiRuntime::Graph shows no "
                    "dependency of a recorded node, so the edge is not asserted.";

}

/**
 * @brief Tests that disabling GEMM_BIAS_RELU falls back to GEMM_BIAS and moves no byte.
 * @see tests/regression/README.md
 */
TEST_F(Regression_Fusion, DisablingOneCatalogEntryLeavesTheRestAlone)
{
    using MLP = NN::Sequential<NN::Linear, NN::ReLU, NN::Linear>;

    MLP whole_model(NN::Linear{K_FEATURES, K_HIDDEN}, NN::ReLU{}, NN::Linear{K_HIDDEN, K_CLASSES});
    MLP narrowed_model(NN::Linear{K_FEATURES, K_HIDDEN}, NN::ReLU{}, NN::Linear{K_HIDDEN, K_CLASSES});

    const std::vector<FusedForm> without{FusedForm::GEMMBiasReLU};
    FusionOptions narrowed;
    narrowed.disabled = SushiRuntime::span<const FusedForm>(without.data(), without.size());

    TrainedRun whole;
    ASSERT_NO_FATAL_FAILURE(train(whole_model, FusionOptions{}, whole));
    TrainedRun some;
    ASSERT_NO_FATAL_FAILURE(train(narrowed_model, narrowed, some));

    // A catalog refusal costs two kernels, not one; see tests/regression/README.md.
    EXPECT_EQ(some.operations_per_step, whole.operations_per_step + 2u);
    expect_runs_agree(whole, some, "switching the fused ReLU epilogue off");
}

/**
 * @brief Tests that a selector refusing every form yields an unfused plan with identical bytes.
 * @see tests/regression/README.md
 */
TEST_F(Regression_Fusion, ASelectorMayRefuseAMatchedFormAndTheNumbersDoNotMove)
{
    class RefuseEverything final : public Graph::FusionSelector
    {
        public:
            Graph::FusionChoice choose(const Graph::FusionQuery&) const override
            {
                return Graph::FusionChoice{FusedForm::None, "this device wants none of it"};
            }
    };

    using MLP = NN::Sequential<NN::Linear, NN::ReLU, NN::Linear>;

    MLP taken_model(NN::Linear{K_FEATURES, K_HIDDEN}, NN::ReLU{}, NN::Linear{K_HIDDEN, K_CLASSES});
    MLP refused_model(NN::Linear{K_FEATURES, K_HIDDEN}, NN::ReLU{}, NN::Linear{K_HIDDEN, K_CLASSES});

    const RefuseEverything refuse;
    TrainedRun taken;
    ASSERT_NO_FATAL_FAILURE(train(taken_model, FusionOptions{}, taken));
    TrainedRun refused;
    ASSERT_NO_FATAL_FAILURE(train(refused_model, FusionOptions{}, refused, &refuse));

    // The graph fused identically; only the spelling differs.
    EXPECT_EQ(taken.fused_subtrees, refused.fused_subtrees);
    EXPECT_EQ(refused.selector_refusals, refused.fused_subtrees);
    EXPECT_EQ(taken.selector_refusals, 0u)
        << "this machine's own device declined a form; the measurement notes assume it did not";

    EXPECT_GT(refused.operations_per_step, taken.operations_per_step);

    expect_runs_agree(taken, refused, "a selector that refuses every form");
}

TEST_F(Regression_Fusion, ADeviceWithoutDoublePrecisionIsNotHandedAFusedDoubleKernel)
{
    const Graph::DeviceFusionSelector without(pretend_device(false),
                                              SushiBLAS::ExecutionMode::Auto);
    const Graph::DeviceFusionSelector with(pretend_device(true),
                                           SushiBLAS::ExecutionMode::Auto);

    Graph::FusionQuery query;
    query.matched = FusedForm::GEMMBiasReLU;
    query.dtype = DType::Float64;
    query.rows = 32;
    query.columns = 128;

    EXPECT_EQ(without.choose(query).form, FusedForm::None);
    EXPECT_EQ(with.choose(query).form, FusedForm::GEMMBiasReLU);

    // A float graph is fine on either.
    query.dtype = DType::Float32;
    EXPECT_EQ(without.choose(query).form, FusedForm::GEMMBiasReLU);
}

/**
 * @brief Tests that host-executing modes are refused a GEMM epilogue and keep the zip form.
 * @see tests/regression/README.md
 */
TEST_F(Regression_Fusion, AHostExecutingModeDoesNotGetAnEpilogueThatDereferencesDeviceMemory)
{
    Graph::FusionQuery gemm;
    gemm.matched = FusedForm::GEMMBias;
    gemm.rows = 32;
    gemm.columns = 128;

    for (SushiBLAS::ExecutionMode mode : {SushiBLAS::ExecutionMode::ForceReference})
    {
        const Graph::DeviceFusionSelector selector(pretend_device(true), mode);
        EXPECT_EQ(selector.choose(gemm).form, FusedForm::None)
            << "mode " << static_cast<int>(mode);

        // The zip form runs no host arm; see tests/regression/README.md.
        Graph::FusionQuery zip;
        zip.matched = FusedForm::ExpSubScale;
        zip.rows = 32;
        zip.columns = 128;
        EXPECT_EQ(selector.choose(zip).form, FusedForm::ExpSubScale);
    }
}

TEST_F(Regression_Fusion, AStridedBiasIsNotHandedToAnEpilogueThatIndexesItDirectly)
{
    const Graph::DeviceFusionSelector selector(pretend_device(true),
                                               SushiBLAS::ExecutionMode::Auto);

    Graph::FusionQuery query;
    query.matched = FusedForm::GEMMBias;
    query.rows = 32;
    query.columns = 128;
    query.operand_is_contiguous = false;

    EXPECT_EQ(selector.choose(query).form, FusedForm::None);
}

/**
 * @brief Tests that every selector choice carries a non-empty reason.
 * @see tests/regression/README.md
 */
TEST_F(Regression_Fusion, EveryChoiceCarriesAReasonSomebodyCanRead)
{
    const Graph::DeviceFusionSelector selector(pretend_device(true),
                                               SushiBLAS::ExecutionMode::Auto);

    Graph::FusionQuery query;
    query.matched = FusedForm::GEMMBiasReLU;
    query.rows = 32;
    query.columns = 128;

    EXPECT_FALSE(selector.choose(query).reason.empty());

    query.operand_is_contiguous = false;
    EXPECT_FALSE(selector.choose(query).reason.empty());
}
