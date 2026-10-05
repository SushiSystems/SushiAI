/****************************************************************/
/* test_fusion_numerics.cpp                                     */
/* SushiAI - https://github.com/SushiSystems/SushiAI            */
/* Copyright (c) 2026 Sushi Systems                             */
/* Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.    */
/* Commercial use requires a licence from Sushi Systems.        */
/****************************************************************/

/**
 * @file test_fusion_numerics.cpp
 * @brief Tests that the fused graph trains byte for byte like the unfused graph it replaces.
 * @author Mustafa Garip
 * @see tests/functional/regression/README.md
 */

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include "../../common/test_utils.hpp"

using namespace SushiAI;

// SaTest keeps SushiAI::Test reachable inside TEST_F; see tests/functional/regression/README.md.
namespace SaTest = SushiAI::Test;
using Graph::FusedForm;
using Graph::FusionOptions;
using Graph::TensorPool;

namespace
{
    class Regression_Fusion : public ::testing::Test {};

    constexpr int64_t BATCH = 8;
    constexpr int64_t FEATURES = 6;
    constexpr int64_t HIDDEN = 12;
    constexpr int64_t CLASSES = 4;
    constexpr int64_t BATCHES = 4;
    constexpr int64_t EPOCHS = 3;

    /** @brief The synthetic problem every comparison below trains on. */
    Data::SyntheticOptions data_options()
    {
        Data::SyntheticOptions options;
        options.batch_size = BATCH;
        options.batches = BATCHES;
        options.features = FEATURES;
        options.classes = CLASSES;
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
     * @brief Trains one model end to end and reports everything comparable.
     * @param selector Null selects this machine's own device.
     * @see tests/functional/regression/README.md
     */
    template<typename M>
    TrainedRun train(M& model, FusionOptions fusion,
                     const Graph::IFusionSelector* selector = nullptr)
    {
        SushiBLAS::Engine engine(SaTest::TestEnvironment::context());

        Train::TrainingGraph built = Train::build_classifier(
            model, BATCH, FEATURES, CLASSES, DType::FLOAT32, fusion);

        TensorPool pool(engine);
        std::vector<Optim::ParameterSlot> slots = Train::bind_parameters(built, pool);
        NN::initialize(engine, model, pool);

        Optim::AdamWOptions adam;
        adam.learning_rate = 0.02;
        Optim::AdamW optimizer(engine, slots, adam);

        Train::TrainerOptions options;
        options.epochs = EPOCHS;
        options.fusion_selector = selector;
        Train::Trainer trainer(engine, built.graph, pool, built.spec, optimizer, options);

        const Data::SyntheticClassification dataset(engine, data_options());

        TrainedRun run;
        run.fused_subtrees = built.fusion.fused().size();
        run.declined = built.fusion.declined().size();

        for (const Graph::FusionSelection& selection : trainer.step().selections())
            if (selection.chosen != selection.matched) ++run.selector_refusals;

        const Train::RunReport report = trainer.fit(dataset);
        run.operations_per_step = report.operations_per_step;

        for (int64_t epoch = 0; epoch < EPOCHS; ++epoch)
            for (std::size_t batch = 0; batch < dataset.batch_count(); ++batch)
            {
                const Train::StepMetrics metrics = trainer.run_step(dataset.batch(batch));
                run.losses.push_back(metrics.loss);
                run.accuracies.push_back(metrics.accuracy);
            }

        for (const Optim::ParameterSlot& slot : slots)
        {
            run.names.push_back(slot.name);
            run.parameters.push_back(SaTest::read_tensor_bytes(slot.parameter));
        }

        run.compiles = trainer.compile_count();
        return run;
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
     * @see tests/functional/regression/README.md
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
        profile.global_mem_cache_line = 128;
        return profile;
    }
}

/**
 * @brief Tests that a ReLU MLP trains to identical bytes with fusion on and off.
 * @see tests/functional/regression/README.md
 */
TEST_F(Regression_Fusion, AReluMlpTrainsIdenticallyFusedAndUnfused)
{
    using Mlp = NN::Sequential<NN::Linear, NN::ReLU, NN::Linear>;

    Mlp fused_model(NN::Linear{FEATURES, HIDDEN}, NN::ReLU{}, NN::Linear{HIDDEN, CLASSES});
    Mlp plain_model(NN::Linear{FEATURES, HIDDEN}, NN::ReLU{}, NN::Linear{HIDDEN, CLASSES});

    const TrainedRun fused = train(fused_model, FusionOptions{});
    const TrainedRun plain = train(plain_model, off());

    ASSERT_GT(fused.fused_subtrees, 0u);
    ASSERT_EQ(plain.fused_subtrees, 0u);

    expect_runs_agree(fused, plain, "fusing the forward and backward graph");
}

/**
 * @brief Tests that a tanh MLP trains to identical bytes with fusion on and off.
 * @see tests/functional/regression/README.md
 */
TEST_F(Regression_Fusion, ATanhMlpTrainsIdenticallyFusedAndUnfused)
{
    using Mlp = NN::Sequential<NN::Linear, NN::Tanh, NN::Linear>;

    Mlp fused_model(NN::Linear{FEATURES, HIDDEN}, NN::Tanh{}, NN::Linear{HIDDEN, CLASSES});
    Mlp plain_model(NN::Linear{FEATURES, HIDDEN}, NN::Tanh{}, NN::Linear{HIDDEN, CLASSES});

    const TrainedRun fused = train(fused_model, FusionOptions{});
    const TrainedRun plain = train(plain_model, off());

    ASSERT_GT(fused.fused_subtrees, 0u);
    expect_runs_agree(fused, plain, "a fused tanh epilogue");
}

TEST_F(Regression_Fusion, ASigmoidMlpTrainsIdenticallyFusedAndUnfused)
{
    using Mlp = NN::Sequential<NN::Linear, NN::Sigmoid, NN::Linear>;

    Mlp fused_model(NN::Linear{FEATURES, HIDDEN}, NN::Sigmoid{}, NN::Linear{HIDDEN, CLASSES});
    Mlp plain_model(NN::Linear{FEATURES, HIDDEN}, NN::Sigmoid{}, NN::Linear{HIDDEN, CLASSES});

    const TrainedRun fused = train(fused_model, FusionOptions{});
    const TrainedRun plain = train(plain_model, off());

    ASSERT_GT(fused.fused_subtrees, 0u);
    expect_runs_agree(fused, plain, "a fused sigmoid epilogue");
}

/**
 * @brief Tests that a GELU MLP trains to identical bytes with fusion on and off.
 * @see tests/functional/regression/README.md
 */
TEST_F(Regression_Fusion, AGeluMlpTrainsIdenticallyFusedAndUnfused)
{
    using Mlp = NN::Sequential<NN::Linear, NN::GELU, NN::Linear>;

    Mlp fused_model(NN::Linear{FEATURES, HIDDEN}, NN::GELU{}, NN::Linear{HIDDEN, CLASSES});
    Mlp plain_model(NN::Linear{FEATURES, HIDDEN}, NN::GELU{}, NN::Linear{HIDDEN, CLASSES});

    const TrainedRun fused = train(fused_model, FusionOptions{});
    const TrainedRun plain = train(plain_model, off());

    // Two layers and the gradient tail fuse; see tests/functional/regression/README.md.
    EXPECT_EQ(fused.fused_subtrees, 3u)
        << "expected gemm+bias+gelu, gemm+bias and the cross-entropy gradient tail";
    ASSERT_EQ(plain.fused_subtrees, 0u);

    expect_runs_agree(fused, plain, "a GELU epilogue spilling its pre-activation");

    // The loss fell, so both runs really trained; see tests/functional/regression/README.md.
    ASSERT_GE(fused.losses.size(), 2u);
    EXPECT_LT(fused.losses.back(), fused.losses.front())
        << "the GELU model did not train at all, so agreeing with the control proves nothing";
}

TEST_F(Regression_Fusion, TheGeluFoldIsReportedAsAFoldAndNotAsARefusal)
{
    using Mlp = NN::Sequential<NN::Linear, NN::GELU, NN::Linear>;
    Mlp model(NN::Linear{FEATURES, HIDDEN}, NN::GELU{}, NN::Linear{HIDDEN, CLASSES});

    const Train::TrainingGraph built =
        Train::build_classifier(model, BATCH, FEATURES, CLASSES);

    ASSERT_EQ(built.fusion.count(FusedForm::GEMM_BIAS_GELU), 1u) << built.fusion.to_string();

    for (const Graph::DeclinedFusion& decline : built.fusion.declined())
        EXPECT_NE(decline.reason, Graph::FusionDecline::GRADIENT_NEEDS_PRE_ACTIVATION)
            << built.fusion.to_string();

    // gelu_backward must read the spilled value; see tests/functional/regression/README.md.
    const Graph::FusedSubtree* gelu = nullptr;
    for (const Graph::FusedSubtree& subtree : built.fusion.fused())
        if (subtree.form == FusedForm::GEMM_BIAS_GELU) gelu = &subtree;

    ASSERT_NE(gelu, nullptr);
    ASSERT_NE(gelu->spilled, Graph::NO_VALUE);

    const Graph::NodeId producer = built.graph.value(gelu->spilled).producer;
    ASSERT_NE(producer, Graph::NO_NODE) << "nothing writes the spill";
    EXPECT_EQ(built.graph.node(producer).op, Ops::GEMM_BIAS_GELU.id);

    bool read_by_the_gradient = false;
    for (const Graph::Node& node : built.graph.nodes())
        if (node.op == Ops::GELU_BACKWARD.id && node.inputs.at(1) == gelu->spilled)
            read_by_the_gradient = true;

    EXPECT_TRUE(read_by_the_gradient)
        << "gelu_backward does not read the spilled pre-activation, so the spill is "
           "buying nothing and the fold is the silently wrong one";
}

/**
 * @brief Tests that the spill buffer holds the host-computed pre-activation.
 * @see tests/functional/regression/README.md
 */
TEST_F(Regression_Fusion, TheSpillHoldsThePreActivationItself)
{
    SushiBLAS::Engine engine(SaTest::TestEnvironment::context());

    constexpr int64_t ROWS = 4;
    constexpr int64_t IN_DIM = 3;
    constexpr int64_t OUT_DIM = 5;

    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);

    const Graph::ValueId x = builder.input(Shape{ROWS, IN_DIM}, "x");
    const Graph::ValueId w = builder.parameter(Shape{OUT_DIM, IN_DIM}, "w");
    const Graph::ValueId b = builder.parameter(Shape{OUT_DIM}, "b");
    const Graph::ValueId pre = builder.bias_add(builder.matmul(x, w, false, true), b);
    const Graph::ValueId out = builder.gelu(pre);

    // `pre` is unobserved; the fold rests on the spill; see tests/functional/regression/README.md.
    const std::vector<Graph::ValueId> observed{out};
    FusionOptions options;
    options.observed = SushiRuntime::span<const Graph::ValueId>(observed.data(), observed.size());

    const Graph::FusionReport report = Graph::fuse(graph, options);
    ASSERT_EQ(report.count(FusedForm::GEMM_BIAS_GELU), 1u) << report.to_string();

    TensorPool pool(engine);
    pool.materialise_leaves(graph);

    SaTest::HostNoise noise(0xC0FFEEu);
    std::vector<double> x_values(static_cast<std::size_t>(ROWS * IN_DIM));
    std::vector<double> w_values(static_cast<std::size_t>(OUT_DIM * IN_DIM));
    std::vector<double> b_values(static_cast<std::size_t>(OUT_DIM));
    noise.fill(SushiRuntime::span<double>(x_values.data(), x_values.size()), 2.0);
    noise.fill(SushiRuntime::span<double>(w_values.data(), w_values.size()), 2.0);
    noise.fill(SushiRuntime::span<double>(b_values.data(), b_values.size()), 2.0);

    SaTest::write_tensor(pool.tensor(x),
                         SushiRuntime::span<const double>(x_values.data(), x_values.size()));
    SaTest::write_tensor(pool.tensor(w),
                         SushiRuntime::span<const double>(w_values.data(), w_values.size()));
    SaTest::write_tensor(pool.tensor(b),
                         SushiRuntime::span<const double>(b_values.data(), b_values.size()));

    Graph::Lowering lowering(engine, pool);
    lowering.record(graph);
    ASSERT_EQ(1u, lowering.emitted_operations())
        << "this machine's device declined the fused form, so the epilogue never ran and "
           "this test is not exercising what it claims to";

    engine.execute();
    SaTest::TestEnvironment::context().wait_all();

    const std::vector<double> spilled = SaTest::read_tensor(pool.tensor(pre));
    const std::vector<double> activated = SaTest::read_tensor(pool.tensor(out));

    ASSERT_EQ(spilled.size(), static_cast<std::size_t>(ROWS * OUT_DIM));

    // The reference is computed on the host; see tests/functional/regression/README.md.
    std::size_t separated = 0;

    for (int64_t row = 0; row < ROWS; ++row)
        for (int64_t col = 0; col < OUT_DIM; ++col)
        {
            double expected = b_values[static_cast<std::size_t>(col)];
            for (int64_t k = 0; k < IN_DIM; ++k)
                expected += x_values[static_cast<std::size_t>(row * IN_DIM + k)]
                          * w_values[static_cast<std::size_t>(col * IN_DIM + k)];

            const std::size_t index = static_cast<std::size_t>(row * OUT_DIM + col);
            EXPECT_NEAR(spilled[index], expected, 1e-5)
                << "the spill at (" << row << ", " << col << ") is not the pre-activation";

            // Only a negative element separates them; see tests/functional/regression/README.md.
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
 * @see tests/functional/regression/README.md
 */
TEST_F(Regression_Fusion, ASelectorRefusalStillFillsTheSpillAndMovesNoBit)
{
    class RefuseEverything final : public Graph::IFusionSelector
    {
        public:
            Graph::FusionChoice choose(const Graph::FusionQuery&) const override
            {
                return Graph::FusionChoice{FusedForm::NONE, "this device wants none of it"};
            }
    };

    using Mlp = NN::Sequential<NN::Linear, NN::GELU, NN::Linear>;

    Mlp taken_model(NN::Linear{FEATURES, HIDDEN}, NN::GELU{}, NN::Linear{HIDDEN, CLASSES});
    Mlp refused_model(NN::Linear{FEATURES, HIDDEN}, NN::GELU{}, NN::Linear{HIDDEN, CLASSES});

    const RefuseEverything refuse;
    const TrainedRun taken = train(taken_model, FusionOptions{});
    const TrainedRun refused = train(refused_model, FusionOptions{}, &refuse);

    EXPECT_EQ(taken.fused_subtrees, refused.fused_subtrees);
    EXPECT_EQ(refused.selector_refusals, refused.fused_subtrees);
    EXPECT_GT(refused.operations_per_step, taken.operations_per_step);

    expect_runs_agree(taken, refused, "a declined GELU epilogue that still spills");
}

/**
 * @brief Tests that disabling the GELU form trains to the same bytes as folding it.
 * @see tests/functional/regression/README.md
 */
TEST_F(Regression_Fusion, DisablingTheGeluFormTrainsToTheSamePlaceAsFoldingIt)
{
    using Mlp = NN::Sequential<NN::Linear, NN::GELU, NN::Linear>;

    Mlp folded_model(NN::Linear{FEATURES, HIDDEN}, NN::GELU{}, NN::Linear{HIDDEN, CLASSES});
    Mlp separate_model(NN::Linear{FEATURES, HIDDEN}, NN::GELU{}, NN::Linear{HIDDEN, CLASSES});

    const std::vector<FusedForm> without{FusedForm::GEMM_BIAS_GELU};
    FusionOptions narrowed;
    narrowed.disabled = SushiRuntime::span<const FusedForm>(without.data(), without.size());

    const TrainedRun folded = train(folded_model, FusionOptions{});
    const TrainedRun separate = train(separate_model, narrowed);

    EXPECT_EQ(folded.fused_subtrees, separate.fused_subtrees)
        << "switching the three-node form off must fall back to gemm+bias, not to nothing";

    // Disabling the form costs exactly two kernels; see tests/functional/regression/README.md.
    EXPECT_EQ(separate.operations_per_step, folded.operations_per_step + 2u);

    expect_runs_agree(folded, separate, "switching the fused GELU epilogue off");
}

/**
 * @brief Tests that the fused ReLU MLP step issues ten fewer operations than the unfused one.
 * @see tests/functional/regression/README.md
 */
TEST_F(Regression_Fusion, TheFusedStepIssuesTenFewerKernelsPerStep)
{
    using Mlp = NN::Sequential<NN::Linear, NN::ReLU, NN::Linear>;

    Mlp fused_model(NN::Linear{FEATURES, HIDDEN}, NN::ReLU{}, NN::Linear{HIDDEN, CLASSES});
    Mlp plain_model(NN::Linear{FEATURES, HIDDEN}, NN::ReLU{}, NN::Linear{HIDDEN, CLASSES});

    const TrainedRun fused = train(fused_model, FusionOptions{});
    const TrainedRun plain = train(plain_model, off());

    EXPECT_EQ(plain.operations_per_step, fused.operations_per_step + 10u);

    // Fusing adds no ordering node and no compile; see tests/functional/regression/README.md.
    EXPECT_EQ(fused.compiles, 1u);
    EXPECT_EQ(plain.compiles, 1u);
}

/**
 * @brief Tests that a fused GEMM depends on the task that writes the bias its epilogue reads.
 * @see tests/functional/regression/README.md
 */
TEST_F(Regression_Fusion, AFusedGemmIsOrderedAgainstTheTaskThatWritesItsBias)
{
    SushiBLAS::Engine engine(SaTest::TestEnvironment::context());

    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueId x = builder.input(Shape{BATCH, FEATURES}, "x");
    const Graph::ValueId w = builder.parameter(Shape{HIDDEN, FEATURES}, "w");
    const Graph::ValueId b = builder.parameter(Shape{HIDDEN}, "b");
    builder.relu(builder.bias_add(builder.matmul(x, w, false, true), b));

    const Graph::FusionReport fusion = Graph::fuse(graph, FusionOptions{});
    ASSERT_EQ(1u, fusion.fused().size()) << fusion.to_string();

    TensorPool pool(engine);
    pool.materialise_leaves(graph);

    // Recorded first: the edge can only run to the GEMM; see tests/functional/regression/README.md.
    SushiBLAS::Tensor& bias = pool.tensor(b);
    engine.elementwise().mul(bias, bias, bias);

    Graph::Lowering lowering(engine, pool);
    lowering.record(graph);
    ASSERT_EQ(1u, lowering.emitted_operations())
        << "this machine's device declined the fused form, so there is no epilogue to "
           "order and this test is not exercising what it claims to";

    auto nodes = engine.get_graph().take_nodes();
    ASSERT_EQ(2u, nodes.size());
    EXPECT_EQ(1, nodes[1]->in_degree.load(std::memory_order_relaxed))
        << "The fused GEMM must depend on the task that wrote the bias its epilogue "
           "dereferences. An in-degree of 0 means the read was never declared to "
           "SushiBLAS, and the optimizer's update is free to land before the forward "
           "pass has read the old value - which still trains, and still reports success.";
}

/**
 * @brief Tests that the reader of the spill depends on the fused GEMM that fills it.
 * @see tests/functional/regression/README.md
 */
TEST_F(Regression_Fusion, WhateverReadsTheSpillIsOrderedBehindTheGemmThatFillsIt)
{
    SushiBLAS::Engine engine(SaTest::TestEnvironment::context());

    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueId x = builder.input(Shape{BATCH, FEATURES}, "x");
    const Graph::ValueId w = builder.parameter(Shape{HIDDEN, FEATURES}, "w");
    const Graph::ValueId b = builder.parameter(Shape{HIDDEN}, "b");
    const Graph::ValueId pre = builder.bias_add(builder.matmul(x, w, false, true), b);
    const Graph::ValueId out = builder.gelu(pre);

    // Emitted after the GELU, as gelu_backward is; see tests/functional/regression/README.md.
    const Graph::ValueId dy = builder.input(Shape{BATCH, HIDDEN}, "dy");
    (void)builder.activation_backward(Ops::GELU_BACKWARD.id, dy, pre);

    const std::vector<Graph::ValueId> observed{out};
    FusionOptions options;
    options.observed = SushiRuntime::span<const Graph::ValueId>(observed.data(), observed.size());

    const Graph::FusionReport fusion = Graph::fuse(graph, options);
    ASSERT_EQ(1u, fusion.count(FusedForm::GEMM_BIAS_GELU)) << fusion.to_string();

    TensorPool pool(engine);
    pool.materialise_leaves(graph);

    Graph::Lowering lowering(engine, pool);
    lowering.record(graph);
    ASSERT_EQ(2u, lowering.emitted_operations())
        << "expected the fused GEMM and the activation gradient; this machine's device may "
           "have declined the fused form, in which case this test is not exercising what "
           "it claims to";

    auto nodes = engine.get_graph().take_nodes();
    ASSERT_EQ(2u, nodes.size());
    EXPECT_EQ(1, nodes[1]->in_degree.load(std::memory_order_relaxed))
        << "The task that reads the spilled pre-activation must depend on the fused GEMM "
           "whose epilogue wrote it. An in-degree of 0 means `.writes(aux)` was never "
           "declared, the two may run concurrently, and the spill buys nothing at all - "
           "while the run still trains and still reports success.";
}

/**
 * @brief Tests that disabling GEMM_BIAS_RELU falls back to GEMM_BIAS and moves no byte.
 * @see tests/functional/regression/README.md
 */
TEST_F(Regression_Fusion, DisablingOneCatalogEntryLeavesTheRestAlone)
{
    using Mlp = NN::Sequential<NN::Linear, NN::ReLU, NN::Linear>;

    Mlp whole_model(NN::Linear{FEATURES, HIDDEN}, NN::ReLU{}, NN::Linear{HIDDEN, CLASSES});
    Mlp narrowed_model(NN::Linear{FEATURES, HIDDEN}, NN::ReLU{}, NN::Linear{HIDDEN, CLASSES});

    const std::vector<FusedForm> without{FusedForm::GEMM_BIAS_RELU};
    FusionOptions narrowed;
    narrowed.disabled = SushiRuntime::span<const FusedForm>(without.data(), without.size());

    const TrainedRun whole = train(whole_model, FusionOptions{});
    const TrainedRun some = train(narrowed_model, narrowed);

    // A catalog refusal costs two kernels, not one; see tests/functional/regression/README.md.
    EXPECT_EQ(some.operations_per_step, whole.operations_per_step + 2u);
    expect_runs_agree(whole, some, "switching the fused ReLU epilogue off");
}

/**
 * @brief Tests that a selector refusing every form yields an unfused plan with identical bytes.
 * @see tests/functional/regression/README.md
 */
TEST_F(Regression_Fusion, ASelectorMayRefuseAMatchedFormAndTheNumbersDoNotMove)
{
    class RefuseEverything final : public Graph::IFusionSelector
    {
        public:
            Graph::FusionChoice choose(const Graph::FusionQuery&) const override
            {
                return Graph::FusionChoice{FusedForm::NONE, "this device wants none of it"};
            }
    };

    using Mlp = NN::Sequential<NN::Linear, NN::ReLU, NN::Linear>;

    Mlp taken_model(NN::Linear{FEATURES, HIDDEN}, NN::ReLU{}, NN::Linear{HIDDEN, CLASSES});
    Mlp refused_model(NN::Linear{FEATURES, HIDDEN}, NN::ReLU{}, NN::Linear{HIDDEN, CLASSES});

    const RefuseEverything refuse;
    const TrainedRun taken = train(taken_model, FusionOptions{});
    const TrainedRun refused = train(refused_model, FusionOptions{}, &refuse);

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
                                              SushiBLAS::ExecutionMode::AUTO);
    const Graph::DeviceFusionSelector with(pretend_device(true),
                                           SushiBLAS::ExecutionMode::AUTO);

    Graph::FusionQuery query;
    query.matched = FusedForm::GEMM_BIAS_RELU;
    query.dtype = DType::FLOAT64;
    query.rows = 32;
    query.columns = 128;

    EXPECT_EQ(without.choose(query).form, FusedForm::NONE);
    EXPECT_EQ(with.choose(query).form, FusedForm::GEMM_BIAS_RELU);

    // A float graph is fine on either.
    query.dtype = DType::FLOAT32;
    EXPECT_EQ(without.choose(query).form, FusedForm::GEMM_BIAS_RELU);
}

/**
 * @brief Tests that host-executing modes are refused a GEMM epilogue and keep the zip form.
 * @see tests/functional/regression/README.md
 */
TEST_F(Regression_Fusion, AHostExecutingModeDoesNotGetAnEpilogueThatDereferencesDeviceMemory)
{
    Graph::FusionQuery gemm;
    gemm.matched = FusedForm::GEMM_BIAS;
    gemm.rows = 32;
    gemm.columns = 128;

    for (SushiBLAS::ExecutionMode mode : {SushiBLAS::ExecutionMode::FORCE_REFERENCE,
                                          SushiBLAS::ExecutionMode::VERIFY})
    {
        const Graph::DeviceFusionSelector selector(pretend_device(true), mode);
        EXPECT_EQ(selector.choose(gemm).form, FusedForm::NONE)
            << "mode " << static_cast<int>(mode);

        // The zip form runs no host arm; see tests/functional/regression/README.md.
        Graph::FusionQuery zip;
        zip.matched = FusedForm::EXP_SUB_SCALE;
        zip.rows = 32;
        zip.columns = 128;
        EXPECT_EQ(selector.choose(zip).form, FusedForm::EXP_SUB_SCALE);
    }
}

TEST_F(Regression_Fusion, AStridedBiasIsNotHandedToAnEpilogueThatIndexesItDirectly)
{
    const Graph::DeviceFusionSelector selector(pretend_device(true),
                                               SushiBLAS::ExecutionMode::AUTO);

    Graph::FusionQuery query;
    query.matched = FusedForm::GEMM_BIAS;
    query.rows = 32;
    query.columns = 128;
    query.operand_is_contiguous = false;

    EXPECT_EQ(selector.choose(query).form, FusedForm::NONE);
}

/**
 * @brief Tests that every selector choice carries a non-empty reason.
 * @see tests/functional/regression/README.md
 */
TEST_F(Regression_Fusion, EveryChoiceCarriesAReasonSomebodyCanRead)
{
    const Graph::DeviceFusionSelector selector(pretend_device(true),
                                               SushiBLAS::ExecutionMode::AUTO);

    Graph::FusionQuery query;
    query.matched = FusedForm::GEMM_BIAS_RELU;
    query.rows = 32;
    query.columns = 128;

    EXPECT_FALSE(selector.choose(query).reason.empty());

    query.operand_is_contiguous = false;
    EXPECT_FALSE(selector.choose(query).reason.empty());
}
