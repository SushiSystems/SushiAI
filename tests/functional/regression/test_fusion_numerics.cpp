/****************************************************************/
/* test_fusion_numerics.cpp                                     */
/* SushiAI - https://github.com/SushiSystems/SushiAI            */
/* Copyright (c) 2026 Sushi Systems                             */
/* Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.    */
/* Commercial use requires a licence from Sushi Systems.        */
/****************************************************************/

// The gate on collapsing the forward and backward graph into fewer kernels.
//
// Fusion is a pure performance change over the only arithmetic in the library
// nothing downstream validates, and every way it can be wrong is silent:
//
//  - an epilogue that adds the bias at the wrong coordinate still produces a
//    loss that falls, because the model just learns a different bias;
//  - an activation folded into a GEMM whose gradient needed the pre-activation
//    trains towards the wrong place and reports nothing at all;
//  - a fused kernel reading memory the dependency tracker cannot see races its
//    own producer, and a race that usually resolves the right way looks exactly
//    like training.
//
// So the fused graph is not tested against a tolerance. It is tested against
// the unfused graph it replaced, byte for byte, from identical state.
//
// WHY BYTE FOR BYTE IS THE RIGHT BAR
//
// The same argument test_fused_optimizer.cpp makes for the optimizer. Every
// element is computed by the same operations in the same order; the one
// plausible source of a moved bit - a compiler contracting `a * b + c` into an
// FMA inside one kernel where it could not across two - is ruled out by
// `-ffp-contract=off`, which the whole stack inherits through SushiRuntime's
// `sushiruntime_deterministic_fp` contract. Each fused functor is written in
// the operand order of the calls it replaces, and where a constant reached a
// kernel through a `float` parameter it still does.
//
// If this fails, the answer is to find which expression moved. Not to widen it.
//
// THE SECOND HALF: THE ACTIVATION THAT NEEDS A SPILL TO BE FUSED
//
// GELU's gradient differentiates at its input, so the epilogue that folds it
// has to write that input to an auxiliary buffer on its way past -
// cuBLASLt's GELU_AUX_BIAS, FusedForm::GEMM_BIAS_GELU here. Every way of
// getting that wrong is silent, and worse than the others: a forward pass that
// matches perfectly while the gradients are taken at the wrong point still
// converges, just somewhere else.
//
// So a forward comparison is not the test and is not treated as one. The test
// is that a GELU model trained end to end reaches byte-identical *parameters*
// after 24 optimizer steps - which is a statement about the gradients, since
// nothing else moves a weight - and it is taken three ways, against three
// different fillings of the same buffer:
//
//   - AGeluMlpTrainsIdenticallyFusedAndUnfused, against the graph with no
//     fusion pass at all;
//   - DisablingTheGeluFormTrainsToTheSamePlaceAsFoldingIt, against the same
//     graph with only this catalog row switched off, so the difference is the
//     entry and nothing else;
//   - ASelectorRefusalStillFillsTheSpillAndMovesNoBit, against a selector that
//     declines the fused arm, so the aux buffer is filled by a separate copy in
//     the lowering rather than from inside the epilogue.
//
// TheSpillHoldsThePreActivationItself then reads the buffer directly and
// compares it with a host-computed pre-activation, because three spellings can
// agree while all three hold the wrong tensor.

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include "../../common/test_utils.hpp"

using namespace SushiAI;

// `Test` alone resolves to ::testing::Test inside a TEST_F body, where the
// fixture's base injects the name; the alias keeps the helpers reachable.
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
     *
     * A fresh Engine, because reproducibility is a property of one - see
     * dataset.hpp on the RNG offset counter. AdamW rather than plain SGD
     * because its division by an accumulated square root is where a one-ULP
     * difference stops staying one ULP.
     *
     * @tparam M The model type.
     * @param model The model to train; its parameters are declared by the trace.
     * @param fusion What the fusion pass is allowed to do.
     * @param selector Which spelling the lowering should use, or null for this
     *        machine's own device.
     * @return The trajectory, the final parameters and the plan's shape.
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
     * @brief A device profile that is not this machine's.
     *
     * The one device here is an OpenCL CPU with no dedicated scratchpad and, as
     * it happens, doubles. A selector rule that can only be exercised on
     * hardware nobody owns is a rule nobody has run, so the declining branches
     * are driven from a described device instead.
     *
     * @param fp64 Whether the pretend device supports double precision.
     * @return A plausible profile.
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

// =========================================================================
// 1. Fusing changes no number
// =========================================================================

TEST_F(Regression_Fusion, AReluMlpTrainsIdenticallyFusedAndUnfused)
{
    // The demo's own model. Both linear layers fuse - the first with its ReLU,
    // the second without one - and so does the cross-entropy gradient's
    // elementwise tail, which is three separate collapses reaching the same
    // arithmetic in one run.
    using Mlp = NN::Sequential<NN::Linear, NN::ReLU, NN::Linear>;

    Mlp fused_model(NN::Linear{FEATURES, HIDDEN}, NN::ReLU{}, NN::Linear{HIDDEN, CLASSES});
    Mlp plain_model(NN::Linear{FEATURES, HIDDEN}, NN::ReLU{}, NN::Linear{HIDDEN, CLASSES});

    const TrainedRun fused = train(fused_model, FusionOptions{});
    const TrainedRun plain = train(plain_model, off());

    ASSERT_GT(fused.fused_subtrees, 0u);
    ASSERT_EQ(plain.fused_subtrees, 0u);

    expect_runs_agree(fused, plain, "fusing the forward and backward graph");
}

TEST_F(Regression_Fusion, ATanhMlpTrainsIdenticallyFusedAndUnfused)
{
    // tanh's gradient reads its own output, so the pre-activation is dead and
    // the whole layer folds. Its epilogue is a different template instantiation
    // from ReLU's and gets its own comparison for that reason.
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

// =========================================================================
// 2. The activation that must NOT be fused
// =========================================================================

TEST_F(Regression_Fusion, AGeluMlpTrainsIdenticallyFusedAndUnfused)
{
    // The trap, end to end, and the claim that matters is about the *gradients*
    // rather than the forward pass. A GEMM_BIAS_GELU that computed the right
    // activation and spilled nothing would match the unfused forward exactly
    // and still differentiate at the wrong tensor - nothing would throw, no
    // value would go non-finite, and the loss would still fall. What it could
    // not do is land the parameters on the same bits after 24 AdamW steps,
    // because a weight only moves by its gradient.
    using Mlp = NN::Sequential<NN::Linear, NN::GELU, NN::Linear>;

    Mlp fused_model(NN::Linear{FEATURES, HIDDEN}, NN::GELU{}, NN::Linear{HIDDEN, CLASSES});
    Mlp plain_model(NN::Linear{FEATURES, HIDDEN}, NN::GELU{}, NN::Linear{HIDDEN, CLASSES});

    const TrainedRun fused = train(fused_model, FusionOptions{});
    const TrainedRun plain = train(plain_model, off());

    // Two linear layers - the first folding its GELU, the second with no
    // activation to fold - and the cross-entropy gradient tail.
    EXPECT_EQ(fused.fused_subtrees, 3u)
        << "expected gemm+bias+gelu, gemm+bias and the cross-entropy gradient tail";
    ASSERT_EQ(plain.fused_subtrees, 0u);

    expect_runs_agree(fused, plain, "a GELU epilogue spilling its pre-activation");

    // And it really did learn something, so "identical" is not two identically
    // broken runs.
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

    // The spilled value is the one the backward rule reads. Checked on the IR
    // rather than through a loss, because "the gradient is taken at the right
    // tensor" has no numerical symptom when it is wrong.
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

TEST_F(Regression_Fusion, TheSpillHoldsThePreActivationItself)
{
    // Read directly, not inferred from a downstream consequence. Everything
    // else about this entry is checked through gradients that agree, and two
    // spellings can agree while both hold the wrong tensor.
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

    // `pre` is deliberately NOT observed: the fold has to survive on the spill
    // and not on a pin, or this test would be pinning the wrong thing.
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

    // The reference the spill is compared against is computed here, on the
    // host, from the inputs - not read out of some other tensor the same kernel
    // wrote.
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

            // And it is emphatically not the *post*-activation, which is the
            // value a spill-less fold would leave for the backward pass. Where
            // that check is made matters: gelu(x) tends to x for large positive
            // x, so the two agree there to within a rounding error and an
            // assertion taken on |x| large would be satisfied by either tensor.
            // On the negative side gelu tends to zero instead, so a
            // sufficiently negative element separates them by about |x|.
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

TEST_F(Regression_Fusion, ASelectorRefusalStillFillsTheSpillAndMovesNoBit)
{
    // The control that makes the case above a test rather than a description.
    // With the fused arm declined, the aux buffer is filled by a separate copy
    // in Lowering::record_fused_gemm instead of by the epilogue - a completely
    // different code path to the same buffer - and the gradients have to come
    // out identical. Delete either write and this comparison fails.
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

TEST_F(Regression_Fusion, DisablingTheGeluFormTrainsToTheSamePlaceAsFoldingIt)
{
    // The A/B the brief for this entry is really about: the same model with the
    // catalog row on and off. Off, the GELU is an ordinary IR node and the
    // pre-activation is an ordinary value; on, it is a second output of a
    // fused GEMM written from inside a device kernel. The parameters after 24
    // steps have to be the same bits, and they are only the same bits if the
    // gradients were.
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

    // Two kernels more, exactly as switching the ReLU form off costs two: the
    // activation becomes an out-of-place IR node again, which is a copy and a
    // gelu. The spill costs nothing on this side, because the value the fused
    // node spilled is the value BIAS_ADD writes anyway.
    EXPECT_EQ(separate.operations_per_step, folded.operations_per_step + 2u);

    expect_runs_agree(folded, separate, "switching the fused GELU epilogue off");
}

// =========================================================================
// 3. What it actually cost, counted
// =========================================================================

TEST_F(Regression_Fusion, TheFusedStepIssuesTenFewerKernelsPerStep)
{
    // The headline claim, counted rather than timed - a wall clock says
    // different things on a loaded machine, and this is the number the change
    // actually controls.
    //
    // Ten, for this model: the first layer's gemm+bias+relu replaces five calls
    // (gemm; copy and ger for the bias; copy and relu) with one, the second
    // layer's gemm+bias replaces three with one, and the cross-entropy
    // gradient's exp/sub/scale replaces five with one.
    using Mlp = NN::Sequential<NN::Linear, NN::ReLU, NN::Linear>;

    Mlp fused_model(NN::Linear{FEATURES, HIDDEN}, NN::ReLU{}, NN::Linear{HIDDEN, CLASSES});
    Mlp plain_model(NN::Linear{FEATURES, HIDDEN}, NN::ReLU{}, NN::Linear{HIDDEN, CLASSES});

    const TrainedRun fused = train(fused_model, FusionOptions{});
    const TrainedRun plain = train(plain_model, off());

    EXPECT_EQ(plain.operations_per_step, fused.operations_per_step + 10u);

    // And no price beside it. Each fused GEMM used to need a host-side ordering
    // node carrying the dependency edges execute_gemm could not declare for a
    // bias its functor dereferences - two layers, two extra nodes. SushiBLAS's
    // seam takes the epilogue's own access set now, so `operations_per_step` is
    // the whole plan again and the two counts cannot drift apart.
    EXPECT_EQ(fused.compiles, 1u);
    EXPECT_EQ(plain.compiles, 1u);
}

TEST_F(Regression_Fusion, AFusedGemmIsOrderedAgainstTheTaskThatWritesItsBias)
{
    // The hazard the fused spelling carries and the unfused one does not: the
    // epilogue dereferences the bias from inside the GEMM kernel, and the
    // optimizer rewrites that same bias inside this same compiled plan.
    // SushiBLAS cannot see into the functor, so the declaration is this
    // library's to make - `gemm_epilogue(f).reads(bias)` in
    // Lowering::record_fused_gemm.
    //
    // Asserted on `in_degree`, never on a loss. With no edge the two tasks may
    // still come out in the right order on any given run, so a loss that falls
    // proves nothing at all about ordering - which is exactly why this failure
    // mode survives an otherwise green suite. The node count is asserted for
    // the same reason it is worth stating: this used to be three nodes, the
    // third being a host-side ordering node that ran nothing and existed only
    // to carry this edge.
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

    // Stands in for the optimizer's update: a task in this same plan that
    // writes the very bias the epilogue reads. Recorded first, so an edge can
    // only run from it to the GEMM.
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

TEST_F(Regression_Fusion, WhateverReadsTheSpillIsOrderedBehindTheGemmThatFillsIt)
{
    // The other half of the epilogue's access set, and the half only the
    // spilling arm has. `.reads(bias)` orders the GEMM behind whoever writes
    // the bias; `.writes(aux)` orders whoever reads the spill behind the GEMM.
    // Without it the backward pass that differentiates at the pre-activation
    // races the kernel that produced it - and a race that usually resolves the
    // right way is indistinguishable from training, which is why this is
    // asserted on in_degree and never on a loss.
    SushiBLAS::Engine engine(SaTest::TestEnvironment::context());

    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueId x = builder.input(Shape{BATCH, FEATURES}, "x");
    const Graph::ValueId w = builder.parameter(Shape{HIDDEN, FEATURES}, "w");
    const Graph::ValueId b = builder.parameter(Shape{HIDDEN}, "b");
    const Graph::ValueId pre = builder.bias_add(builder.matmul(x, w, false, true), b);
    const Graph::ValueId out = builder.gelu(pre);

    // A reader of the pre-activation, emitted after the GELU exactly as the
    // differentiation pass emits gelu_backward.
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

TEST_F(Regression_Fusion, DisablingOneCatalogEntryLeavesTheRestAlone)
{
    // The A/B knob the measurement above needs, and the escape hatch a caller
    // who has measured otherwise on their model needs. Switching the three-op
    // GEMM form off must fall back to the two-op one rather than to nothing.
    using Mlp = NN::Sequential<NN::Linear, NN::ReLU, NN::Linear>;

    Mlp whole_model(NN::Linear{FEATURES, HIDDEN}, NN::ReLU{}, NN::Linear{HIDDEN, CLASSES});
    Mlp narrowed_model(NN::Linear{FEATURES, HIDDEN}, NN::ReLU{}, NN::Linear{HIDDEN, CLASSES});

    const std::vector<FusedForm> without{FusedForm::GEMM_BIAS_RELU};
    FusionOptions narrowed;
    narrowed.disabled = SushiRuntime::span<const FusedForm>(without.data(), without.size());

    const TrainedRun whole = train(whole_model, FusionOptions{});
    const TrainedRun some = train(narrowed_model, narrowed);

    // Two kernels more, not one, and the difference between the two ways of
    // declining is worth knowing. Refused *here*, at the catalog, the ReLU
    // stays an ordinary IR node - and an IR activation is out of place, so it
    // lowers to a copy and a ReLU. Refused by the *selector* instead (the test
    // below), the node is already fused and its output tensor is the
    // activation's own result, so the trailing ReLU runs in place and costs
    // one. Same numbers either way; different launch counts.
    EXPECT_EQ(some.operations_per_step, whole.operations_per_step + 2u);
    expect_runs_agree(whole, some, "switching the fused ReLU epilogue off");
}

// =========================================================================
// 4. The dynamic selector
// =========================================================================

TEST_F(Regression_Fusion, ASelectorMayRefuseAMatchedFormAndTheNumbersDoNotMove)
{
    // The seam's whole promise: the static pass decides what *could* be fused
    // and the selector decides what is, so a device that wants none of it gets
    // an unfused plan out of a fused graph - with the same numbers, because
    // every spelling of a fused node is the same arithmetic.
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

TEST_F(Regression_Fusion, AHostExecutingModeDoesNotGetAnEpilogueThatDereferencesDeviceMemory)
{
    // Under FORCE_REFERENCE and VERIFY the epilogue is invoked on the host by
    // Kernels::Reference::gemm, so any pointer it carries must be host-readable
    // there. SushiAI's own allocations are shared USM and would be; a bound
    // one need not be, and nothing here can ask which it got.
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

        // The zip-based form runs no host arm, so it is unaffected - a blanket
        // refusal here would be a different bug from the one being prevented.
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

TEST_F(Regression_Fusion, EveryChoiceCarriesAReasonSomebodyCanRead)
{
    // A selector nobody can interrogate is a selector nobody will trust, for
    // the same reason the static pass carries a report: both of its answers
    // produce a plan that trains.
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
