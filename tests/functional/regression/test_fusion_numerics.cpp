/**************************************************************************/
/* test_fusion_numerics.cpp                                               */
/**************************************************************************/
/*                          This file is part of:                         */
/*                                 SushiAI                                */
/**************************************************************************/

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
// THE SECOND HALF: THE ACTIVATION THAT CANNOT BE FUSED
//
// GELU's gradient differentiates at its input, so no catalog entry folds it and
// the static pass says so out loud. That is checked twice over: the pass's own
// refusal is a unit test, and here a GELU model is trained end to end and made
// to match the unfused spelling exactly - because "we declined to fuse it" and
// "we declined to fuse it *and the bias still fused correctly around it*" are
// different claims, and only the second one is worth anything.

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
        std::size_t ordering_nodes_per_step = 0;
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
        run.ordering_nodes_per_step = report.ordering_nodes_per_step;

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

TEST_F(Regression_Fusion, AGeluMlpStillTrainsCorrectlyAndItsActivationStaysOutside)
{
    // The trap, end to end. If GELU were folded into the GEMM, the backward
    // pass would differentiate at the activation's *output* where its rule
    // needs the input - nothing would throw, no value would go non-finite, and
    // the loss would still fall. So the check is not "it trained", it is "it
    // trained to exactly the same place as the unfused graph", plus the
    // structural claim that the bias fused and the activation did not.
    using Mlp = NN::Sequential<NN::Linear, NN::GELU, NN::Linear>;

    Mlp fused_model(NN::Linear{FEATURES, HIDDEN}, NN::GELU{}, NN::Linear{HIDDEN, CLASSES});
    Mlp plain_model(NN::Linear{FEATURES, HIDDEN}, NN::GELU{}, NN::Linear{HIDDEN, CLASSES});

    const TrainedRun fused = train(fused_model, FusionOptions{});
    const TrainedRun plain = train(plain_model, off());

    // Two linear layers, both with their bias folded in, and no activation
    // folded anywhere.
    EXPECT_EQ(fused.fused_subtrees, 3u)
        << "expected two gemm+bias layers and the cross-entropy gradient tail";
    EXPECT_GT(fused.declined, 0u) << "the GELU refusal is not in the report";

    expect_runs_agree(fused, plain, "a GELU model whose activation cannot be fused");

    // And it really did learn something, so "identical" is not two identically
    // broken runs.
    ASSERT_GE(fused.losses.size(), 2u);
    EXPECT_LT(fused.losses.back(), fused.losses.front())
        << "the GELU model did not train at all, so agreeing with the control proves nothing";
}

TEST_F(Regression_Fusion, TheGeluRefusalNamesTheReasonRatherThanAUseCount)
{
    using Mlp = NN::Sequential<NN::Linear, NN::GELU, NN::Linear>;
    Mlp model(NN::Linear{FEATURES, HIDDEN}, NN::GELU{}, NN::Linear{HIDDEN, CLASSES});

    const Train::TrainingGraph built =
        Train::build_classifier(model, BATCH, FEATURES, CLASSES);

    bool named = false;
    for (const Graph::DeclinedFusion& decline : built.fusion.declined())
        if (decline.reason == Graph::FusionDecline::GRADIENT_NEEDS_PRE_ACTIVATION)
            named = true;

    EXPECT_TRUE(named) << built.fusion.to_string();
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

    // And the price: one host-side ordering node per fused GEMM epilogue,
    // carrying the dependency edges execute_gemm cannot declare for a bias its
    // functor dereferences. Two layers, two nodes, no kernels.
    EXPECT_EQ(fused.ordering_nodes_per_step, 2u);
    EXPECT_EQ(plain.ordering_nodes_per_step, 0u);

    EXPECT_EQ(fused.compiles, 1u);
    EXPECT_EQ(plain.compiles, 1u);
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
    narrowed.disabled = std::span<const FusedForm>(without.data(), without.size());

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
    EXPECT_EQ(refused.ordering_nodes_per_step, 0u)
        << "an unfused spelling reads no memory the tracker cannot see, so it needs no "
           "ordering node";

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
