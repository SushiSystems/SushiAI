/**************************************************************************/
/* test_all.cpp                                                           */
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
 * @file test_all.cpp
 * @brief Tests the backward rule table: its coverage, and every rule against a central difference
 *        in FLOAT64.
 * @author Mustafa Garip
 */

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <gtest/gtest.h>
#include <SushiRuntime/core/span.hpp>
#include <SushiBLAS/engine/engine.hpp>
#include <SushiBLAS/tensor/tensor.hpp>
#include <SushiAI/autograd/backward.hpp>
#include <SushiAI/autograd/gradcheck.hpp>
#include <SushiAI/autograd/rules/all.hpp>
#include <SushiAI/engine_fixture/engine_test.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/graph/compiled_step.hpp>
#include <SushiAI/graph/fusion.hpp>
#include <SushiAI/graph/fusion_selector.hpp>
#include <SushiAI/graph/tensor_pool.hpp>
#include <SushiAI/seeded_noise/host_noise.hpp>
#include <SushiAI/tensor_values/tensor_values.hpp>

using namespace SushiAI;
// SaTest stays reachable inside a TEST_F body; see tests/core/autograd/README.md.
namespace SaTest = SushiAI::Test;
using Graph::CompiledStep;
using Graph::GraphBuilder;
using Graph::TensorPool;
using Graph::ValueID;

namespace SushiAI
{
    namespace Test
    {
        TEST(Medium_RuleTable, EveryDomainsRulesReachTheCombinedTable)
        {
            // A domain left out of all.hpp still compiles; see tests/core/autograd/README.md.
            constexpr std::size_t K_DOMAINS =
                Autograd::K_ELEMENTWISE_RULES.size() + Autograd::K_LINALG_RULES.size()
                + Autograd::K_ACTIVATION_RULES.size() + Autograd::K_LOSS_RULES.size()
                + Autograd::K_SHAPE_RULES.size() + Autograd::K_SPATIAL_RULES.size()
                + Autograd::K_NORM_RULES.size() + Autograd::K_RANDOM_RULES.size();

            EXPECT_EQ(Autograd::K_RULES.size(), K_DOMAINS);
        }

        TEST(Medium_RuleTable, NoOperationHasTwoRules)
        {
            // The first of two entries would win silently; see tests/core/autograd/README.md.
            for (std::size_t i = 0; i < Autograd::K_RULES.size(); ++i)
                for (std::size_t j = i + 1; j < Autograd::K_RULES.size(); ++j)
                    EXPECT_NE(Autograd::K_RULES[i].op, Autograd::K_RULES[j].op)
                        << "'" << Ops::name_of(Autograd::K_RULES[i].op)
                        << "' has two backward rules";
        }

        TEST(Medium_RuleTable, TheRulesThatExistedBeforeTheSplitStillDo)
        {
            // These fourteen rules predate the split; see tests/core/autograd/README.md.
            for (OpID op : {Ops::K_MATMUL.id, Ops::K_BIAS_ADD.id, Ops::K_ADD.id,
                                 Ops::K_SUB.id, Ops::K_MUL.id, Ops::K_SCALE.id, Ops::K_EXP.id,
                                 Ops::K_RELU.id, Ops::K_GELU.id, Ops::K_SIGMOID.id,
                                 Ops::K_TANH.id, Ops::K_SUM_ROWS.id, Ops::K_ACCUMULATE.id,
                                 Ops::K_NLL_LOSS.id})
                EXPECT_TRUE(Autograd::has_backward_rule(op))
                    << "'" << Ops::name_of(op) << "' lost its rule in the split";
        }

        /** @brief Tests that no identity declared a gradient barrier has a backward rule. */
        TEST(Medium_RuleTable, NoGradientBarrierHasARule)
        {
            std::size_t barriers = 0;
            for (const Ops::OpName& entry : Ops::K_ALL)
            {
                if (!entry.blocks_gradient)
                    continue;

                ++barriers;
                EXPECT_FALSE(Autograd::has_rule(entry.id)) << entry.name;
                EXPECT_FALSE(Autograd::has_backward_rule(entry.id)) << entry.name;
            }

            EXPECT_EQ(barriers, 4u) << "the loop above checked a different set than it names";
        }

        /** @brief Tests that the comparison is a gradient barrier and has no rule row. */
        TEST(Medium_RuleTable, TheComparisonIsABarrierWithoutARule)
        {
            EXPECT_FALSE(Ops::K_LESS_EQUAL.differentiable);
            EXPECT_TRUE(Ops::K_LESS_EQUAL.blocks_gradient);
            EXPECT_FALSE(Autograd::has_rule(Ops::K_LESS_EQUAL.id));
        }

        /** @brief Tests that each ACTIVATIONS row has a rule for its forward identity alone. */
        TEST(Medium_RuleTable, EveryActivationRowHasARuleForItsForwardIdentityOnly)
        {
            for (const Ops::Activation& entry : Ops::K_ACTIVATIONS)
            {
                EXPECT_TRUE(Autograd::has_rule(entry.forward))
                    << "'" << Ops::name_of(entry.forward) << "' has no backward rule";
                EXPECT_FALSE(Autograd::has_rule(entry.backward))
                    << "'" << Ops::name_of(entry.backward) << "' is a gradient with a rule";
            }

            EXPECT_EQ(Autograd::K_ACTIVATION_RULES.size(), Ops::K_ACTIVATIONS.size());
        }

        /** @brief Tests that the normalisation domain rules BATCHNORM2D and not the fold. */
        TEST(Medium_RuleTable, BatchNormFoldHasNoRuleAndBatchNormKeepsIts)
        {
            EXPECT_TRUE(Autograd::has_rule(Ops::K_BATCHNORM2D.id));
            EXPECT_FALSE(Autograd::has_rule(Ops::K_BATCHNORM_FOLD.id));
            EXPECT_FALSE(Autograd::has_backward_rule(Ops::K_BATCHNORM_FOLD.id));
            EXPECT_EQ(Autograd::K_NORM_RULES.size(), 3u);
        }

        /** @brief Tests that LAYER_NORM has the third normalisation rule and its gradient none. */
        TEST(Medium_RuleTable, LayerNormHasItsRuleAndItsGradientHasNone)
        {
            EXPECT_TRUE(Autograd::has_rule(Ops::K_LAYER_NORM.id));
            EXPECT_FALSE(Ops::K_LAYER_NORM_BACKWARD.differentiable);
            EXPECT_FALSE(Autograd::has_rule(Ops::K_LAYER_NORM_BACKWARD.id));
        }

        /** @brief Tests that LOG_SOFTMAX has the second loss rule and its gradient none. */
        TEST(Medium_RuleTable, LogSoftmaxHasItsRuleAndItsGradientHasNone)
        {
            EXPECT_EQ(Autograd::K_LOSS_RULES.size(), 2u);
            EXPECT_TRUE(Ops::K_LOG_SOFTMAX.differentiable);
            EXPECT_TRUE(Autograd::has_rule(Ops::K_LOG_SOFTMAX.id));
            EXPECT_FALSE(Ops::K_LOG_SOFTMAX_BACKWARD.differentiable);
            EXPECT_FALSE(Autograd::has_rule(Ops::K_LOG_SOFTMAX_BACKWARD.id));
        }

        /** @brief Tests that CAUSAL_MASK has one of the six linear-algebra rules. */
        TEST(Medium_RuleTable, TheCausalMaskHasItsRule)
        {
            EXPECT_EQ(Autograd::K_LINALG_RULES.size(), 6u);
            EXPECT_TRUE(Autograd::has_rule(Ops::K_CAUSAL_MASK.id));
        }

        /** @brief Tests that DROPOUT has the one rule of the random domain. */
        TEST(Medium_RuleTable, DropoutHasTheRuleOfTheRandomDomain)
        {
            EXPECT_EQ(Autograd::K_RANDOM_RULES.size(), 1u);
            EXPECT_TRUE(Ops::K_DROPOUT.differentiable);
            EXPECT_TRUE(Autograd::has_rule(Ops::K_DROPOUT.id));
        }

        /** @brief Tests that the row gather and its scatter-add each have a rule. */
        TEST(Medium_RuleTable, TheGatherAndTheScatterAddDifferentiateIntoEachOther)
        {
            EXPECT_TRUE(Ops::K_GATHER_ROWS.differentiable);
            EXPECT_TRUE(Ops::K_SCATTER_ADD_ROWS.differentiable);
            EXPECT_TRUE(Autograd::has_rule(Ops::K_GATHER_ROWS.id));
            EXPECT_TRUE(Autograd::has_rule(Ops::K_SCATTER_ADD_ROWS.id));
        }

        /** @brief Tests that the depthwise forward has the spatial table's ninth rule. */
        TEST(Medium_RuleTable, TheDepthwiseForwardHasItsRuleAndItsGradientsHaveNone)
        {
            EXPECT_EQ(Autograd::K_SPATIAL_RULES.size(), 9u);
            EXPECT_TRUE(Autograd::has_rule(Ops::K_DEPTHWISE_CONV2D.id));

            // The two gradient identities are not differentiable, so they need no row.
            EXPECT_FALSE(Ops::K_DEPTHWISE_CONV2D_BACKWARD_INPUT.differentiable);
            EXPECT_FALSE(Ops::K_DEPTHWISE_CONV2D_BACKWARD_WEIGHT.differentiable);
            EXPECT_FALSE(Autograd::has_rule(Ops::K_DEPTHWISE_CONV2D_BACKWARD_INPUT.id));
            EXPECT_FALSE(Autograd::has_rule(Ops::K_DEPTHWISE_CONV2D_BACKWARD_WEIGHT.id));
        }
    }
}

namespace
{
    class Medium_Autograd : public SaTest::EngineTest {};

    /**
     * @brief Compiles @p graph into a step; a refusal is a failed expectation and yields null.
     * @param engine Must outlive the returned step.
     * @param graph Must outlive the returned step.
     * @param pool Must outlive the returned step.
     * @return The compiled step, or null when CompiledStep::compile refused the graph.
     */
    std::unique_ptr<CompiledStep> compile_or_record(SushiBLAS::Engine& engine,
                                                    const Graph::Graph& graph, TensorPool& pool)
    {
        Result<std::unique_ptr<CompiledStep>> outcome =
            CompiledStep::compile(engine, graph, pool);
        SA_EXPECT_OK(outcome);

        if (!outcome.has_value())
            return nullptr;

        return std::move(outcome).value();
    }

    /**
     * @brief Holds a FLOAT64 graph whose loss and gradient can both be evaluated at a point.
     * @pre The variable under test is the only gradient-requiring leaf; see
     *      tests/core/autograd/README.md.
     */
    class Probe
    {
        public:
            template<typename BuildFn>
            Probe(SushiBLAS::Engine& engine, Shape variable_shape, BuildFn&& build,
                  bool fuse = false)
                : pool_(engine)
            {
                const GraphBuilder builder(graph_);

                variable_ = builder.parameter(variable_shape, "x", DType::Float64);

                const ValueID logits = build(builder, variable_);
                const Shape logit_shape = graph_.value(logits).shape;

                targets_ = builder.input(logit_shape, "targets", DType::Float64);
                loss_ = builder.nll_loss(logits, targets_);

                const std::vector<ValueID> wrt = {variable_};
                const std::vector<ValueID> grads = Autograd::differentiate(
                    graph_, loss_, SushiRuntime::span<const ValueID>(wrt.data(), wrt.size()));

                gradient_ = grads.at(0);
                EXPECT_NE(gradient_, Graph::K_NO_VALUE);

                // Fusion is off unless a case asks; see tests/core/autograd/README.md.
                if (fuse)
                {
                    const ValueID observed[] = {loss_, targets_};
                    Graph::FusionOptions options;
                    options.observed = SushiRuntime::span<const ValueID>(observed, 2);
                    fusion_ = Graph::fuse(graph_, options);
                }

                SA_EXPECT_OK(pool_.materialise_leaves(graph_));
                step_ = compile_or_record(engine, graph_, pool_);
            }

            /**
             * @brief Reports whether the step compiled.
             * @return False after a refusal, which the constructor already recorded.
             */
            bool compiled() const noexcept { return step_ != nullptr; }

            /** @brief What the fusion pass did, when the probe was asked to run it. */
            const Graph::FusionReport& fusion() const noexcept { return fusion_; }

            /** @brief What the lowering actually spelled, so a decline cannot pass unnoticed. */
            SushiRuntime::span<const Graph::FusionSelection> selections() const noexcept
            {
                if (!compiled())
                    return SushiRuntime::span<const Graph::FusionSelection>();

                return step_->selections();
            }

            /** @brief Writes a constant leaf's values. */
            void fill(ValueID id, const std::vector<double>& values)
            {
                SaTest::write_tensor(pool_.tensor(id), values);
            }

            /** @brief Fills a constant leaf from the deterministic host stream. */
            void fill_noise(ValueID id, std::uint64_t seed, double scale = 1.0)
            {
                std::vector<double> values(
                    static_cast<std::size_t>(pool_.tensor(id).num_elements));
                SaTest::HostNoise(seed).fill(values, scale);
                fill(id, values);
            }

            /** @brief Writes an INT32 leaf's values; no task is recorded while the host writes. */
            void fill_indices(ValueID id, const std::vector<std::int32_t>& indices)
            {
                SushiBLAS::Tensor& tensor = pool_.tensor(id);
                ASSERT_EQ(tensor.dtype, DType::Int32);
                ASSERT_EQ(static_cast<std::size_t>(tensor.num_elements), indices.size());

                std::int32_t* const data = tensor.data_as<std::int32_t>();
                for (std::size_t i = 0; i < indices.size(); ++i)
                    data[i] = indices[i];
            }

            /** @brief Fills the target matrix with a balanced one-hot pattern. */
            void fill_one_hot(int64_t rows, int64_t classes)
            {
                std::vector<double> values(static_cast<std::size_t>(rows * classes), 0.0);
                for (int64_t row = 0; row < rows; ++row)
                    values[static_cast<std::size_t>(row * classes + row % classes)] = 1.0;
                fill(targets_, values);
            }

            /** @brief The variable's element count. */
            std::size_t size() const { return static_cast<std::size_t>(
                pool_.tensor(variable_).num_elements); }

            /** @brief The target matrix's value id, for a test that wants to shape it. */
            ValueID targets() const noexcept { return targets_; }

            /** @brief The loss at @p point. */
            double value(SushiRuntime::span<const double> point)
            {
                // A refused compile is already a failed expectation; there is no step to run.
                if (!compiled())
                    return 0.0;

                SaTest::write_tensor(pool_.tensor(variable_), point);
                SA_EXPECT_OK(step_->run());
                return SaTest::read_tensor(pool_.tensor(loss_))[0];
            }

            /** @brief The analytic gradient at @p point. */
            void grad(SushiRuntime::span<const double> point, SushiRuntime::span<double> out)
            {
                if (!compiled())
                    return;

                SaTest::write_tensor(pool_.tensor(variable_), point);
                SA_EXPECT_OK(step_->run());

                const std::vector<double> values = SaTest::read_tensor(pool_.tensor(gradient_));
                ASSERT_EQ(values.size(), out.size());
                for (std::size_t i = 0; i < values.size(); ++i) out[i] = values[i];
            }

            /** @brief A ScalarFunction bound to this probe. */
            Autograd::ScalarFunction value_fn()
            {
                return [this](SushiRuntime::span<const double> point) { return value(point); };
            }

            /** @brief A GradientFunction bound to this probe. */
            Autograd::GradientFunction grad_fn()
            {
                return [this](SushiRuntime::span<const double> point,
                              SushiRuntime::span<double> out)
                {
                    grad(point, out);
                };
            }

        private:
            Graph::Graph graph_;
            TensorPool pool_;
            std::unique_ptr<CompiledStep> step_;
            ValueID variable_ = Graph::K_NO_VALUE;
            ValueID targets_ = Graph::K_NO_VALUE;
            ValueID loss_ = Graph::K_NO_VALUE;
            ValueID gradient_ = Graph::K_NO_VALUE;
            Graph::FusionReport fusion_;
    };

    /** @brief Runs the full-Jacobian check and reports the result on failure. */
    void expect_gradient_matches(Probe& probe, std::vector<double>& point,
                                 const Autograd::GradCheckOptions& options = {})
    {
        ASSERT_LE(point.size(), 200u) << "use the directional mode above 200 inputs";

        const Autograd::GradCheckResult result =
            Autograd::check_full_jacobian(probe.value_fn(), probe.grad_fn(), point, options);

        EXPECT_TRUE(result.passed) << result.to_string();
    }
}

TEST_F(Medium_Autograd, MatmulWithoutTransposes)
{
    ValueID other = Graph::K_NO_VALUE;

    Probe probe(*engine, Shape{4, 5}, [&other](const GraphBuilder& builder, ValueID x)
    {
        other = builder.input(Shape{5, 3}, "b", DType::Float64);
        return builder.matmul(x, other, false, false);
    });

    probe.fill_noise(other, 11, 1.0);
    probe.fill_noise(probe.targets(), 12, 1.0);

    std::vector<double> point(probe.size());
    SaTest::HostNoise(13).fill(point, 1.0);

    expect_gradient_matches(probe, point);
}

TEST_F(Medium_Autograd, MatmulWithATransposedRightOperand)
{
    // The shape Linear actually emits: X [N, in] against W [out, in]^T.
    ValueID weight = Graph::K_NO_VALUE;

    Probe probe(*engine, Shape{4, 5}, [&weight](const GraphBuilder& builder, ValueID x)
    {
        weight = builder.input(Shape{3, 5}, "w", DType::Float64);
        return builder.matmul(x, weight, false, true);
    });

    probe.fill_noise(weight, 21, 1.0);
    probe.fill_noise(probe.targets(), 22, 1.0);

    std::vector<double> point(probe.size());
    SaTest::HostNoise(23).fill(point, 1.0);

    expect_gradient_matches(probe, point);
}

TEST_F(Medium_Autograd, MatmulWithATransposedLeftOperand)
{
    ValueID other = Graph::K_NO_VALUE;

    Probe probe(*engine, Shape{5, 4}, [&other](const GraphBuilder& builder, ValueID x)
    {
        other = builder.input(Shape{5, 3}, "b", DType::Float64);
        return builder.matmul(x, other, true, false);
    });

    probe.fill_noise(other, 31, 1.0);
    probe.fill_noise(probe.targets(), 32, 1.0);

    std::vector<double> point(probe.size());
    SaTest::HostNoise(33).fill(point, 1.0);

    expect_gradient_matches(probe, point);
}

TEST_F(Medium_Autograd, MatmulWithBothOperandsTransposed)
{
    ValueID other = Graph::K_NO_VALUE;

    Probe probe(*engine, Shape{5, 4}, [&other](const GraphBuilder& builder, ValueID x)
    {
        other = builder.input(Shape{3, 5}, "b", DType::Float64);
        return builder.matmul(x, other, true, true);
    });

    probe.fill_noise(other, 41, 1.0);
    probe.fill_noise(probe.targets(), 42, 1.0);

    std::vector<double> point(probe.size());
    SaTest::HostNoise(43).fill(point, 1.0);

    expect_gradient_matches(probe, point);
}

TEST_F(Medium_Autograd, MatmulGradientOfTheRightOperand)
{
    // The weight is the variable, covering dB; see tests/core/autograd/README.md.
    ValueID activations = Graph::K_NO_VALUE;

    Probe probe(*engine, Shape{3, 5}, [&activations](const GraphBuilder& builder, ValueID w)
    {
        activations = builder.input(Shape{4, 5}, "x", DType::Float64);
        return builder.matmul(activations, w, false, true);
    });

    probe.fill_noise(activations, 51, 1.0);
    probe.fill_noise(probe.targets(), 52, 1.0);

    std::vector<double> point(probe.size());
    SaTest::HostNoise(53).fill(point, 1.0);

    expect_gradient_matches(probe, point);
}

TEST_F(Medium_Autograd, BiasAddPassesTheMatrixGradientThrough)
{
    ValueID bias = Graph::K_NO_VALUE;

    Probe probe(*engine, Shape{4, 3}, [&bias](const GraphBuilder& builder, ValueID x)
    {
        bias = builder.input(Shape{3}, "b", DType::Float64);
        return builder.bias_add(x, bias);
    });

    probe.fill_noise(bias, 61, 1.0);
    probe.fill_noise(probe.targets(), 62, 1.0);

    std::vector<double> point(probe.size());
    SaTest::HostNoise(63).fill(point, 1.0);

    expect_gradient_matches(probe, point);
}

TEST_F(Medium_Autograd, BiasAddSumsOverTheBroadcastAxis)
{
    ValueID base = Graph::K_NO_VALUE;

    Probe probe(*engine, Shape{3}, [&base](const GraphBuilder& builder, ValueID bias)
    {
        base = builder.input(Shape{4, 3}, "x", DType::Float64);
        return builder.bias_add(base, bias);
    });

    probe.fill_noise(base, 71, 1.0);
    probe.fill_noise(probe.targets(), 72, 1.0);

    std::vector<double> point(probe.size());
    SaTest::HostNoise(73).fill(point, 1.0);

    expect_gradient_matches(probe, point);
}

TEST_F(Medium_Autograd, ElementwiseAddAndSubtract)
{
    for (int which = 0; which < 2; ++which)
    {
        ValueID other = Graph::K_NO_VALUE;

        Probe probe(*engine, Shape{4, 3}, [&other, which](const GraphBuilder& builder, ValueID x)
        {
            other = builder.input(Shape{4, 3}, "b", DType::Float64);
            return which == 0 ? builder.add(x, other) : builder.sub(x, other);
        });

        probe.fill_noise(other, 81 + static_cast<std::uint64_t>(which), 1.0);
        probe.fill_noise(probe.targets(), 91 + static_cast<std::uint64_t>(which), 1.0);

        std::vector<double> point(probe.size());
        SaTest::HostNoise(101 + static_cast<std::uint64_t>(which)).fill(point, 1.0);

        expect_gradient_matches(probe, point);
    }
}

TEST_F(Medium_Autograd, ElementwiseSubtractFromTheRight)
{
    // dB = -dC, the half the left-hand check cannot reach.
    ValueID other = Graph::K_NO_VALUE;

    Probe probe(*engine, Shape{4, 3}, [&other](const GraphBuilder& builder, ValueID x)
    {
        other = builder.input(Shape{4, 3}, "a", DType::Float64);
        return builder.sub(other, x);
    });

    probe.fill_noise(other, 111, 1.0);
    probe.fill_noise(probe.targets(), 112, 1.0);

    std::vector<double> point(probe.size());
    SaTest::HostNoise(113).fill(point, 1.0);

    expect_gradient_matches(probe, point);
}

TEST_F(Medium_Autograd, ElementwiseMultiply)
{
    ValueID other = Graph::K_NO_VALUE;

    Probe probe(*engine, Shape{4, 3}, [&other](const GraphBuilder& builder, ValueID x)
    {
        other = builder.input(Shape{4, 3}, "b", DType::Float64);
        return builder.mul(x, other);
    });

    probe.fill_noise(other, 121, 1.0);
    probe.fill_noise(probe.targets(), 122, 1.0);

    std::vector<double> point(probe.size());
    SaTest::HostNoise(123).fill(point, 1.0);

    expect_gradient_matches(probe, point);
}

TEST_F(Medium_Autograd, ElementwiseMultiplyFromTheRight)
{
    // Covers dB = dC * A, the right-hand half; see tests/core/autograd/README.md.
    ValueID other = Graph::K_NO_VALUE;

    Probe probe(*engine, Shape{4, 3}, [&other](const GraphBuilder& builder, ValueID x)
    {
        other = builder.input(Shape{4, 3}, "a", DType::Float64);
        return builder.mul(other, x);
    });

    probe.fill_noise(other, 124, 1.0);
    probe.fill_noise(probe.targets(), 125, 1.0);

    std::vector<double> point(probe.size());
    SaTest::HostNoise(126).fill(point, 1.0);

    expect_gradient_matches(probe, point);
}

TEST_F(Medium_Autograd, AccumulateInTheForwardGraph)
{
    // Every probe rewrites the whole variable first; see tests/core/autograd/README.md.
    ValueID other = Graph::K_NO_VALUE;

    Probe probe(*engine, Shape{4, 3}, [&other](const GraphBuilder& builder, ValueID x)
    {
        other = builder.input(Shape{4, 3}, "delta", DType::Float64);
        return builder.accumulate(x, other);
    });

    probe.fill_noise(other, 231, 1.0);
    probe.fill_noise(probe.targets(), 232, 1.0);

    std::vector<double> point(probe.size());
    SaTest::HostNoise(233).fill(point, 1.0);

    expect_gradient_matches(probe, point);
}

TEST_F(Medium_Autograd, Scale)
{
    Probe probe(*engine, Shape{4, 3}, [](const GraphBuilder& builder, ValueID x)
    {
        return builder.scale(x, -2.5);
    });

    probe.fill_noise(probe.targets(), 131, 1.0);

    std::vector<double> point(probe.size());
    SaTest::HostNoise(132).fill(point, 1.0);

    expect_gradient_matches(probe, point);
}

TEST_F(Medium_Autograd, Exp)
{
    Probe probe(*engine, Shape{4, 3}, [](const GraphBuilder& builder, ValueID x)
    {
        return builder.exp(x);
    });

    probe.fill_noise(probe.targets(), 141, 1.0);

    std::vector<double> point(probe.size());
    SaTest::HostNoise(142).fill(point, 1.0);

    expect_gradient_matches(probe, point);
}

TEST_F(Medium_Autograd, ReLUAwayFromItsKink)
{
    Probe probe(*engine, Shape{4, 3}, [](const GraphBuilder& builder, ValueID x)
    {
        return builder.relu(x);
    });

    probe.fill_noise(probe.targets(), 151, 1.0);

    std::vector<double> point(probe.size());
    SaTest::HostNoise(152).fill(point, 1.0);

    // ReLU is sampled clear of its kink at zero; see tests/core/autograd/README.md.
    Autograd::push_off_kinks(point);

    expect_gradient_matches(probe, point);
}

TEST_F(Medium_Autograd, LeakyReLUHasBothOfItsSlopes)
{
    // The sample range puts elements on both slopes; see tests/core/autograd/README.md.
    Probe probe(*engine, Shape{4, 8}, [](const GraphBuilder& builder, ValueID x)
    {
        return builder.log_softmax(builder.leaky_relu(x, 0.1));
    });

    probe.fill_one_hot(4, 8);

    std::vector<double> point(probe.size());
    SaTest::HostNoise(171).fill(point, 1.5);

    // LEAKY_RELU is not differentiable at zero, same as RELU.
    Autograd::push_off_kinks(point);

    expect_gradient_matches(probe, point);
}

TEST_F(Medium_Autograd, LeakyReLUBranchSwapIsCaughtByProbePoints)
{
    // Pins the sign split of the sample used above; see tests/core/autograd/README.md.
    std::vector<double> point(4 * 8);
    SaTest::HostNoise(171).fill(point, 1.5);
    Autograd::push_off_kinks(point);

    const auto negative = std::count_if(point.begin(), point.end(),
                                        [](double v) { return v < 0.0; });
    EXPECT_GE(negative, 10) << "too few negative-side points to exercise the "
                                "slope branch of leaky_relu's gradient";

    const double min_abs = std::abs(*std::min_element(point.begin(), point.end(),
        [](double a, double b) { return std::abs(a) < std::abs(b); }));
    EXPECT_GT(min_abs, 0.05) << "a point too close to zero would sit on the kink "
                                 "push_off_kinks is supposed to have cleared";
}

TEST_F(Medium_Autograd, GELU)
{
    Probe probe(*engine, Shape{4, 3}, [](const GraphBuilder& builder, ValueID x)
    {
        return builder.gelu(x);
    });

    probe.fill_noise(probe.targets(), 161, 1.0);

    std::vector<double> point(probe.size());
    SaTest::HostNoise(162).fill(point, 1.5);

    expect_gradient_matches(probe, point);
}

TEST_F(Medium_Autograd, AFusedGELUEpilogueHasTheSameDerivativeAsAnUnfusedOne)
{
    // The fused epilogue is checked in FLOAT64; see tests/core/autograd/README.md.
    ValueID weight = Graph::K_NO_VALUE;
    ValueID bias = Graph::K_NO_VALUE;

    Probe probe(*engine, Shape{4, 5},
                [&](const GraphBuilder& builder, ValueID x)
                {
                    // x is the variable: matmul -> bias_add -> gelu gives [4, 3] logits.
                    weight = builder.input(Shape{3, 5}, "w", DType::Float64);
                    bias = builder.input(Shape{3}, "b", DType::Float64);
                    return builder.gelu(
                        builder.bias_add(builder.matmul(x, weight, false, true), bias));
                },
                /*fuse=*/true);

    // The fused arm must really have run; see tests/core/autograd/README.md.
    ASSERT_EQ(probe.fusion().count(Graph::FusedForm::GEMMBiasGELU), 1u)
        << probe.fusion().to_string();

    bool spelled = false;
    for (const Graph::FusionSelection& selection : probe.selections())
        if (selection.chosen == Graph::FusedForm::GEMMBiasGELU) spelled = true;

    ASSERT_TRUE(spelled)
        << "this device declined the fused GELU epilogue, so the gradient below is the "
           "unfused one and this case is not checking what it claims to";

    probe.fill_noise(weight, 171, 0.8);
    probe.fill_noise(bias, 172, 0.5);
    probe.fill_one_hot(4, 3);

    std::vector<double> point(probe.size());
    SaTest::HostNoise(173).fill(point, 1.2);

    expect_gradient_matches(probe, point);
}

TEST_F(Medium_Autograd, SigmoidReadsItsForwardOutput)
{
    Probe probe(*engine, Shape{4, 3}, [](const GraphBuilder& builder, ValueID x)
    {
        return builder.sigmoid(x);
    });

    probe.fill_noise(probe.targets(), 171, 1.0);

    std::vector<double> point(probe.size());
    SaTest::HostNoise(172).fill(point, 1.5);

    expect_gradient_matches(probe, point);
}

TEST_F(Medium_Autograd, TanhReadsItsForwardOutput)
{
    Probe probe(*engine, Shape{4, 3}, [](const GraphBuilder& builder, ValueID x)
    {
        return builder.tanh(x);
    });

    probe.fill_noise(probe.targets(), 181, 1.0);

    std::vector<double> point(probe.size());
    SaTest::HostNoise(182).fill(point, 1.5);

    expect_gradient_matches(probe, point);
}

TEST_F(Medium_Autograd, SumRowsBroadcastsItsGradientBack)
{
    Probe probe(*engine, Shape{4, 3}, [](const GraphBuilder& builder, ValueID x)
    {
        // [4, 3] -> [3] -> [1, 3], so the loss can consume it as a one-row batch.
        const ValueID sums = builder.sum_rows(x);
        return builder.bias_add(builder.zeros(Shape{1, 3}, DType::Float64), sums);
    });

    probe.fill_noise(probe.targets(), 191, 1.0);

    std::vector<double> point(probe.size());
    SaTest::HostNoise(192).fill(point, 1.0);

    expect_gradient_matches(probe, point);
}

TEST_F(Medium_Autograd, GatherRowsScatterAddsARepeatedIndexIntoItsTable)
{
    ValueID indices = Graph::K_NO_VALUE;

    Probe probe(*engine, Shape{5, 3}, [&indices](const GraphBuilder& builder, ValueID x)
    {
        indices = builder.input(Shape{6}, "indices", DType::Int32);
        return builder.gather_rows(x, indices).value();
    });

    // Row 3 is read three times and row 1 twice; rows 2 and 4 are read by no index.
    probe.fill_indices(indices, {3, 1, 3, 3, 0, 1});
    probe.fill_noise(probe.targets(), 193, 1.0);

    std::vector<double> point(probe.size());
    SaTest::HostNoise(194).fill(point, 1.0);

    expect_gradient_matches(probe, point);
}

TEST_F(Medium_Autograd, GatherRowsOverARankTwoIndexDifferentiatesThroughItsReshape)
{
    ValueID indices = Graph::K_NO_VALUE;

    Probe probe(*engine, Shape{4, 3}, [&indices](const GraphBuilder& builder, ValueID x)
    {
        // [4, 3] -> [2, 3, 3] -> [6, 3], so the loss can consume the tokens as rows.
        indices = builder.input(Shape{2, 3}, "indices", DType::Int32);
        return builder.reshape(builder.gather_rows(x, indices).value(), Shape{6, 3});
    });

    probe.fill_indices(indices, {2, 2, 0, 3, 2, 1});
    probe.fill_noise(probe.targets(), 195, 1.0);

    std::vector<double> point(probe.size());
    SaTest::HostNoise(196).fill(point, 1.0);

    expect_gradient_matches(probe, point);
}

TEST_F(Medium_Autograd, ScatterAddRowsGathersARepeatedIndexBackOntoItsRows)
{
    ValueID indices = Graph::K_NO_VALUE;

    Probe probe(*engine, Shape{6, 3}, [&indices](const GraphBuilder& builder, ValueID x)
    {
        indices = builder.input(Shape{6}, "indices", DType::Int32);
        return builder.scatter_add_rows(x, indices, 5).value();
    });

    // Row 3 is written three times and row 1 twice; rows 2 and 4 are written by no index.
    probe.fill_indices(indices, {3, 1, 3, 3, 0, 1});
    probe.fill_noise(probe.targets(), 197, 1.0);

    std::vector<double> point(probe.size());
    SaTest::HostNoise(198).fill(point, 1.0);

    expect_gradient_matches(probe, point);
}

TEST_F(Medium_Autograd, ScatterAddRowsOverARankTwoIndexGathersInTheRowsOwnShape)
{
    ValueID indices = Graph::K_NO_VALUE;

    Probe probe(*engine, Shape{2, 3, 3}, [&indices](const GraphBuilder& builder, ValueID x)
    {
        // [2, 3, 3] rows over a [2, 3] index sum into the [4, 3] the loss reads as rows.
        indices = builder.input(Shape{2, 3}, "indices", DType::Int32);
        return builder.scatter_add_rows(x, indices, 4).value();
    });

    probe.fill_indices(indices, {2, 2, 0, 3, 2, 1});
    probe.fill_noise(probe.targets(), 199, 1.0);

    std::vector<double> point(probe.size());
    SaTest::HostNoise(200).fill(point, 1.0);

    expect_gradient_matches(probe, point);
}

TEST_F(Medium_Autograd, FusedCrossEntropyAgainstOneHotTargets)
{
    // The fused rule gives (softmax - Y) / N; see tests/core/autograd/README.md.
    Probe probe(*engine, Shape{4, 5}, [](const GraphBuilder& builder, ValueID x)
    {
        return builder.log_softmax(x);
    });

    probe.fill_one_hot(4, 5);

    std::vector<double> point(probe.size());
    SaTest::HostNoise(201).fill(point, 2.0);

    expect_gradient_matches(probe, point);
}

TEST_F(Medium_Autograd, GradientAccumulationAtAFanOut)
{
    // Two branches over one variable make the gradient an ACCUMULATE chain.
    Probe probe(*engine, Shape{4, 3}, [](const GraphBuilder& builder, ValueID x)
    {
        return builder.add(builder.tanh(x), builder.sigmoid(x));
    });

    probe.fill_noise(probe.targets(), 211, 1.0);

    std::vector<double> point(probe.size());
    SaTest::HostNoise(212).fill(point, 1.2);

    expect_gradient_matches(probe, point);
}

TEST_F(Medium_Autograd, WholeTwoLayerModelUnderRandomDirections)
{
    // Sixteen random directions cost 2k runs, not 2n; see tests/core/autograd/README.md.
    ValueID hidden_weight = Graph::K_NO_VALUE;
    ValueID output_weight = Graph::K_NO_VALUE;
    ValueID inputs = Graph::K_NO_VALUE;

    Probe probe(*engine, Shape{6}, [&](const GraphBuilder& builder, ValueID bias)
    {
        inputs = builder.input(Shape{4, 5}, "x", DType::Float64);
        hidden_weight = builder.input(Shape{6, 5}, "w0", DType::Float64);
        output_weight = builder.input(Shape{3, 6}, "w1", DType::Float64);

        const ValueID hidden = builder.relu(
            builder.bias_add(builder.matmul(inputs, hidden_weight, false, true), bias));

        return builder.log_softmax(builder.matmul(hidden, output_weight, false, true));
    });

    probe.fill_noise(inputs, 221, 1.0);
    probe.fill_noise(hidden_weight, 222, 0.6);
    probe.fill_noise(output_weight, 223, 0.6);
    probe.fill_one_hot(4, 3);

    std::vector<double> point(probe.size());
    SaTest::HostNoise(224).fill(point, 0.5);
    Autograd::push_off_kinks(point);

    const Autograd::GradCheckResult result =
        Autograd::check_directional(probe.value_fn(), probe.grad_fn(), point, 16);

    EXPECT_TRUE(result.passed) << result.to_string();
}

TEST_F(Medium_Autograd, ReshapeMovesItsGradientBackUnchanged)
{
    // The gradient returns in the input shape; see tests/core/autograd/README.md.
    Probe probe(*engine, Shape{3, 8}, [](const GraphBuilder& builder, ValueID x)
    {
        return builder.log_softmax(builder.reshape(builder.reshape(x, Shape{4, 6}), Shape{3, 8}));
    });

    // The fused NLL rule needs one-hot targets; see tests/core/autograd/README.md.
    probe.fill_one_hot(3, 8);

    std::vector<double> point(probe.size());
    SaTest::HostNoise(232).fill(point, 1.0);

    expect_gradient_matches(probe, point);
}

TEST_F(Medium_Autograd, LayoutConvertsGradientIsThePermutedGradient)
{
    // A single layout_convert, so the order matters; see tests/core/autograd/README.md.
    Probe probe(*engine, Shape{2, 3, 4, 5}, [](const GraphBuilder& builder, ValueID x)
    {
        builder.graph().value(x).layout = Graph::TensorLayout::NCHW;
        const ValueID nhwc = builder.layout_convert(x, Graph::TensorLayout::NHWC);
        return builder.log_softmax(builder.reshape(nhwc, Shape{2, 60}));
    });

    probe.fill_one_hot(2, 60);

    std::vector<double> point(probe.size());
    SaTest::HostNoise(291).fill(point, 1.0);

    expect_gradient_matches(probe, point);
}

TEST_F(Medium_Autograd, LayoutConvertRoundTripComposesBothDirections)
{
    // The round trip is a secondary check only; see tests/core/autograd/README.md.
    Probe probe(*engine, Shape{2, 3, 4, 5}, [](const GraphBuilder& builder, ValueID x)
    {
        const ValueID nchw = builder.layout_convert(x, Graph::TensorLayout::NCHW);
        const ValueID back = builder.layout_convert(nchw, Graph::TensorLayout::NHWC);
        return builder.log_softmax(builder.reshape(back, Shape{2, 60}));
    });

    probe.fill_one_hot(2, 60);

    std::vector<double> point(probe.size());
    SaTest::HostNoise(242).fill(point, 1.0);

    expect_gradient_matches(probe, point);
}

TEST_F(Medium_Autograd, AWholeConvolutionDifferentiatesThroughIm2colAndMatmul)
{
    // Single hop: im2col feeds matmul, never col2im; see tests/core/autograd/README.md.
    ValueID weights = Graph::K_NO_VALUE;

    Probe probe(*engine, Shape{2, 5, 5, 3}, [&weights](const GraphBuilder& builder, ValueID x)
    {
        Graph::ConvGeometry g;
        g.kernel_h = 3; g.kernel_w = 3;
        g.pad_top = 1;  g.pad_left = 1;

        const ValueID cols = builder.im2col(x, g);                      // [2*25, 27]
        weights = builder.parameter(Shape{27, 8}, "w", DType::Float64); // 8 output channels
        return builder.log_softmax(builder.matmul(cols, weights));      // [2*25, 8]
    });

    probe.fill_noise(weights, 301, 1.0);
    // A log_softmax probe needs one-hot targets; see tests/core/autograd/README.md.
    probe.fill_one_hot(50, 8);

    std::vector<double> point(probe.size());
    SaTest::HostNoise(302).fill(point, 1.0);

    expect_gradient_matches(probe, point);
}

TEST_F(Medium_Autograd, Im2colsGradientAccumulatesEveryWindowThatReadAPixel)
{
    // Nine windows read each interior pixel; see tests/core/autograd/README.md.
    ValueID weights = Graph::K_NO_VALUE;

    Probe probe(*engine, Shape{1, 4, 4, 2}, [&weights](const GraphBuilder& builder, ValueID x)
    {
        Graph::ConvGeometry g;
        g.kernel_h = 3; g.kernel_w = 3;
        g.pad_top = 1;  g.pad_left = 1;

        const ValueID cols = builder.im2col(x, g);
        weights = builder.parameter(Shape{18, 4}, "w", DType::Float64);
        return builder.log_softmax(builder.matmul(cols, weights)); // [16, 4]
    });

    probe.fill_noise(weights, 311, 1.0);
    probe.fill_one_hot(16, 4);

    std::vector<double> point(probe.size());
    SaTest::HostNoise(312).fill(point, 1.0);

    expect_gradient_matches(probe, point);
}

TEST_F(Medium_Autograd, AnAsymmetricallyPaddedConvolutionDifferentiatesOverItsRealWindowGrid)
{
    // The real grid is 5x4; a symmetric guess is 4x5; see tests/core/autograd/README.md.
    ValueID weights = Graph::K_NO_VALUE;

    Probe probe(*engine, Shape{1, 4, 5, 1}, [&weights](const GraphBuilder& builder, ValueID x)
    {
        Graph::ConvGeometry g;
        g.kernel_h = 3; g.kernel_w = 3;
        g.pad_top = 1;  g.pad_left = 1;

        const ValueID cols = builder.im2col_with_extents(x, g, 5, 4);  // [20, 9]
        weights = builder.parameter(Shape{9, 4}, "w", DType::Float64);
        return builder.log_softmax(builder.matmul(cols, weights));     // [20, 4]
    });

    probe.fill_noise(weights, 331, 1.0);
    probe.fill_one_hot(20, 4);

    std::vector<double> point(probe.size());
    SaTest::HostNoise(332).fill(point, 1.0);

    expect_gradient_matches(probe, point);
}

TEST_F(Medium_Autograd, Col2imsGradientGathersEveryWindowThatWroteAPixel)
{
    // The variable is the column matrix itself; see tests/core/autograd/README.md.
    Probe probe(*engine, Shape{9, 18}, [](const GraphBuilder& builder, ValueID x)
    {
        Graph::ConvGeometry g;
        g.kernel_h = 3; g.kernel_w = 3;
        g.pad_top = 1;  g.pad_left = 1;

        const ValueID image = builder.col2im(x, Shape{1, 3, 3, 2}, g); // [1, 3, 3, 2]
        return builder.log_softmax(builder.reshape(image, Shape{1, 18}));
    });

    probe.fill_one_hot(1, 18);

    std::vector<double> point(probe.size());
    SaTest::HostNoise(321).fill(point, 1.0);

    expect_gradient_matches(probe, point);
}

TEST_F(Medium_Autograd, MaxPoolRoutesEachGradientToTheElementThatWon)
{
    // push_off_kinks leaves no ties in a window; see tests/core/autograd/README.md.
    Probe probe(*engine, Shape({2, 4, 4, 3}), [](const GraphBuilder& b, ValueID x)
    {
        Graph::ConvGeometry g;
        g.kernel_h = 2; g.kernel_w = 2;
        g.stride_h = 2; g.stride_w = 2;
        return b.log_softmax(b.spatial_mean(b.maxpool2d(x, g).output));
    });

    probe.fill_one_hot(2, 3);

    std::vector<double> point(probe.size());
    SaTest::HostNoise(401).fill(point, 1.0);
    Autograd::push_off_kinks(point);

    expect_gradient_matches(probe, point);
}

TEST_F(Medium_Autograd, OverlappingMaxPoolWindowsAccumulateOnADominantPixel)
{
    // At stride 1 a pixel can win several windows; see tests/core/autograd/README.md.
    Probe probe(*engine, Shape({1, 5, 5, 2}), [](const GraphBuilder& b, ValueID x)
    {
        Graph::ConvGeometry g;
        g.kernel_h = 2; g.kernel_w = 2;
        return b.log_softmax(b.spatial_mean(b.maxpool2d(x, g).output));
    });

    probe.fill_one_hot(1, 2);

    std::vector<double> point(probe.size());
    SaTest::HostNoise(411).fill(point, 1.0);
    Autograd::push_off_kinks(point);

    expect_gradient_matches(probe, point);
}

TEST_F(Medium_Autograd, AvgPoolSpreadsItsGradientOverTheWholeWindow)
{
    // One batch row keeps this under 200 inputs; see tests/core/autograd/README.md.
    Probe probe(*engine, Shape({1, 6, 6, 3}), [](const GraphBuilder& b, ValueID x)
    {
        Graph::ConvGeometry g;
        g.kernel_h = 3; g.kernel_w = 3;
        g.stride_h = 3; g.stride_w = 3;
        return b.log_softmax(b.spatial_mean(b.avgpool2d(x, g)));
    });

    probe.fill_one_hot(1, 3);

    std::vector<double> point(probe.size());
    SaTest::HostNoise(421).fill(point, 1.0);

    expect_gradient_matches(probe, point);
}

TEST_F(Medium_Autograd, UpsampleSumsEachSourcePixelsReplicas)
{
    // Scale 2 makes each gradient a sum of four; see tests/core/autograd/README.md.
    Probe probe(*engine, Shape({2, 3, 3, 4}), [](const GraphBuilder& b, ValueID x)
    {
        return b.log_softmax(b.spatial_mean(b.upsample2d(x, 2, 2)));
    });

    probe.fill_one_hot(2, 4);

    std::vector<double> point(probe.size());
    SaTest::HostNoise(431).fill(point, 1.0);

    expect_gradient_matches(probe, point);
}

TEST_F(Medium_Autograd, SpatialMeanBroadcastsScaledByOneOverHW)
{
    // The broadcast is per (n, c); see tests/core/autograd/README.md.
    Probe probe(*engine, Shape({2, 4, 4, 5}), [](const GraphBuilder& b, ValueID x)
    {
        return b.log_softmax(b.spatial_mean(x));
    });

    probe.fill_one_hot(2, 5);

    std::vector<double> point(probe.size());
    SaTest::HostNoise(441).fill(point, 1.0);

    expect_gradient_matches(probe, point);
}

TEST_F(Medium_Autograd, ConcatGivesEachOperandItsOwnChannelRange)
{
    // Operands are reordered against their slices; see tests/core/autograd/README.md.
    Probe probe(*engine, Shape({2, 4, 4, 6}), [](const GraphBuilder& b, ValueID x)
    {
        const ValueID left = b.slice_channels(x, 0, 2);
        const ValueID right = b.slice_channels(x, 2, 4);
        const std::array<ValueID, 2> parts = {right, left};   // deliberately reordered
        return b.log_softmax(b.spatial_mean(b.concat_channels(
            SushiRuntime::span<const ValueID>(parts.data(), parts.size()))));
    });

    probe.fill_one_hot(2, 6);

    std::vector<double> point(probe.size());
    SaTest::HostNoise(451).fill(point, 1.0);

    expect_gradient_matches(probe, point);
}

TEST_F(Medium_Autograd, SliceScattersItsGradientBackIntoZeros)
{
    // Channels outside the slice receive zero; see tests/core/autograd/README.md.
    Probe probe(*engine, Shape({1, 3, 3, 8}), [](const GraphBuilder& b, ValueID x)
    {
        return b.log_softmax(b.spatial_mean(b.slice_channels(x, 3, 2)));
    });

    probe.fill_one_hot(1, 2);

    std::vector<double> point(probe.size());
    SaTest::HostNoise(461).fill(point, 1.0);

    expect_gradient_matches(probe, point);
}

TEST_F(Medium_Autograd, BatchNormsGradientAccountsForTheStatisticsItComputed)
{
    // dx has three terms, not one; see tests/core/autograd/README.md.
    ValueID gamma = Graph::K_NO_VALUE;
    ValueID beta = Graph::K_NO_VALUE;

    Probe probe(*engine, Shape({4, 3, 3, 5}),
                [&gamma, &beta](const GraphBuilder& b, ValueID x)
    {
        gamma = b.parameter(Shape{5}, "gamma", DType::Float64);
        beta = b.parameter(Shape{5}, "beta", DType::Float64);
        return b.log_softmax(b.spatial_mean(b.batchnorm2d(x, gamma, beta, 1e-5).output));
    });

    probe.fill_noise(gamma, 471, 1.0);
    probe.fill_noise(beta, 472, 1.0);
    probe.fill_one_hot(4, 5);

    std::vector<double> point(probe.size());
    SaTest::HostNoise(473).fill(point, 1.0);

    expect_gradient_matches(probe, point);
}

TEST_F(Medium_Autograd, BatchNormsScaleGradientIsTheSumOfDyTimesTheNormalisedInput)
{
    // The same node, differentiated at gamma; see tests/core/autograd/README.md.
    ValueID x = Graph::K_NO_VALUE;
    ValueID beta = Graph::K_NO_VALUE;

    Probe probe(*engine, Shape{5}, [&x, &beta](const GraphBuilder& b, ValueID gamma)
    {
        x = b.input(Shape({4, 3, 3, 5}), "x", DType::Float64);
        beta = b.parameter(Shape{5}, "beta", DType::Float64);
        return b.log_softmax(b.spatial_mean(b.batchnorm2d(x, gamma, beta, 1e-5).output));
    });

    probe.fill_noise(x, 481, 1.0);
    probe.fill_noise(beta, 482, 1.0);
    probe.fill_one_hot(4, 5);

    std::vector<double> point(probe.size());
    SaTest::HostNoise(483).fill(point, 1.0);

    expect_gradient_matches(probe, point);
}

TEST_F(Medium_Autograd, BatchNormsShiftGradientIsThePlainSumOfDy)
{
    ValueID x = Graph::K_NO_VALUE;
    ValueID gamma = Graph::K_NO_VALUE;

    Probe probe(*engine, Shape{5}, [&x, &gamma](const GraphBuilder& b, ValueID beta)
    {
        x = b.input(Shape({4, 3, 3, 5}), "x", DType::Float64);
        gamma = b.parameter(Shape{5}, "gamma", DType::Float64);
        return b.log_softmax(b.spatial_mean(b.batchnorm2d(x, gamma, beta, 1e-5).output));
    });

    probe.fill_noise(x, 491, 1.0);
    probe.fill_noise(gamma, 492, 1.0);
    probe.fill_one_hot(4, 5);

    std::vector<double> point(probe.size());
    SaTest::HostNoise(493).fill(point, 1.0);

    expect_gradient_matches(probe, point);
}

TEST_F(Medium_Autograd, ChannelAffineDifferentiatesAtItsInput)
{
    ValueID gamma = Graph::K_NO_VALUE;
    ValueID beta = Graph::K_NO_VALUE;

    Probe probe(*engine, Shape({2, 4, 4, 6}),
                [&gamma, &beta](const GraphBuilder& b, ValueID x)
    {
        gamma = b.parameter(Shape{6}, "gamma", DType::Float64);
        beta = b.parameter(Shape{6}, "beta", DType::Float64);
        return b.log_softmax(b.spatial_mean(b.channel_affine(x, gamma, beta)));
    });

    probe.fill_noise(gamma, 501, 1.0);
    probe.fill_noise(beta, 502, 1.0);
    probe.fill_one_hot(2, 6);

    std::vector<double> point(probe.size());
    SaTest::HostNoise(503).fill(point, 1.0);

    expect_gradient_matches(probe, point);
}

TEST_F(Medium_Autograd, ChannelAffinesScaleGradientSumsOverNHAndW)
{
    // dgamma sums over N, H and W at once; see tests/core/autograd/README.md.
    ValueID x = Graph::K_NO_VALUE;
    ValueID beta = Graph::K_NO_VALUE;

    Probe probe(*engine, Shape{6}, [&x, &beta](const GraphBuilder& b, ValueID gamma)
    {
        x = b.input(Shape({2, 4, 4, 6}), "x", DType::Float64);
        beta = b.parameter(Shape{6}, "beta", DType::Float64);
        return b.log_softmax(b.spatial_mean(b.channel_affine(x, gamma, beta)));
    });

    probe.fill_noise(x, 511, 1.0);
    probe.fill_noise(beta, 512, 1.0);
    probe.fill_one_hot(2, 6);

    std::vector<double> point(probe.size());
    SaTest::HostNoise(513).fill(point, 1.0);

    expect_gradient_matches(probe, point);
}

TEST_F(Medium_Autograd, ChannelAffinesShiftGradientSumsOverNHAndW)
{
    ValueID x = Graph::K_NO_VALUE;
    ValueID gamma = Graph::K_NO_VALUE;

    Probe probe(*engine, Shape{6}, [&x, &gamma](const GraphBuilder& b, ValueID beta)
    {
        x = b.input(Shape({2, 4, 4, 6}), "x", DType::Float64);
        gamma = b.parameter(Shape{6}, "gamma", DType::Float64);
        return b.log_softmax(b.spatial_mean(b.channel_affine(x, gamma, beta)));
    });

    probe.fill_noise(x, 521, 1.0);
    probe.fill_noise(gamma, 522, 1.0);
    probe.fill_one_hot(2, 6);

    std::vector<double> point(probe.size());
    SaTest::HostNoise(523).fill(point, 1.0);

    expect_gradient_matches(probe, point);
}

TEST_F(Medium_Autograd, DepthwiseConvolutionDifferentiatesOverItsInputAndItsTaps)
{
    // The image and the taps are slices of one flat variable, so one check covers both.
    constexpr int64_t K_IMAGE_ELEMENTS = 2 * 5 * 5 * 3;
    constexpr int64_t K_TAP_ELEMENTS = 3 * 3 * 3;

    Probe probe(*engine, Shape{1, 1, 1, K_IMAGE_ELEMENTS + K_TAP_ELEMENTS},
                [](const GraphBuilder& b, ValueID flat)
    {
        const ValueID image = b.reshape(b.slice_channels(flat, 0, K_IMAGE_ELEMENTS),
                                        Shape{2, 5, 5, 3});
        const ValueID taps = b.reshape(b.slice_channels(flat, K_IMAGE_ELEMENTS, K_TAP_ELEMENTS),
                                       Shape{3, 3, 3});

        Graph::ConvGeometry g;
        g.kernel_h = 3; g.kernel_w = 3;
        g.stride_h = 2; g.stride_w = 2;
        g.pad_top = 1;  g.pad_left = 1;

        const ValueID windows = b.depthwise_conv2d(image, taps, g);   // [2, 3, 3, 3]
        return b.log_softmax(b.spatial_mean(windows));                // [2, 3]
    });

    probe.fill_one_hot(2, 3);

    std::vector<double> point(probe.size());
    SaTest::HostNoise(601).fill(point, 1.0);

    expect_gradient_matches(probe, point);
}
