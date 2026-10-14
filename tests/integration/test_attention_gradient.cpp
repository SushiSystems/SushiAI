/**************************************************************************/
/* test_attention_gradient.cpp                                            */
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
 * @file test_attention_gradient.cpp
 * @brief Checks the gradient of a log-softmax that no cross-entropy absorbed, of a softmax and
 *        of a CausalSelfAttention block against a closed form and central differences.
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
#include <SushiAI/autograd/backward.hpp>
#include <SushiAI/autograd/gradcheck.hpp>
#include <SushiAI/core/common.hpp>
#include <SushiAI/core/ops.hpp>
#include <SushiAI/core/result.hpp>
#include <SushiAI/core/shape.hpp>
#include <SushiAI/graph/builder.hpp>
#include <SushiAI/graph/compiled_step.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/graph/tensor_pool.hpp>
#include <SushiAI/language/blocks/causal_self_attention.hpp>
#include <SushiAI/nn/trace.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/SushiAI.h>
#include <SushiAI/engine_fixture/engine_test.hpp>
#include <SushiAI/tensor_values/tensor_values.hpp>
#include <SushiAI/seeded_noise/host_noise.hpp>

using namespace SushiAI;

// SaTest stays reachable inside a TEST_F body; see tests/integration/README.md.
namespace SaTest = SushiAI::Test;
using Graph::CompiledStep;
using Graph::GraphBuilder;
using Graph::TensorPool;
using Graph::ValueID;
using Language::Blocks::AttentionMask;
using Language::Blocks::CausalSelfAttention;
using SaTest::read_tensor;
using SaTest::write_tensor;

namespace
{
    /** @brief Provides the Engine every gradient case records on. */
    class Integration_AttentionGradient : public SaTest::EngineTest
    {

    };

    /** @brief Bounds the distance of a FLOAT64 gradient from its closed form on the host. */
    constexpr double K_CLOSED_FORM_TOLERANCE = 1e-12;

    /** @brief Holds T, the number of positions of the sequence the block cases read. */
    constexpr int64_t K_LENGTH = 4;

    /** @brief Holds the width of the sequence the block cases read. */
    constexpr int64_t K_DIM = 4;

    /** @brief Holds the head count of the block cases. */
    constexpr int64_t K_HEADS = 2;

    /** @brief Returns @p count values of the deterministic host stream seeded with @p seed. */
    std::vector<double> noise(std::size_t count, std::uint64_t seed)
    {
        std::vector<double> values(count);
        SaTest::HostNoise(seed).fill(values, 1.0);
        return values;
    }

    /** @brief Counts the nodes of @p graph that carry @p op. */
    std::size_t count_nodes(const Graph::Graph& graph, OpID op)
    {
        std::size_t found = 0;
        for (const Graph::Node& node : graph.nodes())
        {
            if (node.op == op)
                ++found;
        }

        return found;
    }

    /** @brief Names the leaf a probe differentiates at and the [N, C] value its loss reads. */
    struct Subject
    {
        ValueID variable = Graph::K_NO_VALUE; /**< The leaf the gradient is taken at. */
        ValueID logits = Graph::K_NO_VALUE;   /**< The [N, C] value the cross-entropy reads. */
    };

    /**
     * @brief Holds a FLOAT64 graph whose loss and gradient can both be evaluated at a point.
     *
     * The loss is the cross-entropy of the logits the build function names, against targets
     * whose row r names class r mod C. Every other leaf is filled from the host stream.
     */
    class Probe
    {
        public:
            /** @brief Builds the graph, differentiates it at its variable and compiles it. */
            template<typename BuildFn>
            Probe(SushiBLAS::Engine& engine, std::uint64_t seed, BuildFn&& build)
                : pool_(engine)
            {
                const GraphBuilder builder(graph_);
                const Subject subject = build(builder);
                variable_ = subject.variable;

                const Shape logit_shape = graph_.value(subject.logits).shape;
                targets_ = builder.input(logit_shape, "targets", DType::Float64);
                loss_ = builder.nll_loss(builder.log_softmax(subject.logits), targets_);

                const std::vector<ValueID> wrt = {variable_};
                gradient_ = Autograd::differentiate(
                    graph_, loss_,
                    SushiRuntime::span<const ValueID>(wrt.data(), wrt.size())).at(0);
                EXPECT_NE(gradient_, Graph::K_NO_VALUE);

                SA_EXPECT_OK(pool_.materialise_leaves(graph_));
                fill_other_leaves(seed, logit_shape);

                Result<std::unique_ptr<CompiledStep>> compiled =
                    CompiledStep::compile(engine, graph_, pool_);
                SA_EXPECT_OK(compiled);
                if (compiled.has_value())
                    step_ = std::move(compiled).value();
            }

            /** @brief Returns the graph the probe built, with its gradient nodes. */
            const Graph::Graph& graph() const noexcept
            {
                return graph_;
            }

            /** @brief Returns how many elements the variable holds. */
            std::size_t variable_size() const
            {
                return static_cast<std::size_t>(graph_.value(variable_).shape.num_elements());
            }

            /** @brief Returns the loss at @p point, or zero when the step did not compile. */
            double value(SushiRuntime::span<const double> point)
            {
                if (step_ == nullptr)
                    return 0.0;

                write_tensor(pool_.tensor(variable_), point);
                SA_EXPECT_OK(step_->run());
                return read_tensor(pool_.tensor(loss_))[0];
            }

            /** @brief Writes the analytic gradient at @p point into @p out. */
            void gradient(SushiRuntime::span<const double> point, SushiRuntime::span<double> out)
            {
                if (step_ == nullptr)
                    return;

                write_tensor(pool_.tensor(variable_), point);
                SA_EXPECT_OK(step_->run());

                const std::vector<double> values = read_tensor(pool_.tensor(gradient_));
                ASSERT_EQ(values.size(), out.size());
                for (std::size_t i = 0; i < values.size(); ++i)
                    out[i] = values[i];
            }

        private:
            /** @brief Writes the one-hot targets, and noise into each leaf but the variable. */
            void fill_other_leaves(std::uint64_t seed, const Shape& logit_shape)
            {
                std::vector<double> one_hot(
                    static_cast<std::size_t>(logit_shape.num_elements()), 0.0);
                for (int64_t row = 0; row < logit_shape[0]; ++row)
                    one_hot[static_cast<std::size_t>(row * logit_shape[1]
                                                     + row % logit_shape[1])] = 1.0;
                write_tensor(pool_.tensor(targets_), one_hot);

                for (ValueID id = 0; id < graph_.value_count(); ++id)
                {
                    const Graph::Lifetime lifetime = graph_.value(id).lifetime;
                    const bool is_leaf = lifetime == Graph::Lifetime::Parameter
                                      || lifetime == Graph::Lifetime::Input;
                    if (!is_leaf || id == variable_ || id == targets_)
                        continue;

                    const std::size_t count =
                        static_cast<std::size_t>(graph_.value(id).shape.num_elements());
                    write_tensor(pool_.tensor(id), noise(count, seed + id));
                }
            }

            Graph::Graph graph_;
            TensorPool pool_;
            std::unique_ptr<CompiledStep> step_;
            ValueID variable_ = Graph::K_NO_VALUE;
            ValueID targets_ = Graph::K_NO_VALUE;
            ValueID loss_ = Graph::K_NO_VALUE;
            ValueID gradient_ = Graph::K_NO_VALUE;
    };

    /** @brief Runs the full-Jacobian check of @p probe at a point drawn from @p seed. */
    void expect_gradient_matches(Probe& probe, std::uint64_t seed)
    {
        std::vector<double> point = noise(probe.variable_size(), seed);
        const Autograd::GradCheckResult result = Autograd::check_full_jacobian(
            [&probe](SushiRuntime::span<const double> at)
            {
                return probe.value(at);
            },
            [&probe](SushiRuntime::span<const double> at, SushiRuntime::span<double> out)
            {
                probe.gradient(at, out);
            },
            point);

        EXPECT_TRUE(result.passed) << result.to_string();
    }

    /** @brief Names which leaf of an attention probe the gradient is taken at. */
    enum class AttentionVariable : std::uint8_t
    {
        Input,            /**< The [1, K_LENGTH, K_DIM] sequence the block reads. */
        ProjectionWeight, /**< The weight of qkv, which every score and every value reads. */
    };

    /**
     * @brief Traces @p block over a FLOAT64 [1, K_LENGTH, K_DIM] sequence and names its
     *        [K_LENGTH, K_DIM] output rows as the logits.
     */
    Subject attention_subject(const GraphBuilder& builder, CausalSelfAttention& block,
                              AttentionVariable variable)
    {
        const Shape sequence{1, K_LENGTH, K_DIM};
        const ValueID x = variable == AttentionVariable::Input
            ? builder.parameter(sequence, "x", DType::Float64)
            : builder.input(sequence, "x", DType::Float64);

        const NN::Trace trace = NN::Trace::evaluation(builder);
        const ValueID y = block.forward(trace, x);
        SA_EXPECT_OK(trace.status());
        EXPECT_FALSE(builder.graph().has_refusal());

        const ValueID at = variable == AttentionVariable::Input
            ? x
            : block.qkv.weight.id_in(builder.graph());
        return Subject{at, builder.reshape(y, Shape{K_LENGTH, K_DIM})};
    }

    /** @brief Checks the gradient of a block with @p mask at @p variable. */
    void expect_attention_gradient_matches(SushiBLAS::Engine& engine, AttentionMask mask,
                                           AttentionVariable variable, std::uint64_t seed)
    {
        CausalSelfAttention block(K_DIM, K_HEADS, mask);
        Probe probe(engine, seed, [&block, variable](const GraphBuilder& builder)
        {
            return attention_subject(builder, block, variable);
        });

        // Each head's softmax is one LOG_SOFTMAX, and none of them feeds a cross-entropy.
        EXPECT_EQ(count_nodes(probe.graph(), Ops::K_LOG_SOFTMAX_BACKWARD.id),
                  static_cast<std::size_t>(K_HEADS));
        ASSERT_NO_FATAL_FAILURE(expect_gradient_matches(probe, seed + 1000));
    }
}

/**
 * @brief Tests the unfused rule against its closed form, with targets that are not one-hot.
 *
 * A SCALE by one sits between LOG_SOFTMAX and NLL_LOSS, so the fused rule does not apply and
 * the gradient is (softmax(x) * rowsum(Y) - Y) / N, which the host computes.
 */
TEST_F(Integration_AttentionGradient, AnUnfusedLogSoftmaxMatchesItsClosedForm)
{
    constexpr std::size_t K_ROWS = 3;
    constexpr std::size_t K_CLASSES = 4;
    const Shape shape{static_cast<int64_t>(K_ROWS), static_cast<int64_t>(K_CLASSES)};

    Graph::Graph graph;
    const GraphBuilder builder(graph);
    const ValueID x = builder.parameter(shape, "x", DType::Float64);
    const ValueID targets = builder.input(shape, "targets", DType::Float64);
    const ValueID detoured = builder.scale(builder.log_softmax(x), 1.0);
    const ValueID loss = builder.nll_loss(detoured, targets);

    const std::vector<ValueID> wrt = {x};
    const std::vector<ValueID> gradients = Autograd::differentiate(
        graph, loss, SushiRuntime::span<const ValueID>(wrt.data(), wrt.size()));
    ASSERT_NE(gradients.at(0), Graph::K_NO_VALUE);
    ASSERT_EQ(count_nodes(graph, Ops::K_LOG_SOFTMAX_BACKWARD.id), 1u);

    // The weights are positive and their rows do not sum to one.
    const std::vector<double> values = noise(K_ROWS * K_CLASSES, 7101);
    std::vector<double> weights = noise(K_ROWS * K_CLASSES, 7102);
    for (double& weight : weights)
        weight = std::abs(weight) + 0.25;

    TensorPool pool(*engine);
    SA_EXPECT_OK(pool.materialise_leaves(graph));
    write_tensor(pool.tensor(x), values);
    write_tensor(pool.tensor(targets), weights);

    std::unique_ptr<CompiledStep> step;
    SA_ASSERT_OK_AND_ASSIGN(step, CompiledStep::compile(*engine, graph, pool));
    SA_EXPECT_OK(step->run());
    SA_EXPECT_OK(step->run());

    const std::vector<double> got = read_tensor(pool.tensor(gradients[0]));
    ASSERT_EQ(got.size(), K_ROWS * K_CLASSES);
    for (std::size_t row = 0; row < K_ROWS; ++row)
    {
        double total = 0.0;
        double weight_sum = 0.0;
        for (std::size_t column = 0; column < K_CLASSES; ++column)
        {
            total += std::exp(values[row * K_CLASSES + column]);
            weight_sum += weights[row * K_CLASSES + column];
        }

        for (std::size_t column = 0; column < K_CLASSES; ++column)
        {
            const std::size_t i = row * K_CLASSES + column;
            const double expected = (std::exp(values[i]) / total * weight_sum - weights[i])
                                  / static_cast<double>(K_ROWS);
            EXPECT_NEAR(got[i], expected, K_CLOSED_FORM_TOLERANCE)
                << "at row " << row << " column " << column;
        }
    }
}

/** @brief Tests the gradient of a [3, 4] softmax that a matrix product reads. */
TEST_F(Integration_AttentionGradient, ASoftmaxGradientMatchesCentralDifferences)
{
    Probe probe(*engine, 7200, [](const GraphBuilder& builder)
    {
        const ValueID x = builder.parameter(Shape{3, 4}, "x", DType::Float64);
        const ValueID mix = builder.input(Shape{4, 3}, "mix", DType::Float64);
        return Subject{x, builder.matmul(builder.softmax(x), mix)};
    });

    EXPECT_EQ(count_nodes(probe.graph(), Ops::K_LOG_SOFTMAX_BACKWARD.id), 1u);
    ASSERT_NO_FATAL_FAILURE(expect_gradient_matches(probe, 7201));
}

/** @brief Tests the gradient of a softmax over the last axis of a [2, 3, 4] value. */
TEST_F(Integration_AttentionGradient, ASoftmaxOverRankThreeDifferentiatesThroughItsReshapes)
{
    Probe probe(*engine, 7300, [](const GraphBuilder& builder)
    {
        const ValueID x = builder.parameter(Shape{2, 3, 4}, "x", DType::Float64);
        return Subject{x, builder.reshape(builder.softmax(x), Shape{6, 4})};
    });

    EXPECT_EQ(count_nodes(probe.graph(), Ops::K_LOG_SOFTMAX_BACKWARD.id), 1u);
    ASSERT_NO_FATAL_FAILURE(expect_gradient_matches(probe, 7301));
}

/** @brief Tests the gradient of a causal attention block at the sequence it reads. */
TEST_F(Integration_AttentionGradient, ACausalAttentionDifferentiatesAtItsInput)
{
    ASSERT_NO_FATAL_FAILURE(expect_attention_gradient_matches(
        *engine, AttentionMask::Causal, AttentionVariable::Input, 7400));
}

/** @brief Tests the gradient of a causal attention block at the weight of its qkv. */
TEST_F(Integration_AttentionGradient, ACausalAttentionDifferentiatesAtItsProjectionWeight)
{
    ASSERT_NO_FATAL_FAILURE(expect_attention_gradient_matches(
        *engine, AttentionMask::Causal, AttentionVariable::ProjectionWeight, 7500));
}

/** @brief Tests the gradient of an attention block without the mask at the sequence it reads. */
TEST_F(Integration_AttentionGradient, AnUnmaskedAttentionDifferentiatesAtItsInput)
{
    ASSERT_NO_FATAL_FAILURE(expect_attention_gradient_matches(
        *engine, AttentionMask::None, AttentionVariable::Input, 7600));
}
