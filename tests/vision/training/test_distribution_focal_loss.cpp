/**************************************************************************/
/* test_distribution_focal_loss.cpp                                       */
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

/**
 * @file test_distribution_focal_loss.cpp
 * @brief Tests the distribution focal loss graph: its nodes, a hand-worked value and its gradient.
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
#include <SushiAI/autograd/gradcheck.hpp>
#include <SushiAI/dtype/dtype.hpp>
#include <SushiAI/ops/ops.hpp>
#include <SushiAI/shape/shape.hpp>
#include <SushiAI/engine_fixture/engine_test.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/graph/builder.hpp>
#include <SushiAI/graph/compiled_step.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/graph/tensor_pool.hpp>
#include <SushiAI/seeded_noise/host_noise.hpp>
#include <SushiAI/tensor_values/tensor_values.hpp>
#include <SushiAI/vision/detect/free/distribution_targets.hpp>
#include <SushiAI/vision/training/distribution_focal_loss.hpp>

using namespace SushiAI;
using Graph::CompiledStep;
using Graph::GraphBuilder;
using Graph::TensorPool;
using Graph::ValueID;
using Vision::Training::emit_distribution_focal_loss;
// SaTest stays reachable inside a TEST_F body; see tests/vision/training/README.md.
namespace SaTest = SushiAI::Test;

namespace
{
    /** @brief Holds the bins per side of the cases. */
    constexpr std::int64_t K_BINS = 4;

    /** @brief Holds the box output of one image on a 1x2 grid, 32 values. */
    const Shape K_BOX_SHAPE{1, 1, 2, 4 * K_BINS};

    /** @brief Holds the foreground anchor's normalized weight. */
    constexpr float K_FOREGROUND_WEIGHT = 0.8f;

    /** @brief Seeds the logits of the gradient check. */
    constexpr std::uint64_t K_LOGIT_SEED = 20261026;

    /** @brief Bounds the error a float32 rounding of each bin weight leaves in a sum near 1. */
    constexpr double K_VALUE_TOLERANCE = 1e-6;

    /** @brief Returns the bin weights of the first anchor's targets; the second is background. */
    std::vector<double> hand_weights()
    {
        const std::vector<float> targets = {0.5f, 1.25f, 2.0f, 2.75f, 1.0f, 1.0f, 1.0f, 1.0f};
        const std::vector<float> normalized = {K_FOREGROUND_WEIGHT, 0.0f};
        std::vector<float> weights(static_cast<std::size_t>(K_BOX_SHAPE.num_elements()));
        Vision::Detect::Free::fill_distribution_weights(targets, normalized, K_BINS, weights);

        return std::vector<double>(weights.begin(), weights.end());
    }

    /** @brief Holds the loss over a FLOAT64 box parameter, compiled with its gradient. */
    class LossProbe
    {
        public:
            /** @brief Builds, differentiates and compiles the loss on @p engine. */
            explicit LossProbe(SushiBLAS::Engine& engine) : pool_(engine)
            {
                const GraphBuilder builder(graph_);
                box_ = builder.parameter(K_BOX_SHAPE, "box", DType::Float64);
                weights_ = builder.input(K_BOX_SHAPE, "weights", DType::Float64);
                loss_ = emit_distribution_focal_loss(builder, graph_, box_, weights_);

                const std::vector<ValueID> wrt = {box_};
                gradient_ = Autograd::differentiate(
                    graph_, loss_,
                    SushiRuntime::span<const ValueID>(wrt.data(), wrt.size())).at(0);
                EXPECT_NE(gradient_, Graph::K_NO_VALUE);

                SA_EXPECT_OK(pool_.materialise_leaves(graph_));
                Result<std::unique_ptr<CompiledStep>> compiled =
                    CompiledStep::compile(engine, graph_, pool_);
                SA_EXPECT_OK(compiled);
                if (compiled.has_value())
                    step_ = std::move(compiled).value();

                SaTest::write_tensor(pool_.tensor(weights_), hand_weights());
            }

            /** @brief Returns whether the step compiled. */
            bool ready() const { return step_ != nullptr; }

            /** @brief Returns the loss at @p point, or zero when the step did not compile. */
            double value(SushiRuntime::span<const double> point)
            {
                if (step_ == nullptr)
                    return 0.0;

                SaTest::write_tensor(pool_.tensor(box_), point);
                SA_EXPECT_OK(step_->run());
                return SaTest::read_tensor(pool_.tensor(loss_))[0];
            }

            /** @brief Writes the analytic gradient at @p point into @p out. */
            void gradient(SushiRuntime::span<const double> point, SushiRuntime::span<double> out)
            {
                if (step_ == nullptr)
                    return;

                SaTest::write_tensor(pool_.tensor(box_), point);
                SA_EXPECT_OK(step_->run());
                const std::vector<double> values = SaTest::read_tensor(pool_.tensor(gradient_));
                ASSERT_EQ(values.size(), out.size());
                for (std::size_t i = 0; i < values.size(); ++i)
                    out[i] = values[i];
            }

        private:
            Graph::Graph graph_;                   /**< @brief Holds the loss graph. */
            TensorPool pool_;                      /**< @brief Holds its tensors. */
            std::unique_ptr<CompiledStep> step_;   /**< @brief Holds the step; null when refused. */
            ValueID box_ = Graph::K_NO_VALUE;      /**< @brief Holds the box parameter. */
            ValueID weights_ = Graph::K_NO_VALUE;  /**< @brief Holds the bin weights. */
            ValueID loss_ = Graph::K_NO_VALUE;     /**< @brief Holds the root. */
            ValueID gradient_ = Graph::K_NO_VALUE; /**< @brief Holds the gradient at the box. */
    };

    class Medium_DistributionFocalLossEndToEnd : public SaTest::EngineTest {};
} // namespace

/** @brief Tests that the loss is one value from one log-softmax, with no NLL and no gather. */
TEST(Medium_DistributionFocalLoss, TheLossIsOneValueFromOneLogSoftmax)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);
    const ValueID box = builder.input(K_BOX_SHAPE, "box", DType::Float64);
    const ValueID weights = builder.input(K_BOX_SHAPE, "weights", DType::Float64);

    const ValueID loss = emit_distribution_focal_loss(builder, graph, box, weights);

    EXPECT_EQ(graph.value(loss).shape.num_elements(), 1);
    EXPECT_EQ(graph.value(loss).dtype, DType::Float64);
    std::size_t log_softmax = 0;
    std::size_t picks = 0;
    for (const Graph::Node& node : graph.nodes())
    {
        log_softmax += node.op == Ops::K_LOG_SOFTMAX.id ? 1 : 0;
        picks += node.op == Ops::K_GATHER_ROWS.id || node.op == Ops::K_NLL_LOSS.id ? 1 : 0;
    }
    EXPECT_EQ(log_softmax, 1u);
    EXPECT_EQ(picks, 0u);
}

/** @brief Tests the value on hand logits: 0.8 / 4 times the sum of four sides' cross entropies. */
TEST_F(Medium_DistributionFocalLossEndToEnd, TheValueIsTheHandWorkedCrossEntropy)
{
    LossProbe probe(*engine);
    ASSERT_TRUE(probe.ready());

    // Left, right and bottom are uniform; top holds probabilities 0.1, 0.2, 0.3, 0.4.
    std::vector<double> logits(static_cast<std::size_t>(K_BOX_SHAPE.num_elements()), 0.0);
    for (std::size_t bin = 0; bin < 4; ++bin)
        logits[4 + bin] = std::log(static_cast<double>(bin + 1));
    // The background anchor's logits are large and read with weight 0.
    for (std::size_t i = 16; i < logits.size(); ++i)
        logits[i] = 50.0 * static_cast<double>(i % 3);

    // Top's target 1.25 weighs bin 1 by 0.75 and bin 2 by 0.25; each other side sums to 1.
    const double top = -(0.75 * std::log(0.2) + 0.25 * std::log(0.3));
    const double expected =
        0.25 * static_cast<double>(K_FOREGROUND_WEIGHT) * (3.0 * std::log(4.0) + top);
    EXPECT_NEAR(probe.value(logits), expected, K_VALUE_TOLERANCE);
}

/** @brief Tests that the analytic gradient of the loss matches a central difference. */
TEST_F(Medium_DistributionFocalLossEndToEnd, TheGradientMatchesACentralDifference)
{
    LossProbe probe(*engine);
    ASSERT_TRUE(probe.ready());

    std::vector<double> point(static_cast<std::size_t>(K_BOX_SHAPE.num_elements()));
    SaTest::HostNoise(K_LOGIT_SEED).fill(point);

    const Autograd::GradCheckResult result = Autograd::check_full_jacobian(
        [&probe](SushiRuntime::span<const double> at) { return probe.value(at); },
        [&probe](SushiRuntime::span<const double> at, SushiRuntime::span<double> out)
        { probe.gradient(at, out); },
        point);

    EXPECT_TRUE(result.passed) << result.to_string();
}
