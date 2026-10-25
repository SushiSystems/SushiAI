/**************************************************************************/
/* test_free_detection_loss.cpp                                           */
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
 * @file test_free_detection_loss.cpp
 * @brief Tests the anchor-free detection loss graph against its host evaluation.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>
#include <gtest/gtest.h>
#include <SushiRuntime/core/span.hpp>
#include <SushiBLAS/engine/engine.hpp>
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
#include <SushiAI/vision/training/free_detection_loss.hpp>
#include <SushiAI/vision/training/free_detection_loss_host.hpp>
#include <SushiAI/vision/training/free_loss_gains.hpp>
#include <SushiAI/vision/training/free_model_facts.hpp>

using namespace SushiAI;
using Graph::CompiledStep;
using Graph::GraphBuilder;
using Graph::TensorPool;
using Graph::ValueID;
using Vision::Training::FreeLevelLossArrays;
using Vision::Training::FreeLevelLossValues;
using Vision::Training::FreeLossGains;
using Vision::Training::FreeLossTermIDs;
using Vision::Training::FreeLossTerms;
using Vision::Training::FreeLossValues;
using Vision::Training::FreeModelFacts;
// SaTest stays reachable inside a TEST_F body; see tests/vision/training/README.md.
namespace SaTest = SushiAI::Test;

namespace
{
    /** @brief Holds the classes of the fixture. */
    constexpr std::int64_t K_CLASSES = 2;

    /** @brief Holds the anchors of the fixture: one image on a 1x2 grid. */
    constexpr std::int64_t K_ROWS = 2;

    /** @brief Seeds the box logits of the reg_max 4 cases. */
    constexpr std::uint64_t K_LOGIT_SEED = 20261025;

    /** @brief Bounds the difference of the graph and the host, both in double. */
    constexpr double K_VALUE_TOLERANCE = 1e-9;

    /** @brief Holds the score logits of both anchors, two classes each. */
    const std::vector<double> K_SCORES = {0.3, -1.2, 0.8, 0.1};

    /** @brief Holds the raw distances at reg_max 1: anchor 0's box is the target moved by 0.1. */
    const std::vector<double> K_RAW_DISTANCES = {0.4, 0.5, 0.7, 0.8, 0.3, 0.2, 0.6, 0.9};

    /** @brief Returns the `[1, 1, 2, channels]` shape of a fixture value. */
    Shape level_shape(std::int64_t channels)
    {
        return Shape{1, 1, K_ROWS, channels};
    }

    /** @brief Holds the leaves of one level: anchor 0 is foreground, anchor 1 background. */
    struct Leaves
    {
        /** @brief Holds the `[1, 1, 2, 2]` soft class targets. */
        std::vector<double> target_scores = {0.7, 0.0, 0.0, 0.0};
        /** @brief Holds 1 on the foreground anchor's four values. */
        std::vector<double> box_mask = {1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0};
        /** @brief Holds each anchor's score sum over the normaliser. */
        std::vector<double> normalized_weight = {0.7, 0.0};
        /** @brief Holds the target and the safe box, grid units. */
        std::vector<double> target_boxes = {0.2, 0.1, 1.3, 1.4, 0.0, 0.0, 1.0, 1.0};
        /** @brief Holds the target's sides around anchor 0. */
        std::vector<double> target_distances = {0.3, 0.4, 0.8, 0.9, 0.0, 0.0, 0.0, 0.0};
        /** @brief Holds the factor of each L1 error. */
        std::vector<double> l1_scale = {0.05, 0.06, 0.05, 0.06, 0.0, 0.0, 0.0, 0.0};
        /** @brief Holds K_SAFE_BOX in both rows. */
        std::vector<double> safe_boxes = {0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0};
        /** @brief Holds both cell centres. */
        std::vector<double> anchor_points = {0.5, 0.5, 1.5, 0.5};
        /** @brief Holds the DFL bin weights; empty at reg_max 1. */
        std::vector<double> distribution_weights;
        /** @brief Holds the bin indices; empty at reg_max 1. */
        std::vector<double> projection;
        /** @brief Holds the `[1]` normaliser. */
        double target_scores_sum = 1.0;
    };

    /** @brief Returns the leaves for @p reg_max bins, with DFL weights above 1. */
    Leaves make_leaves(std::int64_t reg_max)
    {
        Leaves leaves;
        if (reg_max == 1)
            return leaves;

        const std::vector<float> targets(leaves.target_distances.begin(),
                                         leaves.target_distances.end());
        const std::vector<float> weights(leaves.normalized_weight.begin(),
                                         leaves.normalized_weight.end());
        std::vector<float> filled(static_cast<std::size_t>(K_ROWS * 4 * reg_max));
        Vision::Detect::Free::fill_distribution_weights(targets, weights, reg_max, filled);
        leaves.distribution_weights.assign(filled.begin(), filled.end());
        for (std::int64_t bin = 0; bin < reg_max; ++bin)
            leaves.projection.push_back(static_cast<double>(bin));
        return leaves;
    }

    /** @brief Returns the facts of a one-level model of @p reg_max bins. */
    FreeModelFacts make_facts(std::int64_t reg_max)
    {
        FreeModelFacts facts;
        facts.reg_max = reg_max;
        facts.classes = K_CLASSES;
        facts.levels.resize(1);
        facts.levels[0].height = 1;
        facts.levels[0].width = K_ROWS;
        facts.levels[0].stride = 8;
        return facts;
    }

    /** @brief Returns the point of a probe: the box outputs, then the score logits. */
    std::vector<double> make_point(std::int64_t reg_max)
    {
        std::vector<double> point;
        if (reg_max == 1)
        {
            point = K_RAW_DISTANCES;
        }
        else
        {
            point.resize(static_cast<std::size_t>(K_ROWS * 4 * reg_max));
            SaTest::HostNoise(K_LOGIT_SEED).fill(point);
        }

        point.insert(point.end(), K_SCORES.begin(), K_SCORES.end());
        return point;
    }

    /** @brief Holds the loss over FLOAT64 box and score parameters, compiled with its gradient. */
    class LossProbe
    {
        public:
            /** @brief Builds, differentiates and compiles the loss of @p reg_max on @p engine. */
            LossProbe(SushiBLAS::Engine& engine, std::int64_t reg_max, const FreeLossGains& gains)
                : pool_(engine), reg_max_(reg_max), gains_(gains), leaves_(make_leaves(reg_max))
            {
                const GraphBuilder builder(graph_);
                FreeLevelLossValues level;
                level.boxes = builder.parameter(level_shape(4 * reg_max), "boxes", DType::Float64);
                level.scores = builder.parameter(level_shape(K_CLASSES), "scores", DType::Float64);
                level.target_scores = leaf(builder, K_CLASSES, "target_scores");
                level.box_mask = leaf(builder, 4, "box_mask");
                level.normalized_weight = leaf(builder, 1, "normalized_weight");
                level.target_boxes = leaf(builder, 4, "target_boxes");
                level.target_distances = leaf(builder, 4, "target_distances");
                level.l1_scale = leaf(builder, 4, "l1_scale");
                level.safe_boxes = leaf(builder, 4, "safe_boxes");
                level.anchor_points = leaf(builder, 2, "anchor_points");

                FreeLossValues values;
                values.target_scores_sum = builder.input(Shape{1}, "tss", DType::Float64);
                if (reg_max > 1)
                {
                    level.distribution_weights = leaf(builder, 4 * reg_max, "dfl_weights");
                    values.projection =
                        builder.input(Shape{reg_max, 1}, "projection", DType::Float64);
                }
                values.levels.push_back(level);
                level_ = level;
                values_ = values;

                terms_ = Vision::Training::emit_free_detection_loss(
                    builder, graph_, make_facts(reg_max), values, gains);

                const std::vector<ValueID> wrt = {level.boxes, level.scores};
                gradients_ = Autograd::differentiate(
                    graph_, terms_.total,
                    SushiRuntime::span<const ValueID>(wrt.data(), wrt.size()));

                SA_EXPECT_OK(pool_.materialise_leaves(graph_));
                Result<std::unique_ptr<CompiledStep>> compiled =
                    CompiledStep::compile(engine, graph_, pool_);
                SA_EXPECT_OK(compiled);
                if (compiled.has_value())
                    step_ = std::move(compiled).value();

                write_leaves();
            }

            /** @brief Returns whether the step compiled. */
            bool ready() const { return step_ != nullptr; }

            /** @brief Returns the graph's loss at @p point, or zero when it did not compile. */
            double value(SushiRuntime::span<const double> point)
            {
                if (step_ == nullptr)
                    return 0.0;

                run_at(point);
                return SaTest::read_tensor(pool_.tensor(terms_.total))[0];
            }

            /** @brief Writes the analytic gradient at @p point into @p out. */
            void gradient(SushiRuntime::span<const double> point, SushiRuntime::span<double> out)
            {
                if (step_ == nullptr)
                    return;

                run_at(point);
                std::size_t at = 0;
                for (const ValueID id : gradients_)
                {
                    ASSERT_NE(id, Graph::K_NO_VALUE);
                    for (const double value : SaTest::read_tensor(pool_.tensor(id)))
                        out[at++] = value;
                }
                ASSERT_EQ(at, out.size());
            }

            /** @brief Returns the host evaluation of the loss at @p point. */
            FreeLossTerms host(SushiRuntime::span<const double> point) const
            {
                const std::size_t box_values = static_cast<std::size_t>(K_ROWS * 4 * reg_max_);
                FreeLevelLossArrays level;
                level.boxes = point.subspan(0, box_values);
                level.scores = point.subspan(box_values);
                level.target_scores = leaves_.target_scores;
                level.box_mask = leaves_.box_mask;
                level.normalized_weight = leaves_.normalized_weight;
                level.target_boxes = leaves_.target_boxes;
                level.target_distances = leaves_.target_distances;
                level.l1_scale = leaves_.l1_scale;
                level.safe_boxes = leaves_.safe_boxes;
                level.distribution_weights = leaves_.distribution_weights;
                level.anchor_points = leaves_.anchor_points;
                return Vision::Training::evaluate_free_detection_loss(
                    SushiRuntime::span<const FreeLevelLossArrays>(&level, 1), reg_max_, K_CLASSES,
                    1, leaves_.target_scores_sum, gains_);
            }

            /** @brief Returns the graph's value of the term @p id after the last run. */
            double term(ValueID id) const
            {
                return SaTest::read_tensor(pool_.tensor(id))[0];
            }

            /** @brief Returns the ids of the root and the terms. */
            const FreeLossTermIDs& terms() const { return terms_; }

        private:
            /** @brief Declares a FLOAT64 leaf of @p channels named @p name. */
            static ValueID leaf(const GraphBuilder& builder, std::int64_t channels,
                                const char* name)
            {
                return builder.input(level_shape(channels), name, DType::Float64);
            }

            /** @brief Writes every leaf's fixture values into the pool. */
            void write_leaves()
            {
                if (step_ == nullptr)
                    return;

                SaTest::write_tensor(pool_.tensor(level_.target_scores), leaves_.target_scores);
                SaTest::write_tensor(pool_.tensor(level_.box_mask), leaves_.box_mask);
                SaTest::write_tensor(pool_.tensor(level_.normalized_weight),
                                     leaves_.normalized_weight);
                SaTest::write_tensor(pool_.tensor(level_.target_boxes), leaves_.target_boxes);
                SaTest::write_tensor(pool_.tensor(level_.target_distances),
                                     leaves_.target_distances);
                SaTest::write_tensor(pool_.tensor(level_.l1_scale), leaves_.l1_scale);
                SaTest::write_tensor(pool_.tensor(level_.safe_boxes), leaves_.safe_boxes);
                SaTest::write_tensor(pool_.tensor(level_.anchor_points), leaves_.anchor_points);
                const std::vector<double> sum = {leaves_.target_scores_sum};
                SaTest::write_tensor(pool_.tensor(values_.target_scores_sum), sum);
                if (reg_max_ > 1)
                {
                    SaTest::write_tensor(pool_.tensor(level_.distribution_weights),
                                         leaves_.distribution_weights);
                    SaTest::write_tensor(pool_.tensor(values_.projection), leaves_.projection);
                }
            }

            /** @brief Writes @p point into the parameters and runs the step. */
            void run_at(SushiRuntime::span<const double> point)
            {
                const std::size_t box_values = static_cast<std::size_t>(K_ROWS * 4 * reg_max_);
                SaTest::write_tensor(pool_.tensor(level_.boxes), point.subspan(0, box_values));
                SaTest::write_tensor(pool_.tensor(level_.scores), point.subspan(box_values));
                SA_EXPECT_OK(step_->run());
            }

            Graph::Graph graph_;                 /**< @brief Holds the loss graph. */
            TensorPool pool_;                    /**< @brief Holds its tensors. */
            std::unique_ptr<CompiledStep> step_; /**< @brief Holds the step; null when refused. */
            std::int64_t reg_max_ = 1;           /**< @brief Holds the bins per side. */
            FreeLossGains gains_;                /**< @brief Holds the gains of both sides. */
            Leaves leaves_;                      /**< @brief Holds the fixture's leaf values. */
            FreeLevelLossValues level_;          /**< @brief Holds the level's ids. */
            FreeLossValues values_;              /**< @brief Holds the shared ids. */
            FreeLossTermIDs terms_;              /**< @brief Holds the root and the terms. */
            std::vector<ValueID> gradients_;     /**< @brief Holds the box, then score gradient. */
    };

    /** @brief Returns the gains with the L1 term switched on. */
    FreeLossGains with_l1()
    {
        FreeLossGains gains;
        gains.l1_gain = 1.5f;
        return gains;
    }

    /** @brief Runs check_full_jacobian over @p probe at @p point. */
    Autograd::GradCheckResult check_gradient(LossProbe& probe, std::vector<double> point)
    {
        return Autograd::check_full_jacobian(
            [&probe](SushiRuntime::span<const double> at) { return probe.value(at); },
            [&probe](SushiRuntime::span<const double> at, SushiRuntime::span<double> out)
            { probe.gradient(at, out); },
            point);
    }

    class Medium_FreeDetectionLossEndToEnd : public SaTest::EngineTest {};
} // namespace

/** @brief Tests that CIoU's alpha is held by one STOP_GRADIENT and the boxes pass one SELECT. */
TEST(Small_FreeDetectionLoss, AlphaIsHeldAndTheBoxesAreSelected)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);
    FreeLevelLossValues level;
    level.boxes = builder.input(level_shape(4), "boxes", DType::Float64);
    level.scores = builder.input(level_shape(K_CLASSES), "scores", DType::Float64);
    level.target_scores = builder.input(level_shape(K_CLASSES), "ts", DType::Float64);
    level.box_mask = builder.input(level_shape(4), "mask", DType::Float64);
    level.normalized_weight = builder.input(level_shape(1), "nw", DType::Float64);
    level.target_boxes = builder.input(level_shape(4), "tb", DType::Float64);
    level.safe_boxes = builder.input(level_shape(4), "sb", DType::Float64);
    level.anchor_points = builder.input(level_shape(2), "ap", DType::Float64);
    FreeLossValues values;
    values.levels.push_back(level);
    values.target_scores_sum = builder.input(Shape{1}, "tss", DType::Float64);

    const FreeLossTermIDs terms = Vision::Training::emit_free_detection_loss(
        builder, graph, make_facts(1), values, FreeLossGains{});

    EXPECT_EQ(graph.value(terms.total).shape, Shape{1});
    EXPECT_EQ(terms.distribution, Graph::K_NO_VALUE);
    EXPECT_EQ(terms.l1, Graph::K_NO_VALUE);
    std::size_t stops = 0;
    std::size_t selects = 0;
    for (const Graph::Node& node : graph.nodes())
    {
        stops += node.op == Ops::K_STOP_GRADIENT.id ? 1 : 0;
        selects += node.op == Ops::K_SELECT.id ? 1 : 0;
    }
    EXPECT_EQ(stops, 1u);
    EXPECT_EQ(selects, 1u);
}

/** @brief Tests that every term at reg_max 4 equals the host evaluation. */
TEST_F(Medium_FreeDetectionLossEndToEnd, TheDistributionCaseEqualsTheHostEvaluation)
{
    LossProbe probe(*engine, 4, FreeLossGains{});
    ASSERT_TRUE(probe.ready());

    const std::vector<double> point = make_point(4);
    const double total = probe.value(point);
    const FreeLossTerms host = probe.host(point);

    EXPECT_NEAR(total, host.total, K_VALUE_TOLERANCE);
    EXPECT_NEAR(probe.term(probe.terms().box), host.box, K_VALUE_TOLERANCE);
    EXPECT_NEAR(probe.term(probe.terms().classes), host.classes, K_VALUE_TOLERANCE);
    EXPECT_NEAR(probe.term(probe.terms().distribution), host.distribution, K_VALUE_TOLERANCE);
    EXPECT_GT(host.box, 0.0);
    EXPECT_GT(host.distribution, 0.0);
}

/** @brief Tests that every term at reg_max 1 with an L1 gain equals the host evaluation. */
TEST_F(Medium_FreeDetectionLossEndToEnd, TheRawDistanceCaseEqualsTheHostEvaluation)
{
    LossProbe probe(*engine, 1, with_l1());
    ASSERT_TRUE(probe.ready());

    const std::vector<double> point = make_point(1);
    const double total = probe.value(point);
    const FreeLossTerms host = probe.host(point);

    EXPECT_NEAR(total, host.total, K_VALUE_TOLERANCE);
    EXPECT_NEAR(probe.term(probe.terms().box), host.box, K_VALUE_TOLERANCE);
    EXPECT_NEAR(probe.term(probe.terms().classes), host.classes, K_VALUE_TOLERANCE);
    EXPECT_NEAR(probe.term(probe.terms().l1), host.l1, K_VALUE_TOLERANCE);
    EXPECT_GT(host.l1, 0.0);
}

/** @brief Tests the gradient where the aspect ratios agree, so the held alpha changes nothing. */
TEST_F(Medium_FreeDetectionLossEndToEnd, TheBoxClassAndL1GradientMatchesACentralDifference)
{
    LossProbe probe(*engine, 1, with_l1());
    ASSERT_TRUE(probe.ready());

    const Autograd::GradCheckResult result = check_gradient(probe, make_point(1));

    EXPECT_TRUE(result.passed) << result.to_string();
}

/** @brief Tests the class and distribution gradient; the box gain is 0 off equal aspects. */
TEST_F(Medium_FreeDetectionLossEndToEnd, TheClassAndDistributionGradientMatchesACentralDifference)
{
    FreeLossGains gains;
    gains.box_gain = 0.0f;
    LossProbe probe(*engine, 4, gains);
    ASSERT_TRUE(probe.ready());

    const Autograd::GradCheckResult result = check_gradient(probe, make_point(4));

    EXPECT_TRUE(result.passed) << result.to_string();
}
