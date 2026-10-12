/**************************************************************************/
/* test_region_program.cpp                                                */
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
 * @file test_region_program.cpp
 * @brief Tests the graph, the plan and the reports compile_region makes for one region.
 * @author Mustafa Garip
 * @see modules/core/eager/README.md
 */

#include <cstddef>
#include <vector>

#include <gtest/gtest.h>

#include <SushiAI/core/ops.hpp>
#include <SushiAI/eager/partition.hpp>
#include <SushiAI/eager/region_boundary.hpp>
#include <SushiAI/eager/region_program.hpp>
#include <SushiAI/graph/fusion.hpp>
#include <SushiAI/graph/ir.hpp>

#include <common/test_throws.hpp>

#include "eager_test_support.hpp"

using namespace SushiAI;
using Eager::Direction;
using Eager::RegionProgram;
using Graph::FusedForm;

namespace Support = SushiAI::Eager::Support;

namespace
{
    /** @brief Compiles the region named @p name of @p step with @p options. */
    [[nodiscard]] RegionProgram compile(const Support::TracedStep& step, const char* name,
                                        Direction direction,
                                        const Eager::ProgramOptions& options = {})
    {
        const Eager::BoundaryMap boundaries = Support::boundaries_of(step);
        const std::size_t at = Support::region_at(step, name, direction);
        EXPECT_LT(at, step.partition.regions.size()) << name;
        return Eager::compile_region(step.graph, step.partition, step.partition.regions[at].key,
                                     boundaries.regions[at], options);
    }

    /** @brief Counts the nodes of @p graph that run @p op. */
    [[nodiscard]] std::size_t count_op(const Graph::Graph& graph, OpID op)
    {
        std::size_t found = 0;
        for (const Graph::Node& node : graph.nodes())
            found += node.op == op ? 1 : 0;

        return found;
    }
} // namespace

TEST(Unit_RegionProgram, TheGraphHoldsTheRegionsNodesBehindOneLeafPerImport)
{
    const Support::TracedStep step = Support::linear_then_relu(true);

    Eager::ProgramOptions options;
    options.fusion.enabled = false;
    const RegionProgram program = compile(step, "activation#0", Direction::Forward, options);

    // The region holds one RELU and reads one value, the result of the linear layer.
    ASSERT_EQ(program.imports.size(), 1u);
    ASSERT_EQ(program.graph.node_count(), 2u);
    EXPECT_EQ(program.graph.node(0).op, Ops::K_LEAF.id);
    EXPECT_EQ(program.graph.node(0).outputs, program.imports);
    EXPECT_EQ(program.graph.node(1).op, Ops::K_RELU.id);
    EXPECT_EQ(program.graph.node(1).inputs, program.imports);
    EXPECT_EQ(program.graph.node(1).outputs, std::vector<Graph::ValueID>{step.hidden});
}

TEST(Unit_RegionProgram, ValuesKeepTheIdsOfTheStep)
{
    const Support::TracedStep step = Support::linear_then_relu(true);
    const RegionProgram program = compile(step, "activation#0", Direction::Forward);

    ASSERT_GE(program.graph.value_count(), step.graph.value_count());
    EXPECT_EQ(program.graph.value(step.hidden).shape, step.graph.value(step.hidden).shape);
    EXPECT_EQ(program.exports, std::vector<Graph::ValueID>{step.hidden});
}

TEST(Unit_RegionProgram, AnImportAnotherRegionComputesIsAnInputOfTheProgram)
{
    const Support::TracedStep step = Support::linear_then_relu(true);
    const RegionProgram program = compile(step, "activation#0", Direction::Forward);

    const Graph::ValueID projected = program.imports.front();
    EXPECT_EQ(step.graph.value(projected).lifetime, Graph::Lifetime::Activation);
    EXPECT_EQ(program.graph.value(projected).lifetime, Graph::Lifetime::Input);
}

TEST(Unit_RegionProgram, AParameterImportKeepsItsLifetime)
{
    const Support::TracedStep step = Support::linear_then_relu(true);
    const RegionProgram program = compile(step, "linear#0", Direction::Forward);

    for (Graph::ValueID parameter : step.parameters)
        EXPECT_EQ(program.graph.value(parameter).lifetime, Graph::Lifetime::Parameter);
}

TEST(Unit_RegionProgram, NoSharedValueIsPackedIntoAnArena)
{
    const Support::TracedStep step = Support::encoder_head_loss(Support::Head::Tanh);
    const Eager::BoundaryMap boundaries = Support::boundaries_of(step);

    for (std::size_t at = 0; at < step.partition.regions.size(); ++at)
    {
        const RegionProgram program = Eager::compile_region(
            step.graph, step.partition, step.partition.regions[at].key, boundaries.regions[at]);

        for (Graph::ValueID value : program.imports)
            EXPECT_FALSE(program.plan.is_packed(value)) << "import " << value;
        for (Graph::ValueID value : program.exports)
            EXPECT_FALSE(program.plan.is_packed(value)) << "export " << value;
    }
}

TEST(Unit_RegionProgram, AChainInOneRegionFolds)
{
    const Support::TracedStep step = Support::linear_then_relu(false);
    const RegionProgram program = compile(step, "linear#0", Direction::Forward);

    EXPECT_EQ(program.fusion.count(FusedForm::GEMMBiasReLU), 1u);
    EXPECT_EQ(count_op(program.graph, Ops::K_RELU.id), 0u);
}

TEST(Unit_RegionProgram, AChainCutByAScopeFoldsOnlyUpToTheBoundary)
{
    const Support::TracedStep step = Support::linear_then_relu(true);
    const RegionProgram linear = compile(step, "linear#0", Direction::Forward);
    const RegionProgram activation = compile(step, "activation#0", Direction::Forward);

    EXPECT_EQ(linear.fusion.count(FusedForm::GEMMBiasReLU), 0u);
    EXPECT_EQ(linear.fusion.count(FusedForm::GEMMBias), 1u);
    EXPECT_EQ(activation.fusion.count(FusedForm::GEMMBiasReLU), 0u);
    EXPECT_EQ(count_op(activation.graph, Ops::K_RELU.id), 1u);
}

TEST(Unit_RegionProgram, SwitchingFusionOffLeavesTheNodesAsTraced)
{
    const Support::TracedStep step = Support::linear_then_relu(false);

    Eager::ProgramOptions options;
    options.fusion.enabled = false;
    const RegionProgram program = compile(step, "linear#0", Direction::Forward, options);

    EXPECT_EQ(count_op(program.graph, Ops::K_MATMUL.id), 1u);
    EXPECT_EQ(count_op(program.graph, Ops::K_BIAS_ADD.id), 1u);
    EXPECT_EQ(count_op(program.graph, Ops::K_RELU.id), 1u);
}

TEST(Unit_RegionProgram, AKeyThatNamesNoRegionThrows)
{
    const Support::TracedStep step = Support::linear_then_relu(false);
    const Eager::BoundaryMap boundaries = Support::boundaries_of(step);

    SA_EXPECT_ERROR((void)Eager::compile_region(step.graph, step.partition, Eager::K_NO_REGION,
                                                boundaries.regions[0]),
                    "names no region");
}
