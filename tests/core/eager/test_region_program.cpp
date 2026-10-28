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
 */

#include <cstddef>
#include <string_view>
#include <vector>
#include <gtest/gtest.h>
#include "eager_test_support.hpp"
#include <SushiAI/ops/ops.hpp>
#include <SushiAI/eager/partition.hpp>
#include <SushiAI/eager/region_boundary.hpp>
#include <SushiAI/eager/region_program.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/graph/fusion.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/graph/precision.hpp>
#include <SushiAI/dtype/dtype.hpp>

using namespace SushiAI;
using Eager::Direction;
using Eager::RegionProgram;
using Graph::FusedForm;
namespace Support = SushiAI::Eager::Support;

namespace
{
    /** @brief Compiles the region named @p name of @p step with @p options. */
    [[nodiscard]] Result<RegionProgram> compile(const Support::TracedStep& step,
                                                const char* name, Direction direction,
                                                const Eager::ProgramOptions& options = {})
    {
        const Eager::BoundaryMap boundaries = Support::boundaries_of(step);
        const std::size_t at = Support::region_at(step, name, direction);
        EXPECT_LT(at, step.partition.regions.size()) << name;
        return Eager::compile_region(step.graph, step.partition, step.partition.regions[at].key,
                                     boundaries.regions[at], options);
    }

    /** @brief Asks every operation to compute in int32, a conversion the cast pass refuses. */
    class Int32Policy final : public Graph::PrecisionPolicy
    {
        public:
            [[nodiscard]] Graph::PrecisionChoice choose(
                const Graph::PrecisionQuery& query) const override
            {
                (void)query;
                return Graph::PrecisionChoice{DType::Int32, "test asks for int32"};
            }

            [[nodiscard]] std::string_view name() const noexcept override
            {
                return "int32";
            }
    };

    /** @brief Counts the nodes of @p graph that run @p op. */
    [[nodiscard]] std::size_t count_op(const Graph::Graph& graph, OpID op)
    {
        std::size_t found = 0;
        for (const Graph::Node& node : graph.nodes())
            found += node.op == op ? 1 : 0;

        return found;
    }
} // namespace

TEST(Small_RegionProgram, TheGraphHoldsTheRegionsNodesBehindOneLeafPerImport)
{
    const Support::TracedStep step = Support::linear_then_relu(true);

    Eager::ProgramOptions options;
    options.fusion.enabled = false;
    SA_ASSERT_OK_AND_ASSIGN(const RegionProgram program,
                            compile(step, "activation#0", Direction::Forward, options));

    // The region holds one RELU and reads one value, the result of the linear layer.
    ASSERT_EQ(program.imports.size(), 1u);
    ASSERT_EQ(program.graph.node_count(), 2u);
    EXPECT_EQ(program.graph.node(0).op, Ops::K_LEAF.id);
    EXPECT_EQ(program.graph.node(0).outputs, program.imports);
    EXPECT_EQ(program.graph.node(1).op, Ops::K_RELU.id);
    EXPECT_EQ(program.graph.node(1).inputs, program.imports);
    EXPECT_EQ(program.graph.node(1).outputs, std::vector<Graph::ValueID>{step.hidden});
}

TEST(Small_RegionProgram, ValuesKeepTheIdsOfTheStep)
{
    const Support::TracedStep step = Support::linear_then_relu(true);
    SA_ASSERT_OK_AND_ASSIGN(const RegionProgram program,
                            compile(step, "activation#0", Direction::Forward));

    ASSERT_GE(program.graph.value_count(), step.graph.value_count());
    EXPECT_EQ(program.graph.value(step.hidden).shape, step.graph.value(step.hidden).shape);
    EXPECT_EQ(program.exports, std::vector<Graph::ValueID>{step.hidden});
}

TEST(Small_RegionProgram, AnImportAnotherRegionComputesIsAnInputOfTheProgram)
{
    const Support::TracedStep step = Support::linear_then_relu(true);
    SA_ASSERT_OK_AND_ASSIGN(const RegionProgram program,
                            compile(step, "activation#0", Direction::Forward));

    const Graph::ValueID projected = program.imports.front();
    EXPECT_EQ(step.graph.value(projected).lifetime, Graph::Lifetime::Activation);
    EXPECT_EQ(program.graph.value(projected).lifetime, Graph::Lifetime::Input);
}

TEST(Small_RegionProgram, AParameterImportKeepsItsLifetime)
{
    const Support::TracedStep step = Support::linear_then_relu(true);
    SA_ASSERT_OK_AND_ASSIGN(const RegionProgram program,
                            compile(step, "linear#0", Direction::Forward));

    for (Graph::ValueID parameter : step.parameters)
        EXPECT_EQ(program.graph.value(parameter).lifetime, Graph::Lifetime::Parameter);
}

TEST(Small_RegionProgram, NoSharedValueIsPackedIntoAnArena)
{
    const Support::TracedStep step = Support::encoder_head_loss(Support::Head::Tanh);
    const Eager::BoundaryMap boundaries = Support::boundaries_of(step);

    for (std::size_t at = 0; at < step.partition.regions.size(); ++at)
    {
        SA_ASSERT_OK_AND_ASSIGN(const RegionProgram program,
                                Eager::compile_region(step.graph, step.partition,
                                                      step.partition.regions[at].key,
                                                      boundaries.regions[at]));

        for (Graph::ValueID value : program.imports)
            EXPECT_FALSE(program.plan.is_packed(value)) << "import " << value;
        for (Graph::ValueID value : program.exports)
            EXPECT_FALSE(program.plan.is_packed(value)) << "export " << value;
    }
}

TEST(Small_RegionProgram, AChainInOneRegionFolds)
{
    const Support::TracedStep step = Support::linear_then_relu(false);
    SA_ASSERT_OK_AND_ASSIGN(const RegionProgram program,
                            compile(step, "linear#0", Direction::Forward));

    EXPECT_EQ(program.fusion.count(FusedForm::GEMMBiasReLU), 1u);
    EXPECT_EQ(count_op(program.graph, Ops::K_RELU.id), 0u);
}

TEST(Small_RegionProgram, AChainCutByAScopeFoldsOnlyUpToTheBoundary)
{
    const Support::TracedStep step = Support::linear_then_relu(true);
    SA_ASSERT_OK_AND_ASSIGN(const RegionProgram linear,
                            compile(step, "linear#0", Direction::Forward));
    SA_ASSERT_OK_AND_ASSIGN(const RegionProgram activation,
                            compile(step, "activation#0", Direction::Forward));

    EXPECT_EQ(linear.fusion.count(FusedForm::GEMMBiasReLU), 0u);
    EXPECT_EQ(linear.fusion.count(FusedForm::GEMMBias), 1u);
    EXPECT_EQ(activation.fusion.count(FusedForm::GEMMBiasReLU), 0u);
    EXPECT_EQ(count_op(activation.graph, Ops::K_RELU.id), 1u);
}

TEST(Small_RegionProgram, SwitchingFusionOffLeavesTheNodesAsTraced)
{
    const Support::TracedStep step = Support::linear_then_relu(false);

    Eager::ProgramOptions options;
    options.fusion.enabled = false;
    SA_ASSERT_OK_AND_ASSIGN(const RegionProgram program,
                            compile(step, "linear#0", Direction::Forward, options));

    EXPECT_EQ(count_op(program.graph, Ops::K_MATMUL.id), 1u);
    EXPECT_EQ(count_op(program.graph, Ops::K_BIAS_ADD.id), 1u);
    EXPECT_EQ(count_op(program.graph, Ops::K_RELU.id), 1u);
}

TEST(Small_RegionProgram, AKeyThatNamesNoRegionIsFatal)
{
    const Support::TracedStep step = Support::linear_then_relu(false);
    const Eager::BoundaryMap boundaries = Support::boundaries_of(step);

    SA_EXPECT_FATAL((void)Eager::compile_region(step.graph, step.partition, Eager::K_NO_REGION,
                                                boundaries.regions[0]),
                    "names no region");
}

TEST(Small_RegionProgram, AZeroAlignmentComesBackAsAStatus)
{
    const Support::TracedStep step = Support::linear_then_relu(false);

    Eager::ProgramOptions options;
    options.plan.alignment = 0;
    SA_EXPECT_FAILURE(compile(step, "linear#0", Direction::Forward, options),
                      StatusCode::InvalidDescription, "alignment must be positive");
}

TEST(Small_RegionProgram, ANonFloatCastComesBackAsAStatus)
{
    const Support::TracedStep step = Support::linear_then_relu(false);
    const Int32Policy policy;

    Eager::ProgramOptions options;
    options.precision = &policy;
    SA_EXPECT_FAILURE(compile(step, "linear#0", Direction::Forward, options),
                      StatusCode::InvalidDescription, "real floating types");
}
