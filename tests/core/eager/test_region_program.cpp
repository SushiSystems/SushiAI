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
/* Licensed under the Apache License, Version 2.0 (the "License"); you    */
/* may not use this file except in compliance with the License. You may   */
/* obtain a copy of the License at                                        */
/*                                                                        */
/*     https://www.apache.org/licenses/LICENSE-2.0                        */
/*                                                                        */
/* Unless required by applicable law or agreed to in writing, software    */
/* distributed under the License is distributed on an "AS IS" BASIS,      */
/* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or        */
/* implied. See the License for the specific language governing           */
/* permissions and limitations under the License.                         */
/**************************************************************************/


/**
 * @file test_region_program.cpp
 * @brief Tests the graph, the plan and the reports compile_region makes for one region.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <vector>
#include <gtest/gtest.h>
#include "eager_test_support.hpp"
#include <SushiAI/ops/ops.hpp>
#include <SushiAI/eager/partition.hpp>
#include <SushiAI/eager/region_boundary.hpp>
#include <SushiAI/eager/region_program.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/graph/fusion.hpp>
#include <SushiAI/graph/in_memory_precision_policy.hpp>
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

    /** @brief Expects every parameter of @p step to keep the Parameter lifetime in @p program. */
    void expect_parameters_keep_their_lifetime(const Support::TracedStep& step,
                                               const RegionProgram& program)
    {
        for (Graph::ValueID parameter : step.parameters)
            EXPECT_EQ(program.graph.value(parameter).lifetime, Graph::Lifetime::Parameter);
    }

    /** @brief Expects no import and no export of the regions of @p step to be packed. */
    void expect_no_shared_value_is_packed(const Support::TracedStep& step)
    {
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

    /** @brief Counts the nodes of @p graph that run @p op. */
    [[nodiscard]] std::size_t count_op(const Graph::Graph& graph, OpID op)
    {
        std::size_t found = 0;
        for (const Graph::Node& node : graph.nodes())
            found += node.op == op ? 1 : 0;

        return found;
    }
} // namespace

/** @brief Tests that each import is a leaf of the program and the export is computed by it. */
TEST(Small_RegionProgram, TheGraphHoldsTheRegionsNodesBehindOneLeafPerImport)
{
    const Support::TracedStep step = Support::linear_then_relu(true);

    Eager::ProgramOptions options;
    options.fusion.enabled = false;
    SA_ASSERT_OK_AND_ASSIGN(const RegionProgram program,
                            compile(step, "activation#0", Direction::Forward, options));

    // The region reads one value, the result of the linear layer, and computes the step's result.
    ASSERT_EQ(program.imports.size(), 1u);
    const Graph::NodeID import_producer = program.graph.value(program.imports.front()).producer;
    ASSERT_NE(import_producer, Graph::K_NO_NODE);
    EXPECT_EQ(program.graph.node(import_producer).op, Ops::K_LEAF.id);

    ASSERT_EQ(program.exports, std::vector<Graph::ValueID>{step.hidden});
    const Graph::NodeID export_producer = program.graph.value(step.hidden).producer;
    ASSERT_NE(export_producer, Graph::K_NO_NODE);
    EXPECT_NE(program.graph.node(export_producer).op, Ops::K_LEAF.id);
    EXPECT_EQ(program.graph.node(export_producer).inputs, program.imports);
}

/** @brief Tests that a value keeps its id from the step to the program. */
TEST(Small_RegionProgram, ValuesKeepTheIdsOfTheStep)
{
    const Support::TracedStep step = Support::linear_then_relu(true);
    SA_ASSERT_OK_AND_ASSIGN(const RegionProgram program,
                            compile(step, "activation#0", Direction::Forward));

    ASSERT_GE(program.graph.value_count(), step.graph.value_count());
    EXPECT_EQ(program.graph.value(step.hidden).shape, step.graph.value(step.hidden).shape);
    EXPECT_EQ(program.exports, std::vector<Graph::ValueID>{step.hidden});
}

/** @brief Tests that an import another region computes has the Input lifetime in the program. */
TEST(Small_RegionProgram, AnImportAnotherRegionComputesIsAnInputOfTheProgram)
{
    const Support::TracedStep step = Support::linear_then_relu(true);
    SA_ASSERT_OK_AND_ASSIGN(const RegionProgram program,
                            compile(step, "activation#0", Direction::Forward));

    const Graph::ValueID projected = program.imports.front();
    EXPECT_EQ(step.graph.value(projected).lifetime, Graph::Lifetime::Activation);
    EXPECT_EQ(program.graph.value(projected).lifetime, Graph::Lifetime::Input);
}

/** @brief Tests that a parameter import keeps the lifetime it was declared with. */
TEST(Small_RegionProgram, AParameterImportKeepsItsLifetime)
{
    const Support::TracedStep step = Support::linear_then_relu(true);
    SA_ASSERT_OK_AND_ASSIGN(const RegionProgram program,
                            compile(step, "linear#0", Direction::Forward));

    expect_parameters_keep_their_lifetime(step, program);
}

/** @brief Tests that no import and no export of any region is packed into an arena. */
TEST(Small_RegionProgram, NoSharedValueIsPackedIntoAnArena)
{
    expect_no_shared_value_is_packed(Support::encoder_head_loss(Support::Head::Tanh));
}

/** @brief Tests that a chain inside one region folds into one fused form. */
TEST(Small_RegionProgram, AChainInOneRegionFolds)
{
    const Support::TracedStep step = Support::linear_then_relu(false);
    SA_ASSERT_OK_AND_ASSIGN(const RegionProgram program,
                            compile(step, "linear#0", Direction::Forward));

    EXPECT_EQ(program.fusion.count(FusedForm::GEMMBiasReLU), 1u);
    EXPECT_EQ(count_op(program.graph, Ops::K_RELU.id), 0u);
}

/** @brief Tests that a scope boundary stops a fold. */
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

/** @brief Tests that with fusion off the nodes stay as traced. */
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

/** @brief Tests that a key naming no region stops the process. */
TEST(Medium_RegionProgramFatal, AKeyThatNamesNoRegionIsFatal)
{
    const Support::TracedStep step = Support::linear_then_relu(false);
    const Eager::BoundaryMap boundaries = Support::boundaries_of(step);

    SA_EXPECT_FATAL((void)Eager::compile_region(step.graph, step.partition, Eager::K_NO_REGION,
                                                boundaries.regions[0]),
                    "names no region");
}

/** @brief Tests that a zero alignment comes back as a status. */
TEST(Small_RegionProgram, AZeroAlignmentComesBackAsAStatus)
{
    const Support::TracedStep step = Support::linear_then_relu(false);

    Eager::ProgramOptions options;
    options.plan.alignment = 0;
    SA_EXPECT_FAILURE(compile(step, "linear#0", Direction::Forward, options),
                      StatusCode::InvalidDescription, "alignment must be positive");
}

/** @brief Tests that a precision policy asking for int32 comes back as a status. */
TEST(Small_RegionProgram, ANonFloatCastComesBackAsAStatus)
{
    const Support::TracedStep step = Support::linear_then_relu(false);
    Graph::InMemoryPrecisionPolicy policy;
    const Graph::PrecisionChoice int32{DType::Int32, "test asks for int32"};
    policy.script(Ops::K_MATMUL.id, int32);
    policy.script(Ops::K_BIAS_ADD.id, int32);
    policy.script(Ops::K_RELU.id, int32);

    Eager::ProgramOptions options;
    options.precision = &policy;
    SA_EXPECT_FAILURE(compile(step, "linear#0", Direction::Forward, options),
                      StatusCode::InvalidDescription, "real floating types");
}
