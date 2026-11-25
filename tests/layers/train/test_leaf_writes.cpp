/**************************************************************************/
/* test_leaf_writes.cpp                                                   */
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
 * @file test_leaf_writes.cpp
 * @brief Tests mark_leaves_unwritten and require_leaves_written.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>
#include <gtest/gtest.h>
#include <SushiBLAS/tensor/tensor.hpp>
#include <SushiAI/allocation/allocation.hpp>
#include <SushiAI/allocation/host_copy.hpp>
#include <SushiAI/dtype/dtype.hpp>
#include <SushiAI/execution/host_execution_backend.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/graph/builder.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/graph/tensor_pool.hpp>
#include <SushiAI/shape/shape.hpp>
#include <SushiAI/tensor_values/tensor_values.hpp>
#include <SushiAI/train/leaf_writes.hpp>
#include <SushiAI/train/objective.hpp>

using namespace SushiAI;
namespace SaTest = SushiAI::Test;

namespace
{
    /** @brief Holds the extents of the leaves the cases declare. */
    constexpr std::int64_t K_ROWS = 3;

    /** @brief Holds the element types the leaf cases run over. */
    const std::vector<DType> K_LEAF_TYPES = {DType::Float32, DType::Float64, DType::Int32};

    /** @brief Runs one case per element type of a leaf. */
    class Small_LeafWritesType : public ::testing::TestWithParam<DType>
    {

    };

    /** @brief Provides the host engine the leaves are allocated on and the leaves they bind. */
    class Small_LeafWrites : public ::testing::Test
    {
        protected:
            /** @brief Declares the leaf @p role of [K_ROWS] elements of @p dtype and binds it. */
            void declare_bound(const std::string& role, DType dtype)
            {
                const Graph::GraphBuilder builder(graph);
                const Graph::ValueID id = builder.input(Shape{K_ROWS}, role, dtype);
                leaves.add(role, id);

                Result<SushiBLAS::Tensor> made =
                    allocate_tensor(backend.engine(), Shape{K_ROWS}, dtype);
                ASSERT_TRUE(made.has_value());
                SA_EXPECT_OK(pool.bind(id, made.value()));
            }

            /** @brief Declares the leaf @p role and leaves it out of the pool. */
            void declare_unbound(const std::string& role)
            {
                const Graph::GraphBuilder builder(graph);
                leaves.add(role, builder.input(Shape{K_ROWS}, role, DType::Float32));
            }

            /** @brief Writes every element of the float32 leaf @p role, as a feed that wrote it. */
            void write_all(const std::string& role)
            {
                SaTest::write_tensor(pool.tensor(leaves.at(role)),
                                     std::vector<double>(static_cast<std::size_t>(K_ROWS), 1.0));
            }

            /** @brief The host backend the tensors are allocated from. */
            HostExecutionBackend backend;

            /** @brief The graph the leaves are inputs of. */
            Graph::Graph graph;

            /** @brief The leaves the cases declare. */
            Train::ObjectiveLeaves leaves;

            /** @brief Binds the tensor of every declared leaf. */
            Graph::TensorPool pool{backend.engine()};
    };
} // namespace

INSTANTIATE_TEST_SUITE_P(Types, Small_LeafWritesType, ::testing::ValuesIn(K_LEAF_TYPES));

/** @brief Tests that a leaf no feed wrote is refused, whatever its element type. */
TEST_P(Small_LeafWritesType, ALeafNoFeedWroteIsRefused)
{
    HostExecutionBackend backend;
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    Train::ObjectiveLeaves leaves;
    Graph::TensorPool pool(backend.engine());
    const Graph::ValueID id = builder.input(Shape{K_ROWS}, "targets", GetParam());
    leaves.add("targets", id);
    SA_ASSERT_OK_AND_ASSIGN(SushiBLAS::Tensor tensor,
                            allocate_tensor(backend.engine(), Shape{K_ROWS}, GetParam()));
    SA_EXPECT_OK(pool.bind(id, tensor));

    SA_EXPECT_OK(Train::mark_leaves_unwritten(leaves, pool));

    SA_EXPECT_FAILURE(Train::require_leaves_written(leaves, pool), StatusCode::InvalidDescription,
                      "'targets'", "element 0", "unwritten");
}

/** @brief Tests that a leaf holding a value of an earlier step is refused when nothing wrote it. */
TEST_F(Small_LeafWrites, AStaleValueIsNotMistakenForAWrite)
{
    declare_bound("targets", DType::Float32);
    SaTest::write_tensor(pool.tensor(leaves.at("targets")), std::vector<double>{7.0, 7.0, 7.0});

    SA_EXPECT_OK(Train::mark_leaves_unwritten(leaves, pool));

    SA_EXPECT_FAILURE(Train::require_leaves_written(leaves, pool), StatusCode::InvalidDescription,
                      "'targets'");
}

/** @brief Tests that a leaf a feed changed is accepted. */
TEST_F(Small_LeafWrites, ALeafAFeedWroteIsAccepted)
{
    declare_bound("targets", DType::Float32);
    SA_EXPECT_OK(Train::mark_leaves_unwritten(leaves, pool));

    SaTest::write_tensor(pool.tensor(leaves.at("targets")), std::vector<double>{1.0, 0.0, 0.0});

    SA_EXPECT_OK(Train::require_leaves_written(leaves, pool));
}

/** @brief Tests that the refusal names the leaf that was skipped and not the one that was written. */
TEST_F(Small_LeafWrites, OnlyTheSkippedLeafIsNamed)
{
    declare_bound("targets", DType::Float32);
    declare_bound("weights", DType::Float32);
    SA_EXPECT_OK(Train::mark_leaves_unwritten(leaves, pool));
    write_all("targets");

    const Status status = Train::require_leaves_written(leaves, pool);

    SA_EXPECT_FAILURE(status, StatusCode::InvalidDescription, "'weights'", "element 0");
    EXPECT_EQ(status.message().find("'targets'"), std::string_view::npos);
}

/** @brief Tests that a leaf a feed wrote in part is refused, naming its first unwritten element. */
TEST_F(Small_LeafWrites, ALeafWrittenInPartIsRefusedNamingTheElement)
{
    declare_bound("targets", DType::Float32);
    SA_EXPECT_OK(Train::mark_leaves_unwritten(leaves, pool));
    const std::vector<float> written{1.0f, 0.0f};
    SA_EXPECT_OK(write_elements(pool.tensor(leaves.at("targets")), 0,
                                SushiRuntime::span<const float>(written.data(), written.size())));

    SA_EXPECT_FAILURE(Train::require_leaves_written(leaves, pool), StatusCode::InvalidDescription,
                      "'targets'", "element 2", "unwritten");
}

/** @brief Tests that a leaf the pool does not bind is left alone: no feed could have written it. */
TEST_F(Small_LeafWrites, ALeafThePoolDoesNotBindIsSkipped)
{
    declare_unbound("targets");

    SA_EXPECT_OK(Train::mark_leaves_unwritten(leaves, pool));

    SA_EXPECT_OK(Train::require_leaves_written(leaves, pool));
}
