/**************************************************************************/
/* test_broadcast.cpp                                            */
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
 * @file test_broadcast.cpp
 * @brief Tests the backward rules of the four broadcast operations and the sum along against
 *        central differences in FLOAT64.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>
#include <gtest/gtest.h>
#include <SushiAI/dtype/dtype.hpp>
#include <SushiAI/engine_fixture/engine_test.hpp>
#include <SushiAI/graph/builder.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/seeded_noise/host_noise.hpp>
#include <SushiAI/shape/shape.hpp>
#include <support/gradient_probe.hpp>
#include <support/test_names.hpp>
#include "rule_gradient_support.hpp"

using namespace SushiAI;
// SaTest stays reachable inside a TEST_F body; see tests/core/autograd/README.md.
namespace SaTest = SushiAI::Test;
using Graph::GraphBuilder;
using Graph::ValueID;
using SaTest::GradientProbe;
using SaTest::expect_gradient_matches;

namespace
{
    /** @brief The four broadcast operations. */
    enum class BroadcastKind : std::uint8_t
    {
        Add,
        Sub,
        Mul,
        Div,
    };

    /** @brief Names one broadcast on one side and rank, differentiated at one operand. */
    struct BroadcastCase
    {
        std::string name;          /**< The case name. */
        BroadcastKind kind;        /**< The operation. */
        Graph::BroadcastSide side; /**< The side the vector sits on. */
        int rank;                  /**< The rank of the tensor: 2 or 3. */
        bool at_vector;            /**< True to differentiate at the vector, false at the tensor. */
    };

    /** @brief Runs the broadcast case once per row. */
    class Medium_AutogradBroadcast : public SaTest::EngineTest,
                                     public ::testing::WithParamInterface<BroadcastCase>
    {
    };

    /** @brief Names the side and the rank of one sum along. */
    struct SumAlongCase
    {
        const char* name;          /**< The case name. */
        Graph::BroadcastSide side; /**< The side the kept shape sits on. */
        int rank;                  /**< The rank of the tensor: 2 or 3. */
    };

    /** @brief Runs the sum-along case once per row. */
    class Medium_AutogradSumAlong : public SaTest::EngineTest,
                                    public ::testing::WithParamInterface<SumAlongCase>
    {
    };

    /** @brief Returns the shape of a tensor of the given rank: [4, 3] or [2, 4, 3]. */
    Shape tensor_shape_of(int rank)
    {
        return rank == 2 ? Shape{4, 3} : Shape{2, 4, 3};
    }

    /** @brief Returns the shape of a vector at @p side of the tensor of @p rank, one axis short. */
    Shape vector_shape_of(int rank, Graph::BroadcastSide side)
    {
        if (rank == 2)
            return side == Graph::BroadcastSide::Leading ? Shape{4} : Shape{3};

        return side == Graph::BroadcastSide::Leading ? Shape{2, 4} : Shape{4, 3};
    }

    /** @brief Emits the broadcast operation @p kind of @p x and @p vector at @p side. */
    Graph::Traced emit_broadcast(const GraphBuilder& builder, BroadcastKind kind, ValueID x,
                                 ValueID vector, Graph::BroadcastSide side)
    {
        switch (kind)
        {
            case BroadcastKind::Add: return builder.broadcast_add(x, vector, side);
            case BroadcastKind::Sub: return builder.broadcast_sub(x, vector, side);
            case BroadcastKind::Mul: return builder.broadcast_mul(x, vector, side);
            case BroadcastKind::Div: return builder.broadcast_div(x, vector, side);
        }

        return builder.broadcast_add(x, vector, side);
    }

    /** @brief Lists every operation on both sides at both ranks, at the tensor and the vector. */
    std::vector<BroadcastCase> broadcast_cases()
    {
        const std::pair<const char*, BroadcastKind> kinds[] = {
            {"Add", BroadcastKind::Add}, {"Sub", BroadcastKind::Sub},
            {"Mul", BroadcastKind::Mul}, {"Div", BroadcastKind::Div}};
        const std::pair<const char*, Graph::BroadcastSide> sides[] = {
            {"Leading", Graph::BroadcastSide::Leading},
            {"Trailing", Graph::BroadcastSide::Trailing}};

        std::vector<BroadcastCase> cases;
        for (const auto& kind : kinds)
        {
            for (const auto& side : sides)
            {
                for (const int rank : {2, 3})
                {
                    for (const bool at_vector : {false, true})
                    {
                        cases.push_back(BroadcastCase{
                            std::string(kind.first) + side.first + "Rank" + std::to_string(rank)
                                + (at_vector ? "Vector" : "Tensor"),
                            kind.second, side.second, rank, at_vector});
                    }
                }
            }
        }
        return cases;
    }

    /**
     * @brief Returns the values of the leaf that is not differentiated in @p row.
     *
     * A divisor stays away from zero, so the vector holds positive values where it is the divisor;
     * every other leaf is noise.
     */
    std::vector<double> broadcast_other_values(const BroadcastCase& row, std::size_t count)
    {
        const bool is_divisor = row.kind == BroadcastKind::Div && !row.at_vector;
        if (is_divisor)
            return SaTest::positive_point(count, 4);

        std::vector<double> values(count);
        SaTest::HostNoise(721).fill(values, 1.0);
        return values;
    }

    /** @brief Returns the probe point of @p row: positive where the variable is the divisor. */
    std::vector<double> broadcast_point(const BroadcastCase& row, std::size_t count)
    {
        if (row.kind == BroadcastKind::Div && row.at_vector)
            return SaTest::positive_point(count, 5);

        std::vector<double> values(count);
        SaTest::HostNoise(724).fill(values, 1.0);
        return values;
    }
} // namespace

/** @brief Tests that each broadcast differentiates at its tensor and at its vector. */
TEST_P(Medium_AutogradBroadcast, TheBroadcastDifferentiatesAtEachOperand)
{
    const BroadcastCase& row = GetParam();
    const Shape x_shape = tensor_shape_of(row.rank);
    const Shape v_shape = vector_shape_of(row.rank, row.side);
    const Shape variable_shape = row.at_vector ? v_shape : x_shape;
    const int64_t result_rows = x_shape.num_elements() / 3;

    ValueID other = Graph::K_NO_VALUE;

    GradientProbe probe(backend(), variable_shape, [&](const GraphBuilder& b, ValueID variable)
    {
        other = b.input(row.at_vector ? x_shape : v_shape, "other", DType::Float64);
        const ValueID x = row.at_vector ? other : variable;
        const ValueID v = row.at_vector ? variable : other;
        return b.reshape(emit_broadcast(b, row.kind, x, v, row.side), Shape{result_rows, 3});
    });

    const Shape other_shape = row.at_vector ? x_shape : v_shape;
    probe.fill(other, broadcast_other_values(row, static_cast<std::size_t>(
                                                      other_shape.num_elements())));
    probe.fill_noise(probe.targets(), 723, 1.0);

    std::vector<double> point = broadcast_point(row, probe.size());

    expect_gradient_matches(probe, point);
}

INSTANTIATE_TEST_SUITE_P(Cases, Medium_AutogradBroadcast, ::testing::ValuesIn(broadcast_cases()),
                         [](const ::testing::TestParamInfo<BroadcastCase>& info)
                         {
                             return info.param.name;
                         });

/** @brief Tests that the sum along differentiates at each side and rank. */
TEST_P(Medium_AutogradSumAlong, TheSumAlongDifferentiatesAtEachSideAndRank)
{
    const SumAlongCase& row = GetParam();
    const Shape x_shape = tensor_shape_of(row.rank);
    const Shape kept = vector_shape_of(row.rank, row.side);
    const int64_t kept_elements = kept.num_elements();

    GradientProbe probe(backend(), x_shape, [&](const GraphBuilder& b, ValueID x)
    {
        return b.reshape(b.sum_along(x, kept, row.side), Shape{kept_elements, 1});
    });

    probe.fill_noise(probe.targets(), 731, 1.0);

    std::vector<double> point(probe.size());
    SaTest::HostNoise(732).fill(point, 1.0);

    expect_gradient_matches(probe, point);
}

INSTANTIATE_TEST_SUITE_P(
    Cases, Medium_AutogradSumAlong,
    ::testing::Values(SumAlongCase{"LeadingRank2", Graph::BroadcastSide::Leading, 2},
                      SumAlongCase{"TrailingRank2", Graph::BroadcastSide::Trailing, 2},
                      SumAlongCase{"LeadingRank3", Graph::BroadcastSide::Leading, 3},
                      SumAlongCase{"TrailingRank3", Graph::BroadcastSide::Trailing, 3}),
    [](const ::testing::TestParamInfo<SumAlongCase>& info)
    {
        return ::SushiAI::Test::parameter_name(info.param.name);
    });
