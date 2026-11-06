/**************************************************************************/
/* test_arena.cpp                                                         */
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
 * @file test_arena.cpp
 * @brief Tests arena packing: region precision, element offsets, dtype alignment, the pool.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstdint>
#include <vector>
#include <gtest/gtest.h>
#include <SushiBLAS/blas/blas.hpp>
#include <SushiRuntime/region/buffer_region.hpp>
#include <SushiAI/engine_fixture/engine_test.hpp>
#include <SushiAI/engine_fixture/test_graph.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/graph/arena.hpp>
#include <SushiAI/graph/builder.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/graph/memory_plan.hpp>
#include <SushiAI/graph/tensor_pool.hpp>
#include <SushiAI/shape/shape.hpp>
#include <SushiAI/tensor_values/tensor_values.hpp>

using namespace SushiAI;
namespace SaTest = SushiAI::Test;
using Graph::ArenaSet;
using Graph::GraphBuilder;
using Graph::Lifetime;
using Graph::MemoryPlan;
using Graph::PlanOptions;
using Graph::ValueID;

namespace
{
    class Medium_Arena : public SaTest::EngineTest {};

    /** @brief Names the suite of the cases that expect the process to stop. */
    using Medium_ArenaDeathTest = Medium_Arena;

    /** @brief x -> a -> b -> c, whose middle values are packed into one arena. */
    struct Chain
    {
        Graph::Graph graph;
        ValueID x = Graph::K_NO_VALUE;
        ValueID a = Graph::K_NO_VALUE;
        ValueID b = Graph::K_NO_VALUE;

        Chain()
        {
            const GraphBuilder builder(graph);
            x = builder.input(Shape{16}, "x");
            // A unary emitter over a fresh input cannot refuse: value() stops the test if it does.
            a = builder.relu(x).value();
            b = builder.sigmoid(a).value();
        }
    };

    /** @brief Returns the address of the first element of @p tensor. */
    std::uintptr_t address(const SushiBLAS::Tensor& tensor)
    {
        return reinterpret_cast<std::uintptr_t>(tensor.data());
    }

    /**
     * @brief Holds a float32 and a float64 activation packed into one arena with alignment 1, so
     * only the element-width floor holds the two starts apart; see tests/core/graph/README.md.
     */
    class MixedArena
    {
        public:
            /** @brief Plans and allocates the arena on @p engine. */
            explicit MixedArena(SushiBLAS::Engine& engine)
                : graph_(make_graph()), arenas_(ArenaSet::make(engine, plan_of(graph_)).value())
            {

            }

            /** @brief Views the float32 activation. */
            SushiBLAS::Tensor float_view() const
            {
                return arenas_.view(K_NARROW, Shape{5}, DType::Float32);
            }

            /** @brief Views the float64 activation. */
            SushiBLAS::Tensor double_view() const
            {
                return arenas_.view(K_WIDE, Shape{3}, DType::Float64);
            }

        private:
            /// @brief Names the float32 activation of the graph make_graph() builds.
            static constexpr ValueID K_NARROW = 2;

            /// @brief Names the float64 activation of the graph make_graph() builds.
            static constexpr ValueID K_WIDE = 3;

            /** @brief Builds two leaves and one relu over each. */
            static Graph::Graph make_graph()
            {
                Graph::Graph graph;
                const ValueID narrow_leaf =
                    graph.add_value(Shape{5}, DType::Float32, Lifetime::Input, "x");
                graph.add_node(Ops::K_LEAF.id, {}, {narrow_leaf});
                const ValueID wide_leaf =
                    graph.add_value(Shape{3}, DType::Float64, Lifetime::Input, "y");
                graph.add_node(Ops::K_LEAF.id, {}, {wide_leaf});
                const ValueID narrow =
                    graph.add_value(Shape{5}, DType::Float32, Lifetime::Activation, "a");
                graph.add_node(Ops::K_RELU.id, {narrow_leaf}, {narrow});
                const ValueID wide =
                    graph.add_value(Shape{3}, DType::Float64, Lifetime::Activation, "b");
                graph.add_node(Ops::K_RELU.id, {wide_leaf}, {wide});
                return graph;
            }

            /** @brief Plans @p graph with alignment 1. */
            static MemoryPlan plan_of(const Graph::Graph& graph)
            {
                PlanOptions options;
                options.alignment = 1;
                return MemoryPlan::build(graph, options).value();
            }

            /// @brief Holds the graph; declared first so the arenas are built over it.
            Graph::Graph graph_;

            /// @brief Holds the arenas of the plan.
            ArenaSet arenas_;
    };
}

/** @brief Tests that two packed values come out of one arena over one allocation. */
TEST_F(Medium_Arena, TwoPackedTensorsShareOneAllocation)
{
    const Chain chain;
    SA_ASSERT_OK_AND_ASSIGN(const MemoryPlan plan, MemoryPlan::build(chain.graph));
    ASSERT_TRUE(plan.is_packed(chain.a));
    ASSERT_TRUE(plan.is_packed(chain.b));

    SA_ASSERT_OK_AND_ASSIGN(const ArenaSet arenas, ArenaSet::make(*engine, plan));
    EXPECT_EQ(arenas.arena_count(), 1u);

    const SushiBLAS::Tensor first = arenas.view(chain.a, Shape{16}, DType::Float32);
    const SushiBLAS::Tensor second = arenas.view(chain.b, Shape{16}, DType::Float32);

    EXPECT_EQ(first.region().base, second.region().base);
    EXPECT_EQ(first.region().base, second.storage->allocation_base());
}

/** @brief Tests that the regions of two packed views share no byte and sit at their offsets. */
TEST_F(Medium_Arena, TwoPackedTensorsReportRegionsThatShareNoByte)
{
    const Chain chain;
    SA_ASSERT_OK_AND_ASSIGN(const MemoryPlan plan, MemoryPlan::build(chain.graph));
    SA_ASSERT_OK_AND_ASSIGN(const ArenaSet arenas, ArenaSet::make(*engine, plan));

    const SushiRuntime::BufferRegion left =
        arenas.view(chain.a, Shape{16}, DType::Float32).region();
    const SushiRuntime::BufferRegion right =
        arenas.view(chain.b, Shape{16}, DType::Float32).region();

    EXPECT_FALSE(left.overlaps(right));
    EXPECT_FALSE(right.overlaps(left));
    EXPECT_EQ(left.length, 16u * sizeof(float));
    EXPECT_EQ(right.length, 16u * sizeof(float));
    EXPECT_EQ(left.offset, plan.placement(chain.a).offset);
    EXPECT_EQ(right.offset, plan.placement(chain.b).offset);
}

/** @brief Tests that work recorded on two views of one arena writes only into its own bytes. */
TEST_F(Medium_Arena, OpsRecordedOnTwoPackedTensorsDoNotTreadOnEachOther)
{
    // Each view writes only its own bytes; see tests/core/graph/README.md.
    const Chain chain;
    SA_ASSERT_OK_AND_ASSIGN(const ArenaSet arenas,
                            ArenaSet::make(*engine, MemoryPlan::build(chain.graph).value()));

    SushiBLAS::Tensor first = arenas.view(chain.a, Shape{16}, DType::Float32);
    SushiBLAS::Tensor second = arenas.view(chain.b, Shape{16}, DType::Float32);

    std::vector<double> ones(16, 1.0);
    std::vector<double> twos(16, 2.0);
    SaTest::write_tensor(first, ones);
    SaTest::write_tensor(second, twos);

    SaTest::TestGraph recording(*engine);
    SushiBLAS::BLASOps blas(recording.recorder());
    blas.scal(3.0f, first);
    blas.scal(5.0f, second);
    SA_EXPECT_OK(recording.run());
    flush();

    EXPECT_EQ(SaTest::read_tensor(first), std::vector<double>(16, 3.0));
    EXPECT_EQ(SaTest::read_tensor(second), std::vector<double>(16, 10.0));
}

/** @brief Tests that a view addresses the elements at its placement offset. */
TEST_F(Medium_Arena, AViewAddressesItsOwnElements)
{
    const Chain chain;
    SA_ASSERT_OK_AND_ASSIGN(const MemoryPlan plan, MemoryPlan::build(chain.graph));
    SA_ASSERT_OK_AND_ASSIGN(const ArenaSet arenas, ArenaSet::make(*engine, plan));

    const SushiBLAS::Tensor view = arenas.view(chain.b, Shape{16}, DType::Float32);
    const std::size_t offset = plan.placement(chain.b).offset;

    EXPECT_EQ(view.storage_offset, static_cast<int64_t>(offset / sizeof(float)));
    EXPECT_EQ(static_cast<const char*>(view.data())
                  - static_cast<const char*>(view.storage->allocation_base()),
              static_cast<std::ptrdiff_t>(offset));
}

/** @brief Tests that a float64 view at an offset of four bytes is refused as fatal. */
TEST_F(Medium_ArenaDeathTest, AnOffsetThatIsNotAWholeNumberOfElementsIsRefused)
{
    SA_ASSERT_OK_AND_ASSIGN(const auto arena, allocate_arena(*engine, 256));

    // A float64 view needs a whole-element offset; see tests/core/graph/README.md.
    SA_EXPECT_FATAL((void)view_tensor(*engine, arena, 4, Shape{2}, DType::Float64),
                    "counts elements");
}

/** @brief Tests that an offset of four bytes is the valid start of a float32 view. */
TEST_F(Medium_Arena, AnOffsetOfOneFloatIsAValidFloat32Start)
{
    SA_ASSERT_OK_AND_ASSIGN(const auto arena, allocate_arena(*engine, 256));

    const SushiBLAS::Tensor view = view_tensor(*engine, arena, 4, Shape{2}, DType::Float32);

    EXPECT_EQ(view.storage_offset, int64_t{1});
}

/** @brief Tests that a view that would run past the end of its arena is fatal. */
TEST_F(Medium_ArenaDeathTest, AViewThatWouldRunPastTheArenaIsRefused)
{
    SA_ASSERT_OK_AND_ASSIGN(const auto arena, allocate_arena(*engine, 128));
    SA_EXPECT_FATAL((void)view_tensor(*engine, arena, 64, Shape{64}, DType::Float32),
                    "run past the end");
}

/** @brief Tests that the views of two dtypes in one arena start aligned for their own width. */
TEST_F(Medium_Arena, MixedDtypesEachStartAlignedForTheirOwnElementWidth)
{
    const MixedArena mixed(*engine);

    EXPECT_EQ(address(mixed.float_view()) % dtype_size(DType::Float32), 0u);
    EXPECT_EQ(address(mixed.double_view()) % dtype_size(DType::Float64), 0u);
}

/** @brief Tests that two dtypes in one arena share one allocation and no byte. */
TEST_F(Medium_Arena, MixedDtypesShareOneAllocationAndNoByte)
{
    const MixedArena mixed(*engine);
    const SushiBLAS::Tensor as_float = mixed.float_view();
    const SushiBLAS::Tensor as_double = mixed.double_view();

    EXPECT_EQ(as_float.storage->allocation_base(), as_double.storage->allocation_base());
    EXPECT_FALSE(as_float.region().overlaps(as_double.region()));
}

/** @brief Tests that bytes written through each mixed-dtype view read back unchanged. */
TEST_F(Medium_Arena, MixedDtypesRoundTripTheirBytes)
{
    const MixedArena mixed(*engine);
    SushiBLAS::Tensor as_float = mixed.float_view();
    SushiBLAS::Tensor as_double = mixed.double_view();

    SaTest::write_tensor(as_float, std::vector<double>{1.0, 2.0, 3.0, 4.0, 5.0});
    SaTest::write_tensor(as_double, std::vector<double>{-1.5, -2.5, -3.5});

    EXPECT_EQ(SaTest::read_tensor(as_float), (std::vector<double>{1.0, 2.0, 3.0, 4.0, 5.0}));
    EXPECT_EQ(SaTest::read_tensor(as_double), (std::vector<double>{-1.5, -2.5, -3.5}));
}

/** @brief Tests that a pool that adopted a plan refuses a second one as fatal. */
TEST_F(Medium_ArenaDeathTest, AdoptingASecondPlanIsRefused)
{
    const Chain chain;
    Graph::TensorPool pool(*engine);

    SA_EXPECT_OK(pool.adopt(MemoryPlan::build(chain.graph).value()));
    SA_EXPECT_FATAL((void)pool.adopt(MemoryPlan::build(chain.graph).value()), "already adopted");
}

/** @brief Tests that an empty plan packs nothing and allocates every value alone. */
TEST_F(Medium_Arena, TheEmptyPlanPacksNothing)
{
    // The empty plan is the unpacked control; see tests/core/graph/README.md.
    const Chain chain;
    Graph::TensorPool pool(*engine);

    SA_EXPECT_OK(pool.adopt(MemoryPlan{}));
    SA_EXPECT_OK(pool.materialise_leaves(chain.graph));

    SA_ASSERT_OK_AND_ASSIGN(SushiBLAS::Tensor& a,
                            pool.materialise(chain.a, chain.graph.value(chain.a)));
    SA_ASSERT_OK_AND_ASSIGN(SushiBLAS::Tensor& b,
                            pool.materialise(chain.b, chain.graph.value(chain.b)));

    EXPECT_TRUE(pool.has_plan());
    EXPECT_EQ(pool.arenas().arena_count(), 0u);
    EXPECT_NE(a.storage->allocation_base(), b.storage->allocation_base());
    EXPECT_EQ(pool.allocated_bytes(), pool.unpacked_bytes());
}

/** @brief Tests that a planned pool allocates one storage for the values of one arena. */
TEST_F(Medium_Arena, APlannedPoolAllocatesOneStorageForAWholeArena)
{
    const Chain chain;
    Graph::TensorPool pool(*engine);

    SA_EXPECT_OK(pool.adopt(MemoryPlan::build(chain.graph).value()));
    SA_EXPECT_OK(pool.materialise_leaves(chain.graph));

    SA_ASSERT_OK_AND_ASSIGN(SushiBLAS::Tensor& a,
                            pool.materialise(chain.a, chain.graph.value(chain.a)));
    SA_ASSERT_OK_AND_ASSIGN(SushiBLAS::Tensor& b,
                            pool.materialise(chain.b, chain.graph.value(chain.b)));

    EXPECT_EQ(a.storage->allocation_base(), b.storage->allocation_base())
        << "two packed values must come out of the same allocation";
    EXPECT_EQ(pool.arenas().arena_count(), 1u);

    // Bytes are not compared at this tensor size; see tests/core/graph/README.md.
}
