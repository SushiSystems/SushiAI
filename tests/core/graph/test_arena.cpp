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
 * @file test_arena.cpp
 * @brief Tests arena packing: region precision, element offsets, dtype alignment, the pool.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstdint>
#include <vector>
#include <gtest/gtest.h>
#include <SushiBLAS/blas/blas.hpp>
#include <SushiAI/SushiAI.hpp>
#include <SushiAI/engine_fixture/engine_test.hpp>
#include <SushiAI/engine_fixture/test_graph.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
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
            a = builder.relu(x);
            b = builder.sigmoid(a);
        }
    };
}

TEST_F(Medium_Arena, TwoPackedTensorsShareABaseButNotAByte)
{
    const Chain chain;
    const MemoryPlan plan = MemoryPlan::build(chain.graph);

    ASSERT_TRUE(plan.is_packed(chain.a));
    ASSERT_TRUE(plan.is_packed(chain.b));
    ASSERT_NE(plan.placement(chain.a).offset, plan.placement(chain.b).offset);

    SA_ASSERT_OK_AND_ASSIGN(const ArenaSet arenas, ArenaSet::make(*engine, plan));
    ASSERT_EQ(arenas.arena_count(), 1u);

    const SushiBLAS::Tensor first = arenas.view(chain.a, Shape{16}, DType::Float32);
    const SushiBLAS::Tensor second = arenas.view(chain.b, Shape{16}, DType::Float32);

    const SushiRuntime::BufferRegion left = first.region();
    const SushiRuntime::BufferRegion right = second.region();

    // One allocation - if these differed the test would be proving nothing.
    EXPECT_EQ(left.base, right.base);
    EXPECT_EQ(left.base, second.storage->allocation_base());

    // ...and yet no shared byte, in either direction.
    EXPECT_FALSE(left.overlaps(right));
    EXPECT_FALSE(right.overlaps(left));

    EXPECT_EQ(left.length, 16u * sizeof(float));
    EXPECT_EQ(right.length, 16u * sizeof(float));
    EXPECT_EQ(left.offset, plan.placement(chain.a).offset);
    EXPECT_EQ(right.offset, plan.placement(chain.b).offset);
}

TEST_F(Medium_Arena, OpsRecordedOnTwoPackedTensorsDoNotTreadOnEachOther)
{
    // Each view writes only its own bytes; see tests/core/graph/README.md.
    const Chain chain;
    SA_ASSERT_OK_AND_ASSIGN(const ArenaSet arenas,
                            ArenaSet::make(*engine, MemoryPlan::build(chain.graph)));

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

    for (double value : SaTest::read_tensor(first)) EXPECT_DOUBLE_EQ(value, 3.0);
    for (double value : SaTest::read_tensor(second)) EXPECT_DOUBLE_EQ(value, 10.0);
}

TEST_F(Medium_Arena, AViewAddressesItsOwnElements)
{
    const Chain chain;
    const MemoryPlan plan = MemoryPlan::build(chain.graph);
    SA_ASSERT_OK_AND_ASSIGN(const ArenaSet arenas, ArenaSet::make(*engine, plan));

    const SushiBLAS::Tensor view = arenas.view(chain.b, Shape{16}, DType::Float32);
    const std::size_t offset = plan.placement(chain.b).offset;

    EXPECT_EQ(view.storage_offset, static_cast<int64_t>(offset / sizeof(float)));
    EXPECT_EQ(static_cast<const char*>(view.data())
                  - static_cast<const char*>(view.storage->allocation_base()),
              static_cast<std::ptrdiff_t>(offset));
}

TEST_F(Medium_ArenaDeathTest, AnOffsetThatIsNotAWholeNumberOfElementsIsRefused)
{
    SA_ASSERT_OK_AND_ASSIGN(const auto arena, allocate_arena(*engine, 256));

    // A float64 view needs a whole-element offset; see tests/core/graph/README.md.
    SA_EXPECT_FATAL((void)view_tensor(*engine, arena, 4, Shape{2}, DType::Float64),
                    "counts elements");

    // Four bytes are one float32 element, so the same offset is a valid float32 start.
    const SushiBLAS::Tensor view = view_tensor(*engine, arena, 4, Shape{2}, DType::Float32);
    EXPECT_EQ(view.storage_offset, int64_t{1});
}

TEST_F(Medium_ArenaDeathTest, AViewThatWouldRunPastTheArenaIsRefused)
{
    SA_ASSERT_OK_AND_ASSIGN(const auto arena, allocate_arena(*engine, 128));
    SA_EXPECT_FATAL((void)view_tensor(*engine, arena, 64, Shape{64}, DType::Float32),
                    "run past the end");
}

TEST_F(Medium_Arena, MixedDtypesEachStartAlignedForTheirOwnElementWidth)
{
    // Alignment 1 leaves only the element-width floor; see tests/core/graph/README.md.
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

    PlanOptions options;
    options.alignment = 1;

    const MemoryPlan plan = MemoryPlan::build(graph, options);
    ASSERT_EQ(plan.arenas().size(), 1u) << "both are activations, so both are in one arena";

    SA_ASSERT_OK_AND_ASSIGN(const ArenaSet arenas, ArenaSet::make(*engine, plan));

    SushiBLAS::Tensor as_float = arenas.view(narrow, Shape{5}, DType::Float32);
    SushiBLAS::Tensor as_double = arenas.view(wide, Shape{3}, DType::Float64);

    const auto address = [](const SushiBLAS::Tensor& t)
    {
        return reinterpret_cast<std::uintptr_t>(t.data());
    };

    EXPECT_EQ(address(as_float) % dtype_size(DType::Float32), 0u);
    EXPECT_EQ(address(as_double) % dtype_size(DType::Float64), 0u);

    // And the two really are different dtypes in one allocation.
    EXPECT_EQ(as_float.storage->allocation_base(), as_double.storage->allocation_base());
    EXPECT_FALSE(as_float.region().overlaps(as_double.region()));

    // A round trip proves each address is usable; see tests/core/graph/README.md.
    SaTest::write_tensor(as_float, std::vector<double>{1.0, 2.0, 3.0, 4.0, 5.0});
    SaTest::write_tensor(as_double, std::vector<double>{-1.5, -2.5, -3.5});

    EXPECT_EQ(SaTest::read_tensor(as_float), (std::vector<double>{1.0, 2.0, 3.0, 4.0, 5.0}));
    EXPECT_EQ(SaTest::read_tensor(as_double), (std::vector<double>{-1.5, -2.5, -3.5}));
}

TEST_F(Medium_ArenaDeathTest, AdoptingASecondPlanIsRefused)
{
    const Chain chain;
    Graph::TensorPool pool(*engine);

    SA_EXPECT_OK(pool.adopt(MemoryPlan::build(chain.graph)));
    SA_EXPECT_FATAL((void)pool.adopt(MemoryPlan::build(chain.graph)), "already adopted");
}

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

TEST_F(Medium_Arena, APlannedPoolAllocatesOneStorageForAWholeArena)
{
    const Chain chain;
    Graph::TensorPool pool(*engine);

    SA_EXPECT_OK(pool.adopt(MemoryPlan::build(chain.graph)));
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
