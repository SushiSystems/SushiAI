/****************************************************************/
/* test_arena.cpp                                               */
/* SushiAI - https://github.com/SushiSystems/SushiAI            */
/* Copyright (c) 2026 Sushi Systems                             */
/* Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.    */
/* Commercial use requires a licence from Sushi Systems.        */
/****************************************************************/

// The three device-side claims the memory planner rests on, each checked
// rather than assumed:
//
//  1. Two values packed into one arena report NON-OVERLAPPING ResourceRegions
//     over the SAME allocation base. This is the whole reason packing is safe:
//     SushiRuntime's dependency tracker orders two accesses iff their regions
//     overlap, so a region that still named the bare allocation base - which is
//     what every driver used to pass - would make every tensor in an arena
//     alias every other one and serialise the lot.
//  2. A view addresses its own elements. storage_offset counts elements, so an
//     offset that is not a whole number of them is refused rather than
//     truncated into a tensor that quietly starts in the wrong place.
//  3. Mixed dtypes in one arena each start correctly aligned for their own
//     element width - the one part of the offset arithmetic Tensor cannot do
//     for the planner, because by the time a Tensor exists the offset has
//     already been chosen.

#include <cstddef>
#include <cstdint>
#include <vector>
#include <gtest/gtest.h>
#include "../../common/test_throws.hpp"
#include "../../common/test_utils.hpp"

using namespace SushiAI;

namespace SaTest = SushiAI::Test;
using Graph::ArenaSet;
using Graph::GraphBuilder;
using Graph::Lifetime;
using Graph::MemoryPlan;
using Graph::PlanOptions;
using Graph::ValueId;

namespace
{
    class Integration_Arena : public SaTest::EngineTest {};

    /** @brief x -> a -> b -> c, whose middle values are packed into one arena. */
    struct Chain
    {
        Graph::Graph graph;
        ValueId x = Graph::NO_VALUE;
        ValueId a = Graph::NO_VALUE;
        ValueId b = Graph::NO_VALUE;

        Chain()
        {
            const GraphBuilder builder(graph);
            x = builder.input(Shape{16}, "x");
            a = builder.relu(x);
            b = builder.sigmoid(a);
        }
    };
}

// =========================================================================
// 1. Region precision - the property the whole design depends on
// =========================================================================

TEST_F(Integration_Arena, TwoPackedTensorsShareABaseButNotAByte)
{
    const Chain chain;
    const MemoryPlan plan = MemoryPlan::build(chain.graph);

    ASSERT_TRUE(plan.is_packed(chain.a));
    ASSERT_TRUE(plan.is_packed(chain.b));
    ASSERT_NE(plan.placement(chain.a).offset, plan.placement(chain.b).offset);

    const ArenaSet arenas(*engine, plan);
    ASSERT_EQ(arenas.arena_count(), 1u);

    const SushiBLAS::Tensor first = arenas.view(chain.a, Shape{16}, DType::FLOAT32);
    const SushiBLAS::Tensor second = arenas.view(chain.b, Shape{16}, DType::FLOAT32);

    const SushiRuntime::Core::ResourceRegion left = first.region();
    const SushiRuntime::Core::ResourceRegion right = second.region();

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

TEST_F(Integration_Arena, OpsRecordedOnTwoPackedTensorsDoNotTreadOnEachOther)
{
    // The functional half of the claim above: real SushiBLAS work on two views
    // of one arena, run in one step, each landing only in its own bytes.
    const Chain chain;
    const ArenaSet arenas(*engine, MemoryPlan::build(chain.graph));

    SushiBLAS::Tensor first = arenas.view(chain.a, Shape{16}, DType::FLOAT32);
    SushiBLAS::Tensor second = arenas.view(chain.b, Shape{16}, DType::FLOAT32);

    std::vector<double> ones(16, 1.0);
    std::vector<double> twos(16, 2.0);
    SaTest::write_tensor(first, ones);
    SaTest::write_tensor(second, twos);

    engine->blas().scal(3.0f, first);
    engine->blas().scal(5.0f, second);
    flush();

    for (double value : SaTest::read_tensor(first)) EXPECT_DOUBLE_EQ(value, 3.0);
    for (double value : SaTest::read_tensor(second)) EXPECT_DOUBLE_EQ(value, 10.0);
}

// =========================================================================
// 2. Element-counted offsets
// =========================================================================

TEST_F(Integration_Arena, AViewAddressesItsOwnElements)
{
    const Chain chain;
    const MemoryPlan plan = MemoryPlan::build(chain.graph);
    const ArenaSet arenas(*engine, plan);

    const SushiBLAS::Tensor view = arenas.view(chain.b, Shape{16}, DType::FLOAT32);
    const std::size_t offset = plan.placement(chain.b).offset;

    EXPECT_EQ(view.storage_offset, static_cast<int64_t>(offset / sizeof(float)));
    EXPECT_EQ(static_cast<const char*>(view.data())
                  - static_cast<const char*>(view.storage->allocation_base()),
              static_cast<std::ptrdiff_t>(offset));
}

TEST_F(Integration_Arena, AnOffsetThatIsNotAWholeNumberOfElementsIsRefused)
{
    const auto arena = allocate_arena(*engine, 256);

    // Four bytes into a float64 view: representable in bytes, not in the
    // element count Tensor actually stores. Truncating it would put the tensor
    // somewhere the planner never reserved.
    SA_EXPECT_ERROR((void)view_tensor(*engine, arena, 4, Shape{2}, DType::FLOAT64),
                    "counts elements");

    // The same offset is perfectly fine for float32.
    EXPECT_NO_THROW((void)view_tensor(*engine, arena, 4, Shape{2}, DType::FLOAT32));
}

TEST_F(Integration_Arena, AViewThatWouldRunPastTheArenaIsRefused)
{
    const auto arena = allocate_arena(*engine, 128);
    SA_EXPECT_ERROR((void)view_tensor(*engine, arena, 64, Shape{64}, DType::FLOAT32),
                    "run past the end");
}

// =========================================================================
// 3. Mixed dtypes in one arena
// =========================================================================

TEST_F(Integration_Arena, MixedDtypesEachStartAlignedForTheirOwnElementWidth)
{
    // Alignment turned down to 1 so nothing but the element-width floor is
    // holding these apart. With the default alignment the property is true for
    // an uninteresting reason; here it is true only because the planner rounds.
    Graph::Graph graph;

    const ValueId narrow_leaf =
        graph.add_value(Shape{5}, DType::FLOAT32, Lifetime::INPUT, "x");
    graph.add_node(Ops::LEAF.id, {}, {narrow_leaf});

    const ValueId wide_leaf =
        graph.add_value(Shape{3}, DType::FLOAT64, Lifetime::INPUT, "y");
    graph.add_node(Ops::LEAF.id, {}, {wide_leaf});

    const ValueId narrow =
        graph.add_value(Shape{5}, DType::FLOAT32, Lifetime::ACTIVATION, "a");
    graph.add_node(Ops::RELU.id, {narrow_leaf}, {narrow});

    const ValueId wide =
        graph.add_value(Shape{3}, DType::FLOAT64, Lifetime::ACTIVATION, "b");
    graph.add_node(Ops::RELU.id, {wide_leaf}, {wide});

    PlanOptions options;
    options.alignment = 1;

    const MemoryPlan plan = MemoryPlan::build(graph, options);
    ASSERT_EQ(plan.arenas().size(), 1u) << "both are activations, so both are in one arena";

    const ArenaSet arenas(*engine, plan);

    SushiBLAS::Tensor as_float = arenas.view(narrow, Shape{5}, DType::FLOAT32);
    SushiBLAS::Tensor as_double = arenas.view(wide, Shape{3}, DType::FLOAT64);

    const auto address = [](const SushiBLAS::Tensor& t)
    {
        return reinterpret_cast<std::uintptr_t>(t.data());
    };

    EXPECT_EQ(address(as_float) % dtype_size(DType::FLOAT32), 0u);
    EXPECT_EQ(address(as_double) % dtype_size(DType::FLOAT64), 0u);

    // And the two really are different dtypes in one allocation.
    EXPECT_EQ(as_float.storage->allocation_base(), as_double.storage->allocation_base());
    EXPECT_FALSE(as_float.region().overlaps(as_double.region()));

    // A round trip through each proves the addresses are usable, not merely
    // aligned: a float64 read through a mis-rounded start would come back as
    // two halves of neighbouring values rather than as the number written.
    SaTest::write_tensor(as_float, std::vector<double>{1.0, 2.0, 3.0, 4.0, 5.0});
    SaTest::write_tensor(as_double, std::vector<double>{-1.5, -2.5, -3.5});

    EXPECT_EQ(SaTest::read_tensor(as_float), (std::vector<double>{1.0, 2.0, 3.0, 4.0, 5.0}));
    EXPECT_EQ(SaTest::read_tensor(as_double), (std::vector<double>{-1.5, -2.5, -3.5}));
}

// =========================================================================
// 4. The pool's side of the contract
// =========================================================================

TEST_F(Integration_Arena, AdoptingASecondPlanIsRefused)
{
    const Chain chain;
    Graph::TensorPool pool(*engine);

    pool.adopt(MemoryPlan::build(chain.graph));
    SA_EXPECT_ERROR(pool.adopt(MemoryPlan::build(chain.graph)), "already adopted");
}

TEST_F(Integration_Arena, TheEmptyPlanPacksNothing)
{
    // The control an A/B measurement of the whole pass needs, and the escape
    // hatch a caller who wants the old behaviour uses.
    const Chain chain;
    Graph::TensorPool pool(*engine);

    pool.adopt(MemoryPlan{});
    pool.materialise_leaves(chain.graph);

    SushiBLAS::Tensor& a = pool.materialise(chain.a, chain.graph.value(chain.a));
    SushiBLAS::Tensor& b = pool.materialise(chain.b, chain.graph.value(chain.b));

    EXPECT_TRUE(pool.has_plan());
    EXPECT_EQ(pool.arenas().arena_count(), 0u);
    EXPECT_NE(a.storage->allocation_base(), b.storage->allocation_base());
    EXPECT_EQ(pool.allocated_bytes(), pool.unpacked_bytes());
}

TEST_F(Integration_Arena, APlannedPoolAllocatesOneStorageForAWholeArena)
{
    const Chain chain;
    Graph::TensorPool pool(*engine);

    pool.adopt(MemoryPlan::build(chain.graph));
    pool.materialise_leaves(chain.graph);

    SushiBLAS::Tensor& a = pool.materialise(chain.a, chain.graph.value(chain.a));
    SushiBLAS::Tensor& b = pool.materialise(chain.b, chain.graph.value(chain.b));

    EXPECT_EQ(a.storage->allocation_base(), b.storage->allocation_base())
        << "two packed values must come out of the same allocation";
    EXPECT_EQ(pool.arenas().arena_count(), 1u);

    // No byte comparison here on purpose: these tensors are 64 bytes each and
    // the arena aligns every slot to 128, so packing them *costs* memory. The
    // saving is a property of realistic tensor sizes, and it is measured where
    // that holds - see Regression_StaticMemory.
}
