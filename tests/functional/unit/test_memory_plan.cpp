/**************************************************************************/
/* test_memory_plan.cpp                                                   */
/**************************************************************************/
/*                          This file is part of:                         */
/*                                 SushiAI                                */
/**************************************************************************/

// Device-free: the memory planner is a pure function of the IR, which is the
// whole reason the IR is a host-lane structure. No engine, no allocator, no
// SYCL - every case below is a synthetic graph and a table of offsets.
//
// The contract being pinned is one sentence: **two values whose live ranges
// intersect never share a byte, and two whose live ranges do not are free to.**
// The first half is checked exhaustively over every pair, against a liveness
// computed here by an obviously-correct breadth-first search rather than by
// asking the planner what it thinks - a planner checked against its own
// reachability code would agree with itself no matter what it did.
//
// The second half is what the individual cases below are for, and the diamond
// is the one that matters: it is the shape where a planner that overlaps
// disjoint [first_use, last_use] intervals on a linearised schedule gets a
// *wrong* answer that no test written from the linear model would catch.

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <set>
#include <vector>
#include <gtest/gtest.h>
#include <SushiAI/graph/builder.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/graph/memory_plan.hpp>

using namespace SushiAI;
using Graph::GraphBuilder;
using Graph::Lifetime;
using Graph::MemoryPlan;
using Graph::NO_ARENA;
using Graph::NO_NODE;
using Graph::NodeId;
using Graph::PlanOptions;
using Graph::ValueId;

namespace
{
    /** @brief 128 bytes of float32 - exactly one alignment unit, so offsets read cleanly. */
    Shape unit_shape() { return Shape{32}; }

    /**
     * @brief Every node reachable from @p from, by a plain breadth-first walk.
     *
     * Deliberately not the planner's bitset closure: this is the independent
     * second opinion the contract check is measured against.
     */
    std::set<NodeId> descendants(const Graph::Graph& graph, NodeId from)
    {
        std::vector<std::vector<NodeId>> successors(graph.node_count());
        for (NodeId id = 0; id < static_cast<NodeId>(graph.node_count()); ++id)
            for (ValueId in : graph.node(id).inputs)
                if (graph.value(in).producer != NO_NODE)
                    successors[graph.value(in).producer].push_back(id);

        std::set<NodeId> seen;
        std::vector<NodeId> pending{from};

        while (!pending.empty())
        {
            const NodeId here = pending.back();
            pending.pop_back();
            for (NodeId next : successors[here])
                if (seen.insert(next).second) pending.push_back(next);
        }

        return seen;
    }

    /** @brief Every node that writes or reads @p id. */
    std::vector<NodeId> uses_of(const Graph::Graph& graph, ValueId id)
    {
        std::vector<NodeId> uses;
        if (graph.value(id).producer != NO_NODE) uses.push_back(graph.value(id).producer);

        for (NodeId node = 0; node < static_cast<NodeId>(graph.node_count()); ++node)
            for (ValueId in : graph.node(node).inputs)
                if (in == id) uses.push_back(node);

        return uses;
    }

    /** @brief Whether every use of @p earlier is a strict ancestor of @p later's producer. */
    bool finished_before(const Graph::Graph& graph, ValueId earlier, ValueId later)
    {
        const NodeId write = graph.value(later).producer;
        if (write == NO_NODE) return false;

        for (NodeId use : uses_of(graph, earlier))
            if (descendants(graph, use).count(write) == 0) return false;

        return true;
    }

    /** @brief Two values are live together unless one is finished before the other starts. */
    bool live_together(const Graph::Graph& graph, ValueId a, ValueId b)
    {
        return !finished_before(graph, a, b) && !finished_before(graph, b, a);
    }

    /**
     * @brief The planner's actual contract, checked over every pair of values.
     *
     * Exhaustive rather than sampled, because it is only quadratic in a handful
     * of values and a packing bug is exactly the kind that hides in the one
     * pair nobody thought to name.
     */
    void expect_no_live_overlap(const Graph::Graph& graph, const MemoryPlan& plan)
    {
        for (ValueId a = 0; a < static_cast<ValueId>(graph.value_count()); ++a)
            for (ValueId b = a + 1; b < static_cast<ValueId>(graph.value_count()); ++b)
            {
                if (!plan.is_packed(a) || !plan.is_packed(b)) continue;
                if (plan.placement(a).arena != plan.placement(b).arena) continue;
                if (plan.placement(a).buffer == plan.placement(b).buffer) continue;
                if (!live_together(graph, a, b)) continue;

                const auto& pa = plan.placement(a);
                const auto& pb = plan.placement(b);

                EXPECT_TRUE(pa.offset + pa.bytes <= pb.offset || pb.offset + pb.bytes <= pa.offset)
                    << "values " << a << " and " << b << " are live at the same time but share "
                    << "bytes: [" << pa.offset << ", " << pa.offset + pa.bytes << ") against ["
                    << pb.offset << ", " << pb.offset + pb.bytes << ")";
            }
    }
}

// =========================================================================
// 1. Degenerate graphs
// =========================================================================

TEST(Unit_MemoryPlan, AnEmptyGraphAsksForNothing)
{
    const Graph::Graph graph;
    const MemoryPlan plan = MemoryPlan::build(graph);

    EXPECT_EQ(plan.value_count(), 0u);
    EXPECT_TRUE(plan.arenas().empty());
    EXPECT_EQ(plan.arena_bytes(), 0u);
}

TEST(Unit_MemoryPlan, ASingleActivationGetsOneArenaAndOffsetZero)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);

    const ValueId x = builder.input(unit_shape(), "x");
    const ValueId y = builder.relu(x);

    const MemoryPlan plan = MemoryPlan::build(graph);

    ASSERT_EQ(plan.arenas().size(), 1u);
    EXPECT_EQ(plan.arenas()[0].lifetime, Lifetime::ACTIVATION);
    EXPECT_EQ(plan.arenas()[0].buffers, 1u);
    EXPECT_EQ(plan.arena_bytes(), 128u);

    EXPECT_FALSE(plan.is_packed(x)) << "an input leaf is written from outside the step";
    ASSERT_TRUE(plan.is_packed(y));
    EXPECT_EQ(plan.placement(y).offset, 0u);
    EXPECT_EQ(plan.placement(y).bytes, 128u);
}

TEST(Unit_MemoryPlan, AGraphOfOnlyLeavesPacksNothing)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);

    builder.input(unit_shape(), "x");
    builder.parameter(unit_shape(), "w");

    const MemoryPlan plan = MemoryPlan::build(graph);

    EXPECT_TRUE(plan.arenas().empty());
    EXPECT_EQ(plan.arena_bytes(), 0u);
}

// =========================================================================
// 2. The two shapes that decide whether the pass is right
// =========================================================================

TEST(Unit_MemoryPlan, AChainPingPongsBetweenTwoBuffers)
{
    // x -> a -> b -> c -> d. Nothing branches, so `a` is finished before `c`
    // starts and `b` before `d`: four activations need exactly two buffers,
    // which is the most any planner can do with a chain.
    Graph::Graph graph;
    const GraphBuilder builder(graph);

    const ValueId x = builder.input(unit_shape(), "x");
    const ValueId a = builder.relu(x);
    const ValueId b = builder.relu(a);
    const ValueId c = builder.relu(b);
    const ValueId d = builder.relu(c);

    const MemoryPlan plan = MemoryPlan::build(graph);

    EXPECT_TRUE(plan.reuse_enabled());
    EXPECT_EQ(plan.arena_bytes(), 256u) << "four 128-byte activations should fit in two slots";
    EXPECT_EQ(plan.unpacked_bytes(), 512u);

    EXPECT_EQ(plan.placement(a).offset, plan.placement(c).offset);
    EXPECT_EQ(plan.placement(b).offset, plan.placement(d).offset);
    EXPECT_NE(plan.placement(a).offset, plan.placement(b).offset);

    expect_no_live_overlap(graph, plan);
}

TEST(Unit_MemoryPlan, TwoUnorderedBranchesNeverShareBytes)
{
    // THE case. Linearised, the two branches have disjoint [first, last]
    // intervals - a1 dies at node 2, b1 is born at node 3 - so an interval
    // planner overlaps them and is wrong: nothing in the graph orders node 2
    // before node 3, the runtime may run the branches concurrently, and the
    // sharing would both corrupt the result and serialise the parallelism it
    // was meant to be free of.
    Graph::Graph graph;
    const GraphBuilder builder(graph);

    const ValueId x = builder.input(unit_shape(), "x");

    const ValueId a1 = builder.relu(x);
    const ValueId a2 = builder.relu(a1);

    const ValueId b1 = builder.sigmoid(x);
    const ValueId b2 = builder.sigmoid(b1);

    const ValueId out = builder.add(a2, b2);

    const MemoryPlan plan = MemoryPlan::build(graph);

    ASSERT_TRUE(plan.is_packed(a1));
    ASSERT_TRUE(plan.is_packed(b1));

    const auto& left = plan.placement(a1);
    const auto& right = plan.placement(b1);

    EXPECT_TRUE(left.offset + left.bytes <= right.offset
                || right.offset + right.bytes <= left.offset)
        << "the two branches of a diamond are unordered, so a1 and b1 must not share bytes; "
        << "a1 at " << left.offset << ", b1 at " << right.offset;

    EXPECT_TRUE(plan.is_packed(out));
    expect_no_live_overlap(graph, plan);
}

// =========================================================================
// 3. Lifetimes
// =========================================================================

TEST(Unit_MemoryPlan, OnlyTheNonPersistentClassesArePacked)
{
    // All six lifetimes in one graph. The four persistent ones live for the
    // whole run, so packing them would save nothing while merging exactly the
    // dependency-tracker shards that matter most; only ACTIVATION and SCRATCH
    // reach an arena, and they reach different ones.
    Graph::Graph graph;

    const ValueId parameter = graph.add_value(unit_shape(), DType::FLOAT32, Lifetime::PARAMETER, "w");
    graph.add_node(Ops::LEAF.id, {}, {parameter});

    const ValueId input = graph.add_value(unit_shape(), DType::FLOAT32, Lifetime::INPUT, "x");
    graph.add_node(Ops::LEAF.id, {}, {input});

    const ValueId activation = graph.add_value(unit_shape(), DType::FLOAT32, Lifetime::ACTIVATION, "h");
    graph.add_node(Ops::RELU.id, {input}, {activation});

    const ValueId scratch = graph.add_value(unit_shape(), DType::FLOAT32, Lifetime::SCRATCH, "t");
    graph.add_node(Ops::RELU.id, {activation}, {scratch});

    const ValueId gradient = graph.add_value(unit_shape(), DType::FLOAT32, Lifetime::GRADIENT, "dw");
    graph.add_node(Ops::RELU.id, {scratch}, {gradient});

    const ValueId state = graph.add_value(unit_shape(), DType::FLOAT32, Lifetime::OPTIMIZER_STATE, "m");
    graph.add_node(Ops::RELU.id, {gradient}, {state});

    const MemoryPlan plan = MemoryPlan::build(graph);

    EXPECT_FALSE(plan.is_packed(parameter));
    EXPECT_FALSE(plan.is_packed(input));
    EXPECT_FALSE(plan.is_packed(gradient));
    EXPECT_FALSE(plan.is_packed(state));

    ASSERT_TRUE(plan.is_packed(activation));
    ASSERT_TRUE(plan.is_packed(scratch));
    EXPECT_NE(plan.placement(activation).arena, plan.placement(scratch).arena)
        << "one arena per lifetime class, so the tracker keeps a shard per class";

    ASSERT_EQ(plan.arenas().size(), 2u);
    EXPECT_EQ(plan.arenas()[0].lifetime, Lifetime::ACTIVATION);
    EXPECT_EQ(plan.arenas()[1].lifetime, Lifetime::SCRATCH);
}

TEST(Unit_MemoryPlan, AValueNothingWritesKeepsItsOwnAllocation)
{
    // An ACTIVATION with no producer is written from outside the step and never
    // rewritten, so its bytes are live for the whole run whatever its class
    // says. Handing them away would survive exactly one replay.
    Graph::Graph graph;

    const ValueId orphan = graph.add_value(unit_shape(), DType::FLOAT32, Lifetime::ACTIVATION, "?");
    const ValueId out = graph.add_value(unit_shape(), DType::FLOAT32, Lifetime::ACTIVATION, "y");
    graph.add_node(Ops::RELU.id, {orphan}, {out});

    const MemoryPlan plan = MemoryPlan::build(graph);

    EXPECT_FALSE(plan.is_packed(orphan));
    EXPECT_TRUE(plan.is_packed(out));
}

// =========================================================================
// 4. Aliasing
// =========================================================================

TEST(Unit_MemoryPlan, AnAccumulateChainIsOneBufferWithOneLiveRange)
{
    // ACCUMULATE's result IS its accumulator's storage. Planning the two
    // separately would be silently wrong rather than merely wasteful: the
    // planner would think the accumulator died at the ACCUMULATE node and hand
    // its bytes to `sink`, which is produced by the node still reading them.
    Graph::Graph graph;
    const GraphBuilder builder(graph);

    const ValueId x = builder.input(unit_shape(), "x");
    const ValueId first = builder.relu(x);
    const ValueId second = builder.sigmoid(x);
    const ValueId accumulated = builder.accumulate(first, second);
    const ValueId sink = builder.relu(accumulated);

    const MemoryPlan plan = MemoryPlan::build(graph);

    EXPECT_EQ(plan.buffer_of(accumulated), plan.buffer_of(first));
    EXPECT_EQ(plan.placement(accumulated).offset, plan.placement(first).offset);
    EXPECT_EQ(plan.placement(accumulated).arena, plan.placement(first).arena);

    EXPECT_NE(plan.placement(sink).offset, plan.placement(first).offset)
        << "sink is produced by the node that last reads the accumulator's bytes";

    // The alias class counts once, not twice.
    ASSERT_EQ(plan.arenas().size(), 1u);
    EXPECT_EQ(plan.arenas()[0].buffers, 3u);
}

// =========================================================================
// 5. Mixed dtypes
// =========================================================================

TEST(Unit_MemoryPlan, EveryStartIsAWholeNumberOfItsOwnElements)
{
    // Alignment turned down to 1 so the element-width floor is the only thing
    // left doing the work. A 20-byte float32 buffer at offset 0 leaves the next
    // free byte at 20, which is not a whole number of float64 elements; the
    // float64 buffer therefore has to start at 24, or its storage_offset would
    // not be representable at all.
    Graph::Graph graph;

    const ValueId narrow_leaf = graph.add_value(Shape{5}, DType::FLOAT32, Lifetime::INPUT, "x");
    graph.add_node(Ops::LEAF.id, {}, {narrow_leaf});

    const ValueId wide_leaf = graph.add_value(Shape{2}, DType::FLOAT64, Lifetime::INPUT, "y");
    graph.add_node(Ops::LEAF.id, {}, {wide_leaf});

    const ValueId narrow = graph.add_value(Shape{5}, DType::FLOAT32, Lifetime::ACTIVATION, "a");
    graph.add_node(Ops::RELU.id, {narrow_leaf}, {narrow});

    const ValueId wide = graph.add_value(Shape{2}, DType::FLOAT64, Lifetime::ACTIVATION, "b");
    graph.add_node(Ops::RELU.id, {wide_leaf}, {wide});

    PlanOptions options;
    options.alignment = 1;

    const MemoryPlan plan = MemoryPlan::build(graph, options);

    EXPECT_EQ(plan.placement(narrow).offset % dtype_size(DType::FLOAT32), 0u);
    EXPECT_EQ(plan.placement(wide).offset % dtype_size(DType::FLOAT64), 0u);

    EXPECT_EQ(plan.placement(narrow).offset, 0u);
    EXPECT_EQ(plan.placement(wide).offset, 24u)
        << "20 bytes of float32 rounded up to the next whole float64";
}

// =========================================================================
// 6. Knobs, refusals and malformed input
// =========================================================================

TEST(Unit_MemoryPlan, ReuseCanBeSwitchedOffAndSaysSo)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);

    const ValueId x = builder.input(unit_shape(), "x");
    const ValueId a = builder.relu(x);
    const ValueId b = builder.relu(a);
    const ValueId c = builder.relu(b);
    (void)c;

    PlanOptions options;
    options.reuse = false;

    const MemoryPlan plan = MemoryPlan::build(graph, options);

    EXPECT_FALSE(plan.reuse_enabled());
    EXPECT_FALSE(plan.reuse_note().empty());
    EXPECT_EQ(plan.arena_bytes(), 384u) << "three activations, three slots";
}

TEST(Unit_MemoryPlan, TooManyNodesDeclinesOverlapRatherThanApproximatingIt)
{
    // The refusal is the point. A planner that quietly fell back to overlapping
    // linear intervals here would produce a plan that is wrong in exactly the
    // way the diamond case above describes, and would report a saving for it.
    Graph::Graph graph;
    const GraphBuilder builder(graph);

    const ValueId x = builder.input(unit_shape(), "x");
    ValueId last = builder.relu(x);
    for (int i = 0; i < 4; ++i) last = builder.relu(last);
    (void)last;

    PlanOptions options;
    options.reuse_node_limit = 2;

    const MemoryPlan plan = MemoryPlan::build(graph, options);

    EXPECT_FALSE(plan.reuse_enabled());
    EXPECT_NE(plan.reuse_note().find("reuse_node_limit"), std::string_view::npos);
    EXPECT_EQ(plan.arena_bytes(), plan.unpacked_bytes()) << "declined means no saving at all";
}

TEST(Unit_MemoryPlan, APinnedValueKeepsItsOwnAllocation)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);

    const ValueId x = builder.input(unit_shape(), "x");
    const ValueId a = builder.relu(x);
    const ValueId b = builder.relu(a);

    const ValueId pinned[] = {a};

    PlanOptions options;
    options.pinned = std::span<const ValueId>(pinned, 1);

    const MemoryPlan plan = MemoryPlan::build(graph, options);

    EXPECT_FALSE(plan.is_packed(a));
    EXPECT_TRUE(plan.is_packed(b));
}

TEST(Unit_MemoryPlan, PinningAValueThatDoesNotExistThrows)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);
    (void)builder.input(unit_shape(), "x");

    const ValueId pinned[] = {99};

    PlanOptions options;
    options.pinned = std::span<const ValueId>(pinned, 1);

    EXPECT_THROW((void)MemoryPlan::build(graph, options), Error);
}

TEST(Unit_MemoryPlan, ZeroAlignmentThrows)
{
    Graph::Graph graph;
    PlanOptions options;
    options.alignment = 0;

    EXPECT_THROW((void)MemoryPlan::build(graph, options), Error);
}

TEST(Unit_MemoryPlan, AGraphThatIsNotInTopologicalOrderThrows)
{
    // Not reachable through GraphBuilder, which always emits a producer before
    // anything can read it - but the whole pass rests on the property, so it is
    // checked rather than assumed. A closure built over a back edge would
    // silently miss ordering and the plan would overlap things it must not.
    Graph::Graph graph;

    const ValueId a = graph.add_value(unit_shape(), DType::FLOAT32, Lifetime::ACTIVATION, "a");
    const ValueId b = graph.add_value(unit_shape(), DType::FLOAT32, Lifetime::ACTIVATION, "b");

    graph.add_node(Ops::RELU.id, {a}, {b});
    graph.add_node(Ops::RELU.id, {b}, {a});

    EXPECT_THROW((void)MemoryPlan::build(graph), Error);
}

TEST(Unit_MemoryPlan, ThePlanIsDeterministicForTheSameGraph)
{
    // Byte-identical offsets across two builds, because a plan that shuffled
    // would move every tensor and take the library's bit-for-bit replay
    // guarantee with it.
    Graph::Graph graph;
    const GraphBuilder builder(graph);

    const ValueId x = builder.input(unit_shape(), "x");
    const ValueId a = builder.relu(x);
    const ValueId b = builder.sigmoid(x);
    const ValueId c = builder.add(a, b);
    const ValueId d = builder.tanh(c);
    (void)d;

    const MemoryPlan first = MemoryPlan::build(graph);
    const MemoryPlan second = MemoryPlan::build(graph);

    ASSERT_EQ(first.value_count(), second.value_count());
    for (ValueId id = 0; id < static_cast<ValueId>(first.value_count()); ++id)
    {
        EXPECT_EQ(first.placement(id).arena, second.placement(id).arena) << "value " << id;
        EXPECT_EQ(first.placement(id).offset, second.placement(id).offset) << "value " << id;
    }
}
