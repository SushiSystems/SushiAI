# Eager front end: plan for the host-lane units

**Status:** In progress. The owner approved the `eager` module and the `Provenance` parameter on 2026-10-07 and left the run mode to the orchestrator. Covers step 3 of `SPEC.md` section 12 and nothing else.

**Goal:** Build the four units of the eager front end that need no engine: scope tracing, gradient
provenance, the region partitioner and the structure key.

**Spec:** `docs/agent/2026_10_07_EAGER_FRONT_END/SPEC.md`, sections 3, 5 and 6.

**Why this slice first:** it depends on nothing in SushiBLAS, so it can land while the recorder is
still being designed there. Every unit is host lane and is tested without a device.

## Global constraints

- C++17, `cpp-code-style`, `source-comments`. New names follow the skill: no `I` prefix on the
  interface, constants spelled `K_UPPER_SNAKE`. The rest of SushiAI still uses the older spelling;
  the estate audit's finding C11 covers that and this plan does not touch it.
- Errors are raised with `SA_THROW_IF`, as everywhere else in the library, until the owner decides
  finding C4.
- Build and test only through `sa`. One build per task.
- No file under `src/graph/lowering.cpp`, `src/optim/`, `src/nn/init.cpp` or `src/data/` is edited.
- Each task adds its changelog line and its README section in the same change.

## Two decisions this plan asks the owner to confirm

1. **A new module, `eager`**, at `include/SushiAI/eager/` and `src/eager/`. It depends on `graph`,
   `autograd` and `core`; nothing depends on it. The spec's `Eager::` namespace implies it, but a
   module boundary is the owner's call.
2. **`Autograd::differentiate` gains one trailing parameter**, `Provenance* provenance = nullptr`.
   Source compatible; the exported symbol changes. Task 1 explains why a structural guess is not
   good enough.

## File map

| File | Holds |
| --- | --- |
| `include/SushiAI/autograd/provenance.hpp` | `Provenance`: which forward node each backward node came from |
| `src/autograd/differentiator.{hpp,cpp}` | Fills a `Provenance` when one is passed |
| `include/SushiAI/eager/segment.hpp` | `Segment` |
| `include/SushiAI/eager/scope_trace.hpp`, `src/eager/scope_trace.cpp` | `ScopeTrace` |
| `include/SushiAI/eager/scope.hpp` | `Scope`, the RAII guard over a `ScopeTrace` |
| `include/SushiAI/eager/partition.hpp` | `RegionKey`, `Direction`, `RegionInfo`, `Partition` |
| `include/SushiAI/eager/region_partitioner.hpp` | `RegionPartitioner`, the interface |
| `include/SushiAI/eager/scope_partitioner.hpp`, `src/eager/scope_partitioner.cpp` | `ScopePartitioner` |
| `include/SushiAI/eager/region_signature.hpp`, `src/eager/region_signature.cpp` | `RegionSignature`, `signature_of()` |
| `include/SushiAI/eager/README.md` | The module's facts and the reasons the comments cite |
| `tests/functional/unit/test_provenance.cpp`, `test_scope_trace.cpp`, `test_scope_partitioner.cpp`, `test_region_signature.cpp` | One file per unit |
| `src/CMakeLists.txt`, `tests/CMakeLists.txt`, `include/SushiAI/SushiAI.h` | Wiring; the configure step checks the umbrella header lists every public header |

## Task 1: gradient provenance

**Acceptance:** `sa test --suite unit --filter Unit_Provenance` passes.

**Why:** the partitioner must put a backward node in the backward region of the scope whose forward
node it differentiates. Guessing that from operands is wrong for the common case: the weight
gradient of a `Linear` in scope `k` reads that layer's input, which scope `k - 1` produced. Only the
reverse walk knows which forward node a rule was running for.

**Interface produced:**

```cpp
namespace SushiAI::Autograd
{
    struct Provenance
    {
        std::size_t forward_node_count = 0;
        std::vector<Graph::NodeId> origin;   // origin[i] is for node forward_node_count + i

        Graph::NodeId origin_of(Graph::NodeId backward_node) const;
    };

    std::vector<Graph::ValueId> differentiate(Graph::Graph& graph, Graph::ValueId loss,
                                              SushiRuntime::span<const Graph::ValueId> wrt,
                                              Graph::ValueId seed_factor = Graph::NO_VALUE,
                                              Provenance* provenance = nullptr);
}
```

**Rule:** every node appended while the walk is at forward node `f` (the rule's own emissions and
the `ACCUMULATE` nodes `resolve()` emits for `f`'s outputs) has origin `f`. An `ACCUMULATE` emitted
by the closing `resolve()` of a `wrt` value has the producer of that value as its origin, which for
a parameter is its `LEAF` node.

**Tests** (`Unit_Provenance`, built on a two-`Linear` graph with cross-entropy, no engine):

- `EveryBackwardNodeHasAForwardOrigin`: `origin.size()` equals the number of appended nodes and
  every origin is below `forward_node_count`.
- `ARuleEmitsUnderTheNodeItDifferentiates`: the node producing the first layer's weight gradient
  has the first `MATMUL` (or its `BIAS_ADD`, whichever rule emitted it) as origin, asserted by op.
- `AFanOutAccumulateBelongsToTheProducer`: a value read twice gets an `ACCUMULATE` whose origin is
  that value's producer.
- `AParameterReadTwiceAccumulatesUnderItsLeaf`: the closing accumulate has the `LEAF` as origin.
- `PassingNoProvenanceChangesNothing`: the graph differentiated with and without the parameter has
  the same nodes, op by op.

## Task 2: scope tracing

**Acceptance:** `sa test --suite unit --filter Unit_ScopeTrace` passes.

**Interface produced:**

```cpp
namespace SushiAI::Eager
{
    struct Segment
    {
        std::string name;        // "encoder", "encoder/attention", "" outside any scope
        std::uint32_t occurrence; // nth segment with this name in this trace, from 0
        Graph::NodeId begin;     // first node
        Graph::NodeId end;       // one past the last node
    };

    class ScopeTrace final
    {
        public:
            explicit ScopeTrace(const Graph::Graph& graph) noexcept;
            void open(std::string_view name);
            void close();
            std::vector<Segment> seal();
            std::size_t depth() const noexcept;
    };

    class Scope final   // constructor opens, destructor closes; not copyable, not movable
    {
        public:
            Scope(ScopeTrace& trace, std::string_view name);
            ~Scope();
    };
}
```

A segment is a maximal run of consecutively traced nodes with the same innermost open scope. Opening
or closing a scope ends the current segment; a segment with no node is dropped. Segments are
therefore contiguous and ordered, which is what lets the partitioner number them without checking
dependencies.

**Tests:**

- `NodesOutsideAnyScopeFormOneUnnamedSegment`
- `AScopeSplitsTheTraceIntoThreeSegments`: before, inside, after.
- `ANestedScopeIsNamedByItsPath`: `"outer/inner"`, and the outer scope resumes as a new segment
  with occurrence 1.
- `AScopeOpenedTwiceGetsRisingOccurrences`: a loop that opens `"cell"` three times.
- `AnEmptyScopeLeavesNoSegment`
- `SealWithAScopeStillOpenThrows`, `CloseWithNoScopeOpenThrows`
- `SegmentsCoverEveryNodeExactlyOnce`: the union of `[begin, end)` is `[0, node_count)`.

## Task 3: the partitioner

**Acceptance:** `sa test --suite unit --filter Unit_ScopePartitioner` passes. Waits on tasks 1 and 2.

**Interface produced:**

```cpp
namespace SushiAI::Eager
{
    using RegionKey = std::uint64_t;
    inline constexpr RegionKey K_NO_REGION = 0;

    enum class Direction : std::uint8_t { Forward, Backward };

    struct RegionInfo
    {
        RegionKey key;
        std::string name;          // the segment's name and occurrence, "head#0"
        Direction direction;
    };

    struct Partition
    {
        std::vector<RegionKey> region_of_node;   // K_NO_REGION for a LEAF node
        std::vector<RegionInfo> regions;         // ascending key
        RegionKey optimizer_key;                 // reserved, above every region
    };

    class RegionPartitioner
    {
        public:
            virtual ~RegionPartitioner() = default;
            virtual Partition assign(const Graph::Graph& graph,
                                     SushiRuntime::span<const Segment> segments,
                                     const Autograd::Provenance& provenance) const = 0;
    };

    class ScopePartitioner final : public RegionPartitioner { /* assign() */ };
}
```

**Rule, with `m` segments in trace order:** a forward node of segment `k` gets key `k`. A backward
node whose origin lies in segment `k` gets key `2m - k`. For `k = m` the two coincide, so the last
segment's forward and backward nodes share one region; when the caller scopes the loss, that region
is the loss and its seed. A backward node whose origin is a `LEAF` gets the highest key among the
regions of its operands' producers. `optimizer_key` is `2m`. `LEAF` nodes get `K_NO_REGION`: the
lowering records nothing for them.

This restates the table in `SPEC.md` section 5 with the loss as an ordinary last segment: the spec's
`n` forward scopes plus the loss are `m = n + 1` segments, and `2n + 2 - k` is `2m - k`.

**Tests:**

- `AnUnscopedStepIsOneRegion`: `m = 1`, every non-leaf node has key 1, `optimizer_key` is 2.
- `ForwardKeysFollowTraceOrder`
- `BackwardKeysMirrorForwardKeys`: three segments give backward keys 5, 4 and 3.
- `TheLastSegmentSharesItsRegionWithItsBackward`
- `EveryOperandComesFromALowerOrEqualKey`: for every node and every operand with a producer in a
  region, `key(producer) <= key(node)`. This is the property the runtime's ordering needs, and it is
  asserted over the whole differentiated graph, not over chosen nodes.
- `AWeightGradientLandsInItsOwnLayersBackwardRegion`: the case task 1 exists for.
- `ASharedParametersAccumulateSitsAboveBothContributors`
- `LeavesBelongToNoRegion`

## Task 4: the structure key

**Acceptance:** `sa test --suite unit --filter Unit_RegionSignature` passes. Waits on task 3.

**Interface produced:**

```cpp
namespace SushiAI::Eager
{
    struct RegionSignature
    {
        std::uint64_t hash;
        std::vector<OpID> ops;     // the region's nodes, in order

        bool operator==(const RegionSignature& other) const noexcept;
        bool operator!=(const RegionSignature& other) const noexcept;
    };

    RegionSignature signature_of(const Graph::Graph& graph, const Partition& partition,
                                 RegionKey region,
                                 SushiRuntime::span<const std::uint64_t> external_identity);
}
```

`external_identity[v]` is the caller's stable identity for a value the region reads and does not
produce: the address of a parameter's storage, of a session-owned input buffer, or of the storage
a boundary value is bound to. The hash covers what `SPEC.md` section 6 lists. An operand produced
inside the region is hashed by its position in the region; every other operand by its identity
alone. Value ids and node ids never enter the hash, because a change in an earlier region shifts
them.

The first implementation hashed a boundary operand by its producer's region name and the
producer's index inside that region. The review showed that the index moves whenever the producer
region grows, so every reader of a changed region changed with it, which is the property this unit
exists to prevent. The rule above replaced it on 2026-10-07.

**Tests:**

- `TheSameTraceGivesTheSameSignature`: two graphs built by the same code.
- `AChangedAttributeChangesTheHash`, `AChangedShapeChangesTheHash`, `AChangedDtypeChangesTheHash`,
  `AChangedOpChangesTheHash`
- `ADifferentParameterChangesTheHash`: same structure, another identity for the parameter.
- `AnEarlierRegionGrowingDoesNotChangeALaterRegion`: add a layer to segment 1; segment 2's
  signature is unchanged although every one of its ids moved. This is the property the whole
  design rests on.
- `ARewiredOperandChangesTheHash`: `add(a, b)` against `add(b, a)`.
- `EqualHashesWithDifferentOpsCompareUnequal`: built by hand, to pin the collision guard.

## What this plan does not build

`Session`, `Step`, `StepExecutor`, `RegionExecutor`, `RegionEmitter`, the flush points and the
phases. They need the SushiBLAS recorder and get their own plan once its signatures are fixed.

## Order and builds

The plan runs as one wave. The orchestrator first writes what every task shares: the public
headers of all four units, the module README, and the lines in `src/CMakeLists.txt`,
`tests/CMakeLists.txt` and `SushiAI.h`. With the interfaces fixed, the four tasks hold disjoint
files (one source and one test file each; task 1 holds the two differentiator files) and run in
parallel, each followed by its own reviewer. The orchestrator closes the wave with one `sa build`
and `sa test --suite unit`.

`sa build` compiles SushiBLAS and SushiRuntime from source, now with the CUDA target, and this
machine runs one build at a time. Each build is announced to the sushiblas session first.
