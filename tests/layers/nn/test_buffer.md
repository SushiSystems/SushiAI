# tests/layers/nn/test_buffer.cpp


Buffer: its description, its one leaf per graph and its identity.

Suites: `Small_Buffer`, `Small_BufferDeathTest`.

Tests Buffer: its description, its one leaf per graph and its identity.

This file needs no device. `NN::Buffer` in `modules/layers/nn/include/SushiAI/nn/buffer.hpp` is state a model
owns that takes no gradient. `define()` records its shape, starting value and element type and
returns a `Status`. `read()` declares a leaf of lifetime `Graph::Lifetime::Buffer` in the graph
of the builder it is given.

## K_EXACT_TOLERANCE

`K_EXACT_TOLERANCE` is 1e-15, the absolute tolerance for a double that is stored and read back
unchanged.

## read_undefined

`read_undefined` calls `read()` on a buffer under the name "running_mean" with a builder over
a fresh graph. It is the statement of the death test.

## ANewBufferIsUndefined

A default-constructed buffer reports `declared()` false and the state `Graph::K_NO_STATE`.

## DefineRecordsTheDescription

After `define([8], 1.0, FLOAT64)` the buffer is declared and returns that shape, that element
type and the starting value 1.0.

## DefineDefaultsToFloat32

`define([8], 0.0)` with no element type records `DType::Float32`.

## ADefinedBufferAddsNoLeafUntilItIsRead

After `define()` alone a graph holds no value and `id_in(graph)` is `Graph::K_NO_VALUE`.

## DefineDrawsTheIdentity

Two buffers that were only defined have states other than `K_NO_STATE`, and the two states
differ. A second identical `define()` and a `read()` leave the first buffer's state unchanged.

## ReadDeclaresABufferLeaf

`read()` adds one value to the graph. It has the lifetime `BUFFER`, `requires_grad` false and
the shape [8] and type FLOAT64 that were defined. `id_in(graph)` and
`graph.state_leaf(buffer.state())` both return it.

## ASecondReadInOneGraphReturnsTheFirstLeaf

A second `read()` through the same builder returns the first leaf. The graph's value count
does not grow and `state_leaves()` holds one pair.

## ABufferReadInTwoGraphsHasALeafInEach

The first graph holds a padding input, so the buffer's leaf there is value 1, and in the
second graph it is value 0. `id_in` returns each graph's own leaf, and the buffer's state is
the same after both reads.

## ACopiedBufferHasANewStateID

A copy of a defined and read buffer has the original's shape and starting value and a state
that is neither `K_NO_STATE` nor the original's. `copy.id_in(graph)` is `K_NO_VALUE` until the
copy is read, and its leaf then differs from the original's, which keeps its own.

## AMovedBufferKeepsItsStateID

A buffer move-constructed from a read one has the same state and the same leaf in the graph.
Reading it again returns that leaf and adds no value.

## AMovedFromBufferIsUndefined

After a move construction and after a move assignment the source reports `declared()` false
and the state `K_NO_STATE`. The assigned target is declared and has a state.

## ACopyAssignedBufferHasANewStateID

A defined [2] buffer copy-assigned from a [8] FLOAT64 one takes that shape and type. Its state
differs from the state it had before and from the source's.

## ACopyOfAnUndefinedBufferHasNoIdentity

A copy of a buffer that was never defined is not declared and has the state `K_NO_STATE`.

## DefiningTwiceWithOneDescriptionIsAccepted

A second `define()` with the same shape, value and type returns ok and leaves the description
as it was.

## DefiningWithAnotherShapeFails

`define([4], 0.0)` after `define([8], 0.0)` returns `InvalidDescription` with a message
containing "Buffer was defined as", "[8]" and "[4]". The shape stays [8].

## DefiningWithAnotherElementTypeFails

`define([8], 0.0, FLOAT64)` after a FLOAT32 definition returns `InvalidDescription` with a
message naming both types. The type stays FLOAT32.

## DefiningWithAnotherStartingValueFails

`define([8], 1.0)` after `define([8], 0.0)` returns `InvalidDescription` with a message
containing "Buffer was defined as" and "is now defined as". The starting value stays 0.0.

## ReadingBeforeDefineIsFatal

The case is in the suite `Small_BufferDeathTest`. `read()` on an undefined buffer stops the
process with a report containing "'running_mean'" and "before define()".
