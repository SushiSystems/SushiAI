# tests/core/eager/test_region_boundary.cpp


Suites: `Unit_RegionBoundary`.

Tests which values `find_boundaries` lists as the imports and exports of a region.

This file needs no device. Every case traces a small graph on the host, partitions it with
`ScopePartitioner` and reads the `BoundaryMap`. `modules/core/eager/README.md`, under
`region_boundary.hpp`, says when a buffer is shared and why a region exports every value of a
shared buffer.

## eager_test_support.hpp

The host-lane tests of the module share the traced steps of `eager_test_support.hpp`.
`encoder_head_loss(head, extra)` traces scopes "encoder", "head" and "loss" over an [8, 6]
input, differentiates the loss and partitions the result; `extra` adds a scope "extra" between
the head and the loss, which turns three segments into four. `linear_then_relu(split)` traces
`relu(linear(x))` with no backward pass, as one segment or as two. `presets_of` gives each leaf
the id `K_LEAF_STORAGE` plus its position among the leaves and each parameter gradient
`K_GRADIENT_STORAGE` plus the parameter's position, which is the storage a `Session` would
supply. The positions are the same in every variant of a step, so two traces of one model give
their leaves the same ids. `observed_of` returns the loss and the gradients, the values a session
observes.

## AForwardRegionExportsWhatALaterRegionReads

The encoder's result is read by the head, so the encoder's forward region exports it and the
head's forward region imports it. The head's result is read by the loss region.

## AnObservedValueIsExportedWithNoReader

One region holds the whole step. Its result is exported when the caller observes it and is not
exported when nobody does.

## AnExportIsNamedByItsPlaceInTheRegion

The region holds K_MATMUL, K_BIAS_ADD and K_RELU; the three K_LEAF nodes are in no region. The
observed result is therefore at node 2, result 0. The position counts the region's own nodes,
which is what keeps it stable when an earlier region grows.

## ImportsAreListedOnceInReadOrder

`add(add(b, a), add(a, b))` reads `b` first and each leaf twice. The imports are `b`, `a`.

## EveryImportIsALeafOrAnExportOfALowerKey

The case walks every region of a four-segment differentiated step. Each import that a node
produces comes from a region with a lower key, and that region lists it as an export. `checked`
counts the imports the loop compared, so a map with no boundary value would fail the last
assertion.

## AnAccumulateSharesTheBufferOfItsFirstOperand

K_ACCUMULATE writes its result into its first operand. The result and that operand have one
buffer and the second operand has its own. The result is observed, so both values of the buffer
are exports and the second operand is not.

## AnAccumulateIntoALeafThrows

An accumulate whose first operand is an input leaf would add into storage the session does not
own.

## APartitionOfAnotherGraphThrows

A partition with one node too few is refused before any node is read.
