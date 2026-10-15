# tests/layers/nn/test_concat.cpp


The Concat node's shape check, its port and the node it traces.

Suites: `Unit_Concat`.

Tests the Concat node's shape check, its port and the node it traces.

This file needs no device. `NN::Concat` in `modules/layers/nn/include/SushiAI/nn/concat.hpp` is an `NN::Node`
that joins several feature maps along the channel axis. `Concat::create` takes one per-sample
shape per input and returns a `Result`, and `forward` returns a `Status`.

## create_over

`create_over` calls `Concat::create` over a vector of shapes viewed as an `NN::ShapeSpan`.

## span_of

`span_of` views a vector of ids as the `NN::ValueSpan` one `forward` call takes.

## CreateSumsTheChannelsOfItsInputs

Over [8, 6, 3], [8, 6, 5] and [8, 6, 1] the node has one port named "output" of shape
[8, 6, 9].

## CreateAcceptsOneInput

Over the one shape [8, 6, 3] the port's shape is [8, 6, 3].

## CreateRejectsUnequalHeight

[8, 6, 3] with [4, 6, 3] fails with `InvalidDescription` and a message containing "concat
input 1" and "H and W".

## CreateRejectsUnequalWidth

A third input of [8, 12, 3] fails with a message containing "concat input 2" and "H and W".

## CreateRejectsAnInputThatIsNotAFeatureMap

[8, 6, 3] with [144] fails with a message containing "concat input 1" and "rank-3".

## CreateRejectsAFirstInputThatIsNotAFeatureMap

[144] with [8, 6, 3] fails with a message containing "concat input 0" and "rank-3".

## CreateRejectsNoInput

No shape at all fails with a message containing "at least one input".

## ForwardEmitsOneConcatChannelsReadingEveryInputInOrder

`forward` over the values b and a adds one `Ops::K_CONCAT_CHANNELS` node whose inputs are b then
a. It appends one id to `outputs`, the node's one output.

## ForwardResultIsThePortShapeWithABatchAxis

Over [2, 8, 6, 3] and [2, 8, 6, 5] the traced value has the shape [2, 8, 6, 8], and the port
reports [8, 6, 8].

## ForwardRejectsAnInputCountItWasNotBuiltOver

A node built over two shapes and given one value fails with a message containing "input count
is 2" and "given is 1". `outputs` stays empty and the graph gains no node.

## TheInputCountIsTheNumberOfShapesItWasBuiltOver

`input_count()` is 1 for a node built over one shape and 3 for a node built over three.

## ForwardReturnsTheFailureOfValuesThatDoNotJoin

A node built over two [8, 6, 3] shapes is given a [2, 8, 6, 3] value and a [2, 4, 6, 3] value.
`forward` returns an `InvalidDescription` failure, appends nothing to `outputs` and adds no
node.

## TheFactoryBuildsANodeFromARowWithNoKey

`NN::make_concat_node` over [8, 6, 3] and [8, 6, 5] with the attributes `{}` returns a
`Concat` with an input count of 2 and one port of shape [8, 6, 8].

## TheFactoryRefusesAKeyAndShapesThatDoNotJoin

The attributes `{"axis": 3}` fail with a message containing the context "layer 2 'concat'",
"'axis'" and "no key". The shapes [8, 6, 3] and [4, 6, 5] with no attribute fail with the
context and "H and W".

## TheStateWalkVisitsNothing

After a trace, `walk_state("merge", visitor)` calls neither the parameter callback nor the
buffer callback.
