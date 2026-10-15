# tests/layers/nn/test_add.cpp


That the Add node takes exactly two rank-3 inputs of one shape, emits one ADD reading them in order, returns a failure for values that do not add, and what its factory refuses.

Suites: `Unit_Add`.

Tests `NN::Add` in `modules/layers/nn/include/SushiAI/nn/add.hpp` and its factory
`NN::make_add_node`. Unless a case says otherwise, the node is built over two [8, 6, 3] shapes
and traced over two [2, 8, 6, 3] inputs.

## CreateKeepsTheSharedShapeAsThePort

The node has one port, named "output", of the shape [8, 6, 3].

## CreateRejectsOneInput

`create` over one shape fails with `InvalidDescription`, and the message holds "takes 2 inputs"
and "was given 1".

## CreateRejectsThreeInputs

`create` over three shapes fails with "takes 2 inputs" and "was given 3".

## CreateRejectsNoInput

`create` over an empty list fails with "takes 2 inputs" and "was given 0".

## CreateRejectsUnequalChannels

[8, 6, 3] with [8, 6, 5] fails with `InvalidDescription` and "share one shape".

## CreateRejectsUnequalHeight

[8, 6, 3] with [4, 6, 3] fails with "share one shape".

## CreateRejectsUnequalWidth

[8, 6, 3] with [8, 12, 3] fails with "share one shape".

## CreateRejectsAnInputThatIsNotAFeatureMap

[8, 6, 3] with [144] fails with a message holding "add input 1" and "rank-3".

## CreateRejectsAFirstInputThatIsNotAFeatureMap

[144] with [144] fails with a message holding "add input 0" and "rank-3".

## TheInputCountIsTwo

`input_count()` is 2.

## ForwardEmitsOneAddReadingBothInputsInOrder

`forward` over the inputs b and a adds one node. Its operation is `Ops::K_ADD`, its inputs are
b then a, and its one output is the one value `forward` returned.

## ForwardResultIsThePortShapeWithABatchAxis

The traced value has the shape [2, 8, 6, 3], and the port's shape stays [8, 6, 3].

## ForwardRejectsAnInputCountItWasNotBuiltOver

`forward` over one input fails with `InvalidDescription`, "input count is 2" and "given is 1".
It returns no output and adds no node.

## ForwardReturnsTheFailureOfValuesThatDoNotAdd

`forward` over a [2, 8, 6, 3] value and a [2, 8, 6, 5] value returns an `InvalidDescription`
status, no output, and adds no node.

## TheFactoryBuildsANodeFromARowWithNoKey

`make_add_node` over two [8, 6, 3] shapes and an empty row returns an `Add` with two inputs and
one port of the shape [8, 6, 3].

## TheFactoryRefusesAKeyAndShapesThatDoNotMatch

A row with the key "alpha" fails with a message holding "layer 3 'add'", "'alpha'" and
"no key". An empty row over [8, 6, 3] and [8, 6, 5] fails with "layer 3 'add'" and
"share one shape".

## TheStateWalkVisitsNothing

After a trace, `walk_state` calls neither the parameter visitor nor the buffer visitor.
