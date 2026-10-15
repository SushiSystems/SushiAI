# tests/layers/nn/test_module_list.cpp


Suites: `Small_ModuleList`, `Small_ModuleListDeathTest`.

Tests ModuleList: its storage, its index names and its place in the state walk.

This file needs no device. `NN::ModuleList<M>` in `modules/layers/nn/include/SushiAI/nn/module_list.hpp` holds
children of one type whose count is known at run time.

Three `static_assert` lines at the top of the file hold that `ModuleList<Linear>` is a
`ModuleContainer` and that neither `Detail::HasStateWalk` nor `Detail::HasParameterWalk`
detects it, so the walk reaches it through `visit_children`.

## Stage

`Stage` lists a `ModuleList<Linear>` under the child name "blocks" and a `Linear{4, 2}` under
"head".

## Counter

`Counter` holds one `NN::Buffer` listed as "count", so a list of it carries state that is not
a parameter.

## parameter_names

`parameter_names` collects what `NN::for_each_parameter` yields as "name:extents" strings.

## read_past_the_end

`read_past_the_end` builds a list of two layers and reads `list[2]`. It is the statement of
the death test.

## ANewListIsEmpty

A new list has size 0, `visit_children` calls its visitor zero times, and `begin()` equals
`end()`.

## PushBackAppendsInOrder

After two `push_back` calls the size is 2, element 0 has 8 output features and element 1
has 3. Indexing a const reference to the list reads element 1's 8 input features.

## IterationFollowsIndexOrder

A range-for over three layers reads the output widths 8, 3 and 1.

## VisitChildrenPairsEachChildWithItsIndex

`visit_children` over two layers passes the indices 0 and 1 with the addresses of `list[0]` and
`list[1]`.

## TheWalkNamesChildrenByIndex

Two traced layers, the second without a bias, walk as "0.weight:[8, 4]", "0.bias:[8]" and
"1.weight:[3, 8]", and `parameter_count` is 3.

## AListUnderAChildFieldIsNamedByFieldThenIndex

A traced `Stage` with two blocks walks as "blocks.0.weight", "blocks.0.bias",
"blocks.1.weight", "blocks.1.bias", "head.weight" and "head.bias", each with its extents.

## TheWalkNamesBuffersByIndex

Of three `Counter` children, 0 and 2 have their buffer defined. `for_each_buffer` yields
"0.count" and "2.count": child 1 yields nothing and still keeps its index. `parameter_count`
is 0.

## GrowingTheListKeepsATracedChildsStateIdentity

The first child is traced, then 64 more layers are appended, which moves a one-element
vector's storage at least once. The first child's weight keeps its state and its leaf in the
graph, and its bias keeps its leaf in the graph.

## AnIndexPastTheEndIsFatal

The case is in the suite `Small_ModuleListDeathTest`. `list[2]` on a list of two stops the
process with a report containing "index 2" and "holds 2 children".
