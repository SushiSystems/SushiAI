# tests/layers/train/test_objective_context.cpp


Suites: `Small_ObjectiveContext`, `Small_ObjectiveContextFind` and
`Medium_ObjectiveContextDeathTest`.

Tests `Train::list_output_shapes` and `Train::find_output_shape`. No case needs an engine.

## ListsEveryOutputWithItsExtentsInPublishedOrder

For a `ModelOutputs` holding `coarse` of shape [2, 4, 4, 14] and `fine` of shape [2, 8, 8, 14],
the list has two entries with those names and shapes, in that order.

## AnEmptyOutputSetListsNothing

An empty `ModelOutputs` over an empty graph gives an empty list.

## AnOutputOfAnotherGraphStopsTheProcess

`list_output_shapes` over a graph that does not hold the value an output names stops the process
with "Value 0 does not exist". A death test, so it is `Medium_`.

## AProbeTraceGivesTheShapesWithoutAnEngine

For a `ModelGraph` of two `Linear` nodes, 6 to 5 and 5 to 3, that publishes both,
`Train::build_probe_graph` over [4, 6] gives a graph whose listed outputs are `hidden` of shape
[4, 5] and `logits` of shape [4, 3].

## FindRow

`FindRow` holds the name asked for, whether an entry is returned and its shape.

## AnOutputIsFoundByItsName

Over the entries `coarse` and `fine`, `find_output_shape` returns the entry named `fine` with its
shape [2, 8, 8, 14] for the first row, and null for `medium` in the second.
