# tests/layers/train/test_objective_context.cpp


Suites: `Medium_ObjectiveContext`.

Tests `Train::list_output_shapes` and `Train::find_output_shape`. No case needs an engine.

## ListsEveryOutputWithItsExtentsInPublishedOrder

For a `ModelOutputs` holding `coarse` of shape [2, 4, 4, 14] and `fine` of shape [2, 8, 8, 14],
the list has two entries with those names and shapes, in that order.

## AnEmptyOutputSetListsNothing

An empty `ModelOutputs` over an empty graph gives an empty list.

## AProbeTraceGivesTheShapesWithoutAnEngine

For a `ModelGraph` of two `Linear` nodes, 6 to 5 and 5 to 3, that publishes both,
`Train::build_probe_graph` over [4, 6] gives a graph whose listed outputs are `hidden` of shape
[4, 5] and `logits` of shape [4, 3].

## FindsAnOutputByItsName

`find_output_shape` returns the entry named `fine` with its shape, and null for `medium`.
