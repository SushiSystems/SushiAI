# tests/layers/train/test_objective_registry.cpp


Suites: `Small_ObjectiveRegistry`.

Tests `Train::ObjectiveRegistry`, `Train::register_objectives` and `Train::make_objective`. The
file's stand-in factory returns a `CrossEntropyObjective` when the context holds a task with a
`type`, one output named `logits`, the input sample [6], a `weight` attribute and a registry
that holds the row `stand_in`. Otherwise it fails with `Unsupported` and names the field.

## TheLayersTierAddsItsFiveObjectives

After `register_objectives` the registry holds five rows in the order `cross_entropy`,
`binary_cross_entropy`, `multi_label`, `mean_squared_error`, `sum`. `find("cross_entropy")`
returns a row and `find("anchor")` is null.

## ATypeIsHeldOnce

A second `add` of `stand_in` fails with `InvalidDescription`, `'stand_in'` and "already
registered", and the registry holds one row.

## TheFactoryReceivesTheContextAndItsRegistry

`make_objective` for `stand_in` with the attribute `weight` returns an objective named
"cross_entropy".

## AnUnknownTypeListsWhatTheRegistryHolds

`make_objective` for `yolo` fails with `InvalidDescription` and a message containing `'yolo'`,
`'cross_entropy'` and `'stand_in'`.

## TheFailureOfAFactoryIsReturned

`make_objective` for `stand_in` with empty attributes and the label "the objective" fails with
`Unsupported`, "the objective" and "other attributes".
