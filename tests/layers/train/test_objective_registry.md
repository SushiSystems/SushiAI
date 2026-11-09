# tests/layers/train/test_objective_registry.cpp


Suites: `Small_ObjectiveRegistry`, `Small_ObjectiveRegistryFind` and `Medium_ObjectiveRegistryDeathTest`.

Tests `Train::ObjectiveRegistry`, `Train::register_objectives` and `Train::make_objective`. Every
case runs on the real rows of the layers tier; no stand-in factory is registered. The context is
a batch of 4 over 3 classes with the input sample [6] and one output `output` of shape [4, 3].

## FindRow

`FindRow` holds a type asked of the registry and whether it holds a row for it.

## TheLayersTierAddsItsFiveObjectivesInOrder

After `register_objectives` the registry holds five rows in the order `cross_entropy`,
`binary_cross_entropy`, `multi_label`, `mean_squared_error`, `sum`.

## FindReturnsAHeldTypeAndNullForAnother

`find("cross_entropy")` returns a row and `find("anchor")`, a domain's type, is null.

## ATypeIsHeldOnce

A second `add` of the held `cross_entropy` row fails with `InvalidDescription`,
`'cross_entropy'` and "already registered", and the registry still holds five rows.

## ARowWithoutATypeStopsTheProcess

`ObjectiveRegistry::add` stops the process on a row whose type is empty, with "non-empty type". A
death test, so it is `Medium_`.

## ARowWithoutAFactoryStopsTheProcessNamingIt

A row with a type and no factory stops the process, and the message names the type and says it has
no factory.

## AnUnknownTypeListsWhatTheRegistryHolds

`make_objective` for `yolo` fails with `InvalidDescription` and a message containing `'yolo'`,
`'cross_entropy'` and `'sum'`.

## TheFailureOfAFactoryIsReturned

`make_objective` for `cross_entropy` with the key `label_smoothing` and the label "the objective"
returns the failure of the real factory: `InvalidDescription`, "the objective" and
`'label_smoothing'`.

## TheFactoryReceivesTheRegistryItWasCalledWith

The `sum` factory refuses a context with no registry. `make_objective` sets the registry on the
context it hands over, so `sum` with `{"terms": []}` reaches its own rule and fails with
`InvalidDescription` and "at least one term", not with the refusal for a missing registry.
