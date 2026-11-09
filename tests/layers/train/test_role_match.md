# tests/layers/train/test_role_match.cpp


Suites: `Small_RoleMatchAccepted`, `Small_RoleMatchRefused`.

Tests `Train::match_roles`. The dataset is a `Data::InMemoryDataset` of no batch that publishes
the roles a row gives it; its name is "in-memory". The cases use two roles: `labels`, one int32
per sample over a class count, and `boxes`, a float32 [n, 4] value per sample.

## AcceptRow

`AcceptRow` holds the roles a step requires and the roles a dataset publishes.

## RefuseRow

`RefuseRow` holds the same two lists, the code of the refusal and three texts its message holds.

## ADatasetThatPublishesTheRequiredRolesIsAccepted

`match_roles` is ok in each row: `boxes` of 8 and then `labels` over 4 against a dataset that
publishes `labels` over 4 and `boxes` of 8; no required role against a dataset that publishes
none; no required role against one that publishes `labels`; and `labels` required over 0
classes, which accepts whatever count is published, against `labels` over 5.

## ARoleThatDoesNotMatchIsRefusedWithBothDescriptions

`match_roles` fails with the row's code and a message that holds `'in-memory'` and the row's
texts.

- `AMissingRoleNamesWhatIsPublished`: `boxes` required of a dataset that publishes only
  `labels` fails with `InvalidDescription`, `'boxes'`, "float32 [8, 4] per sample" and "publishes
  'labels'".
- `AMissingRoleOfADatasetThatPublishesNothing`: `labels` required fails with `'labels'` and
  "(none)".
- `AnotherShape`: `boxes` of 8 against `boxes` of 16 fails with `Incompatible`, "as float32
  [16, 4] per sample" and "compiled for float32 [8, 4] per sample".
- `AnotherElementType`: int32 `labels` against float32 `labels` fails with `Incompatible`, "as 4
  classes of float32 [] per sample" and "compiled for 4 classes of int32 [] per sample".
- `AnotherClassCount`: `labels` over 4 against `labels` over 5 fails with `Incompatible`, "as 5
  classes of int32 [] per sample" and "compiled for 4 classes of int32 [] per sample".
- `TheFirstRoleThatDoesNotMatch`: `boxes` and then `labels` over 4 against `labels` over 5 fails
  with `InvalidDescription` and a message naming `'boxes'`.
