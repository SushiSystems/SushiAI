# tests/layers/train/test_role_match.cpp


Suites: `Unit_RoleMatch`.

Tests `Train::match_roles`. The dataset is a local `Published` class named "published" that
publishes the roles it is built with and holds no batch. The cases use two roles: `labels`,
one int32 per sample over a class count, and `boxes`, a float32 [n, 4] value per sample.

## ADatasetThatPublishesEveryRoleIsAccepted

A dataset that publishes `labels` over 4 classes and `boxes` of 8 matches a step that requires
`boxes` of 8 and then `labels` over 4.

## AStepThatReadsNoRoleAcceptsAnyDataset

An empty list of required roles matches a dataset that publishes no role and one that
publishes `labels`.

## AMissingRoleIsNamedWithTheDatasetAndWhatItPublishes

Requiring `boxes` of a dataset that publishes only `labels` fails with `InvalidDescription`
and a message containing `'published'`, `'boxes'`, "float32 [8, 4] per sample" and "publishes
'labels'". Requiring `labels` of a dataset that publishes nothing fails with "(none)".

## AnotherShapeIsRefusedWithBothShapes

`boxes` of 8 against a dataset that publishes `boxes` of 16 fails with `Incompatible`, "as
float32 [16, 4] per sample" and "compiled for float32 [8, 4] per sample".

## AnotherElementTypeIsRefusedWithBothTypes

Int32 `labels` against float32 `labels` fails with `Incompatible`, "as 4 classes of float32 []
per sample" and "compiled for 4 classes of int32 [] per sample".

## AnotherClassCountIsRefusedWithBothCounts

`labels` over 4 classes against `labels` over 5 fails with `Incompatible`, "as 5 classes of
int32 [] per sample" and "compiled for 4 classes of int32 [] per sample".

## ARequiredClassCountOfZeroAcceptsAnyCount

`labels` required over 0 classes matches a dataset that publishes `labels` over 5.

## TheFirstRoleThatDoesNotMatchIsReported

Requiring `boxes` and then `labels` over 4 of a dataset that publishes `labels` over 5 fails
with `InvalidDescription` and a message naming `'boxes'`.
