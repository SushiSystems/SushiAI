# tests/core/data/test_roles.cpp


Suites: `Small_DataRoles`.

Tests the spelling of the target roles that more than one module reads.

This file needs no device. A role is matched by its text, so a dataset and an objective that
spell one role differently would not find each other's target.

## EachRoleHasItsSpelling

`K_ROLE_LABELS`, `K_ROLE_BOXES`, `K_ROLE_CLASSES`, `K_ROLE_VALID`, `K_ROLE_MASK` and
`K_ROLE_VALUES` are "labels", "boxes", "classes", "valid", "mask" and "values".
