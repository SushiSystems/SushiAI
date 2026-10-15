# tests/layers/train/test_metrics.cpp


Suites: `Small_Metrics`.

Tests `Train::Metrics`, `Train::describe_metric_value` and `Train::only_output`.

## MetricsAreKeptInOrderAndFoundByName

A new `Metrics` is empty. After `accuracy` 0.75 and `perplexity` 12.5 are added, `all()` holds
them in that order, `find("perplexity")` has the value 12.5 and `find("recall")` is null.
`value_or` returns 0.75 for `accuracy` and the fallback -1 for `recall`. After `clear()` the
value is empty and `find("accuracy")` is null.

## AnEmptyNameADuplicateAndAnOverflowAreRefused

`add` fails with `InvalidDescription` for an empty name ("non-empty name") and for a second `a`
("'a'", "twice"), and `a` keeps the value 1. After `K_METRIC_CAPACITY` metrics the next `add`
fails with "at most 8" and `'i'`, and the size stays `K_METRIC_CAPACITY`.

## AFractionPrintsAsAPercentage

`describe_metric_value` gives "92.40%" for the fraction 0.9240 and "12.5" for the plain number
12.5.

## OnlyOutputReturnsTheOneTensor

`only_output` over a view with one output returns the address of that output's tensor.

## OnlyOutputRefusesAnyOtherCount

`only_output` fails with `InvalidDescription` for no output ("published 0"), for two outputs
("published 2") and for one output whose tensor is null ("'output'", "no tensor").
