# tests/layers/train/test_metrics.cpp


Suites: `Small_Metrics`, `Small_MetricsRefusal`, `Small_MetricsDescribe`,
`Small_MetricsOnlyOutput`.

Tests `Train::Metrics`, `Train::describe_metric_value` and `Train::only_output`. A
`static_assert` holds that `K_METRIC_CAPACITY` is 8, because the overflow row fills eight names.

## AddRow

`AddRow` holds the names a set already holds, the name of the metric added and two texts of the
refusal.

## DescribeRow

`DescribeRow` holds a metric and the text it prints as.

## OutputsRow

`OutputsRow` holds the outputs of a step, each as a name and whether it holds a tensor, and two
texts of the refusal.

## ANewSetIsEmpty

A new `Metrics` has no metric in `all()`.

## MetricsAreKeptInTheOrderTheyWereAdded

After `accuracy` 0.75 and `perplexity` 12.5 are added, `all()` holds them in that order.

## AMetricIsFoundByName

`find("perplexity")` has the value 12.5 and `find("recall")` is null.

## ValueOrFallsBackForAMissingName

`value_or` returns 0.75 for `accuracy` and the fallback -1 for `recall`.

## ClearRemovesEveryMetric

After `clear()` the set is empty and `find("accuracy")` is null.

## AMetricTheSetCannotHoldIsRefused

`add` fails with `InvalidDescription`. The rows are an empty name ("non-empty name"), a second
`a` ("'a'", "twice") and a ninth metric `i` into a set holding `a` to `h` ("at most 8", `'i'`).

## ARefusedDuplicateKeepsTheFirstValue

After the refusal of a second `a`, `a` still has the value 1.

## ARefusedOverflowLeavesTheSetFull

After the refusal of the ninth metric the size is `K_METRIC_CAPACITY`.

## AFractionPrintsAsAPercentage

`describe_metric_value` gives "92.40%" for the fraction 0.9240 and "12.5" for the plain number
12.5.

## OnlyOutputReturnsTheOneTensor

`only_output` over a view with one output returns the address of that output's tensor.

## OnlyOutputRefusesAnyOtherCount

`only_output` fails with `InvalidDescription`. The rows are no output ("published 0"), two
outputs ("published 2") and one output whose tensor is null ("'output'", "no tensor").
