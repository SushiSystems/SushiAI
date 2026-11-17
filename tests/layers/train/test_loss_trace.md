# tests/layers/train/test_loss_trace.cpp

Suites: `Small_LossTrace` and `Medium_LossTraceDeathTest`.

Tests `Train::LossTrace` of `loss_trace.hpp`: the loss value of an objective, the named values it
keeps for `declare_metric_values`, and the trace of each term of a composite. Value ids are written
as numbers; no graph is built.

## ADefaultTraceHoldsNothing

A default trace has `K_NO_VALUE` as its loss, no kept value and, for any position, a term trace
that holds nothing: the trace an evaluation graph passes.

## ATraceOfALossReadsAsTheLoss

A trace built over the value 7 returns 7 from `loss()`, and is not convertible to a value id.

## KeptValuesAreFoundByNameInOrder

Two values kept as `rows` and `weights` are found by name, a name never kept is absent, and `all()`
lists them in the order they were kept.

## TermsAreFoundByPosition

A trace with two terms returns the first term's kept `rows` from position 0 and the second term's
loss from position 1, which holds no `rows`.

## APositionPastTheTermsAnswersAnEmptyTrace

A trace with one term answers position 1 with a trace of no loss and no value.

## KeepingANameTwiceIsFatalNamingIt

`Medium_LossTraceDeathTest`. A second `add` of `rows` stops the process and the report contains
"kept the value 'rows' twice".

## ReadingAMissingNameIsFatalListingTheHeldOnes

`Medium_LossTraceDeathTest`. `at("columns")` on a trace that holds `rows` stops the process and the
report names `columns` and lists `'rows'`.
