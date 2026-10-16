# tests/layers/layer_traces/test_trace_summary.cpp


Suites: `Small_LayerTracesSummary`.

Tests `state_record`, `state_records` and `trace_summary` in
`modules/layers/layer_traces/include/SushiAI/layer_traces/trace_summary.hpp`. Each model is a
`NN::ModelGraph` built in the case; a trace needs no device.

## ends_with

Returns whether a string ends with a suffix, so a case can pin a record without its prefix.

## AStateRecordNamesItsKind

A parameter of [4, 3] reads "0.weight [4, 3] parameter" and a buffer of [8] reads
"1.running_mean [8] buffer".

## TheSummaryListsTheOpsTheOutputAndTheState

A model of one 6-to-4 linear layer publishing "logits" yields the output "logits [2, 4]", two
state records ending in "weight [4, 6] parameter" and "bias [4] parameter", equal to what
`state_records` lists, and an op list holding a K_MATMUL.

## TheTraceRunsOverTheFixedBatch

A flatten over the sample [2, 2, 3] yields "flat [2, 12]" and no state, so the input carries
`K_TRACE_BATCH`, which is 2, rows in front of the sample.
