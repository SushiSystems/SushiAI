# tests/layers/layer_traces/test_trace_summary.cpp


Suites: `Small_LayerTracesSummary`, `Small_LayerTracesStateRecord`.

Tests `state_record`, `state_records` and `trace_summary` in
`modules/layers/layer_traces/include/SushiAI/layer_traces/trace_summary.hpp`. Each model is a
`NN::ModelGraph` built in the case; a trace needs no device.

## ends_with

Returns whether a string ends with a suffix, so a case can pin a record without its prefix.

## RecordRow

Holds a state entry and the record written for it.

## summary_of_linear_model

Traces a model of one 6-to-4 linear layer publishing "logits" and returns its summary.

## AStateRecordNamesItsKind

Run once for each row of `Small_LayerTracesStateRecord`: a parameter of [4, 3] reads
"0.weight [4, 3] parameter" and a buffer of [8] reads "1.running_mean [8] buffer".

## TheSummaryListsTheOutputs

The summary of a linear model publishing "logits" lists the output "logits [2, 4]".

## TheSummaryListsTheParameters

The summary lists two state records ending in "weight [4, 6] parameter" and "bias [4] parameter".
The ops of the trace are how the model computes, not a promise of the summary, and the state is
not compared with `state_records`, which would compare the code with itself.

## TheTraceRunsOverTheFixedBatch

A flatten over the sample [2, 2, 3] yields "flat [2, 12]" and no state, so the input carries two
rows in front of the sample.
