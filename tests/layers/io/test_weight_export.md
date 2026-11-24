# tests/layers/io/test_weight_export.cpp

Suites: `Small_WeightExport`.

Tests `IO::export_state_weights`, in `modules/layers/io/include/SushiAI/io/weight_export.hpp`: a convolution, a depthwise convolution, a batch norm and a linear, written as safetensors and as PyTorch and read back by `import_state_weights`.

## KernelsTakeTheSourceLayout

A kernel whose entry states a layout is written in the layout of the source. The depthwise kernel comes out as [2, 1, 3, 3], a record of another entry keeps its bytes, and one note names `num_batches_tracked`.

## SafetensorsRoundTripsEveryRecord

The records are exported, written as safetensors, read and imported. Name, shape and bytes equal the records.

## TorchRoundTripsEveryRecordThroughAMap

The records are exported under a name map that renames them, written as a PyTorch file, read and imported through the same map. The renamed names are the ones written, and the import returns the records.

## RecordsThatDoNotFitAreNamedInOneMessage

A record with no entry and a record of another shape fail together, in one message that names both.
