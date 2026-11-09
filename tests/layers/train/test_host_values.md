# tests/layers/train/test_host_values.cpp


Suites: `Small_HostValues`.

Tests `Train::require_host_reals` and `Train::host_real` on the two element types they read.

The tensors are allocated from the engine of a `HostExecutionBackend`, so the file is small. The
refusal of an int32 tensor and of another element count is pinned in `test_binary_confusion.cpp`;
the cases here pin the reads.

## ReadRow

`ReadRow` holds an element type, the elements written, the index read and the double the read
returns.

## AnElementIsReadAsTheDoubleItsTypeHolds

The tensor is accepted for its own element count and the element reads back as the row's
double, compared for equality because a read rounds nothing. The rows are element 1 of the
float32 values 0.5, -1.25 and 3.0, which reads -1.25; the float32 value 0.1, which reads as the
float nearest 0.1 widened to a double and differs from 0.1; and the float64 value 0.1, which
reads back as the double 0.1. A read that went through float for both types would fail the third
row.
