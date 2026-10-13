# Plan-packed scratch: report

**Status:** Task 1 done. Tasks 2 to 8 not started.

## Task 1: the share of scratch

Measured 2026-10-13 with one step of YOLOv4-tiny under the default memory limit of `sa`:
`Integration_DarknetYolov4TinyTraining.TheFirstConvolutionReceivesAGradient`, which passed.

| Pool | Total | Arenas | Own values | Ones | Scratch | Share of scratch |
| --- | --- | --- | --- | --- | --- | --- |
| Probe | 128.2 MB | 75.8 MB | 5.6 MB | 0.004 MB | 46.7 MB | 36.5% |
| Training | 581.7 MB | 409.4 MB | 30.4 MB | 1.7 MB | 140.2 MB | 24.1% |

Each pool has one arena. The probe pool's holds 194 buffers that would take 498.4 MB unpacked;
the training pool's holds 319 buffers that would take 1008.6 MB unpacked.

Scratch is above a tenth of the total in both pools, so the stop condition of SPEC section 7.1
does not apply.

## What the share means for the full-size suites

The arena is 70% of the training pool and scratch is 24%. If YOLOv4 at 608 by 608 splits the
same way, removing all scratch would take its 11.1 GB step to about 8.4 GB, which is still above
the default limit of 7.6 GB. That split is not measured on the full-size model, and packed
scratch does not go to zero. Packing scratch alone is therefore unlikely to make the 12 failing
cases pass.

## Not measured

- The node count of the full-size training graphs against `PlanOptions::reuse_node_limit`. The
  print added for task 1 does not report it.
- The split of the full-size models. They do not fit under the default limit and are not run.

## Build

The first build of task 1's code failed: `tests/vision_support/darknet_training.cpp` added a
`std::string_view` to a string literal. The two lines now build a `std::string` first. The
build after that passed.
