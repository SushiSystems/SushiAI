# Plan-packed scratch: report

**Status:** Task 1 done, and the full-size plans measured. Tasks 2 to 8 not started.

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

## The plans of the full-size graphs

Measured 2026-10-13 on the host: `ThePlansOfTheStepOverlapLiveRanges` traces both graphs of a
model, builds their memory plans and prints them. Nothing is allocated on the device, so the
full-size models run under the default limit. All three cases passed.

| Model | Graph | Nodes | Values | Arenas | Unpacked | Own values |
| --- | --- | --- | --- | --- | --- | --- |
| YOLOv4-tiny | Probe | 198 | 260 | 75.8 MB | 500.1 MB | 28.7 MB |
| YOLOv4-tiny | Training | 395 | 495 | 409.4 MB | 1008.6 MB | 55.1 MB |
| YOLOv4 | Probe | 1164 | 1491 | 719.0 MB | 8023.3 MB | 268.7 MB |
| YOLOv4 | Training | 2314 | 2855 | 5436.7 MB | 16723.9 MB | 544.0 MB |
| YOLOv7 | Probe | 850 | 1128 | 640.6 MB | 5825.9 MB | 166.6 MB |
| YOLOv7 | Training | 1644 | 2100 | 4420.2 MB | 11886.3 MB | 340.3 MB |

The arenas of YOLOv4-tiny equal the ones the device run allocated, so the host figures are the
ones a pool would allocate. "Own values" counts every unpacked value of the plan, parameters
included; a pool allocates fewer of them because the model state owns the parameters.

What this settles:

- **The node limit is not the cause.** The largest graph has 2314 nodes against a limit of 8192,
  and every plan overlaps live ranges.
- **The arenas of a YOLOv4 step are 6.2 GB**, 5.4 GB for training and 0.7 GB for the probe.
  A YOLOv7 step asks for 5.1 GB.

## An estimate of the full-size scratch

Scratch is not part of a plan, so the host run does not report it. On YOLOv4-tiny scratch is
0.34 of the training arena and 0.62 of the probe arena. If YOLOv4 keeps those ratios, its scratch
is about 1.9 GB and 0.4 GB, 2.3 GB together. This is an estimate, not a measurement.

With it, a YOLOv4 step is about 6.2 GB of arenas, 2.3 GB of scratch and up to 0.8 GB of own
values, 9.3 GB before the weights and the optimizer's state. The measured commit was 11.1 GB.

Packing the scratch would bring that estimate to about 7 GB plus whatever the scratch still
needs inside the arena. The default limit is 7.6 GB. The work is needed for the 12 failing
cases, and by this estimate it is not certain to be enough alone. The probe pool, about 1.2 GB
with its scratch, is the next largest part; whether a step can do without it is not examined.

## Not measured

- The scratch of the full-size models. They do not fit under the default limit and are not run.
- What the remaining 1.8 GB of the 11.1 GB commit is.

## Build

The first build of task 1's code failed: `tests/vision_support/darknet_training.cpp` added a
`std::string_view` to a string literal. The two lines now build a `std::string` first. The
build after that passed.
