# tests

The tests mirror the source. The test of a public header
`modules/<tier>/<module>/include/SushiAI/.../<name>.hpp` is
`tests/<tier>/<module>/.../test_<name>.cpp`. The test of a header under
`applications/<app>/<path>/` is `tests/applications/<app>/<path>/test_<name>.cpp`. A unit whose
tests do not fit one file keeps them in a folder named like that file, such as
`tests/layers/train/test_trainer/`, one file per behaviour.

## Programs

Each module with tests builds one program through `sushiai_add_test_program`
(`cmake/SushiAITestProgram.cmake`). A module builds `sushiai_<tier>_<module>_tests` and an
application `sushiai_<app>_tests`. Run one through the
CLI, for example `sa run sushiai_core_graph_tests -- --gtest_filter=Small_Ir.*`.

A case's suite prefix gives its size, which is its ctest label and sets its timeout: `Small_` is
`small` (60 s), `Medium_` is `medium` (300 s) and `Large_` is `large` (900 s). A small case uses
one process and no device, a medium one a CPU SYCL device, threads or files, and a large one a
real GPU or a run longer than medium allows. A suite without one of these prefixes fails the
configure. A case named in `SUSHIAI_QUARANTINED_TESTS` also carries the label `quarantined`.

The code the tests of several modules share lives in internal modules, such as
`modules/core/engine_fixture/`, which holds `main()`. They are never installed. Each one's README
says what it holds.

## Fixtures and environment

A program defines `SA_TEST_FIXTURE_DIR` as the `fixtures/` folder beside its `CMakeLists.txt`
and `SA_REPOSITORY_ROOT` as the root of the repository. The YOLO26 reference tensors are not
committed. `tools/reference/export_yolo26_reference.py` writes them into the fixture folders of
`tests/layers/nn/`, `tests/vision/detect/` and `tests/vision/registration/`, and a case whose
file is absent reports itself skipped.

The Darknet cases of `sushiai_vision_importers_tests` read their dumps from
`SA_REFERENCE_DATA_DIR`. The training cases of the full YOLOv3, YOLOv4 and YOLOv7 models need
several gigabytes of device memory.

## The SaTest alias

Inside a `TEST_F` body the fixture's base class injects the name `Test`, so `Test` alone resolves
to `::testing::Test`. A file that uses `TEST_F` declares `namespace SaTest = SushiAI::Test` to
reach the helpers in `SushiAI::Test`.
