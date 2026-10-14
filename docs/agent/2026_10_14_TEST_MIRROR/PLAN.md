# Test mirror

**Status:** Open, phase 1 of 8

Moves every test of SushiAI into the mirror of `repository-layout/TESTS.md`: one test file per
public header at `tests/<tier>/<module>/test_<name>.cpp`, one test program per module, the kind
of a test as a label, and the code the tests share in internal modules named for what they do.
The owner chose on 2026-10-14: existing tests are moved, split and renamed, and headers with no
test are listed in `docs/design/REMAINING_WORK.md`, not given new tests; one program per module;
the work runs inline in one session.

## Rules this plan follows

- `refactor` in `testing`: no new test, every existing test green before and after.
- A test that crosses modules goes to the highest-tier module it uses, in the test file of the
  unit whose behaviour it checks.
- A header of a domain module sits at `<SushiAI/<tier>/<module>/<name>.hpp>`, where the skill has
  `<SushiAI/<module>/<name>.hpp>`. Its test goes to `tests/<tier>/<module>/test_<name>.cpp`;
  `check_test_mirror.py` reports those as untested until the owner rules on the header paths
  (`api-stability`). The count is reported as a known deviation.
- Suite prefixes `Unit_`, `Integration_` and `Regression_` stay and set the ctest label.

## Starting point

329 C++ test files with 3698 cases: 253 under `modules/<tier>/<module>/tests/`, the rest under
`tests/unit/`, `tests/integration/`, `tests/regression/`, plus shared code in `tests/common/`,
`tests/vision_support/` and `tests/integration/decoder_model.hpp`. Five programs:
`sushiai_tests`, `sushiai_vision_tests`, `sushiai_language_tests`, `sushiai_program_tests` and
the package test. `check_test_mirror.py` reports 372 untested units, 78 orphan tests and 9
folders with no counterpart.

## Shared test code

Each part is an internal module in the lowest tier its dependencies allow. Its headers are
never installed, it is built only with the tests, and its README says so first.

| Module | Tier | From |
| --- | --- | --- |
| `engine_fixture` | core | `test_environment.hpp`, `test_main.cpp`, `EngineTest` |
| `failure_expectations` | core | `test_status.hpp`, `test_throws.hpp` |
| `tensor_values` | core | `write_tensor`, `read_tensor`, `read_tensor_bytes`, `first_difference` |
| `seeded_noise` | core | `HostNoise` |
| `scratch_files` | core | `ScratchFile` |
| `model_text` | core | `model_text.hpp` |
| `layer_traces` | layers | `MinimalCase`, `TraceSummary` and their functions |
| `reference_tensors` | layers | `reference_fixture.hpp` |
| `classifier_objective` | layers | `classifier_objective.hpp` |
| `vision/darknet_reference` | vision | `darknet_reference.*` |
| `vision/darknet_import_cases` | vision | `darknet_import.*` |
| `vision/darknet_forward_cases` | vision | `darknet_forward.*` |
| `vision/darknet_training_cases` | vision | `darknet_training.*` |
| `language/decoder_models` | language | `decoder_model.hpp` |

## Phases

1. Copy `check_test_mirror.py` into `tools/testing/` and record the starting report.
2. CMake: an `INTERNAL` module kind that is never installed and never joins a tier library, and
   `sushiai_add_test_program`, one program per module with the labels from the suite prefix.
   The CLI's suite labels stay.
3. The shared test code into the modules above.
4. Core tier: each module's tests to `tests/core/<module>/`, renamed and merged per header.
5. Layers tier, then the files of `tests/unit/`, `tests/integration/` and `tests/regression/`
   that belong to core or layers.
6. Vision tier and its cross-module files, the Darknet suites among them.
7. Language tier and its cross-module files; the programs' tests to `tests/applications/`.
8. The test manuals into `tests/<owner>/README.md`, fixtures into `tests/<owner>/fixtures/`,
   the untested headers into `REMAINING_WORK.md`, the old folders removed, the changelog line.

Each phase ends with one build and the tests of what it moved, run through `sa`, and one commit.
