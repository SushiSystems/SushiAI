# Test tree

## Remaining work

| ID | Pri | State | Title | Done when | Blocked by | Source | Opened |
| --- | --- | --- | --- | --- | --- | --- | --- |
| RW-116 | P2 | unmeasured | Run the Darknet parity scenarios | the scenarios of YOLOv4-tiny, YOLOv7-tiny, YOLOv3, YOLOv4 and YOLOv7 in `tests/vision/importers/test_darknet_training/` and `test_darknet_import/` have run… |  | `docs/agent/2026_10_09_DARKNET_YOLO/REPORT.md` §32 | 2026-10-25 |
| RW-117 | P2 | open | Name every suite after its unit | `Small_Diagnostics` in `test_backward.cpp` and `test_builder.cpp` is named after the unit each file tests |  |  | 2026-10-19 |
| RW-118 | P2 | open | Give every test folder a README | the 28 test folders that hold sources and no README have one, and each README with a test-file table lists its pages |  |  | 2026-10-25 |
| RW-119 | P2 | open | Move three tests into their mirror files | the cases of `check.hpp`, `load_sample_mask.hpp` and `perplexity.hpp` leave `test_status.cpp`, `test_fit_mask.cpp` and `test_next_token_targets.cpp` |  |  | 2026-10-19 |
| RW-159 | P2 | open | Build the CLI seams the M3 resizes need and move the colour-name test to a checker | the CLI tests named in the test review decision 16 (REPORT.md as of `a87d12c8`) run small and the raw rich colour rule is a checker under tools/ |  | `cli/tests/test_config.py` | 2026-11-07 |
| RW-168 | P2 | open | Give every test a name and suite that check_test_odr accepts | check_test_odr.py prints nothing for this repository and its known findings are gone |  |  | 2026-11-11 |
| RW-169 | P2 | open | Bring the one-step cluster run test back under the medium limit | Medium_ClusterRun.RegressionOneStepFromAFileMatchesTheSingleMachineStep runs under 5 s alone and leaves tests/budget_exemptions.toml |  | `tests/applications/programs/source/test_cluster_run.cpp` | 2026-11-11 |
| RW-175 | P2 | open | Say in the test summary how many cases a missing reference directory skipped | A run without SA_REFERENCE_DATA_DIR prints the count of skipped Darknet cases and the variable, and the gating line does not read as all passed |  | `modules/core/darknet_reference/README.md` | 2026-11-14 |
| RW-176 | P1 | open | Find why four core medium shards abort on the Linux CI job | sushiai_core_tests.medium_shard_5of13 to 8of13 finish on the Linux job, or the cause of the abort is named in a known issue |  | `.github/workflows/ci.yml` | 2026-11-14 |
| RW-177 | P2 | open | Judge the time limits of a sanitizer run with a sanitizer factor | Under address-undefined and thread the time reader multiplies each limit by a factor its settings name, and the sanitizer jobs pass their time gate |  | `tools/testing/test_budget.py` | 2026-11-14 |
| RW-178 | P2 | open | Bring the cases that broke a limit on the CI runners within it | The two DetectAnchorReview cases, Yolov5nBiasInit LeavesTheBoxEntriesAtZero and DropoutMask TheKeptFraction run within their limits on CI |  | `tests/budget_exemptions.toml` | 2026-11-14 |

## Known issues

| ID | Origin | Symptom | Workaround | Fix | Source | Opened |
| --- | --- | --- | --- | --- | --- | --- |
| KI-19 | dependency | SushiCore: sa test --filter with a regex becomes a GoogleTest filter that matches no test | Run the program with sa run PROGRAM -- --gtest_filter=PATTERN |  | `cli/sushiai/services/project.py` | 2026-11-14 |
| KI-20 | dependency | SushiCore: a rerun that writes no file crashes the time reader on its results folder | none |  | `.github/workflows/ci.yml` | 2026-11-14 |
