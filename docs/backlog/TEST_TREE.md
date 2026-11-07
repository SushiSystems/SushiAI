# Test tree

## Remaining work

| ID | Pri | State | Title | Done when | Blocked by | Source | Opened |
| --- | --- | --- | --- | --- | --- | --- | --- |
| RW-116 | P2 | unmeasured | Run the Darknet parity scenarios | the scenarios of YOLOv4-tiny, YOLOv7-tiny, YOLOv3, YOLOv4 and YOLOv7 in `tests/vision/importers/test_darknet_training/` and `test_darknet_import/` have run… |  | `docs/agent/2026_10_09_DARKNET_YOLO/REPORT.md` §32 | 2026-10-25 |
| RW-117 | P2 | open | Name every suite after its unit | `Small_Diagnostics` in `test_backward.cpp` and `test_builder.cpp` is named after the unit each file tests |  |  | 2026-10-19 |
| RW-118 | P2 | open | Give every test folder a README | the 28 test folders that hold sources and no README have one, and each README with a test-file table lists its pages |  |  | 2026-10-25 |
| RW-119 | P2 | open | Move three tests into their mirror files | the cases of `check.hpp`, `load_sample_mask.hpp` and `perplexity.hpp` leave `test_status.cpp`, `test_fit_mask.cpp` and `test_next_token_targets.cpp` |  |  | 2026-10-19 |
| RW-151 | P2 | blocked | Set the size timeouts to the testing skill's 1 s, 10 s and 120 s | SushiAITestProgram.cmake, cli/tests/conftest.py and tests/README.md give small 1 s, medium 10 s and large 120 s, and the gating run passes | external: the M jobs of the test review resize the tests that run longer | `docs/agent/2026_11_05_TEST_REVIEW/REPORT.md` | 2026-11-06 |
| RW-152 | P3 | open | Explain why yolov4-tiny's second mask differs by one anchor | DiffersFromTheSecondMaskOfYolov4TinyByOneAnchor names the source of the difference or the anchor is fixed |  | `docs/agent/2026_11_05_TEST_REVIEW/REPORT.md` | 2026-11-06 |
| RW-158 | P2 | open | Build the program seams the M2 resizes need, then resize those tests to Small_ | the thirteen Medium_ program tests listed in REPORT.md decision 15 run as Small_ through value functions or in-memory ports |  | `docs/agent/2026_11_05_TEST_REVIEW/REPORT.md` | 2026-11-07 |
| RW-159 | P2 | open | Build the CLI seams the M3 resizes need and move the colour-name test to a checker | the CLI tests named in REPORT.md decision 16 run small and the raw rich colour rule is a checker under tools/ |  | `docs/agent/2026_11_05_TEST_REVIEW/REPORT.md` | 2026-11-07 |
| RW-160 | P2 | open | Build the darknet_reference and distribution seams the M7 and M9 records need | the tests named in REPORT.md decision 17 are applied as their records say, with no runtime made by a test |  | `docs/agent/2026_11_05_TEST_REVIEW/REPORT.md` | 2026-11-07 |
| RW-161 | P1 | open | Put Session, CompiledStep and record_and_run on a host and a runtime ExecutionBackend | No sushiai source calls Engine::runtime() or is_host(); the StepTasks contract runs Small_ on the host adapter, Medium_ on the runtime one |  | `docs/agent/2026_11_05_TEST_REVIEW/REPORT.md` | 2026-11-07 |

## Known issues

| ID | Origin | Symptom | Workaround | Fix | Source | Opened |
| --- | --- | --- | --- | --- | --- | --- |
