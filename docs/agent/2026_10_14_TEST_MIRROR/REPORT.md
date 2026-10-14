# Test mirror: report

**Status:** Done, except the cases listed under "Not run".

## What moved

Every C++ test of SushiAI sits in the mirror of the source. A module's tests are in
`tests/<tier>/<module>/` and build into `sushiai_<tier>_<module>_tests`. The tests of
`applications/programs` are in `tests/applications/programs/` and build into
`sushiai_programs_tests`. The programs `sushiai_tests`, `sushiai_vision_tests`,
`sushiai_language_tests` and `sushiai_program_tests` are gone, and so are `tests/unit/`,
`tests/integration/`, `tests/regression/`, `tests/common/` and `tests/fixtures/`.

| Step | Commit |
| --- | --- |
| Test mirror checker | `d82c6d8` |
| Shared test code into sixteen internal modules | `db82502` |
| Core module tests | `08dba3e` |
| Checker copied again from SushiSkills, accepting tier include paths | `c9414f7` |
| Layers module tests | `82a1e91` |
| Cross-module tests of core and layers | `4f7b4e1` |
| Vision and language tests, public position feeds, fixture split | `07d8b07` |
| Program tests, manuals, references, changelog | this commit |

A file that tested several headers was split by case, and each case went to the test file of
the unit it checks. A unit whose tests do not fit one file got a test folder:
`tests/core/graph/test_lowering/`, `tests/layers/train/test_trainer/`,
`tests/vision/importers/test_darknet_import/` and `test_darknet_training/`. Suites that a split
left in two files of one program were renamed after their unit, because GTest refuses one suite
name over two fixture classes.

The owner ruled four points on the way:

- shared code goes into internal modules named for what they hold;
- untested headers are listed, not given tests;
- the position feeds of `language/generation` and the seeded sequence of `vision/detect`
  become public headers;
- the YOLO26 reference set is split between the tests that read each file.

## Measured

`check_test_mirror.py` before and after:

| Rule | Before | After |
| --- | --- | --- |
| `rule_untested_unit` | 372 | 121 |
| `rule_orphan_test` | 78 | 2 |
| `rule_mirror_folders` | 9 | 4 |

The two orphans are the registration tests, and the four folders are the empty
`tests/functional/` the work did not create. Both are explained in
`docs/design/REMAINING_WORK.md` under "Test tree".

Cases per program, each run through `sa run` after its move:

| Tier | Cases | Before the move |
| --- | --- | --- |
| core and layers | 1933 | 1933 in `sushiai_tests` |
| vision, without the six heavy suites | 1473 | 1473 in `sushiai_vision_tests`, same filter |
| language | 263 | 263 in `sushiai_language_tests` |

## Not run

The suites `Integration_DarknetYolov3Training`, `Integration_DarknetYolov4Training`,
`Integration_DarknetYolov7Training` and the three `Forward` suites of the same models need
several gigabytes of device memory. The machine stopped a full vision run on 2026-10-14, so
they were left out of every run after the move.
`Integration_DarknetYolov7Training.TheCostOfOneBatchFallsUnderATenth` failed before the work
began and is in `docs/reference/KNOWN_ISSUES.md`.
