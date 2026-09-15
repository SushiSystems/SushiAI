# Task 2 Report: CLI Doxyfile Service Update

## Completed Work

1. Inspected [project.py](file:///D:/Projects/sushiai/cli/sushiai/services/project.py#L553-L563) in `cli/sushiai/services/project.py`.
2. Updated `doxygen()` in [project.py](file:///D:/Projects/sushiai/cli/sushiai/services/project.py#L558-L561) to look up `root / "docs" / "Doxyfile"` first and fall back to `root / "Doxyfile"` when `docs/Doxyfile` does not exist.
3. Verified `sa doxygen` execution.
4. Executed `pytest cli/tests -q`.

## Incomplete Work

- No git commit created (retained working tree changes as instructed).
- No other files touched.

## Code Diff

```diff
diff --git a/cli/sushiai/services/project.py b/cli/sushiai/services/project.py
index b3500f3..4c65d51 100644
--- a/cli/sushiai/services/project.py
+++ b/cli/sushiai/services/project.py
@@ -555,6 +555,8 @@ def doxygen() -> int:
     root = find_project_root()
     cfg = load_config()
     env = load_build_env(cfg, _build_dir(root))
-    doxyfile = root / "Doxyfile"
+    doxyfile = root / "docs" / "Doxyfile"
+    if not doxyfile.exists():
+        doxyfile = root / "Doxyfile"
     (root / "docs").mkdir(exist_ok=True)
     return _DRIVER.doxygen(cfg, doxyfile, root, env)
```

## Verification Output

### `sa doxygen`

```
───────────────────────────── Doxygen Generation ──────────────────────────────
[INFO] Executing: D:/Projects/sushistack/dependencies/tools/doxygen\doxygen.EXE
docs/Doxyfile
```

### `pytest cli/tests -q` (with `ONEAPI_DEVICE_SELECTOR=opencl:cpu`)

```
..............................                                           [100%]
30 passed in 51.58s
```

All 30 collected tests pass (including the 12 core unit tests in `test_config.py` and `test_console_lazy.py`, plus `test_suites.py`, `test_demo.py`, and `test_train.py`).

### `pytest cli/tests -q` (without device selector)

```
=========================== short test summary info ===========================
FAILED cli\tests\test_demo.py::test_sa_demo_mlp_runs_and_reports_one_compile
FAILED cli\tests\test_demo.py::test_sa_demo_mlp_precision_flag_reaches_the_demo[fp32]
FAILED cli\tests\test_demo.py::test_sa_demo_mlp_precision_flag_reaches_the_demo[mixed]
FAILED cli\tests\test_demo.py::test_sa_demo_mlp_loss_scale_flag_reaches_the_demo_and_is_exact
FAILED cli\tests\test_demo.py::test_sa_demo_mlp_rejects_a_non_dyadic_loss_scale
FAILED cli\tests\test_demo.py::test_sa_demo_mlp_activation_flag_reaches_the_demo[relu]
FAILED cli\tests\test_demo.py::test_sa_demo_mlp_activation_flag_reaches_the_demo[gelu]
FAILED cli\tests\test_train.py::test_sushiai_train_runs_and_reports_one_compile
FAILED cli\tests\test_train.py::test_sushiai_inference_loads_the_checkpoint_train_wrote
9 failed, 21 passed in 9.44s
```
