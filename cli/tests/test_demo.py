"""End-to-end check that `sa demo mlp` still runs, and still compiles once.

Unlike the other tests here this one is not pure logic: it shells out to the
installed `sa` and to a real build tree, because the thing worth guarding is
the whole path - CLI -> demo binary -> printed contract. A demo nobody runs
rots into a lie, and this one's output is quoted in docs/README.md.

It skips (rather than fails) when `sa` is not installed or the tree has not
been built, since neither is true in a clean checkout. The same claim is
pinned inside the C++ suite, where it always runs: see the
`Integration_Demo.MlpRunsAndReportsOneCompile` CTest entry in
tests/CMakeLists.txt.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

# cli/tests/test_demo.py -> cli/tests -> cli -> <repo root>
REPO_ROOT = Path(__file__).resolve().parents[2]

# The demo prints this twice: once in the run report and once in the banner.
# Either match is enough; what matters is that the number is 1.
_COMPILE_ONCE = re.compile(r"compiles\s*:\s*1\b")

# Deliberately tiny - two steps on the synthetic fallback. --mnist-dir points
# at a directory that cannot exist so the fallback is the path under test
# regardless of whether the developer happens to have MNIST downloaded.
_DEMO_ARGS = [
    "--epochs", "1",
    "--batches", "2",
    "--batch-size", "8",
    "--hidden", "16",
    "--features", "6",
    "--classes", "4",
    "--mnist-dir", "does-not-exist",
]


@pytest.mark.skipif(shutil.which("sa") is None, reason="the `sa` CLI is not on PATH")
@pytest.mark.skipif(not (REPO_ROOT / "build").is_dir(),
                    reason="no build tree — run `sa build` first")
def test_sa_demo_mlp_runs_and_reports_one_compile():
    """`sa demo mlp` exits 0 and reports exactly one compile."""
    completed = subprocess.run(
        ["sa", "demo", "mlp", *_DEMO_ARGS],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=900,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert _COMPILE_ONCE.search(completed.stdout), completed.stdout
