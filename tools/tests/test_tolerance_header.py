# test_tolerance_header.py
# SushiSkills - https://github.com/SushiSystems/SushiSkills
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under the Apache License 2.0. See LICENSE.
"""Compiles tools/testing/support/tolerance.hpp into a self-check and runs it.

The compiler is $CXX, or the first of c++, g++, clang++ and cl on the path. With none, the test
is skipped, never passed.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

K_HEADER = Path(__file__).resolve().parents[1] / "testing" / "support" / "tolerance.hpp"
K_COMPILERS = ("c++", "g++", "clang++", "cl")
K_CHECK = r"""
#include "tolerance.hpp"
#include <cstdio>
#include <limits>

using namespace TestSupport;

static int failures = 0;

static void expect(bool condition, char const* promise)
{
    std::printf("%s %s\n", condition ? "PASS" : "FAIL", promise);
    if (!condition)
    {
        ++failures;
    }
}

int main()
{
    double const nan = std::numeric_limits<double>::quiet_NaN();
    double const inf = std::numeric_limits<double>::infinity();
    expect(unit_roundoff<float>() == 0x1p-24, "unit roundoff of float is 2^-24");
    expect(unit_roundoff<double>() == 0x1p-53, "unit roundoff of double is 2^-53");
    expect(measure_error(1.0, 1.0, 0.0).is_within(), "an exact result passes a zero bound");
    expect(!measure_error(1.0, 1.0 + 0x1p-52, 0.0).is_within(), "an inexact result fails a zero bound");
    double const u = unit_roundoff<float>();
    expect(measure_sum_error<float>(1.0 + 16 * 4 * u, 1.0, 1.0, 4).is_within(), "an error at the threshold passes");
    expect(!measure_sum_error<float>(1.0 + 17 * 4 * u, 1.0, 1.0, 4).is_within(), "an error past the threshold fails");
    expect(measure_sum_error<float>(1.0 + 4 * u, 1.0, 1.0, 4).ratio == 1.0, "the ratio is error over n u sum");
    expect(!measure_error(nan, 1.0, 1.0).is_within(), "a NaN where a number is expected fails");
    expect(!measure_error(1.0, nan, 1.0).is_within(), "a number where a NaN is expected fails");
    expect(measure_error(nan, nan, 0.0).is_within(), "a NaN where a NaN is expected passes");
    expect(measure_error(inf, inf, 0.0).is_within(), "a matching infinity passes");
    expect(!measure_error(inf, 1.0, 1.0).is_within(), "an infinity where a number is expected fails");
    expect(measure_relative_error<double>(2.0 + 0x1p-51, 2.0, 2).ratio == 1.0,
           "the relative bound scales with the reference");
    expect(measure_error(1.5, 1.0, 1.0, 0.25).describe() == "error 0.5, bound 1, ratio 0.5 of 0.25",
           "describe names error, bound, ratio and threshold");
    return failures == 0 ? 0 : 1;
}
"""


K_TEST_SIZE = "medium"


def find_compiler() -> str | None:
    """Returns the compiler to use, or None when the machine has none."""
    named = os.environ.get("CXX")
    if named:
        return named
    for candidate in K_COMPILERS:
        found = shutil.which(candidate)
        if found:
            return found
    return None


def _run_self_check(compiler: str) -> tuple[dict[str, str], str]:
    """Builds the self-check with the compiler and runs it once.

    Returns each promise's result, PASS or FAIL, and the output of a build or run that did not finish.
    """
    with tempfile.TemporaryDirectory() as folder:
        source = Path(folder) / "check.cpp"
        source.write_text(K_CHECK, encoding="utf-8")
        program = Path(folder) / "check.exe"
        if Path(compiler).stem.lower() == "cl":
            command = [compiler, "/nologo", "/std:c++17", "/EHsc", f"/I{K_HEADER.parent}", str(source),
                       f"/Fe{program}", f"/Fo{folder}\\"]
        else:
            command = [compiler, "-std=c++17", f"-I{K_HEADER.parent}", str(source), "-o", str(program)]
        built = subprocess.run(command, capture_output=True, text=True, cwd=folder, check=False)
        if built.returncode != 0:
            return {}, built.stdout + built.stderr
        ran = subprocess.run([str(program)], capture_output=True, text=True, check=False)
    results = {}
    for line in ran.stdout.splitlines():
        status, _, promise = line.partition(" ")
        results[promise] = status
    return results, "" if ran.returncode == 0 else ran.stdout + ran.stderr


class ToleranceHeaderTest(unittest.TestCase):
    """Compiles the self-check once and checks each promise of the header against its result."""

    @classmethod
    def setUpClass(cls) -> None:
        compiler = find_compiler()
        if compiler is None:
            raise unittest.SkipTest("no C++ compiler on this machine; set CXX")
        cls.results, cls.output = _run_self_check(compiler)

    def assert_holds(self, promise: str) -> None:
        """Fails unless the self-check reports the promise as PASS; the output explains a build that failed."""
        self.assertEqual(self.results.get(promise), "PASS", self.output or promise)

    def test_unit_roundoff_of_float_is_2_pow_minus_24(self) -> None:
        self.assert_holds("unit roundoff of float is 2^-24")

    def test_unit_roundoff_of_double_is_2_pow_minus_53(self) -> None:
        self.assert_holds("unit roundoff of double is 2^-53")

    def test_an_exact_result_passes_a_zero_bound(self) -> None:
        self.assert_holds("an exact result passes a zero bound")

    def test_an_inexact_result_fails_a_zero_bound(self) -> None:
        self.assert_holds("an inexact result fails a zero bound")

    def test_an_error_at_the_threshold_passes(self) -> None:
        self.assert_holds("an error at the threshold passes")

    def test_an_error_past_the_threshold_fails(self) -> None:
        self.assert_holds("an error past the threshold fails")

    def test_the_ratio_is_error_over_n_u_sum(self) -> None:
        self.assert_holds("the ratio is error over n u sum")

    def test_a_nan_where_a_number_is_expected_fails(self) -> None:
        self.assert_holds("a NaN where a number is expected fails")

    def test_a_number_where_a_nan_is_expected_fails(self) -> None:
        self.assert_holds("a number where a NaN is expected fails")

    def test_a_nan_where_a_nan_is_expected_passes(self) -> None:
        self.assert_holds("a NaN where a NaN is expected passes")

    def test_a_matching_infinity_passes(self) -> None:
        self.assert_holds("a matching infinity passes")

    def test_an_infinity_where_a_number_is_expected_fails(self) -> None:
        self.assert_holds("an infinity where a number is expected fails")

    def test_the_relative_bound_scales_with_the_reference(self) -> None:
        self.assert_holds("the relative bound scales with the reference")

    def test_describe_names_error_bound_ratio_and_threshold(self) -> None:
        self.assert_holds("describe names error, bound, ratio and threshold")


if __name__ == "__main__":
    unittest.main()
