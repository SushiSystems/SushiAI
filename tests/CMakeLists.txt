# CMakeLists.txt
# SushiAI - https://github.com/SushiSystems/SushiAI
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under the Apache License 2.0. See LICENSE.

# ============================================================================
# SushiAI test pipeline
#
# A case's size comes from its GoogleTest suite-name prefix, Small_, Medium_ or
# Large_, and nothing else; cmake/SushiAITestProgram.cmake reads it and gives the
# case its label and timeout. A suite without one of these prefixes fails the
# configure.
#
# One test is not a GTest case: the `sa demo mlp` smoke test at the bottom runs
# the demo binary itself, because the thing being guarded is the demo's printed
# contract and no in-process test can see that. It carries the `medium`
# label explicitly.
#
# This whole directory is behind SA_BUILD_TESTS (root CMakeLists.txt).
# ============================================================================

cmake_minimum_required(VERSION 3.22) # gtest_discover_tests(TEST_FILTER ...)

# GTest::gtest is found in modules/CMakeLists.txt, which the internal modules need first.
include(GoogleTest)
include(SushiAITestProgram)

# The tests of each module, under tests/<tier>/<module>/; one program per tier.
add_subdirectory(core)
add_subdirectory(layers)
add_subdirectory(applications)
if(SA_BUILD_VISION)
    add_subdirectory(vision)
endif()
if(SA_BUILD_LANGUAGE)
    add_subdirectory(language)
endif()
# Every test folder has recorded its object library; each tier links them into one program.
sushiai_add_tier_test_programs()

# --- The `sa demo mlp` smoke test ---
# A demo whose output nobody checks rots into a lie, and this one's output is
# quoted in docs/guides/DEMO.md. The run is deliberately tiny (2 steps on the
# synthetic fallback, pointed at a directory that cannot hold MNIST) so it
# costs about as much as one GTest case.
#
# The assertion is PASS_REGULAR_EXPRESSION rather than the exit code because
# the compile-count line IS the contract - and CTest ignores a test's exit code
# once that property is set, so a bare exit-0 check would be silently dropped
# anyway. `sa demo mlp`'s exit status is covered from the CLI side, in
# cli/tests/test_demo.py.
#
# The regex's trailing [^0-9] is load-bearing: CMake's regex has no multiline
# mode, so `$` would anchor to the end of the whole output rather than the end
# of the line, and without it "compiles : 12" would satisfy a bare "...: 1".
# The demo reads MNIST, so it exists only in a build with the vision domain.
if(TARGET sushiai_demo_mlp)
    add_test(NAME Medium_Demo.MLPRunsAndReportsOneCompile
        COMMAND sushiai_demo_mlp
            --epochs 1 --batches 2 --batch-size 8 --hidden 16
            --features 6 --classes 4 --mnist-dir does-not-exist
    )
    set_tests_properties(Medium_Demo.MLPRunsAndReportsOneCompile PROPERTIES
        LABELS "medium"
    TIMEOUT 300
        PASS_REGULAR_EXPRESSION "compiles[ \t]*:[ \t]*1[^0-9]"
    )
endif()
