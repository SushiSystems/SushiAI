# ============================================================================
# SushiAI test pipeline
#
# Three GTest tiers, distinguished by the suite-name prefix and nothing else:
# `Unit_*` is device-free host-lane work (shapes, graph topology, the
# differentiation transform, the parameter walk, the gradient checker's own
# arithmetic); `Integration_*` runs the real Engine through both siblings;
# `Regression_*` pins a guarantee whose failure is silent, so a future change
# cannot quietly take it away.
#
# THE PREFIX IS THE ONLY THING THAT ASSIGNS A CTEST LABEL to a GTest case. A
# suite named without one is discovered by none of the gtest_discover_tests
# blocks below, so it never runs under `sa test` and never runs in CI - it is
# invisible rather than failing. Adding a fourth tier means a fourth block here
# AND a matching entry in cli/sushiai/services/project.py's Suite enum and
# _SUITE_LABEL_REGEX.
#
# One test is not a GTest case: the `sa demo mlp` smoke test at the bottom runs
# the demo binary itself, because the thing being guarded is the demo's printed
# contract and no in-process test can see that. It carries the `integration`
# label explicitly.
#
# This whole directory is behind SA_BUILD_TESTS (root CMakeLists.txt).
# ============================================================================

cmake_minimum_required(VERSION 3.22) # gtest_discover_tests(TEST_FILTER ...)

find_package(GTest CONFIG REQUIRED)
include(GoogleTest)

add_executable(sushiai_tests
    common/test_main.cpp

    functional/unit/test_version.cpp
    functional/unit/test_shape.cpp
    functional/unit/test_ops.cpp
    functional/unit/test_ir.cpp
    functional/unit/test_memory_plan.cpp
    functional/unit/test_precision.cpp
    functional/unit/test_fusion.cpp
    functional/unit/test_module.cpp
    functional/unit/test_backward.cpp
    functional/unit/test_gradcheck.cpp
    functional/unit/test_diagnostics.cpp
    functional/unit/test_loss_scaling.cpp
    functional/unit/test_checkpoint_format.cpp
    functional/unit/test_layer_stack.cpp
    functional/unit/test_config.cpp
    functional/unit/test_objective.cpp

    functional/integration/test_engine_smoke.cpp
    functional/integration/test_arena.cpp
    functional/integration/test_forward.cpp
    functional/integration/test_autograd_numeric.cpp
    functional/integration/test_training.cpp
    functional/integration/test_mixed_precision.cpp
    functional/integration/test_resilience.cpp
    functional/integration/test_registries.cpp
    functional/integration/test_checkpoint.cpp

    functional/regression/test_compile_once.cpp
    functional/regression/test_determinism.cpp
    functional/regression/test_fused_optimizer.cpp
    functional/regression/test_fusion_numerics.cpp
    functional/regression/test_learning_curve.cpp
    functional/regression/test_log_softmax_stability.cpp
    functional/regression/test_loss_scaling.cpp
    functional/regression/test_mixed_precision.cpp
    functional/regression/test_static_memory.cpp
)

# GTest::gtest_main is deliberately absent: common/test_main.cpp supplies main()
# so the RuntimeContext is built once per binary rather than once per test.
# Linking both is a duplicate-symbol error at link time.
target_link_libraries(sushiai_tests PRIVATE
    sushiai
    GTest::gtest
)

sushiai_apply_build_policy(sushiai_tests)

gtest_discover_tests(sushiai_tests
    DISCOVERY_MODE PRE_TEST
    TEST_FILTER "Unit_*"
    PROPERTIES LABELS "unit"
)
gtest_discover_tests(sushiai_tests
    DISCOVERY_MODE PRE_TEST
    TEST_FILTER "Integration_*"
    PROPERTIES LABELS "integration"
)
gtest_discover_tests(sushiai_tests
    DISCOVERY_MODE PRE_TEST
    TEST_FILTER "Regression_*"
    PROPERTIES LABELS "regression"
)

# --- The `sa demo mlp` smoke test ---
# A demo whose output nobody checks rots into a lie, and this one's output is
# quoted in docs/README.md. The run is deliberately tiny (2 steps on the
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
add_test(NAME Integration_Demo.MlpRunsAndReportsOneCompile
    COMMAND sushiai_demo_mlp
        --epochs 1 --batches 2 --batch-size 8 --hidden 16
        --features 6 --classes 4 --mnist-dir does-not-exist
)
set_tests_properties(Integration_Demo.MlpRunsAndReportsOneCompile PROPERTIES
    LABELS "integration"
    PASS_REGULAR_EXPRESSION "compiles[ \t]*:[ \t]*1[^0-9]"
)

# --- Post-Build: Copy the runtime DLL ---
if(WIN32)
    add_custom_command(TARGET sushiai_tests POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:SushiRuntime::SushiRuntime>"
            $<TARGET_FILE_DIR:sushiai_tests>
        COMMENT "Copying runtime DLL to test directory..."
    )
endif()
