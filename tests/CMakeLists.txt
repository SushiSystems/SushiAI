# ============================================================================
# SushiAI test pipeline
#
# Two tiers, distinguished by the GTest suite-name prefix and nothing else:
# `Unit_*` is device-free host-lane work (shapes, graph topology, the
# differentiation transform, the parameter walk, the gradient checker's own
# arithmetic); `Integration_*` runs the real Engine through both siblings.
#
# THE PREFIX IS THE ONLY THING THAT ASSIGNS A CTEST LABEL. A suite named
# without one is discovered by neither gtest_discover_tests block below, so it
# never runs under `sa test` and never runs in CI - it is invisible rather than
# failing. Adding a third tier means a third block here AND a matching entry in
# cli/sushiai/services/project.py's Suite enum and _SUITE_LABEL_REGEX.
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
    functional/unit/test_module.cpp
    functional/unit/test_backward.cpp
    functional/unit/test_gradcheck.cpp

    functional/integration/test_engine_smoke.cpp
    functional/integration/test_forward.cpp
    functional/integration/test_autograd_numeric.cpp
    functional/integration/test_training.cpp
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

# --- Post-Build: Copy the runtime DLL ---
if(WIN32)
    add_custom_command(TARGET sushiai_tests POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:SushiRuntime::SushiRuntime>"
            $<TARGET_FILE_DIR:sushiai_tests>
        COMMENT "Copying runtime DLL to test directory..."
    )
endif()
