# ============================================================================
# SushiAI test pipeline
#
# AI-0 scope reset: two tests only, proving the skeleton actually works -
# a version sanity check and a real Engine smoke test through both sibling
# dependencies. Grows with AI-1 onward. Mirrors sushiblas's Unit_*/Integration_*
# CTest-label convention (gtest_discover_tests TEST_FILTER) so `sa test`
# eventually maps onto it the same way `sb test` does.
# ============================================================================

cmake_minimum_required(VERSION 3.22) # gtest_discover_tests(TEST_FILTER ...)

find_package(GTest CONFIG REQUIRED)
include(GoogleTest)

add_executable(sushiai_tests
    functional/unit/test_version.cpp
    functional/integration/test_engine_smoke.cpp
)

target_link_libraries(sushiai_tests PRIVATE
    sushiai
    GTest::gtest
    GTest::gtest_main
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
