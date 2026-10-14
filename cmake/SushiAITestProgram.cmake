# SushiAITestProgram.cmake
# SushiAI - https://github.com/SushiSystems/SushiAI
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.

# sushiai_add_test_program() builds the one test program of a module or an
# application, so every program is linked, labelled and placed the same way.

include(GoogleTest)

if(NOT DEFINED SUSHIAI_TIER_ORDER)
    include(SushiAILayers)
endif()

# The suite prefix of a GTest case and the ctest label it carries. A case
# whose suite has none of these prefixes is discovered by no block and never
# runs; cli/sushiai/services/project.py selects the same labels.
set(SUSHIAI_TEST_LABELS
    Unit         unit
    Integration  integration
    Regression   regression
)

# sushiai_add_test_program(OWNER <module key> | APPLICATION <name>
#     SOURCES test_<name>.cpp ... [DEPENDS <module key> ...]
#     [LINKS <target> ...])
#
# Called from tests/<tier>/<module>/CMakeLists.txt with the key the module has
# in SUSHIAI_MODULE_TIERS, or from tests/applications/<name>/CMakeLists.txt.
# The program is sushiai_<tier>_<module>_tests for a module and
# sushiai_<name>_tests for an application. It links the library of the
# owner's tier, GTest, the internal module engine_fixture, which holds main(),
# and every module DEPENDS names.
# Its cases are labelled by suite prefix, and SA_TEST_FIXTURE_DIR names the
# fixtures/ folder beside its CMakeLists.txt.
function(sushiai_add_test_program)
    cmake_parse_arguments(ARG "" "OWNER;APPLICATION" "SOURCES;DEPENDS;LINKS" ${ARGN})

    if(NOT ARG_SOURCES OR (NOT ARG_OWNER AND NOT ARG_APPLICATION))
        message(FATAL_ERROR
            "sushiai_add_test_program: SOURCES and one of OWNER or APPLICATION are required.")
    endif()

    if(ARG_OWNER)
        sushiai_lookup_pair(SUSHIAI_MODULE_TIERS "${ARG_OWNER}" tier)
        if("${tier}" STREQUAL "")
            message(FATAL_ERROR
                "sushiai_add_test_program: '${ARG_OWNER}' is not in SUSHIAI_MODULE_TIERS "
                "(cmake/SushiAILayers.cmake).")
        endif()
        sushiai_lookup_pair(SUSHIAI_TIER_LIBRARY "${tier}" library)

        string(REPLACE "/" "_" flat "${ARG_OWNER}")
        list(FIND SUSHIAI_DOMAIN_TIERS "${tier}" domain_position)
        if(domain_position EQUAL -1)
            set(program "sushiai_${tier}_${flat}_tests")
        else()
            set(program "sushiai_${flat}_tests")
        endif()
    else()
        set(library sushiai)
        set(program "sushiai_${ARG_APPLICATION}_tests")
    endif()

    add_executable(${program} ${ARG_SOURCES})

    sushiai_module_target(engine_fixture fixture_target)
    target_link_libraries(${program} PRIVATE ${library} GTest::gtest ${fixture_target})

    foreach(dependency IN LISTS ARG_DEPENDS)
        if(ARG_OWNER)
            sushiai_check_module_edge("${ARG_OWNER}" "${dependency}")
        endif()
        sushiai_module_target("${dependency}" dependency_target)
        target_link_libraries(${program} PRIVATE ${dependency_target})
    endforeach()

    foreach(link IN LISTS ARG_LINKS)
        target_link_libraries(${program} PRIVATE ${link})
    endforeach()

    target_compile_definitions(${program} PRIVATE
        SA_TEST_FIXTURE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/fixtures")

    sushiai_apply_build_policy(${program})

    set(labels ${SUSHIAI_TEST_LABELS})
    while(labels)
        list(POP_FRONT labels prefix label)
        gtest_discover_tests(${program}
            DISCOVERY_MODE PRE_TEST
            TEST_FILTER "${prefix}_*"
            PROPERTIES LABELS "${label}")
    endwhile()

    if(WIN32)
        add_custom_command(TARGET ${program} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "$<TARGET_FILE:SushiRuntime::SushiRuntime>"
                $<TARGET_FILE_DIR:${program}>
            COMMENT "Copying the runtime DLL beside ${program}...")
    endif()
endfunction()
