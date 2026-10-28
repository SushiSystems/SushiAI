# SushiAITestProgram.cmake
# SushiAI - https://github.com/SushiSystems/SushiAI
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.

# sushiai_add_test_program() builds the tests of a module or an application, so every
# program is linked, labelled and placed the same way. A module's tests build an object
# library, and sushiai_add_tier_test_programs() links the libraries of each tier into one
# program: a program repeats the device link of the library it links, so one program per tier
# pays it once. An application keeps a program of its own.

include(GoogleTest)

if(NOT DEFINED SUSHIAI_TIER_ORDER)
    include(SushiAILayers)
endif()

# A test's size comes from its GoogleTest suite-name prefix, Small_, Medium_ or Large_, and each
# size has its own label and timeout in seconds; see the testing skill. A test named here keeps its
# size and gains the label quarantined, which leaves it out of every gating run. Each name is
# Suite.Case and has its known issue in docs/backlog/.
set(SUSHIAI_QUARANTINED_TESTS "")

# Fails the configure when a suite in `sources` has no size prefix: it would register no test.
function(sushiai_check_suite_sizes)
    foreach(source IN LISTS ARGN)
        if(IS_ABSOLUTE "${source}")
            set(path "${source}")
        else()
            set(path "${CMAKE_CURRENT_SOURCE_DIR}/${source}")
        endif()
        file(READ "${path}" text)
        string(REGEX MATCHALL "(TEST|TEST_F|TEST_P|TYPED_TEST|TYPED_TEST_P)\\([ \t\r\n]*[A-Za-z0-9_]+"
            declarations "${text}")
        foreach(declaration IN LISTS declarations)
            string(REGEX REPLACE "^[A-Z_]+\\([ \t\r\n]*" "" suite "${declaration}")
            if(NOT suite MATCHES "^(Small|Medium|Large)_")
                message(FATAL_ERROR
                    "${source}: the suite ${suite} has no size prefix (Small_, Medium_ or "
                    "Large_); tests/README.md gives the sizes.")
            endif()
        endforeach()
    endforeach()
endfunction()

# Registers the cases of `program` in one block per size, each with its label and timeout. A
# GoogleTest filter after '-' excludes, so the quarantined cases of a size register apart.
function(sushiai_discover_sized_tests program)
    set(groups "Small_|small|60" "Medium_|medium|300" "Large_|large|900")
    foreach(group IN LISTS groups)
        string(REPLACE "|" ";" parts "${group}")
        list(GET parts 0 prefix)
        list(GET parts 1 label)
        list(GET parts 2 timeout)

        set(quarantined "")
        foreach(name IN LISTS SUSHIAI_QUARANTINED_TESTS)
            if(name MATCHES "^${prefix}")
                list(APPEND quarantined "${name}" "*/${name}")
            endif()
        endforeach()
        list(JOIN quarantined ":" quarantined_filter)

        # An instantiated suite is named <prefix>/<suite>, so its pattern carries the slash.
        set(filter "${prefix}*:*/${prefix}*")
        if(quarantined_filter)
            string(APPEND filter "-${quarantined_filter}")
            gtest_discover_tests(${program}
                DISCOVERY_MODE PRE_TEST
                TEST_FILTER "${quarantined_filter}"
                PROPERTIES LABELS "${label}\;quarantined" TIMEOUT ${timeout})
        endif()
        gtest_discover_tests(${program}
            DISCOVERY_MODE PRE_TEST
            TEST_FILTER "${filter}"
            PROPERTIES LABELS "${label}" TIMEOUT ${timeout})
    endforeach()
endfunction()

# sushiai_add_test_program(OWNER <module key> | APPLICATION <name>
#     SOURCES test_<name>.cpp ... [DEVICE_SOURCES test_<name>.cpp ...]
#     [DEPENDS <module key> ...] [LINKS <target> ...] [INCLUDES <folder> ...])
#
# Called from tests/<tier>/<module>/CMakeLists.txt with the key the module has
# in SUSHIAI_MODULE_TIERS, or from tests/applications/<name>/CMakeLists.txt.
# A module's tests build the object library sushiai_<tier>_<module>_test_objects, or
# sushiai_<module>_test_objects in a domain tier, linked into sushiai_<tier>_tests; an
# application's build the program sushiai_<name>_tests. Each links the library of the
# owner's tier, GTest, the internal module engine_fixture, which holds main(),
# and every module DEPENDS names.
# Every case carries the size its suite-name prefix gives; CTest labels it with that size and
# enforces the size's timeout. SA_TEST_FIXTURE_DIR names the
# fixtures/ folder beside its CMakeLists.txt, and SA_REPOSITORY_ROOT names the
# root of the repository. DEVICE_SOURCES names the SOURCES that instantiate a
# device kernel; only those compile for the device (cmake/SushiAISyclTarget.cmake).
function(sushiai_add_test_program)
    cmake_parse_arguments(ARG "" "OWNER;APPLICATION" "SOURCES;DEVICE_SOURCES;DEPENDS;LINKS;INCLUDES"
        ${ARGN})

    if(NOT ARG_SOURCES OR (NOT ARG_OWNER AND NOT ARG_APPLICATION))
        message(FATAL_ERROR
            "sushiai_add_test_program: SOURCES and one of OWNER or APPLICATION are required.")
    endif()

    sushiai_check_suite_sizes(${ARG_SOURCES})

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
            set(program "sushiai_${tier}_${flat}_test_objects")
        else()
            set(program "sushiai_${flat}_test_objects")
        endif()
        add_library(${program} OBJECT ${ARG_SOURCES})
        set_property(GLOBAL APPEND PROPERTY SUSHIAI_TEST_TIERS ${tier})
        set_property(GLOBAL APPEND PROPERTY SUSHIAI_TEST_OBJECTS_${tier} ${program})
        set_property(GLOBAL PROPERTY SUSHIAI_TEST_LIBRARY_${tier} ${library})
    else()
        set(library sushiai)
        set(program "sushiai_${ARG_APPLICATION}_tests")
        add_executable(${program} ${ARG_SOURCES})
    endif()

    if(ARG_DEVICE_SOURCES)
        sushiai_check_device_sources(${program} "${ARG_SOURCES}" "${ARG_DEVICE_SOURCES}")
        sushiai_apply_sycl(${program} DEVICE_SOURCES ${ARG_DEVICE_SOURCES})
    endif()

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

    if(ARG_INCLUDES)
        target_include_directories(${program} PRIVATE ${ARG_INCLUDES})
    endif()

    target_compile_definitions(${program} PRIVATE
        SA_TEST_FIXTURE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/fixtures"
        SA_REPOSITORY_ROOT="${PROJECT_SOURCE_DIR}")

    sushiai_apply_build_policy(${program})

    if(ARG_APPLICATION)
        sushiai_finish_test_program(${program})
    endif()
endfunction()

# Registers the tests of `program` by size and copies the runtime and log DLLs beside it.
function(sushiai_finish_test_program program)
    sushiai_discover_sized_tests(${program})

    if(WIN32)
        add_custom_command(TARGET ${program} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "$<TARGET_FILE:SushiRuntime::shared_library>"
                "$<TARGET_FILE:SushiLog::SushiLog>"
                $<TARGET_FILE_DIR:${program}>
            COMMENT "Copying the runtime and log DLLs beside ${program}...")
    endif()
endfunction()

# sushiai_add_tier_test_programs()
#
# Builds sushiai_<tier>_tests from the object libraries sushiai_add_test_program() recorded for
# each tier. An object library hands on the links of what it links but not the objects of
# another object library, so the module object libraries its members link are linked here too.
# Called once, after every test folder is read.
function(sushiai_add_tier_test_programs)
    get_property(tiers GLOBAL PROPERTY SUSHIAI_TEST_TIERS)
    list(REMOVE_DUPLICATES tiers)
    sushiai_module_target(engine_fixture fixture_target)
    foreach(tier IN LISTS tiers)
        get_property(objects GLOBAL PROPERTY SUSHIAI_TEST_OBJECTS_${tier})
        get_property(library GLOBAL PROPERTY SUSHIAI_TEST_LIBRARY_${tier})
        set(program sushiai_${tier}_tests)
        add_executable(${program})
        target_link_libraries(${program} PRIVATE ${objects} ${library} GTest::gtest ${fixture_target})
        set(linked_objects "")
        foreach(object IN LISTS objects)
            get_target_property(links ${object} LINK_LIBRARIES)
            foreach(link IN LISTS links)
                if(TARGET ${link})
                    get_target_property(link_type ${link} TYPE)
                    if(link_type STREQUAL "OBJECT_LIBRARY")
                        list(APPEND linked_objects ${link})
                    endif()
                endif()
            endforeach()
        endforeach()
        list(REMOVE_DUPLICATES linked_objects)
        target_link_libraries(${program} PRIVATE ${linked_objects})
        sushiai_apply_sycl(${program})
        sushiai_apply_build_policy(${program})
        sushiai_finish_test_program(${program})
    endforeach()
endfunction()
