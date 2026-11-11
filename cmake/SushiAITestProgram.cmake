# SushiAITestProgram.cmake
# SushiAI - https://github.com/SushiSystems/SushiAI
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under the Apache License 2.0. See LICENSE.

# sushiai_add_test_program() builds the tests of a module or an application, so every
# program is linked, labelled and placed the same way. A module's tests build an object
# library, and sushiai_add_tier_test_programs() links the libraries of each tier into one
# program: a program repeats the device link of the library it links, so one program per tier
# pays it once. An application keeps a program of its own.

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

# The cases one CTest test runs, per size. A process pays its start-up (the runtime, the device
# discovery, the logger) once, so a case run alone is mostly start-up: a small case averaged
# 55 ms alone in the measured run of 2026-11-10, most of it start-up. A shard runs its share of a
# program's cases in one process, and the share is sized to the cost of a case of its size. A
# small case is a few milliseconds of work, so 300 of them are a few seconds and the run still
# spreads over every core; a medium case reaches a device or a thread and takes up to 5 s, so 25
# of them keep a shard well under a minute; a large case takes up to 120 s and runs alone.
set(SUSHIAI_SHARD_CASES_SMALL 300)
set(SUSHIAI_SHARD_CASES_MEDIUM 25)
set(SUSHIAI_SHARD_CASES_LARGE 1)

# One group per size: suite prefix, CTest label, timeout in seconds per case, cases per shard.
set(SUSHIAI_SIZE_GROUPS
    "Small_|small|1|${SUSHIAI_SHARD_CASES_SMALL}"
    "Medium_|medium|10|${SUSHIAI_SHARD_CASES_MEDIUM}"
    "Large_|large|120|${SUSHIAI_SHARD_CASES_LARGE}")

# Where each shard writes its per-case GoogleTest XML; the time reader reads the folder, so the
# per-case limits of the testing skill still apply to a case run inside a shard.
set(SUSHIAI_GTEST_RESULTS_DIR "${CMAKE_BINARY_DIR}/test_results/gtest")

# Counts the cases of the sources after `program` per size and lists their suites, and keeps both
# for `program`. A TEST_P or TYPED_TEST counts once however many instances it gets, so the count is
# a floor.
function(sushiai_record_cases program)
    foreach(group IN LISTS SUSHIAI_SIZE_GROUPS)
        string(REPLACE "|" ";" parts "${group}")
        list(GET parts 0 prefix)
        set(count_${prefix} 0)
    endforeach()
    set(suites "")
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
            list(APPEND suites "${suite}")
            if(suite MATCHES "^(Small_|Medium_|Large_)")
                math(EXPR count_${CMAKE_MATCH_1} "${count_${CMAKE_MATCH_1}} + 1")
            endif()
        endforeach()
    endforeach()
    foreach(group IN LISTS SUSHIAI_SIZE_GROUPS)
        string(REPLACE "|" ";" parts "${group}")
        list(GET parts 0 prefix)
        set_property(GLOBAL PROPERTY SUSHIAI_CASES_${program}_${prefix} ${count_${prefix}})
    endforeach()
    set_property(GLOBAL PROPERTY SUSHIAI_SUITES_${program} ${suites})
endfunction()

# Gives `program` the cases and suites of the object libraries named after it.
function(sushiai_sum_cases program)
    set(suites "")
    foreach(group IN LISTS SUSHIAI_SIZE_GROUPS)
        string(REPLACE "|" ";" parts "${group}")
        list(GET parts 0 prefix)
        set(total 0)
        foreach(object IN LISTS ARGN)
            get_property(count GLOBAL PROPERTY SUSHIAI_CASES_${object}_${prefix})
            math(EXPR total "${total} + ${count}")
        endforeach()
        set_property(GLOBAL PROPERTY SUSHIAI_CASES_${program}_${prefix} ${total})
    endforeach()
    foreach(object IN LISTS ARGN)
        get_property(object_suites GLOBAL PROPERTY SUSHIAI_SUITES_${object})
        list(APPEND suites ${object_suites})
    endforeach()
    set_property(GLOBAL PROPERTY SUSHIAI_SUITES_${program} ${suites})
endfunction()

# Sets `result` to the number of shards that hold `cases` cases at `per_shard` cases each, at
# least one.
function(sushiai_shard_count cases per_shard result)
    math(EXPR shards "(${cases} + ${per_shard} - 1) / ${per_shard}")
    if(shards LESS 1)
        set(shards 1)
    endif()
    set(${result} ${shards} PARENT_SCOPE)
endfunction()

# Sets `result` to the GoogleTest patterns of the quarantined cases of `program` that carry
# `prefix`; a name whose suite the program does not hold is left out.
function(sushiai_quarantined_patterns program prefix result)
    get_property(suites GLOBAL PROPERTY SUSHIAI_SUITES_${program})
    set(patterns "")
    foreach(name IN LISTS SUSHIAI_QUARANTINED_TESTS)
        string(REGEX REPLACE "\\..*$" "" suite "${name}")
        if(name MATCHES "^${prefix}" AND suite IN_LIST suites)
            # An instantiated suite is named <prefix>/<suite>, so its pattern carries the slash.
            list(APPEND patterns "${name}" "*/${name}")
        endif()
    endforeach()
    set(${result} "${patterns}" PARENT_SCOPE)
endfunction()

# Registers one CTest test that runs `program` with `filter`, as shard `index` of `shards`
# (GoogleTest splits the filtered cases round-robin by GTEST_TOTAL_SHARDS and GTEST_SHARD_INDEX).
# It writes its per-case XML under SUSHIAI_GTEST_RESULTS_DIR. The shuffle and its seed reach it
# through GTEST_SHUFFLE and GTEST_RANDOM_SEED of the caller's environment, and a death test starts
# its child outside the sharding, so both keep working inside a shard.
function(sushiai_add_shard_test program name filter labels timeout index shards)
    add_test(NAME ${name}
        COMMAND ${program} --gtest_filter=${filter}
            --gtest_output=xml:${SUSHIAI_GTEST_RESULTS_DIR}/${name}.xml)
    set_tests_properties(${name} PROPERTIES
        LABELS "${labels}"
        TIMEOUT ${timeout}
        ENVIRONMENT "GTEST_TOTAL_SHARDS=${shards};GTEST_SHARD_INDEX=${index}")
    # A large shard runs on the real accelerator, of which the resource file CTest gets holds one
    # slot each, so two large shards never share one.
    if("large" IN_LIST labels)
        set_tests_properties(${name} PROPERTIES RESOURCE_GROUPS "gpus:1")
    endif()
endfunction()

# Registers the cases of `program` of one size group in shards, with the group's label and a
# timeout of the case timeout times the cases of a shard. The quarantined cases of the size run
# apart in one test labelled quarantined, and a GoogleTest filter after '-' leaves them out of
# the shards.
function(sushiai_register_size_group program group)
    string(REPLACE "|" ";" parts "${group}")
    list(GET parts 0 prefix)
    list(GET parts 1 label)
    list(GET parts 2 case_timeout)
    list(GET parts 3 per_shard)

    get_property(cases GLOBAL PROPERTY SUSHIAI_CASES_${program}_${prefix})
    sushiai_quarantined_patterns(${program} ${prefix} quarantined)

    # An instantiated suite is named <prefix>/<suite>, so its pattern carries the slash.
    set(filter "${prefix}*:*/${prefix}*")
    if(quarantined)
        list(LENGTH quarantined quarantined_entries)
        math(EXPR quarantined_cases "${quarantined_entries} / 2")
        list(JOIN quarantined ":" quarantined_filter)
        string(APPEND filter "-${quarantined_filter}")
        math(EXPR quarantined_timeout "${case_timeout} * ${quarantined_cases}")
        sushiai_add_shard_test(${program} ${program}.${label}_quarantined "${quarantined_filter}"
            "${label};quarantined" ${quarantined_timeout} 0 1)
        math(EXPR cases "${cases} - ${quarantined_cases}")
    endif()
    if(cases LESS 1)
        return()
    endif()

    sushiai_shard_count(${cases} ${per_shard} shards)
    math(EXPR shard_cases "(${cases} + ${shards} - 1) / ${shards}")
    math(EXPR timeout "${case_timeout} * ${shard_cases}")
    math(EXPR last "${shards} - 1")
    foreach(index RANGE ${last})
        sushiai_add_shard_test(${program} ${program}.${label}_shard_${index}of${shards}
            "${filter}" "${label}" ${timeout} ${index} ${shards})
    endforeach()
endfunction()

# Registers the cases of `program` in shards, one block per size.
function(sushiai_register_sized_tests program)
    foreach(group IN LISTS SUSHIAI_SIZE_GROUPS)
        sushiai_register_size_group(${program} "${group}")
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
# Every case carries the size its suite-name prefix gives. The cases of a program run in shards,
# CTest tests labelled with that size and timed out at the size's timeout per case they hold. SA_TEST_FIXTURE_DIR names the
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

    sushiai_record_cases(${program} ${ARG_SOURCES})

    if(ARG_DEVICE_SOURCES)
        sushiai_check_device_sources(${program} "${ARG_SOURCES}" "${ARG_DEVICE_SOURCES}")
        sushiai_apply_sycl(${program} DEVICE_SOURCES ${ARG_DEVICE_SOURCES})
    endif()

    sushiai_module_target(engine_fixture fixture_target)
    target_link_libraries(${program} PRIVATE ${library} GTest::gtest ${fixture_target})
    # The shared test helpers copied from SushiSkills, included as <support/tolerance.hpp>, and
    # the repository's own under tests/support/, included as <support/<name>.hpp>.
    target_include_directories(${program} PRIVATE
        "${PROJECT_SOURCE_DIR}/tools/testing"
        "${PROJECT_SOURCE_DIR}/tests")

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

# Registers the tests of `program` in shards by size and copies the runtime, log and GTest DLLs beside it,
# with the Release DLLs of the modules that name one in SUSHIAI_RELEASE_RUNTIME_DLLS.
# gtest.dll comes from the configuration the program links: a Debug tree links GTest's Release
# build, and vcpkg's own copy, which runs before this one, puts the Debug DLL there.
function(sushiai_finish_test_program program)
    sushiai_register_sized_tests(${program})
    get_property(release_runtime_dlls GLOBAL PROPERTY SUSHIAI_RELEASE_RUNTIME_DLLS)

    if(WIN32)
        add_custom_command(TARGET ${program} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "$<TARGET_FILE:SushiRuntime::shared_library>"
                "$<TARGET_FILE:SushiLog::SushiLog>"
                "$<TARGET_FILE:GTest::gtest>"
                ${release_runtime_dlls}
                $<TARGET_FILE_DIR:${program}>
            COMMENT "Copying the runtime, log and GTest DLLs beside ${program}...")
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
        sushiai_sum_cases(${program} ${objects})
        sushiai_apply_sycl(${program})
        sushiai_apply_build_policy(${program})
        sushiai_finish_test_program(${program})
    endforeach()
endfunction()
