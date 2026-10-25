# SushiAIModule.cmake
# SushiAI - https://github.com/SushiSystems/SushiAI
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.

# sushiai_add_module() declares one module of modules/<tier>/<module>/, and
# sushiai_add_library() archives the modules of one or more tiers into a
# library a consumer links. The shape follows SushiEngine's Module.cmake; what
# is added here is what a package needs: the export set, the installed headers
# and the umbrella check.

if(NOT DEFINED SUSHIAI_TIER_ORDER)
    include(SushiAILayers)
endif()

# Looks a key up in a flat list of key/value pairs; empty when absent.
function(sushiai_lookup_pair pairs_variable key output_variable)
    set(${output_variable} "" PARENT_SCOPE)
    set(pairs ${${pairs_variable}})
    list(LENGTH pairs count)
    set(index 0)
    while(index LESS count)
        math(EXPR value_slot "${index} + 1")
        list(GET pairs ${index} listed_key)
        if("${listed_key}" STREQUAL "${key}")
            list(GET pairs ${value_slot} listed_value)
            set(${output_variable} "${listed_value}" PARENT_SCOPE)
            return()
        endif()
        math(EXPR index "${index} + 2")
    endwhile()
endfunction()

# Ranks a tier: its position in SUSHIAI_TIER_ORDER, with every domain tier at
# the rank directly above the last tier that is not a domain.
function(sushiai_tier_rank tier output_variable)
    list(FIND SUSHIAI_TIER_ORDER "${tier}" position)
    if(position EQUAL -1)
        message(FATAL_ERROR
            "Tier '${tier}' is not in SUSHIAI_TIER_ORDER (cmake/SushiAILayers.cmake).")
    endif()
    list(FIND SUSHIAI_DOMAIN_TIERS "${tier}" domain_position)
    if(NOT domain_position EQUAL -1)
        list(LENGTH SUSHIAI_TIER_ORDER tier_count)
        list(LENGTH SUSHIAI_DOMAIN_TIERS domain_count)
        math(EXPR position "${tier_count} - ${domain_count}")
    endif()
    set(${output_variable} ${position} PARENT_SCOPE)
endfunction()

# Names the target of a module key: sushiai_module_<key>, with `_` for `/`. The
# word `module` keeps a module apart from a program of the same name, as
# `train` and sushiai_train are.
function(sushiai_module_target key output_variable)
    string(REPLACE "/" "_" flat "${key}")
    set(${output_variable} "sushiai_module_${flat}" PARENT_SCOPE)
endfunction()

# Sets `output_variable` to the key of module `name` of `tier` in
# SUSHIAI_MODULE_TIERS, and fails the configure when the table lists it under
# another tier. A module of a domain tier is keyed <tier>/<name>.
function(sushiai_module_key name tier output_variable)
    list(FIND SUSHIAI_DOMAIN_TIERS "${tier}" domain_position)
    if(domain_position EQUAL -1)
        set(key "${name}")
    else()
        set(key "${tier}/${name}")
    endif()

    sushiai_lookup_pair(SUSHIAI_MODULE_TIERS "${key}" listed_tier)
    if(NOT "${listed_tier}" STREQUAL "${tier}")
        message(FATAL_ERROR
            "Module ${key}: SUSHIAI_MODULE_TIERS lists it in tier '${listed_tier}', and its "
            "CMakeLists.txt says '${tier}'. Add or correct its row in "
            "cmake/SushiAILayers.cmake.")
    endif()

    set(${output_variable} "${key}" PARENT_SCOPE)
endfunction()

# Fails the configure when `consumer` may not depend on `dependency`.
function(sushiai_check_module_edge consumer dependency)
    sushiai_lookup_pair(SUSHIAI_MODULE_TIERS "${dependency}" dependency_tier)
    if("${dependency_tier}" STREQUAL "")
        message(FATAL_ERROR
            "sushiai_add_module(${consumer}): depends on '${dependency}', which is not in "
            "SUSHIAI_MODULE_TIERS (cmake/SushiAILayers.cmake).")
    endif()
    sushiai_lookup_pair(SUSHIAI_MODULE_TIERS "${consumer}" consumer_tier)
    sushiai_tier_rank("${consumer_tier}" consumer_rank)
    sushiai_tier_rank("${dependency_tier}" dependency_rank)
    if(dependency_rank GREATER consumer_rank)
        message(FATAL_ERROR
            "sushiai_add_module(${consumer}): ${consumer} (${consumer_tier}) depends on "
            "${dependency} (${dependency_tier}), which is a higher tier. Dependencies point "
            "down the order of cmake/SushiAILayers.cmake.")
    endif()
    list(FIND SUSHIAI_DOMAIN_TIERS "${consumer_tier}" consumer_is_domain)
    list(FIND SUSHIAI_DOMAIN_TIERS "${dependency_tier}" dependency_is_domain)
    if(NOT consumer_is_domain EQUAL -1 AND NOT dependency_is_domain EQUAL -1
            AND NOT "${consumer_tier}" STREQUAL "${dependency_tier}")
        message(FATAL_ERROR
            "sushiai_add_module(${consumer}): ${consumer} is of domain ${consumer_tier} and "
            "${dependency} of domain ${dependency_tier}. A domain does not depend on another.")
    endif()
endfunction()

# sushiai_promote_imported(<target> ...)
#
# Makes each imported target, and every imported target its link interface
# names, visible from every directory. It is called from the directory whose
# find_package created them, which is the only one CMake lets do so.
function(sushiai_promote_imported)
    foreach(external IN LISTS ARGN)
        if(NOT TARGET ${external})
            continue()
        endif()

        get_target_property(aliased ${external} ALIASED_TARGET)
        get_target_property(imported ${external} IMPORTED)
        get_target_property(global ${external} IMPORTED_GLOBAL)
        if(aliased OR NOT imported OR global)
            continue()
        endif()

        set_target_properties(${external} PROPERTIES IMPORTED_GLOBAL TRUE)

        get_target_property(interface ${external} INTERFACE_LINK_LIBRARIES)
        if(interface)
            sushiai_promote_imported(${interface})
        endif()
    endforeach()
endfunction()

# sushiai_check_device_sources(<target> <sources> <device sources>)
#
# Stops the configure when a device source of <target> is not one of its sources, so a
# misspelt name cannot leave a kernel unit compiled for the host alone.
function(sushiai_check_device_sources target sources device_sources)
    foreach(device_source IN LISTS device_sources)
        list(FIND sources "${device_source}" found)
        if(found EQUAL -1)
            message(FATAL_ERROR "sushiai_add_module(${target}): the device source "
                "${device_source} is not in SOURCES.")
        endif()
    endforeach()
endfunction()

# sushiai_add_module(NAME <module> TIER <tier>
#     [SOURCES source/<file>.cpp ...] [DEVICE_SOURCES source/<file>.cpp ...]
#     [PUBLIC_DEPENDS <module key> ...] [PRIVATE_DEPENDS <module key> ...]
#     [EXTERNAL_PUBLIC <target> ...] [EXTERNAL_PRIVATE_INCLUDES <target> ...]
#     [EXTERNAL_PRIVATE_LINKS <target> ...] [PUBLIC_DEFINITIONS <name> ...]
#     [TESTS tests/<file>.cpp ...] [INTERNAL])
#
# INTERNAL declares a module only the tests use; see sushiai_add_internal_module.
# Called from modules/<tier>/<module>/CMakeLists.txt. With SOURCES the module
# is an OBJECT library, without them an INTERFACE one. DEVICE_SOURCES names the
# SOURCES that instantiate a device kernel; only those compile for the device
# (cmake/SushiAISyclTarget.cmake). Its public headers are
# include/ and its private ones source/. A dependency is a key of
# SUSHIAI_MODULE_TIERS, and an edge the tier order refuses stops the configure.
# EXTERNAL_PRIVATE_INCLUDES takes the include directories and definitions of a
# header-only target as values, so the target's name stays out of every link
# interface. EXTERNAL_PRIVATE_LINKS takes the same of a compiled target and
# hands the target to the library of the module's tier, which links it PRIVATE:
# the module's sources alone see its headers, and a consumer of the static
# library inherits the link and nothing else. The objects, the include root
# and the test files are handed to the library of the module's tier.
# PUBLIC_DEFINITIONS are compile definitions of the module and of everything that links it,
# its tier library and that library's installed consumers included. A dependency named in the
# global property SUSHIAI_UNBUILT_MODULES, a module an option leaves out, is checked and not
# linked.
function(sushiai_add_module)
    set(multi_value_keywords SOURCES DEVICE_SOURCES PUBLIC_DEPENDS PRIVATE_DEPENDS EXTERNAL_PUBLIC
        EXTERNAL_PRIVATE_INCLUDES EXTERNAL_PRIVATE_LINKS PUBLIC_DEFINITIONS TESTS)
    cmake_parse_arguments(ARG "INTERNAL" "NAME;TIER" "${multi_value_keywords}" ${ARGN})

    if(NOT ARG_NAME OR NOT ARG_TIER)
        message(FATAL_ERROR "sushiai_add_module: NAME and TIER are required.")
    endif()

    if(ARG_INTERNAL)
        if(ARG_EXTERNAL_PRIVATE_INCLUDES OR ARG_EXTERNAL_PRIVATE_LINKS OR ARG_TESTS
                OR ARG_PUBLIC_DEFINITIONS)
            message(FATAL_ERROR "sushiai_add_module(${ARG_NAME}): an INTERNAL module takes "
                "no EXTERNAL_PRIVATE_INCLUDES, EXTERNAL_PRIVATE_LINKS, PUBLIC_DEFINITIONS or "
                "TESTS.")
        endif()
        sushiai_add_internal_module(NAME ${ARG_NAME} TIER ${ARG_TIER}
            SOURCES ${ARG_SOURCES}
            DEVICE_SOURCES ${ARG_DEVICE_SOURCES}
            PUBLIC_DEPENDS ${ARG_PUBLIC_DEPENDS}
            PRIVATE_DEPENDS ${ARG_PRIVATE_DEPENDS}
            EXTERNAL_PUBLIC ${ARG_EXTERNAL_PUBLIC})
        return()
    endif()

    sushiai_module_key("${ARG_NAME}" "${ARG_TIER}" key)

    sushiai_lookup_pair(SUSHIAI_TIER_LIBRARY "${ARG_TIER}" library)
    sushiai_module_target("${key}" target)
    set(include_root "${CMAKE_CURRENT_SOURCE_DIR}/include")

    if(ARG_SOURCES)
        add_library(${target} OBJECT ${ARG_SOURCES})
        set(public_scope PUBLIC)
        target_include_directories(${target}
            PUBLIC "$<BUILD_INTERFACE:${include_root}>"
            PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/source")
        set_target_properties(${target} PROPERTIES
            CXX_STANDARD 17
            CXX_STANDARD_REQUIRED ON)
        sushiai_apply_build_policy(${target})
        sushiai_check_device_sources(${target} "${ARG_SOURCES}" "${ARG_DEVICE_SOURCES}")
        sushiai_apply_sycl(${target} DEVICE_SOURCES ${ARG_DEVICE_SOURCES})
        set_property(GLOBAL APPEND PROPERTY SUSHIAI_OBJECTS_${library}
            "$<TARGET_OBJECTS:${target}>")
    else()
        add_library(${target} INTERFACE)
        set(public_scope INTERFACE)
        target_include_directories(${target}
            INTERFACE "$<BUILD_INTERFACE:${include_root}>")
    endif()

    if(ARG_PUBLIC_DEFINITIONS)
        target_compile_definitions(${target} ${public_scope} ${ARG_PUBLIC_DEFINITIONS})
        set_property(GLOBAL APPEND PROPERTY SUSHIAI_PUBLIC_DEFINITIONS_${library}
            ${ARG_PUBLIC_DEFINITIONS})
    endif()

    foreach(dependency IN LISTS ARG_PUBLIC_DEPENDS ARG_PRIVATE_DEPENDS)
        sushiai_check_module_edge("${key}" "${dependency}")
    endforeach()

    get_property(unbuilt_modules GLOBAL PROPERTY SUSHIAI_UNBUILT_MODULES)
    if(unbuilt_modules)
        list(REMOVE_ITEM ARG_PUBLIC_DEPENDS ${unbuilt_modules})
        list(REMOVE_ITEM ARG_PRIVATE_DEPENDS ${unbuilt_modules})
    endif()

    foreach(dependency IN LISTS ARG_PUBLIC_DEPENDS)
        sushiai_module_target("${dependency}" dependency_target)
        target_link_libraries(${target} ${public_scope} ${dependency_target})
    endforeach()

    if(ARG_PRIVATE_DEPENDS AND NOT ARG_SOURCES)
        message(FATAL_ERROR
            "sushiai_add_module(${key}): a module without SOURCES has no private side.")
    endif()

    foreach(dependency IN LISTS ARG_PRIVATE_DEPENDS)
        sushiai_module_target("${dependency}" dependency_target)
        target_link_libraries(${target} PRIVATE ${dependency_target})
    endforeach()

    sushiai_promote_imported(${ARG_EXTERNAL_PUBLIC} ${ARG_EXTERNAL_PRIVATE_LINKS})

    foreach(external IN LISTS ARG_EXTERNAL_PUBLIC)
        target_link_libraries(${target} ${public_scope} ${external})
        set_property(GLOBAL APPEND PROPERTY SUSHIAI_PUBLIC_LINKS_${library} ${external})
    endforeach()

    if(ARG_EXTERNAL_PRIVATE_LINKS AND NOT ARG_SOURCES)
        message(FATAL_ERROR
            "sushiai_add_module(${key}): a module without SOURCES has no private side.")
    endif()

    foreach(external IN LISTS ARG_EXTERNAL_PRIVATE_LINKS)
        set_property(GLOBAL APPEND PROPERTY SUSHIAI_PRIVATE_LINKS_${library} ${external})
    endforeach()

    foreach(external IN LISTS ARG_EXTERNAL_PRIVATE_INCLUDES ARG_EXTERNAL_PRIVATE_LINKS)
        target_include_directories(${target} PRIVATE
            "$<TARGET_PROPERTY:${external},INTERFACE_INCLUDE_DIRECTORIES>")
        target_compile_definitions(${target} PRIVATE
            "$<TARGET_PROPERTY:${external},INTERFACE_COMPILE_DEFINITIONS>")
    endforeach()

    set_property(GLOBAL APPEND PROPERTY SUSHIAI_INCLUDE_ROOTS_${library} "${include_root}")

    foreach(test IN LISTS ARG_TESTS)
        set_property(GLOBAL APPEND PROPERTY SUSHIAI_TESTS_${library}
            "${CMAKE_CURRENT_SOURCE_DIR}/${test}")
    endforeach()

    # A header's page (<name>.md) sits beside it and is not installed.
    install(DIRECTORY "${include_root}/" DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
        FILES_MATCHING PATTERN "*.hpp")
endfunction()

# Links one dependency of an internal module: an internal dependency itself, a
# library module for compiling only and its tier library for linking, so no
# object is linked twice.
function(sushiai_link_internal_dependency target scope dependency)
    get_property(internal_modules GLOBAL PROPERTY SUSHIAI_INTERNAL_MODULES)
    sushiai_module_target("${dependency}" dependency_target)

    list(FIND internal_modules "${dependency}" internal_position)
    if(NOT internal_position EQUAL -1)
        target_link_libraries(${target} ${scope} ${dependency_target})
        return()
    endif()

    sushiai_lookup_pair(SUSHIAI_MODULE_TIERS "${dependency}" dependency_tier)
    sushiai_lookup_pair(SUSHIAI_TIER_LIBRARY "${dependency_tier}" dependency_library)
    target_link_libraries(${target} ${scope}
        "$<COMPILE_ONLY:${dependency_target}>"
        "$<LINK_ONLY:${dependency_library}>")
endfunction()

# The INTERNAL form of sushiai_add_module: a module only the tests use. It
# joins no tier library and installs nothing: with SOURCES it is a STATIC
# library of its own, without them an INTERFACE one. It compiles against the
# include roots of its dependencies alone.
function(sushiai_add_internal_module)
    cmake_parse_arguments(ARG "" "NAME;TIER"
        "SOURCES;DEVICE_SOURCES;PUBLIC_DEPENDS;PRIVATE_DEPENDS;EXTERNAL_PUBLIC" ${ARGN})

    sushiai_module_key("${ARG_NAME}" "${ARG_TIER}" key)
    sushiai_module_target("${key}" target)
    set(include_root "${CMAKE_CURRENT_SOURCE_DIR}/include")

    if(ARG_SOURCES)
        add_library(${target} STATIC ${ARG_SOURCES})
        set(public_scope PUBLIC)
        target_include_directories(${target}
            PUBLIC "${include_root}"
            PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/source")
        set_target_properties(${target} PROPERTIES
            CXX_STANDARD 17
            CXX_STANDARD_REQUIRED ON)
        sushiai_apply_build_policy(${target})
        sushiai_check_device_sources(${target} "${ARG_SOURCES}" "${ARG_DEVICE_SOURCES}")
        sushiai_apply_sycl(${target} DEVICE_SOURCES ${ARG_DEVICE_SOURCES})
    else()
        add_library(${target} INTERFACE)
        set(public_scope INTERFACE)
        target_include_directories(${target} INTERFACE "${include_root}")
    endif()

    set_property(GLOBAL APPEND PROPERTY SUSHIAI_INTERNAL_MODULES "${key}")

    if(ARG_PRIVATE_DEPENDS AND NOT ARG_SOURCES)
        message(FATAL_ERROR
            "sushiai_add_module(${key}): a module without SOURCES has no private side.")
    endif()

    foreach(dependency IN LISTS ARG_PUBLIC_DEPENDS)
        sushiai_check_module_edge("${key}" "${dependency}")
        sushiai_link_internal_dependency(${target} ${public_scope} "${dependency}")
    endforeach()

    foreach(dependency IN LISTS ARG_PRIVATE_DEPENDS)
        sushiai_check_module_edge("${key}" "${dependency}")
        sushiai_link_internal_dependency(${target} PRIVATE "${dependency}")
    endforeach()

    foreach(external IN LISTS ARG_EXTERNAL_PUBLIC)
        target_link_libraries(${target} ${public_scope} ${external})
    endforeach()
endfunction()

# Fails the configure when a header under the include roots of `library`,
# other than the umbrella itself, is not named by its umbrella. install(DIRECTORY) ships every header, and a
# consumer that includes the umbrella never compiles one the umbrella leaves
# out, so nothing else in the repository would notice it is broken.
function(sushiai_check_umbrella library umbrella)
    get_property(include_roots GLOBAL PROPERTY SUSHIAI_INCLUDE_ROOTS_${library})
    file(READ "${umbrella}" umbrella_text)

    set(unreferenced "")
    foreach(include_root IN LISTS include_roots)
        file(GLOB_RECURSE headers CONFIGURE_DEPENDS
            RELATIVE "${include_root}" "${include_root}/*.hpp")
        foreach(header IN LISTS headers)
            string(FIND "${umbrella_text}" "<${header}>" found)
            if(found EQUAL -1 AND NOT "${include_root}/${header}" STREQUAL "${umbrella}")
                list(APPEND unreferenced "${header}")
            endif()
        endforeach()
    endforeach()

    if(unreferenced)
        string(REPLACE ";" "\n    " report "${unreferenced}")
        message(FATAL_ERROR
            "These public headers are installed but not included by ${umbrella}:\n"
            "    ${report}\n"
            "Add each to the umbrella, or move it out of the module's include/ tree if it "
            "is not public.")
    endif()
endfunction()

# sushiai_add_library(NAME <target> EXPORT_NAME <name> UMBRELLA <header>
#     [DEPENDS <library target> ...])
#
# Archives the objects of every module whose tier names <target> in
# SUSHIAI_TIER_LIBRARY into one STATIC library, aliased SushiAI::<name> and
# installed into the export set SushiAITargets. It is called after the
# modules, since it reads what they handed over. The module targets stay out
# of the link interface, so the export set names the library alone.
function(sushiai_add_library)
    cmake_parse_arguments(ARG "" "NAME;EXPORT_NAME;UMBRELLA" "DEPENDS" ${ARGN})

    if(NOT ARG_NAME OR NOT ARG_EXPORT_NAME OR NOT ARG_UMBRELLA)
        message(FATAL_ERROR "sushiai_add_library: NAME, EXPORT_NAME and UMBRELLA are required.")
    endif()

    get_property(objects GLOBAL PROPERTY SUSHIAI_OBJECTS_${ARG_NAME})
    get_property(include_roots GLOBAL PROPERTY SUSHIAI_INCLUDE_ROOTS_${ARG_NAME})
    get_property(private_links GLOBAL PROPERTY SUSHIAI_PRIVATE_LINKS_${ARG_NAME})

    add_library(${ARG_NAME} STATIC ${objects})
    add_library(SushiAI::${ARG_EXPORT_NAME} ALIAS ${ARG_NAME})

    foreach(include_root IN LISTS include_roots)
        target_include_directories(${ARG_NAME} PUBLIC "$<BUILD_INTERFACE:${include_root}>")
    endforeach()
    target_include_directories(${ARG_NAME} PUBLIC
        "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>")

    # PUBLIC: the external targets its modules link publicly, so a consumer inherits their
    # include paths and, through each SushiRuntime target, the deterministic floating-point
    # contract.
    get_property(public_links GLOBAL PROPERTY SUSHIAI_PUBLIC_LINKS_${ARG_NAME})
    list(REMOVE_DUPLICATES public_links)
    target_link_libraries(${ARG_NAME} PUBLIC ${public_links} ${ARG_DEPENDS})

    # PUBLIC: the definitions its modules declare, so they reach an installed consumer too.
    get_property(public_definitions GLOBAL PROPERTY SUSHIAI_PUBLIC_DEFINITIONS_${ARG_NAME})
    if(public_definitions)
        list(REMOVE_DUPLICATES public_definitions)
        target_compile_definitions(${ARG_NAME} PUBLIC ${public_definitions})
    endif()

    if(private_links)
        list(REMOVE_DUPLICATES private_links)
        target_link_libraries(${ARG_NAME} PRIVATE ${private_links})
    endif()

    # Restated on the target, so an add_subdirectory consumer at another
    # standard still links the library its ABI was decided under.
    set_target_properties(${ARG_NAME} PROPERTIES
        EXPORT_NAME ${ARG_EXPORT_NAME}
        LINKER_LANGUAGE CXX
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED ON)

    sushiai_apply_sycl_link(${ARG_NAME})

    sushiai_check_umbrella(${ARG_NAME} "${ARG_UMBRELLA}")

    install(TARGETS ${ARG_NAME}
        EXPORT SushiAITargets
        LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
        ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
        RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
        INCLUDES DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}")
endfunction()
