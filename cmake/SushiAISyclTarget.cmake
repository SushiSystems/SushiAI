# SushiAISyclTarget.cmake
# SushiAI - https://github.com/SushiSystems/SushiAI
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.

# The SYCL policy of SushiAI's targets, in one place. A translation unit is compiled for the
# device only when it instantiates a kernel, and its module names it in DEVICE_SOURCES. Every
# other unit is compiled for the host alone, with the toolchain's SYCL headers on its include
# path, because SushiBLAS and SushiRuntime headers include <sycl/sycl.hpp>. Every library links
# with -fsycl, so an executable pulls the device images out of the objects that carry them.
# SushiEngine's cmake/SyclTarget.cmake draws the same line, and
# docs/agent/2026_10_26_COMPILE_TIME/PLAN.md holds the measurements behind it.
#
# The file name carries the SushiAI prefix because the module path is shared with the SushiBLAS
# and SushiRuntime checkouts, whose own SyclTarget.cmake defines other functions.
#
# Included after BLAS.cmake: the device units take the device options SushiBLAS was compiled
# with, read from its target, so both libraries' device images are built for the same targets.

# SA_SYCL_INCLUDE_DIR: the toolchain's include directory, where <sycl/sycl.hpp> lives. -fsycl
# puts it on the search path; a host unit compiled without -fsycl needs it named. Derived from
# the compiler, since the compiler decides which toolchain's headers are the right ones.
if(SR_SYCL_TOOLCHAIN STREQUAL "intel-llvm" AND NOT DEFINED SA_SYCL_INCLUDE_DIR)
    get_filename_component(_sa_sycl_bin "${CMAKE_CXX_COMPILER}" DIRECTORY)
    get_filename_component(_sa_sycl_root "${_sa_sycl_bin}" DIRECTORY)
    if(EXISTS "${_sa_sycl_root}/include/sycl/sycl.hpp")
        set(SA_SYCL_INCLUDE_DIR "${_sa_sycl_root}/include" CACHE INTERNAL
            "SYCL headers, for translation units compiled without -fsycl")
    endif()
    unset(_sa_sycl_bin)
    unset(_sa_sycl_root)
endif()

# SA_SYCL_PER_SOURCE: whether device compilation can be given to single sources. It can on the
# intel-llvm toolchain, where -fsycl is a per-unit option, once the SYCL headers are found.
# Elsewhere add_sycl_to_target() works on whole targets, and every unit takes the device pass.
if(SR_SYCL_TOOLCHAIN STREQUAL "intel-llvm" AND SA_SYCL_INCLUDE_DIR)
    set(SA_SYCL_PER_SOURCE ON)
else()
    set(SA_SYCL_PER_SOURCE OFF)
    message(STATUS "SushiAI: every unit takes the device pass "
                   "(toolchain='${SR_SYCL_TOOLCHAIN}', sycl headers='${SA_SYCL_INCLUDE_DIR}')")
endif()

# sushiai_sycl_device_options(<out_compile> <out_link>)
#   Returns the options a device unit compiles with and a library links with: -fsycl and the
#   device targets and flags SushiBLAS publishes, or -fsycl alone when it publishes none.
function(sushiai_sycl_device_options out_compile out_link)
    get_target_property(compile SushiBLAS::sushiblas INTERFACE_COMPILE_OPTIONS)
    get_target_property(link SushiBLAS::sushiblas INTERFACE_LINK_OPTIONS)
    if(NOT compile)
        set(compile -fsycl)
    endif()
    if(NOT link)
        set(link -fsycl)
    endif()
    set(${out_compile} "${compile}" PARENT_SCOPE)
    set(${out_link} "${link}" PARENT_SCOPE)
endfunction()

# SushiAI::blas: SushiBLAS as SushiAI's modules link it. SushiBLAS publishes its device options
# as compile options of `sushiblas`, which would put every unit that links it through the device
# pass. In the build tree this target takes the library's link, include directories and
# definitions, and SushiRuntime, whose include directories and floating-point contract
# SushiBLAS would otherwise carry. An installed consumer links SushiBLAS whole, as before.
if(NOT TARGET sushiai_blas)
    add_library(sushiai_blas INTERFACE)
    add_library(SushiAI::blas ALIAS sushiai_blas)
    set_target_properties(sushiai_blas PROPERTIES EXPORT_NAME blas)
    if(SA_SYCL_PER_SOURCE)
        target_link_libraries(sushiai_blas INTERFACE
            "$<BUILD_INTERFACE:$<LINK_ONLY:SushiBLAS::sushiblas>>"
            "$<BUILD_INTERFACE:SushiRuntime::SushiRuntime>"
            "$<INSTALL_INTERFACE:SushiBLAS::sushiblas>")
        target_include_directories(sushiai_blas INTERFACE
            "$<BUILD_INTERFACE:$<TARGET_PROPERTY:SushiBLAS::sushiblas,INTERFACE_INCLUDE_DIRECTORIES>>")
        target_compile_definitions(sushiai_blas INTERFACE
            "$<BUILD_INTERFACE:$<TARGET_PROPERTY:SushiBLAS::sushiblas,INTERFACE_COMPILE_DEFINITIONS>>")
    else()
        target_link_libraries(sushiai_blas INTERFACE SushiBLAS::sushiblas)
    endif()
    install(TARGETS sushiai_blas EXPORT SushiAITargets)
endif()

# sushiai_apply_sycl(<target> [DEVICE_SOURCES <src>...])
#   Applies the SYCL policy to a module's library <target>. The DEVICE_SOURCES, which must be
#   sources of <target> in the current directory, compile for the device with the options of
#   sushiai_sycl_device_options(); the other units compile for the host. The SYCL headers are a
#   PUBLIC system include directory of the build tree, so a unit that includes this module's
#   headers finds them too. Without per-source compilation every unit takes the device pass,
#   and -fsycl is PUBLIC.
function(sushiai_apply_sycl target)
    cmake_parse_arguments(ARG "" "" "DEVICE_SOURCES" ${ARGN})

    if(NOT SA_SYCL_PER_SOURCE)
        add_sycl_to_target(TARGET ${target})
        if(SR_SYCL_TOOLCHAIN STREQUAL "intel-llvm")
            target_compile_options(${target} PUBLIC -fsycl)
            target_link_options(${target} PUBLIC -fsycl)
        endif()
        return()
    endif()

    target_include_directories(${target} SYSTEM PUBLIC "$<BUILD_INTERFACE:${SA_SYCL_INCLUDE_DIR}>")

    if(ARG_DEVICE_SOURCES)
        sushiai_sycl_device_options(device_compile device_link)
        add_sycl_to_target(TARGET ${target} SOURCES ${ARG_DEVICE_SOURCES})
        set_source_files_properties(${ARG_DEVICE_SOURCES}
            DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
            PROPERTIES COMPILE_OPTIONS "${device_compile}")
    endif()
endfunction()

# sushiai_apply_sycl_link(<target>)
#   Applies the SYCL policy to a library an executable links: the device link options, PUBLIC,
#   so the executable's link pulls the device images out of the objects that carry them, and the
#   SYCL headers as a PUBLIC system include directory of the build tree.
function(sushiai_apply_sycl_link target)
    if(NOT SA_SYCL_PER_SOURCE)
        sushiai_apply_sycl(${target})
        return()
    endif()

    sushiai_sycl_device_options(device_compile device_link)
    target_link_options(${target} PUBLIC ${device_link})
    target_include_directories(${target} SYSTEM PUBLIC "$<BUILD_INTERFACE:${SA_SYCL_INCLUDE_DIR}>")
endfunction()
