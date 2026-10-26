# SushiAISyclTarget.cmake
# SushiAI - https://github.com/SushiSystems/SushiAI
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.

# The SYCL policy of SushiAI's targets, in one place. A translation unit is compiled for the
# device only when it instantiates a kernel, and its module or test program names it in
# DEVICE_SOURCES. Every other unit is compiled for the host alone. SushiBLAS publishes the rest:
# linking SushiBLAS::sushiblas gives a unit the SYCL headers and gives a link the device options,
# and SushiBLAS::device_code holds the compile options of a unit that instantiates a kernel.
# docs/agent/2026_10_26_COMPILE_TIME/PLAN.md holds the measurements behind the split.
#
# SushiRuntime_SYCL_TOOLCHAIN comes from the installed SushiRuntime package. The file name carries
# the SushiAI prefix so that it cannot collide with another repository's SyclTarget.cmake.

# SA_SYCL_PER_SOURCE: whether device compilation can be given to single sources. It can on the
# intel-llvm toolchain, where -fsycl is a per-unit option and SushiBLAS publishes its device
# options apart. Elsewhere add_sycl_to_target() works on whole targets, and every unit takes the
# device pass.
if(SushiRuntime_SYCL_TOOLCHAIN STREQUAL "intel-llvm" AND TARGET SushiBLAS::device_code)
    set(SA_SYCL_PER_SOURCE ON)
else()
    set(SA_SYCL_PER_SOURCE OFF)
    message(STATUS "SushiAI: every unit takes the device pass (toolchain='${SushiRuntime_SYCL_TOOLCHAIN}')")
endif()

# sushiai_apply_sycl(<target> [DEVICE_SOURCES <src>...])
#   Applies the SYCL policy to <target>. The DEVICE_SOURCES, which must be sources of <target>
#   in the current directory, compile with the options of SushiBLAS::device_code; the other units
#   compile for the host. Without per-source compilation every unit takes the device pass, and
#   -fsycl is PUBLIC.
function(sushiai_apply_sycl target)
    cmake_parse_arguments(ARG "" "" "DEVICE_SOURCES" ${ARGN})

    if(NOT SA_SYCL_PER_SOURCE)
        add_sycl_to_target(TARGET ${target})
        if(SushiRuntime_SYCL_TOOLCHAIN STREQUAL "intel-llvm")
            target_compile_options(${target} PUBLIC -fsycl)
            target_link_options(${target} PUBLIC -fsycl)
        endif()
        return()
    endif()

    if(ARG_DEVICE_SOURCES)
        get_target_property(device_compile SushiBLAS::device_code INTERFACE_COMPILE_OPTIONS)
        add_sycl_to_target(TARGET ${target} SOURCES ${ARG_DEVICE_SOURCES})
        set_source_files_properties(${ARG_DEVICE_SOURCES}
            DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
            PROPERTIES COMPILE_OPTIONS "${device_compile}")
    endif()
endfunction()
