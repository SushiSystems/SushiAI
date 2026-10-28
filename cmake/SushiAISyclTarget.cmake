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

# sushiai_device_link: the device optimisation level of a link, -O3 in every build type. A Debug
# build compiles host code at -O0, and the device passes and the device link would follow it; the
# kernels the tests run would then be too slow to test. On the link -O3 reaches only the device
# link, since the host objects are compiled. SushiAIBuildPolicy.cmake applies this target to every
# target of the tree, executables included, which are where the device link runs.
add_library(sushiai_device_link INTERFACE)
if(SA_SYCL_PER_SOURCE)
    target_link_options(sushiai_device_link INTERFACE -O3)
endif()

# sushiai_sycl_device_optimisation(<device_options> <out_compile>)
#   Returns the device optimisation level of a compile, -O3 in every build type, as the options to
#   add to a device unit; see sushiai_device_link for why. The level is given per device target,
#   by its triple, because the driver refuses the bare spelling when it compiles for more than one
#   target and matches a GPU alias only by triple. <device_options> is the option list of
#   SushiBLAS::device_code, which names the targets.
function(sushiai_sycl_device_optimisation device_options out_compile)
    set(_compile "")
    string(REGEX MATCH "-fsycl-targets=[^;]*" _targets "${device_options}")
    string(REPLACE "-fsycl-targets=" "" _targets "${_targets}")
    string(REPLACE "," ";" _targets "${_targets}")
    foreach(_target IN LISTS _targets)
        if(_target MATCHES "^nvidia_gpu_")
            set(_target nvptx64-nvidia-cuda)
        elseif(_target MATCHES "^amd_gpu_")
            set(_target amdgcn-amd-amdhsa)
        endif()
        list(APPEND _compile "SHELL:-Xsycl-target-frontend=${_target} -O3")
    endforeach()
    list(REMOVE_DUPLICATES _compile)
    set(${out_compile} "${_compile}" PARENT_SCOPE)
endfunction()

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
        sushiai_sycl_device_optimisation("${device_compile}" optimise_compile)
        add_sycl_to_target(TARGET ${target} SOURCES ${ARG_DEVICE_SOURCES})
        set_source_files_properties(${ARG_DEVICE_SOURCES}
            DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
            PROPERTIES COMPILE_OPTIONS "${device_compile};${optimise_compile}")
    endif()
endfunction()
