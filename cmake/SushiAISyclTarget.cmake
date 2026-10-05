# SushiAISyclTarget.cmake
# SushiAI - https://github.com/SushiSystems/SushiAI
# Copyright (c) 2026 Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.

# SushiAISyclTarget.cmake — the single source of truth for turning a target's
# sources into SYCL device code against the runtime. Like SushiBLAS (and
# unlike SushiEngine's header-only shape), SushiAI compiles its own
# SYCL-templated code into the `sushiai` library, so the flags below are
# applied PUBLIC: examples/tests just link `sushiai` and inherit them.
#
# Filename IS SushiAI-prefixed, unlike Vcpkg.cmake/Toolchain.cmake/Runtime.cmake:
# this file defines a per-repo-named function (sushiai_apply_sycl vs
# sushiblas_apply_sycl), so it belongs with the BuildPolicy family, not the
# genuinely-idempotent-across-repos files. Originally left unprefixed by
# mistake and caught immediately by a fresh sushiai configure: sushiblas's own
# `include(SyclTarget)` (run when BLAS.cmake add_subdirectory's it) found this
# file first on the shared CMAKE_MODULE_PATH and failed with "Unknown CMake
# command sushiblas_apply_sycl" - the exact same class of bug SB-6 found and
# fixed for the BuildPolicy family.

# sushiai_apply_sycl(<target> [SOURCES <src>...])
#   Applies the SYCL device-compilation policy to <target>:
#     * the toolchain's add_sycl_to_target() (pass SOURCES so AdaptiveCpp/oneAPI
#       can find the device code; harmless where it is a no-op stub),
#     * an explicit -fsycl on the intel-llvm path, where add_sycl_to_target() is a
#       no-op stub and the flag must be added ourselves - PUBLIC, so it reaches
#       whatever links <target> too.
function(sushiai_apply_sycl target)
    cmake_parse_arguments(ARG "" "" "SOURCES" ${ARGN})

    if(ARG_SOURCES)
        add_sycl_to_target(TARGET ${target} SOURCES ${ARG_SOURCES})
    else()
        add_sycl_to_target(TARGET ${target})
    endif()

    if(SR_SYCL_TOOLCHAIN STREQUAL "intel-llvm")
        target_compile_options(${target} PUBLIC -fsycl)
        target_link_options(${target} PUBLIC -fsycl)
    endif()
endfunction()
