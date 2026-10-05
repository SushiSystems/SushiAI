# Runtime.cmake
# SushiAI - https://github.com/SushiSystems/SushiAI
# Copyright (c) 2026 Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.

# Runtime — D3 layered resolution of the SushiRuntime dependency.
#
# SushiAI consumes the live SushiRuntime checkout through exactly the three
# modes D3 defines, tried in order, so that whichever one resolves, every
# consumer links the identical namespaced alias afterwards:
#   1. TARGET-guard — a superproject that already brought SushiRuntime in.
#   2. find_package(SushiRuntime QUIET) — an installed package, the
#      out-of-tree case.
#   3. add_subdirectory of a sibling checkout — the actual case on a fresh
#      SushiStack workspace: no installed package exists yet, so this fires.
# sushiruntime's own src/CMakeLists.txt unconditionally defines the
# SushiRuntime::SushiRuntime ALIAS regardless of how it's brought in, so all
# three modes converge on the same link name with no extra plumbing here.
#
# Resolved BEFORE cmake/BLAS.cmake (see root CMakeLists.txt) deliberately: if
# SushiRuntime is resolved here first, SushiBLAS's own internal Runtime.cmake
# (added transitively when BLAS.cmake's add_subdirectory runs, since
# SushiBLAS's root CMakeLists.txt puts its own cmake/ on the same shared
# CMAKE_MODULE_PATH and calls include(Runtime) too) finds the target already
# defined and skips re-adding it, instead of both trying to add_subdirectory
# the same sushiruntime checkout into the same build-tree path.
#
# Filename intentionally NOT SushiAI-prefixed — see cmake/Vcpkg.cmake's header
# comment; the TARGET-guard/cache-var contract here is identical across every
# repo that carries a Runtime.cmake, which is what makes that safe.
if(NOT TARGET SushiRuntime::SushiRuntime)
    find_package(SushiRuntime QUIET)
endif()

if(NOT TARGET SushiRuntime::SushiRuntime)
    # Inside a SushiStack workspace every module is cloned beside the others,
    # so the runtime sibling is <workspace>/sushiruntime. Prefer
    # SUSHISTACK_HOME when set AND the module actually lives there - `ss link`
    # lets a module be registered from an out-of-workspace checkout (this dev
    # machine's own setup: SUSHISTACK_HOME is set globally, but sushiruntime/
    # sushiblas/sushiai are flat siblings under D:/Projects, not cloned inside
    # the workspace directory), so checking existence before preferring the
    # workspace path avoids pointing add_subdirectory at a directory that
    # simply isn't there. Otherwise fall back to the in-tree sibling
    # ../sushiruntime, which is the same path in a workspace and the right
    # standalone default.
    if(DEFINED ENV{SUSHISTACK_HOME} AND EXISTS "$ENV{SUSHISTACK_HOME}/sushiruntime")
        set(_SA_DEFAULT_RUNTIME "$ENV{SUSHISTACK_HOME}/sushiruntime")
    else()
        set(_SA_DEFAULT_RUNTIME "${CMAKE_CURRENT_SOURCE_DIR}/../sushiruntime")
    endif()
    set(SUSHIRUNTIME_DIR "${_SA_DEFAULT_RUNTIME}"
        CACHE PATH "Path to the SushiRuntime sibling checkout")

    add_subdirectory("${SUSHIRUNTIME_DIR}" "${CMAKE_BINARY_DIR}/sushiruntime")
endif()
