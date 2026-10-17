# Log.cmake
# SushiAI - https://github.com/SushiSystems/SushiAI
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.

# SushiLog, the one logger of the process, exposed as SushiLog::SushiLog. The CLI resolves its
# pin in cli/sushistack.deps.toml and passes SUSHILOG_DIR: a checkout is built in this tree, an
# install prefix is found as a package. Nothing is downloaded here. Included before Runtime, so
# SushiRuntime and SushiBLAS find the target and add no second copy.
if(NOT TARGET SushiLog::SushiLog)
    if(SUSHILOG_DIR AND EXISTS "${SUSHILOG_DIR}/CMakeLists.txt")
        set(SL_BUILD_TESTS OFF)
        add_subdirectory("${SUSHILOG_DIR}" "${CMAKE_BINARY_DIR}/sushilog")
    else()
        find_package(SushiLog CONFIG HINTS "${SUSHILOG_DIR}")
    endif()
endif()
if(NOT TARGET SushiLog::SushiLog)
    message(FATAL_ERROR
        "SushiLog was not found (SUSHILOG_DIR='${SUSHILOG_DIR}'). Run `sa setup`, which "
        "fetches the version cli/sushistack.deps.toml pins, or set SUSHILOG_DIR to a SushiLog "
        "checkout or install prefix.")
endif()
message(STATUS "Found SushiLog: ${SUSHILOG_DIR}")
