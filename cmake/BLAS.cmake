# BLAS.cmake
# SushiAI - https://github.com/SushiSystems/SushiAI
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under the Apache License 2.0. See LICENSE.

# SushiBLAS as a package. The CLI builds or reuses it and passes SushiBLAS_DIR.
find_package(SushiBLAS CONFIG REQUIRED)
message(STATUS "Found SushiBLAS ${SushiBLAS_VERSION}: ${SushiBLAS_DIR}")
