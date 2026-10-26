# BLAS.cmake
# SushiAI - https://github.com/SushiSystems/SushiAI
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.

# SushiBLAS as a package. The CLI builds or reuses it and passes SushiBLAS_DIR.
find_package(SushiBLAS CONFIG REQUIRED)
message(STATUS "Found SushiBLAS ${SushiBLAS_VERSION}: ${SushiBLAS_DIR}")
