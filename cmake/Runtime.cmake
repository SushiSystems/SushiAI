# Runtime.cmake
# SushiAI - https://github.com/SushiSystems/SushiAI
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.

# SushiRuntime as a package. The CLI builds or reuses it and passes SushiRuntime_DIR.
find_package(SushiRuntime CONFIG REQUIRED)
message(STATUS "Found SushiRuntime ${SushiRuntime_VERSION}: ${SushiRuntime_DIR}")
