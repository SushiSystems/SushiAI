# Runtime.cmake
# SushiAI - https://github.com/SushiSystems/SushiAI
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under the Apache License 2.0. See LICENSE.

# SushiRuntime as a package. The CLI builds or reuses it and passes SushiRuntime_DIR.
find_package(SushiRuntime CONFIG REQUIRED)
message(STATUS "Found SushiRuntime ${SushiRuntime_VERSION}: ${SushiRuntime_DIR}")
