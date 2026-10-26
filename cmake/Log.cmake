# Log.cmake
# SushiAI - https://github.com/SushiSystems/SushiAI
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.

# SushiLog, the one logger of the process. The CLI builds or reuses its package and passes
# SushiLog_DIR; nothing is built in this tree and nothing is downloaded.
find_package(SushiLog CONFIG REQUIRED)
message(STATUS "Found SushiLog ${SushiLog_VERSION}: ${SushiLog_DIR}")
