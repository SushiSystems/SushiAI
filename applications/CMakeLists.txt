# CMakeLists.txt
# SushiAI - https://github.com/SushiSystems/SushiAI
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.

# The executables: the two programs behind `sa train` and `sa inference`, and
# the examples.

add_subdirectory(programs)

# Both examples read MNIST, which the vision domain owns.
if(SA_BUILD_VISION)
    add_subdirectory(examples)
endif()
