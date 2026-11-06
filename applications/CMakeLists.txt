# CMakeLists.txt
# SushiAI - https://github.com/SushiSystems/SushiAI
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under the Apache License 2.0. See LICENSE.

# The executables: the two programs behind `sa train` and `sa inference`, and
# the examples.

add_subdirectory(programs)

# Both examples read MNIST, which the vision domain owns.
if(SA_BUILD_VISION)
    add_subdirectory(examples)
endif()
