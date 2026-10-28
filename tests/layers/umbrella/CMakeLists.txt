# CMakeLists.txt
# SushiAI - https://github.com/SushiSystems/SushiAI
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.

# The test program of the module umbrella; see cmake/SushiAITestProgram.cmake. The test includes
# both umbrellas as a consumer does, and SushiBLAS's instantiates kernels.

sushiai_add_test_program(OWNER umbrella
    SOURCES test_SushiAI/test_engine_smoke.cpp test_SushiAI/test_noexcept_surface.cpp
    DEVICE_SOURCES test_SushiAI/test_engine_smoke.cpp)

# The distribution case compiles only where the library carries SA_DISTRIBUTED to its consumers.
if(SA_ENABLE_DISTRIBUTED)
    target_compile_definitions(sushiai_layers_umbrella_test_objects PRIVATE SA_TEST_DISTRIBUTED_BUILD)
endif()
