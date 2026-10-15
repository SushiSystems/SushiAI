# SushiAISanitizers.cmake
# SushiAI - https://github.com/SushiSystems/SushiAI
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.

# Defines the INTERFACE target sushiai_sanitizers, which carries the sanitizers SA_SANITIZE
# names. Empty, the default, adds nothing. `sa build --sanitize` sets it: "address,undefined"
# for the AddressSanitizer and UndefinedBehaviorSanitizer lane, "thread" for the
# ThreadSanitizer lane. Every target links it through SushiAIBuildPolicy.cmake, so a lane
# changes no target's CMakeLists.txt.
#
# The sanitizers instrument host code only. `-Xarch_host` keeps them off the SYCL device
# compile, which has no sanitizer runtime for every device a build targets. An undefined
# behaviour report stops the program, so it fails the test instead of passing with a log line.

set(SA_SANITIZE "" CACHE STRING
    "Sanitizers for host code, as -fsanitize= takes them: empty, address,undefined or thread.")

add_library(sushiai_sanitizers INTERFACE)

if(SA_SANITIZE)
    if(MSVC OR CMAKE_CXX_COMPILER_FRONTEND_VARIANT STREQUAL "MSVC")
        message(FATAL_ERROR "SA_SANITIZE=${SA_SANITIZE} needs a GNU-style compiler driver.")
    endif()
    target_compile_options(sushiai_sanitizers INTERFACE
        "SHELL:-Xarch_host -fsanitize=${SA_SANITIZE}"
        "SHELL:-Xarch_host -fno-omit-frame-pointer")
    target_link_options(sushiai_sanitizers INTERFACE "-fsanitize=${SA_SANITIZE}")
    if(SA_SANITIZE MATCHES "undefined")
        target_compile_options(sushiai_sanitizers INTERFACE
            "SHELL:-Xarch_host -fno-sanitize-recover=undefined")
    endif()
endif()
