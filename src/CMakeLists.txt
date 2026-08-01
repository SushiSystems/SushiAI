cmake_minimum_required(VERSION 3.20)

# --- Define SushiAI Library ---
# AI-0 scope reset: deliberately minimal. Grows as AI-1 (graph-native
# autograd) and beyond land - see docs/slop/make_sushiai_great_again.md.
set(SUSHIAI_SOURCES
    version.cpp
)

add_library(sushiai STATIC ${SUSHIAI_SOURCES})

# A namespaced alias so an in-tree consumer (add_subdirectory, D3's
# TARGET-guard/sibling modes) and an out-of-tree one (find_package) spell the
# dependency identically - mirrors SushiBLAS::sushiblas exactly (SB-6).
add_library(SushiAI::sushiai ALIAS sushiai)

# --- Include Directories ---
target_include_directories(sushiai PUBLIC
    "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>"
    "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
)

# --- Compile Policy ---
# Warnings, architecture targeting and sanitizers, PRIVATE to this tree.
sushiai_apply_build_policy(sushiai)

# --- Link Libraries ---
# PUBLIC so consumers (examples, tests, and SushiAI's own future autograd
# core) inherit both siblings' include paths and, transitively, SushiRuntime's
# `sushiruntime_deterministic_fp` floating-point contract - SushiAI does not
# need (or own) a copy of that guarantee itself.
target_link_libraries(sushiai PUBLIC
    SushiRuntime::SushiRuntime
    SushiBLAS::sushiblas
)

# --- Properties ---
set_target_properties(sushiai PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
)

# SYCL device-compilation policy (see cmake/SyclTarget.cmake). version.cpp
# itself has no SYCL content yet, but future sources will, and -fsycl on the
# intel-llvm path must be applied PUBLIC so examples/tests inherit it from
# linking sushiai rather than reapplying it themselves.
sushiai_apply_sycl(sushiai SOURCES ${SUSHIAI_SOURCES})

# --- Install & export (SB-6 packaging shape, from day one) ---
install(TARGETS sushiai
    EXPORT SushiAITargets
    LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
    INCLUDES DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
)

install(DIRECTORY "${PROJECT_SOURCE_DIR}/include/"
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
)

install(EXPORT SushiAITargets
    FILE SushiAITargets.cmake
    NAMESPACE SushiAI::
    DESTINATION "${SA_INSTALL_CMAKEDIR}"
)

# The same targets, usable from a build tree without installing.
export(EXPORT SushiAITargets
    FILE "${PROJECT_BINARY_DIR}/SushiAITargets.cmake"
    NAMESPACE SushiAI::
)
