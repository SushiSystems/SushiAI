cmake_minimum_required(VERSION 3.20)

# SushiAI — a head; SushiRuntime is the backbone, SushiBLAS is where all
# compute is drawn from (SushiAI never touches SYCL directly - see
# docs/slop/make_sushiai_great_again.md). Mirrors SushiBLAS's own root
# CMakeLists.txt ordering exactly: Vcpkg + Toolchain run BEFORE project() to
# shape the toolchain file and pick the default SYCL compiler ahead of
# project()'s probe; everything else (the two sibling dependencies, the SYCL
# target policy, packaging) runs AFTER. See cmake/ for each stage.
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# --- Pre-project: toolchain shaping -------------------------------------------
include(Vcpkg)      # (Windows) locate vcpkg + set the toolchain file
include(Toolchain)  # pick the default SYCL compiler (clang++) before the probe

project(SushiAI VERSION 0.1.0 LANGUAGES CXX C)

# --- Post-project: sibling dependencies (D3 layered resolution) ---------------
# SushiRuntime resolved first so SushiBLAS's own internal Runtime.cmake
# TARGET-guards onto the same target instead of re-adding it (see
# cmake/Runtime.cmake's header comment).
include(Runtime)  # add_subdirectory(sushiruntime) -> SushiRuntime::SushiRuntime
include(BLAS)     # add_subdirectory(sushiblas)    -> SushiBLAS::sushiblas
include(SushiAISyclTarget)  # sushiai_apply_sycl() helper

# Compile policy. SushiAIBuildPolicy composes the warning, architecture, and
# sanitizer targets and exposes sushiai_apply_build_policy(<target>), which
# every target in the tree calls - see cmake/SushiAIBuildPolicy.cmake.
include(SushiAIBuildPolicy)

# --- Policies ---
if(POLICY CMP0135)
  cmake_policy(SET CMP0135 NEW)
endif()

# --- Options ---
# Named SA_* rather than SUSHIAI_*, matching SA_INSTALL_CMAKEDIR below and the
# SB_*/SR_* cache variables the sibling repos use; the sushiai_ prefix rule in
# CONTRIBUTING.md §4 governs CMake functions and targets, which share one
# global namespace under D3's sibling add_subdirectory, not cache entries.
#
# Defaulted ON because `sa test` expects a test binary to exist, but it is an
# option at all so a consumer that pulls this tree in with add_subdirectory is
# not forced to compile - and, through gtest_discover_tests, to enumerate and
# register - someone else's suite. SushiRuntime's SR_BUILD_TESTS defaults OFF
# for the opposite reason (its own docs call that a wart, since a direct cmake
# build then has nothing for ctest to run); ON is the honest default here.
option(SA_BUILD_TESTS "Build the SushiAI test suite" ON)

# --- Configuration ---
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# --- Install layout & package location (SB-6 packaging shape, from day one) ---
include(GNUInstallDirs)
set(SA_INSTALL_CMAKEDIR "${CMAKE_INSTALL_LIBDIR}/cmake/SushiAI"
    CACHE STRING "Where the SushiAI CMake package files are installed")

# The version literals in include/SushiAI/version.hpp are hand-written, so
# they are greppable and need no generated header on the include path. That
# trade is only safe if they cannot silently disagree with the version the
# build and the package advertise — so the disagreement is a configure-time
# error, here, rather than a consumer discovering it later. Mirrors
# sushiblas's own version-drift check exactly.
file(STRINGS "${CMAKE_CURRENT_SOURCE_DIR}/include/SushiAI/version.hpp"
     SA_VERSION_STRING_LINE
     REGEX "^#define SUSHIAI_VERSION_STRING ")
string(REGEX MATCH "\"([0-9]+\\.[0-9]+\\.[0-9]+)\"" _sa_unused "${SA_VERSION_STRING_LINE}")
if(NOT CMAKE_MATCH_1 STREQUAL PROJECT_VERSION)
    message(FATAL_ERROR
        "Version mismatch: project() says ${PROJECT_VERSION}, "
        "include/SushiAI/version.hpp says ${CMAKE_MATCH_1}. "
        "Update both — they are the same fact stated twice.")
endif()

# --- Subdirectories ---
add_subdirectory(src)
add_subdirectory(examples)

# --- Package files (SB-6 packaging shape, from day one) ---
include(CMakePackageConfigHelpers)

configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/SushiAIConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/SushiAIConfig.cmake"
    INSTALL_DESTINATION "${SA_INSTALL_CMAKEDIR}"
)

write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/SushiAIConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMinorVersion
)

install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/SushiAIConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/SushiAIConfigVersion.cmake"
    DESTINATION "${SA_INSTALL_CMAKEDIR}"
)

# --- Testing ---
# enable_testing() lives inside the guard too: with tests off there is nothing
# to register, and calling it anyway would leave a CTestTestfile.cmake that
# reports success over an empty suite - which is exactly the reading a CI lane
# must never get.
if(SA_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
