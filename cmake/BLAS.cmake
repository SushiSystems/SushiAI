# BLAS — D3 layered resolution of the SushiBLAS dependency.
#
# Same three-mode pattern as cmake/Runtime.cmake, re-templated for SushiBLAS:
# TARGET-guard, then find_package(SushiBLAS QUIET) (SB-6's install/export
# set), then a sibling add_subdirectory. SushiBLAS's own src/CMakeLists.txt
# defines the SushiBLAS::sushiblas ALIAS regardless of resolution mode.
#
# Included AFTER cmake/Runtime.cmake (see root CMakeLists.txt) so that by the
# time this add_subdirectory runs SushiRuntime::SushiRuntime already exists —
# SushiBLAS's own internal Runtime.cmake then TARGET-guards onto it instead of
# trying to add_subdirectory the same sushiruntime checkout a second time.
#
# This file is SushiAI-specific (no sibling repo currently needs to resolve
# SushiBLAS as a dependency), so there is no cross-repo filename-collision
# concern to design around here, unlike Vcpkg.cmake/Runtime.cmake/Toolchain.cmake.
if(NOT TARGET SushiBLAS::sushiblas)
    find_package(SushiBLAS QUIET)
endif()

if(NOT TARGET SushiBLAS::sushiblas)
    # See cmake/Runtime.cmake's header comment: SUSHISTACK_HOME can be set
    # globally without the module actually being cloned inside it (`ss link`),
    # so check existence before preferring that path over the plain sibling.
    if(DEFINED ENV{SUSHISTACK_HOME} AND EXISTS "$ENV{SUSHISTACK_HOME}/sushiblas")
        set(_SA_DEFAULT_BLAS "$ENV{SUSHISTACK_HOME}/sushiblas")
    else()
        set(_SA_DEFAULT_BLAS "${CMAKE_CURRENT_SOURCE_DIR}/../sushiblas")
    endif()
    set(SUSHIBLAS_DIR "${_SA_DEFAULT_BLAS}"
        CACHE PATH "Path to the SushiBLAS sibling checkout")

    add_subdirectory("${SUSHIBLAS_DIR}" "${CMAKE_BINARY_DIR}/sushiblas")
endif()
