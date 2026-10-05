/****************************************************************/
/* version.hpp                                                  */
/* SushiAI - https://github.com/SushiSystems/SushiAI            */
/* Copyright (c) 2026 Sushi Systems                             */
/* Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.    */
/* Commercial use requires a licence from Sushi Systems.        */
/****************************************************************/

/**
 * @file version.hpp
 * @brief Defines compile-time version macros and declarations for SushiAI.
 * @author Mustafa Garip
 */

#pragma once

/** @brief Holds the major version component. */
#define SUSHIAI_VERSION_MAJOR 0
/** @brief Holds the minor version component. */
#define SUSHIAI_VERSION_MINOR 1
/** @brief Holds the patch version component. */
#define SUSHIAI_VERSION_PATCH 0

/** @brief Holds the dotted version literal. */
#define SUSHIAI_VERSION_STRING "0.1.0"

/**
 * @brief Encodes the version components into a single comparable integer.
 */
#define SUSHIAI_VERSION_ENCODE(major, minor, patch) \
    ((major) * 10000 + (minor) * 100 + (patch))

/** @brief Encodes current version components using SUSHIAI_VERSION_ENCODE. */
#define SUSHIAI_VERSION                              \
    SUSHIAI_VERSION_ENCODE(SUSHIAI_VERSION_MAJOR,    \
                           SUSHIAI_VERSION_MINOR,    \
                           SUSHIAI_VERSION_PATCH)

namespace SushiAI
{
    /**
     * @brief Returns the library version string.
     * @return A null-terminated literal owned by the library; never null.
     */
    const char* version_string() noexcept;
}
