/**************************************************************************/
/* version.hpp                                                            */
/**************************************************************************/
/*                          This file is part of:                         */
/*                                 SushiAI                                */
/*                https://github.com/SushiSystems/SushiAI                 */
/*                         https://sushisystems.io                        */
/**************************************************************************/
/* Copyright (c) 2026-present Mustafa Garip & Sushi Systems               */
/*                                                                        */
/* Licensed under the Apache License, Version 2.0 (the "License");        */
/* you may not use this file except in compliance with the License.       */
/* You may obtain a copy of the License at                                */
/*                                                                        */
/*     http://www.apache.org/licenses/LICENSE-2.0                         */
/*                                                                        */
/* Unless required by applicable law or agreed to in writing, software    */
/* distributed under the License is distributed on an "AS IS" BASIS,      */
/* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or        */
/* implied. See the License for the specific language governing           */
/* permissions and limitations under the License.                         */
/**************************************************************************/

#pragma once

/**
 * @file version.hpp
 * @brief The library's version.
 *
 * SushiAI is STATIC, like SushiBLAS: there is no separate loaded artifact
 * whose version could drift from these macros at runtime, so a single
 * compile-time version is the whole story here.
 *
 * The literals here are checked against the project's CMake version at
 * configure time (see the root CMakeLists.txt), so this header and the build
 * cannot drift apart.
 */

/** @brief Major version. */
#define SUSHIAI_VERSION_MAJOR 0
/** @brief Minor version. */
#define SUSHIAI_VERSION_MINOR 1
/** @brief Patch version. */
#define SUSHIAI_VERSION_PATCH 0

/** @brief Dotted version, as a literal. */
#define SUSHIAI_VERSION_STRING "0.1.0"

/**
 * @brief The version as one comparable integer.
 *
 * Encoded so ordinary integer comparison orders releases:
 * `#if SUSHIAI_VERSION >= SUSHIAI_VERSION_ENCODE(0, 2, 0)`.
 */
#define SUSHIAI_VERSION_ENCODE(major, minor, patch) \
    ((major) * 10000 + (minor) * 100 + (patch))

/** @brief The version, encoded by SUSHIAI_VERSION_ENCODE. */
#define SUSHIAI_VERSION                              \
    SUSHIAI_VERSION_ENCODE(SUSHIAI_VERSION_MAJOR,    \
                           SUSHIAI_VERSION_MINOR,    \
                           SUSHIAI_VERSION_PATCH)

namespace SushiAI
{
    /**
     * @brief The library's version, dotted (e.g. "0.1.0").
     * @return A null-terminated literal owned by the library; never null.
     */
    const char* version_string() noexcept;
}
