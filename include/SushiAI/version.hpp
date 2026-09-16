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
