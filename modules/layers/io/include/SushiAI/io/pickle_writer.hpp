/**************************************************************************/
/* pickle_writer.hpp                                                      */
/**************************************************************************/
/*                          This file is part of:                         */
/*                                 SushiAI                                */
/*                 https://github.com/SushiSystems/SushiAI                */
/*                         https://sushisystems.io                        */
/**************************************************************************/
/* Copyright (c) 2026-present Mustafa Garip & Sushi Systems               */
/*                                                                        */
/* Licensed under the Apache License, Version 2.0 (the "License"); you    */
/* may not use this file except in compliance with the License. You may   */
/* obtain a copy of the License at                                        */
/*                                                                        */
/*     https://www.apache.org/licenses/LICENSE-2.0                        */
/*                                                                        */
/* Unless required by applicable law or agreed to in writing, software    */
/* distributed under the License is distributed on an "AS IS" BASIS,      */
/* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or        */
/* implied. See the License for the specific language governing           */
/* permissions and limitations under the License.                         */
/**************************************************************************/

/**
 * @file pickle_writer.hpp
 * @brief Declares the writer of a pickle stream of protocol 2, the inverse of read_pickle.
 * @author Mustafa Garip
 * @see docs/design/MODEL_IMPORT/README.md
 */

#pragma once

#include <cstddef>
#include <vector>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/io/pickle_reader.hpp>

namespace SushiAI
{
    namespace IO
    {
        /**
         * @brief Writes @p root as a pickle stream of protocol 2 that read_pickle reads back.
         * A Call is callable, argument tuple, REDUCE, then SETITEMS, APPENDS and BUILD; a
         * PersistentID is its id and BINPERSID. No memo: a node named twice is written twice.
         * @return The stream; Unsupported naming the path of a Set; InvalidDescription naming
         *         the path of a Call with no Global, a PersistentID without exactly one id, or
         *         a Global that is not "module name".
         */
        [[nodiscard]] Result<std::vector<std::byte>> write_pickle(const PickleValue& root) noexcept;
    } // namespace IO
} // namespace SushiAI
