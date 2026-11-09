/**************************************************************************/
/* compile_sample.cpp                                                     */
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
 * @file compile_sample.cpp
 * @brief Checks at compile time the shape of the sample record and the sample list.
 * @author Mustafa Garip
 */

#include <cstdint>
#include <filesystem>
#include <optional>
#include <type_traits>
#include <vector>
#include <SushiAI/vision/datasets/sample.hpp>

using namespace SushiAI::Vision::Datasets;

namespace
{
    static_assert(std::is_aggregate_v<Sample>,
                  "a reader fills a sample field by field");
    static_assert(std::is_aggregate_v<SampleList>,
                  "a reader fills a sample list field by field");
    static_assert(std::is_same_v<decltype(Sample::file), std::uint32_t>,
                  "a sample names its pixel file by an index into the files of its list");
    static_assert(std::is_same_v<decltype(Sample::mask), std::optional<std::uint32_t>>,
                  "a sample names its mask file by the same index, or by none");
    static_assert(std::is_same_v<decltype(SampleList::files), std::vector<std::filesystem::path>>,
                  "the files of a list are paths");
    static_assert(std::is_same_v<decltype(SampleList::samples), std::vector<Sample>>,
                  "the samples of a list are sample records");
} // namespace
