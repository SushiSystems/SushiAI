/**************************************************************************/
/* compile_language.cpp                                                */
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
 * @file compile_language.cpp
 * @brief Checks at compile time that the language umbrella header alone names the core registry and the domain collator.
 * @author Mustafa Garip
 */

#include <type_traits>
#include <SushiAI/language.hpp>

using namespace SushiAI;

namespace
{
    static_assert(std::is_default_constructible_v<NN::NodeRegistry>,
                  "the umbrella names the core node registry");
    static_assert(std::is_abstract_v<Language::Datasets::Collator>,
                  "the umbrella names the collator interface of the domain");
    static_assert(std::is_base_of_v<Language::Datasets::Collator,
                                    Language::Datasets::NextTokenCollator>,
                  "the umbrella names the next-token collator beside it");
} // namespace
