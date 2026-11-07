/**************************************************************************/
/* compile_partner_source.cpp                                             */
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


/**
 * @file compile_partner_source.cpp
 * @brief Checks at compile time that PartnerSource is destroyed through its base pointer.
 * @author Mustafa Garip
 */

#include <type_traits>
#include <SushiAI/vision/imaging/partner_source.hpp>

static_assert(std::has_virtual_destructor_v<SushiAI::Vision::Imaging::PartnerSource>,
              "a PartnerSource held by a base pointer must run the destructor of its type");
