/**************************************************************************/
/* head_contract.cpp                                                      */
/**************************************************************************/
/*                          This file is part of:                         */
/*                                 SushiAI                                */
/*                 https://github.com/SushiSystems/SushiAI                */
/*                         https://sushisystems.io                        */
/**************************************************************************/
/* Copyright (c) 2026-present Mustafa Garip & Sushi Systems               */
/*                                                                        */
/* Licensed under the PolyForm Noncommercial License 1.0.0 (the           */
/* "License"); you may not use this file except in compliance with the    */
/* License. You may obtain a copy of the License at                       */
/*                                                                        */
/*     https://polyformproject.org/licenses/noncommercial/1.0.0           */
/*                                                                        */
/* Noncommercial use is free. Commercial use requires a separate licence  */
/* from Sushi Systems; see COMMERCIAL.md. The software is provided        */
/* "as is", without warranty of any kind.                                 */
/**************************************************************************/

/**
 * @file head_contract.cpp
 * @brief Implements the port names and the class bias an anchor-free head and its training share.
 * @author Mustafa Garip
 * @see modules/vision/detect/README.md
 */

#include <SushiAI/vision/detect/free/head_contract.hpp>

#include <cmath>

namespace SushiAI
{
    namespace Vision::Detect::Free
    {
        std::string detect_port_name(std::string_view copy, std::size_t level,
                                     std::string_view role)
        {
            std::string name(copy);
            name.append(".level").append(std::to_string(level)).append(".").append(role);

            return name;
        }

        double detect_class_bias(std::int64_t classes, double stride)
        {
            const double cells = K_DETECT_REFERENCE_IMAGE_SIDE / stride;

            return std::log(K_DETECT_OBJECTS_PER_IMAGE / static_cast<double>(classes) /
                            (cells * cells));
        }
    } // namespace Vision::Detect::Free
} // namespace SushiAI
