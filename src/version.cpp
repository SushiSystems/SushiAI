/****************************************************************/
/* version.cpp                                                  */
/* SushiAI - https://github.com/SushiSystems/SushiAI            */
/* Copyright (c) 2026 Sushi Systems                             */
/* Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.    */
/* Commercial use requires a licence from Sushi Systems.        */
/****************************************************************/

/**
 * @file version.cpp
 * @brief Implements runtime version query functions.
 * @author Mustafa Garip
 */

#include <SushiAI/version.hpp>

namespace SushiAI
{
    const char* version_string() noexcept
    {
        return SUSHIAI_VERSION_STRING;
    }
}
