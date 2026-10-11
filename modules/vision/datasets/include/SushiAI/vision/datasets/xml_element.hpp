/**************************************************************************/
/* xml_element.hpp                                                        */
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
 * @file xml_element.hpp
 * @brief Declares the element tree of an XML document and the parser of the subset VOC uses.
 * @author Mustafa Garip
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include <SushiAI/core/result.hpp>

/** @brief Holds the readers of the on-disk layouts vision datasets are kept in. */
namespace SushiAI::Vision::Datasets
{
    /** @brief Holds how many elements may be open at once in a document parse_xml accepts. */
    inline constexpr std::int32_t K_XML_MAX_DEPTH = 64;

    /** @brief Holds one element: its name, its text and the elements inside it. */
    struct XMLElement
    {
        /** @brief Holds the name of the tag. */
        std::string name;

        /** @brief Holds the character data, entities replaced and outer white space dropped. */
        std::string text;

        /** @brief Holds the elements inside this one, in document order. */
        std::vector<XMLElement> children;

        /** @brief Holds the one-based line the opening tag starts on. */
        std::size_t line = 0;

        /** @brief Returns the first child named @p child_name, or null when there is none. */
        [[nodiscard]] const XMLElement* find_child(std::string_view child_name) const noexcept;
    };

    /**
     * @brief Parses a document of elements and text into its root element.
     *
     * README.md, section xml_element.hpp, lists what is accepted. Attributes are read and
     * dropped. Anything else is refused.
     * @return The root; InvalidFile, opening with "line <n>", for what the subset lacks;
     *         OutOfMemory.
     */
    [[nodiscard]] Result<XMLElement> parse_xml(std::string_view text);
} // namespace SushiAI::Vision::Datasets
