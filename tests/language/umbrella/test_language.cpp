/**************************************************************************/
/* test_language.cpp                                                      */
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
 * @file test_language.cpp
 * @brief Tests that the language umbrella header alone reaches the core and the domain.
 * @author Mustafa Garip
 */

#include <cstdint>
#include <memory>
#include <gtest/gtest.h>
#include <SushiAI/language.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>

using namespace SushiAI;

/** @brief Tests that a registry of the core rows takes the language rows through the umbrella. */
TEST(Small_LanguageUmbrella, TheUmbrellaRegistersTheLanguageRows)
{
    NN::NodeRegistry registry;
    SA_EXPECT_OK(NN::register_layers(registry));
    SA_EXPECT_OK(Language::register_nodes(registry));

    EXPECT_NE(registry.find_node("transformer_block"), nullptr);
    EXPECT_EQ(Language::Text::K_END_OF_TEXT, Language::Text::K_BYTE_TOKEN_COUNT);
}

/** @brief Tests that a collator of the domain is built and read through the umbrella. */
TEST(Small_LanguageUmbrella, TheUmbrellaBuildsADomainCollator)
{
    SA_ASSERT_OK_AND_ASSIGN(
        const std::unique_ptr<Language::Datasets::NextTokenCollator> collator,
        Language::Datasets::NextTokenCollator::make(4, 16));

    const Language::Datasets::Collator& seam = *collator;
    EXPECT_EQ(seam.task(), Language::Datasets::K_TASK_NEXT_TOKEN);
    EXPECT_EQ(seam.window_length(), std::int64_t{4});
}
