/**************************************************************************/
/* test_collator.cpp                                                      */
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
 * @file test_collator.cpp
 * @brief Tests that a task is read and sized through the Collator interface alone.
 * @author Mustafa Garip
 */

#include <cstdint>
#include <memory>
#include <gtest/gtest.h>
#include <SushiAI/data/dataset.hpp>
#include <SushiAI/data/roles.hpp>
#include <SushiAI/data/target.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/language/datasets/collator.hpp>
#include <SushiAI/language/datasets/next_token_collator.hpp>

using namespace SushiAI;
using Language::Datasets::Collator;
using Language::Datasets::NextTokenCollator;

/** @brief Tests that a caller sizes the tokens of a batch from the interface alone. */
TEST(Small_LanguageDatasetsCollator, ACallerSizesTheTokensFromTheInterface)
{
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<NextTokenCollator> owned,
                            NextTokenCollator::make(3, 50));
    const Collator& collator = *owned;
    const std::int64_t rows = 3;

    EXPECT_EQ(rows * collator.tokens_per_window(), 12);
    ASSERT_EQ(collator.targets().size(), 1u);
    EXPECT_EQ(collator.targets()[0].role, Data::K_ROLE_LABELS);
    EXPECT_EQ(collator.targets()[0].classes, collator.vocabulary_size());
}
