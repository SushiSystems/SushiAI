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
 * @file test_collator.cpp
 * @brief Tests that a task is read and collated through the Collator interface alone.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <vector>
#include <gtest/gtest.h>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/core/shape.hpp>
#include <SushiAI/core/status.hpp>
#include <SushiAI/data/dataset.hpp>
#include <SushiAI/data/roles.hpp>
#include <SushiAI/data/target.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/language/datasets/collator.hpp>
#include <SushiAI/language/text/token.hpp>

using namespace SushiAI;
using Language::Datasets::Collator;
using Language::Text::TokenID;

namespace
{
    /** @brief Writes nothing; records the token count of every collate() call. */
    class CountingCollator final : public Collator
    {
        public:
            /** @brief Builds the collator; @p destroyed is set when it is destroyed. */
            explicit CountingCollator(bool& destroyed)
                : destroyed_(destroyed),
                  targets_{Data::TargetDescription{std::string(Data::K_ROLE_LABELS), Shape{4},
                                                   DType::Int32, 10}}
            {

            }

            /** @brief Marks the collator as destroyed. */
            ~CountingCollator() override
            {
                destroyed_ = true;
            }

            CountingCollator(const CountingCollator&) = delete;
            CountingCollator& operator=(const CountingCollator&) = delete;
            CountingCollator(CountingCollator&&) = delete;
            CountingCollator& operator=(CountingCollator&&) = delete;

            [[nodiscard]] std::string_view task() const noexcept override
            {
                return "counting";
            }

            [[nodiscard]] std::int64_t window_length() const noexcept override
            {
                return 4;
            }

            [[nodiscard]] std::int64_t tokens_per_window() const noexcept override
            {
                return 5;
            }

            [[nodiscard]] std::int32_t vocabulary_size() const noexcept override
            {
                return 10;
            }

            [[nodiscard]] const std::vector<Data::TargetDescription>& targets()
                const noexcept override
            {
                return targets_;
            }

            /** @brief Records the token count; refuses one that is not whole windows. */
            [[nodiscard]] Status collate(SushiRuntime::span<const TokenID> tokens,
                                         Data::Batch&) const override
            {
                calls.push_back(tokens.size());
                if (tokens.size() % static_cast<std::size_t>(tokens_per_window()) != 0)
                {
                    return Status::from_message(StatusCode::Incompatible, []()
                    {
                        return std::string("not whole windows");
                    });
                }
                return Status();
            }

            /** @brief Holds the token count of every collate() call, in order. */
            mutable std::vector<std::size_t> calls;

        private:
            /** @brief Holds the flag the destructor sets. */
            bool& destroyed_;

            /** @brief Holds the one labels role. */
            std::vector<Data::TargetDescription> targets_;
    };

    /** @brief Collates @p rows windows of zeros through @p collator, as a dataset does. */
    Status collate_rows(const Collator& collator, std::int64_t rows)
    {
        const std::vector<TokenID> tokens(
            static_cast<std::size_t>(rows * collator.tokens_per_window()), 0);
        Data::Batch batch;
        return collator.collate(SushiRuntime::span<const TokenID>(tokens.data(), tokens.size()),
                                batch);
    }
} // namespace

/** @brief Tests that a caller sizes the tokens of a batch from the interface alone. */
TEST(Small_LanguageDatasetsCollator, ACallerSizesTheTokensFromTheInterface)
{
    bool destroyed = false;
    const CountingCollator counting(destroyed);
    const Collator& collator = counting;

    SA_EXPECT_OK(collate_rows(collator, 3));

    ASSERT_EQ(counting.calls.size(), 1u);
    EXPECT_EQ(counting.calls[0], 15u);
    EXPECT_EQ(collator.task(), "counting");
    ASSERT_EQ(collator.targets().size(), 1u);
    EXPECT_EQ(collator.targets()[0].role, Data::K_ROLE_LABELS);
    EXPECT_EQ(collator.targets()[0].classes, collator.vocabulary_size());
}

/** @brief Tests that a refusal of collate() reaches the caller through the interface. */
TEST(Small_LanguageDatasetsCollator, ARefusalReachesTheCaller)
{
    bool destroyed = false;
    const CountingCollator counting(destroyed);
    const Collator& collator = counting;
    const std::vector<TokenID> partial(7, 0);
    Data::Batch batch;

    SA_EXPECT_FAILURE(
        collator.collate(SushiRuntime::span<const TokenID>(partial.data(), partial.size()),
                         batch),
        StatusCode::Incompatible, "not whole windows");
}

/** @brief Tests that a collator owned through the interface runs its own destructor. */
TEST(Small_LanguageDatasetsCollator, DestroyingThroughTheInterfaceRunsTheDerivedDestructor)
{
    bool destroyed = false;
    std::unique_ptr<Collator> owned = std::make_unique<CountingCollator>(destroyed);

    owned.reset();

    EXPECT_TRUE(destroyed);
}
