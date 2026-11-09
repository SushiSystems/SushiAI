/**************************************************************************/
/* test_in_memory_one_position_step.cpp                                  */
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
 * @file test_in_memory_one_position_step.cpp
 * @brief Tests that InMemoryOnePositionStep scores the successor of the id it is fed and keeps
 *        what it was fed.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>
#include <gtest/gtest.h>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/data/dataset.hpp>
#include <SushiAI/data/input.hpp>
#include <SushiAI/data/materialised_dataset.hpp>
#include <SushiAI/dtype/dtype.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/shape/shape.hpp>
#include <SushiAI/train/evaluation_step.hpp>
#include <SushiAI/train/steps.hpp>
#include "position_feed_test_support.hpp"

using namespace SushiAI;
using namespace SushiAI::Language::Generation;

namespace
{
    /** @brief Holds T of the step under test. */
    constexpr std::int64_t K_WINDOW = 4;

    /** @brief Holds V of the step under test. */
    constexpr std::int64_t K_VOCABULARY = 5;

    /** @brief Names the output of the step under test. */
    constexpr std::string_view K_OUTPUT = "scores";

    /** @brief Builds the step under test and the id feed it is run with. */
    class Small_InMemoryOnePositionStep : public Support::HostFeedTest
    {
        protected:
            /** @brief Allocates the [1, 1] int32 id feed. */
            void SetUp() override
            {
                SA_EXPECT_OK(Data::prepare_batch(
                    engine(), 1, Data::InputDescription{Shape{1}, DType::Int32}, {}, id));
            }

            /** @brief Runs @p step with the id @p value fed as the model input. */
            Status run_with(InMemoryOnePositionStep& step, std::int32_t value)
            {
                id.inputs.data_as<std::int32_t>()[0] = value;
                const Train::NamedFeed fed[] = {{Train::K_MODEL_INPUT_NAME, &id.inputs}};
                return step.run(SushiRuntime::span<const Train::NamedFeed>(fed, 1));
            }

            /// @brief Holds the [1, 1] int32 id every run is fed.
            Data::Batch id;
    };

    /** @brief Returns the step under test: the model input and a mask are fed. */
    InMemoryOnePositionStep make_step(bool carries_state)
    {
        return InMemoryOnePositionStep(
            {Support::leaf_of(Train::K_MODEL_INPUT_NAME, 1, DType::Int32),
             Support::leaf_of("mask", K_WINDOW, DType::Float32)},
            std::string(K_OUTPUT), K_VOCABULARY, carries_state);
    }
} // namespace

/** @brief Tests that a run scores 1 at the token after the fed id, modulo V, and 0 elsewhere. */
TEST_F(Small_InMemoryOnePositionStep, ARunScoresTheSuccessorOfTheFedId)
{
    InMemoryOnePositionStep step = make_step(true);

    SA_EXPECT_OK(run_with(step, 4));

    const LogitSpan scores = step.logits(K_OUTPUT);
    ASSERT_EQ(scores.size(), static_cast<std::size_t>(K_VOCABULARY));
    EXPECT_EQ(scores[0], 1.0f);
    EXPECT_EQ(scores[1], 0.0f);
}

/** @brief Tests that the fed ids are kept in order and the resets are counted. */
TEST_F(Small_InMemoryOnePositionStep, TheFedIdsAndTheResetsAreKept)
{
    InMemoryOnePositionStep step = make_step(true);

    SA_EXPECT_OK(step.reset());
    SA_EXPECT_OK(run_with(step, 2));
    SA_EXPECT_OK(run_with(step, 0));

    EXPECT_EQ(step.fed_ids(), (std::vector<std::int32_t>{2, 0}));
    EXPECT_EQ(step.resets(), 1u);
}

/** @brief Tests that a run fed no model input is refused. */
TEST_F(Small_InMemoryOnePositionStep, ARunWithoutAModelInputIsRefused)
{
    InMemoryOnePositionStep step = make_step(true);

    SA_EXPECT_FAILURE(step.run(SushiRuntime::span<const Train::NamedFeed>()),
                      StatusCode::InvalidDescription, "no model input");
}

/** @brief Tests that the output is answered as [1, 1, V] float32 and another name is refused. */
TEST_F(Small_InMemoryOnePositionStep, TheOutputIsAnsweredByItsNameAlone)
{
    const InMemoryOnePositionStep step = make_step(true);

    SA_ASSERT_OK_AND_ASSIGN(const LeafFacts facts, step.output(K_OUTPUT));

    EXPECT_EQ(facts.shape[2], K_VOCABULARY);
    EXPECT_EQ(facts.dtype, DType::Float32);
    SA_EXPECT_FAILURE(step.output("other"), StatusCode::InvalidDescription, "'other'");
}

/** @brief Tests that the step reports the state flag it was given. */
TEST_F(Small_InMemoryOnePositionStep, TheStepReportsTheStateFlagItWasGiven)
{
    const InMemoryOnePositionStep stateless = make_step(false);

    EXPECT_FALSE(stateless.carries_state());
}
