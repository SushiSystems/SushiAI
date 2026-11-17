/**************************************************************************/
/* test_data_normalise.cpp                                                */
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
 * @file test_data_normalise.cpp
 * @brief Tests how a data block takes the input normalisation its model file states.
 * @author Mustafa Garip
 */

#include <gtest/gtest.h>
#include <SushiAI/config/data_normalise.hpp>
#include <SushiAI/config/input_normalise.hpp>
#include <SushiAI/config/model_config.hpp>
#include <SushiAI/config/value.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>

using namespace SushiAI;
using namespace SushiAI::Config;

namespace
{
    /** @brief Returns a normalisation that swaps the channels and divides by 2, 4 and 8. */
    InputNormalise swapped_by_two_four_eight()
    {
        InputNormalise normalise;
        normalise.channels = ChannelOrder::Bgr;
        normalise.deviation = {2.0, 4.0, 8.0};
        return normalise;
    }

    /** @brief Returns a model whose input states @p normalise. */
    ModelSpec model_normalising(const InputNormalise& normalise)
    {
        ModelSpec model;
        model.input_normalise = normalise;
        return model;
    }

    /** @brief Returns a data block of two keys. */
    Value data_block()
    {
        return Value::object({"type", "batch_size"}, {Value::string("rings"), Value::integer(4)});
    }

    /** @brief Returns a data block of two keys and @p normalise as a third. */
    Value data_block_normalising(const Value& normalise)
    {
        return Value::object({"type", "batch_size", "normalise"},
                             {Value::string("rings"), Value::integer(4), normalise});
    }
} // namespace

/** @brief Tests that a model without a normalisation leaves the data block as it is. */
TEST(Small_DataNormalise, AModelWithoutANormalisationLeavesTheBlock)
{
    SA_ASSERT_OK_AND_ASSIGN(const Value taken,
                            normalise_data_to_model(data_block(), ModelSpec(), "model.json"));

    EXPECT_FALSE(taken.has("normalise"));
    EXPECT_EQ(taken.keys().size(), 2u);
}

/** @brief Tests that a block without a normalisation takes the model's, after its own keys. */
TEST(Small_DataNormalise, ABlockWithoutANormalisationTakesTheModels)
{
    const InputNormalise stated = swapped_by_two_four_eight();

    SA_ASSERT_OK_AND_ASSIGN(
        const Value taken,
        normalise_data_to_model(data_block(), model_normalising(stated), "model.json"));

    ASSERT_EQ(taken.keys().size(), 3u);
    EXPECT_EQ(taken.keys()[2], "normalise");
    SA_ASSERT_OK_AND_ASSIGN(const Value& object, taken.at("normalise", "test"));
    SA_ASSERT_OK_AND_ASSIGN(const InputNormalise read,
                            parse_input_normalise(object, "the data block's 'normalise'"));
    EXPECT_TRUE(read == stated);
}

/** @brief Tests that a block stating the model's normalisation keeps it. */
TEST(Small_DataNormalise, ABlockStatingTheModelsNormalisationKeepsIt)
{
    const InputNormalise stated = swapped_by_two_four_eight();

    SA_ASSERT_OK_AND_ASSIGN(
        const Value taken,
        normalise_data_to_model(data_block_normalising(input_normalise_value(stated)),
                                model_normalising(stated), "model.json"));

    EXPECT_EQ(taken.keys().size(), 3u);
}

/** @brief Tests that a block stating another normalisation than the model's is refused. */
TEST(Small_DataNormalise, ABlockStatingAnotherNormalisationIsRefused)
{
    const Value other = input_normalise_value(InputNormalise());

    SA_EXPECT_FAILURE(
        normalise_data_to_model(data_block_normalising(other),
                                model_normalising(swapped_by_two_four_eight()), "model.json"),
        StatusCode::InvalidDescription, "'model.json' states a 'normalise'", "states another");
}

/** @brief Tests that a block whose normalisation is malformed is refused with the key. */
TEST(Small_DataNormalise, ABlockWithAMalformedNormalisationIsRefused)
{
    const Value malformed = Value::object({"gamma"}, {Value::real(2.2)});

    SA_EXPECT_FAILURE(
        normalise_data_to_model(data_block_normalising(malformed),
                                model_normalising(swapped_by_two_four_eight()), "model.json"),
        StatusCode::InvalidDescription, "the data block's 'normalise'", "does not accept 'gamma'");
}
