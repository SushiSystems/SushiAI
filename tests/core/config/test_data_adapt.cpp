/**************************************************************************/
/* test_data_adapt.cpp                                                    */
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
 * @file test_data_adapt.cpp
 * @brief Tests that a data block takes both the fit and the normalisation its model states.
 * @author Mustafa Garip
 */

#include <string>

#include <gtest/gtest.h>
#include <SushiAI/config/data_adapt.hpp>
#include <SushiAI/config/input_normalise.hpp>
#include <SushiAI/config/model_config.hpp>
#include <SushiAI/config/value.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>

using namespace SushiAI;
using namespace SushiAI::Config;

namespace
{
    /** @brief Returns a data block of two keys. */
    Value data_block()
    {
        return Value::object({"type", "batch_size"}, {Value::string("rings"), Value::integer(4)});
    }

    /** @brief Returns a model whose input states a resize fit and the default normalisation. */
    ModelSpec model_fitting_and_normalising()
    {
        ModelSpec model;
        model.input_fit = std::string(K_FIT_RESIZE);
        model.input_normalise = InputNormalise();
        return model;
    }
} // namespace

/** @brief Tests that a model stating neither leaves the data block as it is. */
TEST(Small_DataAdapt, AModelStatingNeitherLeavesTheBlock)
{
    SA_ASSERT_OK_AND_ASSIGN(const Value adapted,
                            adapt_data_to_model(data_block(), ModelSpec(), "model.json"));

    EXPECT_EQ(adapted.keys().size(), 2u);
}

/** @brief Tests that a block takes the fit, then the normalisation, after its own keys. */
TEST(Small_DataAdapt, ABlockTakesTheFitThenTheNormalisation)
{
    SA_ASSERT_OK_AND_ASSIGN(
        const Value adapted,
        adapt_data_to_model(data_block(), model_fitting_and_normalising(), "model.json"));

    ASSERT_EQ(adapted.keys().size(), 4u);
    EXPECT_EQ(adapted.keys()[2], "fit");
    EXPECT_EQ(adapted.keys()[3], "normalise");
}

/** @brief Tests that a block stating another fit is refused before the normalisation is read. */
TEST(Small_DataAdapt, ABlockStatingAnotherFitIsRefused)
{
    const Value stated = Value::object({"type", "fit"},
                                       {Value::string("rings"), Value::string(std::string(K_FIT_LETTERBOX))});

    SA_EXPECT_FAILURE(adapt_data_to_model(stated, model_fitting_and_normalising(), "model.json"),
                      StatusCode::InvalidDescription, "'model.json' fits its images by 'resize'",
                      "another 'fit'");
}

/** @brief Tests that a block stating another normalisation is refused. */
TEST(Small_DataAdapt, ABlockStatingAnotherNormalisationIsRefused)
{
    InputNormalise other;
    other.deviation = {2.0, 4.0, 8.0};
    const Value stated = Value::object({"type", "normalise"},
                                       {Value::string("rings"), input_normalise_value(other)});

    SA_EXPECT_FAILURE(adapt_data_to_model(stated, model_fitting_and_normalising(), "model.json"),
                      StatusCode::InvalidDescription, "'model.json' states a 'normalise'",
                      "states another");
}
