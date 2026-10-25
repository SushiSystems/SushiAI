/**************************************************************************/
/* test_data_fit.cpp                                                      */
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
 * @file test_data_fit.cpp
 * @brief Tests how a data block takes the image fit its model file states.
 * @author Mustafa Garip
 */

#include <string>

#include <gtest/gtest.h>
#include <SushiAI/config/data_fit.hpp>
#include <SushiAI/config/model_config.hpp>
#include <SushiAI/config/value.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>

using namespace SushiAI;
using namespace SushiAI::Config;

namespace
{
    /** @brief Returns a data block of two keys, with "fit" as a third when @p fit is not empty. */
    Value data_block(const std::string& fit)
    {
        if (fit.empty())
            return Value::object({"type", "batch_size"},
                                 {Value::string("rings"), Value::integer(4)});

        return Value::object({"type", "batch_size", "fit"},
                             {Value::string("rings"), Value::integer(4), Value::string(fit)});
    }

    /** @brief Returns a model whose input states @p fit. */
    ModelSpec model_fitting(const std::string& fit)
    {
        ModelSpec model;
        model.input_fit = fit;
        return model;
    }
} // namespace

/** @brief Tests that a model without a fit leaves the data block as it is. */
TEST(Small_DataFit, AModelWithoutAFitLeavesTheBlock)
{
    SA_ASSERT_OK_AND_ASSIGN(const Value fitted,
                            fit_data_to_model(data_block(""), model_fitting(""), "model.json"));

    EXPECT_FALSE(fitted.has("fit"));
    EXPECT_EQ(fitted.keys().size(), 2u);
}

/** @brief Tests that a block without a fit takes the model's, after its own keys. */
TEST(Small_DataFit, ABlockWithoutAFitTakesTheModels)
{
    SA_ASSERT_OK_AND_ASSIGN(
        const Value fitted,
        fit_data_to_model(data_block(""), model_fitting(std::string(K_FIT_RESIZE)), "model.json"));

    ASSERT_EQ(fitted.keys().size(), 3u);
    EXPECT_EQ(fitted.keys()[2], "fit");
    EXPECT_EQ(fitted.at("fit", "test").as_string("test"), K_FIT_RESIZE);
    EXPECT_EQ(fitted.at("batch_size", "test").as_int("test"), 4);
}

/** @brief Tests that a block stating the model's fit is kept and one stating another refused. */
TEST(Small_DataFit, ABlockStatingAnotherFitIsRefused)
{
    const ModelSpec model = model_fitting(std::string(K_FIT_RESIZE));

    SA_ASSERT_OK_AND_ASSIGN(
        const Value same,
        fit_data_to_model(data_block(std::string(K_FIT_RESIZE)), model, "model.json"));
    EXPECT_EQ(same.at("fit", "test").as_string("test"), K_FIT_RESIZE);

    SA_EXPECT_FAILURE(
        fit_data_to_model(data_block(std::string(K_FIT_LETTERBOX)), model, "model.json"),
        StatusCode::InvalidDescription, "'model.json' fits its images by 'resize'",
        "another 'fit'");
}
