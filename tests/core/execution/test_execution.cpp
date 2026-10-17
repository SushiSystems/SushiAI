/**************************************************************************/
/* test_execution.cpp                                                     */
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
 * @file test_execution.cpp
 * @brief Tests that record_and_run runs what a recording recorded and discards what a failed
 *        one left.
 * @author Mustafa Garip
 */

#include <SushiAI/execution/execution.hpp>

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include <gtest/gtest.h>
#include <SushiBLAS/random/random.hpp>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/allocation/allocation.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/engine_fixture/engine_test.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/tensor_values/tensor_values.hpp>

using SushiAI::Result;
using SushiAI::Status;
using SushiAI::StatusCode;
using SushiAI::Detail::record_and_run;
namespace SaTest = SushiAI::Test;

namespace
{
    /** @brief Provides the engine the record_and_run cases record on. */
    class Medium_ExecutionEngine : public SaTest::EngineTest
    {

    };

    /** @brief Holds the element count of the tensors the record_and_run cases fill. */
    constexpr std::int64_t K_FILL_ELEMENTS = 4;

    /** @brief Holds the error allowed in a filled element; a constant fill stores the value. */
    constexpr double K_FILL_TOLERANCE = 0.0;

    /** @brief Allocates a float32 vector of K_FILL_ELEMENTS whose elements are all @p value. */
    Result<SushiBLAS::Tensor> filled_vector(SushiBLAS::Engine& engine, double value)
    {
        const std::int64_t dims[1] = {K_FILL_ELEMENTS};
        SA_TRY_ASSIGN(SushiBLAS::Tensor tensor,
                      SushiAI::allocate_tensor(engine,
                                               SushiRuntime::span<const std::int64_t>(dims, 1)));

        const std::vector<double> values(static_cast<std::size_t>(K_FILL_ELEMENTS), value);
        SaTest::write_tensor(tensor,
                             SushiRuntime::span<const double>(values.data(), values.size()));
        return std::move(tensor);
    }

    /** @brief Expects every element of @p tensor to be @p value within K_FILL_TOLERANCE. */
    void expect_all(const SushiBLAS::Tensor& tensor, double value)
    {
        for (double element : SaTest::read_tensor(tensor))
            EXPECT_NEAR(element, value, K_FILL_TOLERANCE);
    }
} // namespace

TEST_F(Medium_ExecutionEngine, RecordAndRunRunsWhatASuccessfulRecordingRecorded)
{
    SA_ASSERT_OK_AND_ASSIGN(SushiBLAS::Tensor target, filled_vector(*engine, 1.0));

    SA_EXPECT_OK(record_and_run(*engine, [&]()
    {
        SushiBLAS::RandomOps random(*engine);
        random.constant(target, 2.0);
        return Status();
    }));

    expect_all(target, 2.0);
}

TEST_F(Medium_ExecutionEngine, RecordAndRunDiscardsWhatAFailedRecordingRecorded)
{
    SA_ASSERT_OK_AND_ASSIGN(SushiBLAS::Tensor discarded, filled_vector(*engine, 1.0));
    SA_ASSERT_OK_AND_ASSIGN(SushiBLAS::Tensor kept, filled_vector(*engine, 1.0));

    const Status failed = record_and_run(*engine, [&]() -> Status
    {
        SushiBLAS::RandomOps random(*engine);
        random.constant(discarded, 5.0);
        throw std::invalid_argument("axpy: shapes disagree");
    });
    SA_EXPECT_FAILURE(failed, StatusCode::InvalidDescription, "axpy: shapes disagree");

    SA_EXPECT_OK(record_and_run(*engine, [&]()
    {
        SushiBLAS::RandomOps random(*engine);
        random.constant(kept, 2.0);
        return Status();
    }));

    expect_all(kept, 2.0);
    expect_all(discarded, 1.0);
}

TEST_F(Medium_ExecutionEngine, RecordAndRunDiscardsWhenTheRecordingReturnsAFailure)
{
    SA_ASSERT_OK_AND_ASSIGN(SushiBLAS::Tensor discarded, filled_vector(*engine, 1.0));

    const Status failed = record_and_run(*engine, [&]()
    {
        SushiBLAS::RandomOps random(*engine);
        random.constant(discarded, 5.0);
        return Status::from_message(StatusCode::Unsupported, []()
        {
            return std::string("no host path");
        });
    });
    SA_EXPECT_FAILURE(failed, StatusCode::Unsupported, "no host path");

    SA_EXPECT_OK(SushiAI::Detail::run_recorded(*engine));
    expect_all(discarded, 1.0);
}
