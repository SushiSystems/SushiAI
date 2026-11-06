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
 * @file test_execution.cpp
 * @brief Tests that record_and_run runs what a recording recorded on a graph of its own and
 *        drops what a failed one left.
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
#include <SushiBLAS/engine/task_recorder.hpp>
#include <SushiBLAS/random/random.hpp>
#include <SushiRuntime/sushiruntime.hpp>
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
    /** @brief Provides the engine the record_and_run and run_once cases record on. */
    class Medium_ExecutionEngine : public SaTest::EngineTest
    {

    };

    /** @brief Holds the element count of the tensors the record_and_run cases fill. */
    constexpr std::int64_t K_FILL_ELEMENTS = 4;

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

    /** @brief Expects every element of @p tensor to equal @p value. */
    void expect_all(const SushiBLAS::Tensor& tensor, double value)
    {
        EXPECT_EQ(SaTest::read_tensor(tensor),
                  std::vector<double>(static_cast<std::size_t>(K_FILL_ELEMENTS), value));
    }
} // namespace

/** @brief Tests that record_and_run runs what a successful recording recorded. */
TEST_F(Medium_ExecutionEngine, RecordAndRunRunsWhatASuccessfulRecordingRecorded)
{
    SA_ASSERT_OK_AND_ASSIGN(SushiBLAS::Tensor target, filled_vector(*engine, 1.0));

    SA_EXPECT_OK(record_and_run(*engine, [&](SushiBLAS::TaskRecorder& recorder)
    {
        SushiBLAS::RandomOps random(recorder, engine->rng());
        random.constant(target, 2.0);
        return Status();
    }));

    expect_all(target, 2.0);
}

/** @brief Tests that a recording that throws is discarded and its failure is returned. */
TEST_F(Medium_ExecutionEngine, RecordAndRunDiscardsWhatAThrowingRecordingRecorded)
{
    SA_ASSERT_OK_AND_ASSIGN(SushiBLAS::Tensor discarded, filled_vector(*engine, 1.0));

    const Status failed = record_and_run(*engine, [&](SushiBLAS::TaskRecorder& recorder) -> Status
    {
        SushiBLAS::RandomOps random(recorder, engine->rng());
        random.constant(discarded, 5.0);
        throw std::invalid_argument("axpy: shapes disagree");
    });
    SA_EXPECT_FAILURE(failed, StatusCode::InvalidDescription, "axpy: shapes disagree");

    flush();
    expect_all(discarded, 1.0);
}

/** @brief Tests that a recording after a failed one runs and leaves the failed one's work out. */
TEST_F(Medium_ExecutionEngine, RecordAndRunRunsARecordingThatFollowsAFailedOne)
{
    SA_ASSERT_OK_AND_ASSIGN(SushiBLAS::Tensor discarded, filled_vector(*engine, 1.0));
    SA_ASSERT_OK_AND_ASSIGN(SushiBLAS::Tensor kept, filled_vector(*engine, 1.0));

    const Status failed = record_and_run(*engine, [&](SushiBLAS::TaskRecorder& recorder) -> Status
    {
        SushiBLAS::RandomOps random(recorder, engine->rng());
        random.constant(discarded, 5.0);
        throw std::invalid_argument("axpy: shapes disagree");
    });
    ASSERT_FALSE(failed.is_ok());

    SA_EXPECT_OK(record_and_run(*engine, [&](SushiBLAS::TaskRecorder& recorder)
    {
        SushiBLAS::RandomOps random(recorder, engine->rng());
        random.constant(kept, 2.0);
        return Status();
    }));

    expect_all(kept, 2.0);
    expect_all(discarded, 1.0);
}

/** @brief Tests that a recording that returns a failure is discarded and the failure returned. */
TEST_F(Medium_ExecutionEngine, RecordAndRunDiscardsWhenTheRecordingReturnsAFailure)
{
    SA_ASSERT_OK_AND_ASSIGN(SushiBLAS::Tensor discarded, filled_vector(*engine, 1.0));

    const Status failed = record_and_run(*engine, [&](SushiBLAS::TaskRecorder& recorder)
    {
        SushiBLAS::RandomOps random(recorder, engine->rng());
        random.constant(discarded, 5.0);
        return Status::from_message(StatusCode::Unsupported, []()
        {
            return std::string("no host path");
        });
    });
    SA_EXPECT_FAILURE(failed, StatusCode::Unsupported, "no host path");

    flush();
    expect_all(discarded, 1.0);
}
