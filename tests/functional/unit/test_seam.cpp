/**************************************************************************/
/* test_seam.cpp                                                          */
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
 * @file test_seam.cpp
 * @brief Tests the Status the seam gives each exception type, and what its guards pass through.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstdint>
#include <exception>
#include <new>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>
#include <gtest/gtest.h>
#include <SushiBLAS/core/errors.hpp>
#include <SushiBLAS/engine/math/random.hpp>
#include <SushiRuntime/core/error.hpp>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/core/allocation.hpp>
#include <SushiAI/core/common.hpp>
#include <SushiAI/core/result.hpp>
#include <SushiAI/core/seam.hpp>
#include <SushiAI/core/status.hpp>
#include "../../common/test_status.hpp"
#include "../../common/test_utils.hpp"

using SushiAI::Result;
using SushiAI::Status;
using SushiAI::StatusCode;
using SushiAI::Detail::guard;
using SushiAI::Detail::guard_value;
using SushiAI::Detail::record_and_run;
using SushiAI::Detail::SeamPhase;
using SushiRuntime::Core::Errc;

namespace SaTest = SushiAI::Test;

namespace
{
    /** @brief Provides the engine the record_and_run cases record on. */
    class Unit_SeamEngine : public SaTest::EngineTest
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

    /** @brief Returns the Status a guard gives a call that throws @p thrown under @p phase. */
    template<typename Thrown>
    Status status_of_throw(SeamPhase phase, const Thrown& thrown) noexcept
    {
        return guard(phase, [&]() -> Status
        {
            throw thrown;
        });
    }

    /** @brief Builds the exception SushiRuntime throws for @p code, with the message @p text. */
    SushiRuntime::Core::SushiException runtime_failure(Errc code, std::string_view text)
    {
        return SushiRuntime::Core::SushiException(
            SushiRuntime::Core::Error{code, std::string(text)});
    }

    /** @brief Derives from std::exception alone, as an exception of SYCL does. */
    class ForeignException final : public std::exception
    {
        public:
            /** @brief Returns the fixed text of this exception. */
            [[nodiscard]] const char* what() const noexcept override
            {
                return "the device queue faulted";
            }
    };

} // namespace

TEST(Unit_Seam, GuardReturnsTheStatusOfACallThatDoesNotThrow)
{
    SA_EXPECT_OK(guard(SeamPhase::Record, []()
    {
        return Status();
    }));

    const Status failed = guard(SeamPhase::Record, []()
    {
        return Status::from_message(StatusCode::InvalidFile, []()
        {
            return std::string("wrong magic");
        });
    });
    SA_EXPECT_FAILURE(failed, StatusCode::InvalidFile, "wrong magic");
}

TEST(Unit_Seam, GuardValueReturnsTheValueOfACallThatDoesNotThrow)
{
    const Result<int> result = guard_value(SeamPhase::Allocate, []()
    {
        return 7;
    });

    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result.value(), 7);
}

TEST(Unit_Seam, GuardValueReturnsTheStatusOfWhatTheCallThrew)
{
    const Result<int> result = guard_value(SeamPhase::Allocate, []() -> int
    {
        throw std::bad_alloc();
    });

    EXPECT_FALSE(result.has_value());
    EXPECT_EQ(result.status().code(), StatusCode::OutOfMemory);
}

TEST(Unit_Seam, UnsupportedExecutionIsUnsupportedThoughItIsAnInvalidArgument)
{
    SA_EXPECT_FAILURE(
        status_of_throw(SeamPhase::Record, SushiBLAS::UnsupportedExecution("no host path")),
        StatusCode::Unsupported, "no host path");
}

TEST(Unit_Seam, InvalidArgumentIsInvalidDescription)
{
    SA_EXPECT_FAILURE(
        status_of_throw(SeamPhase::Record, std::invalid_argument("gemm: shapes disagree")),
        StatusCode::InvalidDescription, "gemm: shapes disagree");
}

TEST(Unit_Seam, PlainRuntimeErrorTakesItsCodeFromThePhase)
{
    const std::runtime_error error("the storage rule refused it");

    SA_EXPECT_FAILURE(status_of_throw(SeamPhase::Allocate, error), StatusCode::OutOfMemory,
                      "the storage rule refused it");
    SA_EXPECT_FAILURE(status_of_throw(SeamPhase::Record, error), StatusCode::DeviceFailed,
                      "the storage rule refused it");
    SA_EXPECT_FAILURE(status_of_throw(SeamPhase::Execute, error), StatusCode::DeviceFailed,
                      "the storage rule refused it");
}

TEST(Unit_Seam, RuntimeFailureThatTheUserCanActOnMapsByItsCode)
{
    SA_EXPECT_FAILURE(
        status_of_throw(SeamPhase::Execute, runtime_failure(Errc::allocation_failed, "no USM")),
        StatusCode::OutOfMemory, "no USM");
    SA_EXPECT_FAILURE(
        status_of_throw(SeamPhase::Execute, runtime_failure(Errc::kernel_failed, "a fault")),
        StatusCode::DeviceFailed, "a fault");
    SA_EXPECT_FAILURE(
        status_of_throw(SeamPhase::Execute, runtime_failure(Errc::device_lost, "unplugged")),
        StatusCode::DeviceFailed, "unplugged");
    SA_EXPECT_FAILURE(
        status_of_throw(SeamPhase::Allocate, runtime_failure(Errc::unsupported, "no fp16")),
        StatusCode::Unsupported, "no fp16");
}

TEST(Unit_Seam, RuntimeFailureWithTheOkCodeTakesItsCodeFromThePhase)
{
    SA_EXPECT_FAILURE(
        status_of_throw(SeamPhase::Allocate, runtime_failure(Errc::ok, "no cause named")),
        StatusCode::OutOfMemory, "no cause named");
    SA_EXPECT_FAILURE(
        status_of_throw(SeamPhase::Execute, runtime_failure(Errc::ok, "no cause named")),
        StatusCode::DeviceFailed, "no cause named");
}

TEST(Unit_Seam, RuntimeFailureThatReportsADefectIsFatal)
{
    SA_EXPECT_FATAL(
        (void)status_of_throw(SeamPhase::Execute, runtime_failure(Errc::invalid_graph, "cyclic")),
        "SushiRuntime reported invalid_graph", "cyclic");
    SA_EXPECT_FATAL(
        (void)status_of_throw(SeamPhase::Execute, runtime_failure(Errc::compile_failed, "deps")),
        "SushiRuntime reported compile_failed", "deps");
    SA_EXPECT_FATAL(
        (void)status_of_throw(SeamPhase::Execute, runtime_failure(Errc::invalid_access, "host")),
        "SushiRuntime reported invalid_access", "host");
    SA_EXPECT_FATAL(
        (void)status_of_throw(SeamPhase::Execute,
                              runtime_failure(Errc::dangling_resource, "freed")),
        "SushiRuntime reported dangling_resource", "freed");
}

TEST(Unit_Seam, ErrorOfAHostUnitIsFatalWithItsText)
{
    SA_EXPECT_FATAL(
        (void)status_of_throw(SeamPhase::Record, SushiAI::Error("value 9 is out of range")),
        "value 9 is out of range");
}

TEST(Unit_Seam, BadAllocIsOutOfMemoryInEveryPhase)
{
    EXPECT_EQ(status_of_throw(SeamPhase::Record, std::bad_alloc()).code(),
              StatusCode::OutOfMemory);
    EXPECT_EQ(status_of_throw(SeamPhase::Execute, std::bad_alloc()).code(),
              StatusCode::OutOfMemory);
}

TEST(Unit_Seam, AnyOtherStandardExceptionIsDeviceFailed)
{
    SA_EXPECT_FAILURE(status_of_throw(SeamPhase::Allocate, ForeignException()),
                      StatusCode::DeviceFailed, "the device queue faulted");
    SA_EXPECT_FAILURE(status_of_throw(SeamPhase::Record, std::logic_error("a broken rule")),
                      StatusCode::DeviceFailed, "a broken rule");
}

TEST(Unit_Seam, ExceptionOfAnUnknownTypeIsDeviceFailed)
{
    SA_EXPECT_FAILURE(status_of_throw(SeamPhase::Allocate, 3), StatusCode::DeviceFailed,
                      "an exception of an unknown type");
}

TEST_F(Unit_SeamEngine, RecordAndRunRunsWhatASuccessfulRecordingRecorded)
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

TEST_F(Unit_SeamEngine, RecordAndRunDiscardsWhatAFailedRecordingRecorded)
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

TEST_F(Unit_SeamEngine, RecordAndRunDiscardsWhenTheRecordingReturnsAFailure)
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
