/**************************************************************************/
/* test_gradient_clip.cpp                                                 */
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
 * @file test_gradient_clip.cpp
 * @brief Tests that an update is clipped to a global gradient norm by tasks of its own step.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <SushiBLAS/tensor/tensor.hpp>
#include <SushiAI/config/value.hpp>
#include <SushiAI/allocation/allocation.hpp>
#include <SushiAI/dtype/dtype.hpp>
#include <SushiAI/execution/execution_backend.hpp>
#include <SushiAI/execution/host_execution_backend.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/shape/shape.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/engine_fixture/engine_test.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/optim/adamw.hpp>
#include <SushiAI/optim/gradient_clip.hpp>
#include <SushiAI/optim/optimizer.hpp>
#include <SushiAI/optim/optimizer_registry.hpp>
#include <SushiAI/optim/sgd.hpp>
#include <SushiAI/tensor_values/tensor_values.hpp>
#include <support/tolerance.hpp>

using namespace SushiAI;
// SaTest stays reachable inside a TEST_F body; see tests/core/optim/README.md.
namespace SaTest = SushiAI::Test;

namespace
{
    /**
     * @brief Bounds the absolute error of a float32 value of order one after one update; 32 unit
     * roundoffs of float, the smallest power of two that covers 1e-6.
     */
    constexpr double K_UPDATE_TOLERANCE = 32.0 * TestSupport::unit_roundoff<float>();

    /** @brief Returns one slot per entry of @p gradients on @p backend, each parameter all zeros. */
    std::vector<Optim::ParameterSlot> make_slots(
        ExecutionBackend& backend, const std::vector<std::vector<double>>& gradients)
    {
        std::vector<Optim::ParameterSlot> slots;
        for (std::size_t i = 0; i < gradients.size(); ++i)
        {
            const Shape shape{static_cast<std::int64_t>(gradients[i].size())};
            Result<SushiBLAS::Tensor> parameter =
                allocate_tensor(backend.engine(), shape, DType::Float32);
            Result<SushiBLAS::Tensor> gradient =
                allocate_tensor(backend.engine(), shape, DType::Float32);
            EXPECT_TRUE(parameter.status().is_ok());
            EXPECT_TRUE(gradient.status().is_ok());

            SaTest::write_tensor(parameter.value(),
                                 std::vector<double>(gradients[i].size(), 0.0));
            SaTest::write_tensor(gradient.value(), gradients[i]);
            slots.push_back(Optim::ParameterSlot{"slot" + std::to_string(i),
                                                 std::move(parameter).value(),
                                                 std::move(gradient).value()});
        }
        return slots;
    }

    /** @brief Returns an SGD of rate 1 with no momentum and no decay over @p slots. */
    std::unique_ptr<Optim::SGD> make_sgd(ExecutionBackend& backend,
                                         std::vector<Optim::ParameterSlot> slots)
    {
        Optim::SGDOptions options;
        options.learning_rate = 1.0;
        Result<std::unique_ptr<Optim::SGD>> made =
            Optim::SGD::make(backend, std::move(slots), options);
        EXPECT_TRUE(made.status().is_ok());
        return std::move(made).value();
    }

    /** @brief Returns a matcher for a tensor read as doubles within the update tolerance. */
    auto holds(const std::vector<double>& expected)
    {
        return ::testing::Pointwise(::testing::DoubleNear(K_UPDATE_TOLERANCE), expected);
    }

    /** @brief Parses @p text as the attributes of an optimizer. */
    Config::Value attributes(std::string_view text)
    {
        return SushiAI::Test::require_value(Config::parse_json(text, "<test>"));
    }

    /** @brief Provides the engine and backend of the program to the cases that run a kernel. */
    class Medium_GradientClip : public SaTest::EngineTest
    {
    };

    /** @brief Provides the host backend to the cases that decide on values. */
    class Small_GradientClip : public ::testing::Test
    {
        protected:
            /** @brief Holds the host adapter the optimizers allocate from. */
            HostExecutionBackend backend;
    };

    /** @brief Provides the host backend to the cases that stop the process. */
    class Medium_GradientClipFatal : public Small_GradientClip
    {
    };

    /** @brief Names one clip norm and the parameter an SGD ends at; a norm of 0 means no clip. */
    struct ClipRow
    {
        double max_norm;                /**< The clip norm, or 0 for none. */
        std::vector<double> expected;   /**< The parameter after one step over the gradient 3, 4. */
    };

    /** @brief Provides the engine and backend of the program to each clip row. */
    class Medium_GradientClipRow : public Medium_GradientClip,
                                   public ::testing::WithParamInterface<ClipRow>
    {
    };

    /** @brief Provides the host backend to each optimizer type. */
    class Small_GradientClipType : public Small_GradientClip,
                                   public ::testing::WithParamInterface<std::string_view>
    {
    };

    /** @brief Names one norm the clip check refuses. */
    class Small_GradientClipNorm : public ::testing::TestWithParam<double>
    {
    };
} // namespace

/** @brief Tests that a gradient of norm 5 clipped to 1 moves the parameter by a fifth of it. */
TEST_F(Medium_GradientClip, ANormAboveTheThresholdScalesTheUpdate)
{
    std::vector<Optim::ParameterSlot> slots = make_slots(backend(), {{3.0, 4.0, 0.0, 0.0}});
    const SushiBLAS::Tensor parameter = slots.front().parameter;
    const std::unique_ptr<Optim::SGD> optimizer = make_sgd(backend(), std::move(slots));

    SA_EXPECT_OK(optimizer->enable_gradient_clip(1.0));
    SA_EXPECT_OK(optimizer->step());
    flush();

    EXPECT_THAT(SaTest::read_tensor(parameter), holds({-0.6, -0.8, 0.0, 0.0}));
    ASSERT_NE(optimizer->gradient_clip(), nullptr);
    EXPECT_NEAR(optimizer->gradient_clip()->applied_scale(), 0.2, K_UPDATE_TOLERANCE);
}

/** @brief Tests that a gradient whose norm is below the threshold is applied as it is. */
TEST_F(Medium_GradientClip, ANormBelowTheThresholdLeavesTheUpdate)
{
    std::vector<Optim::ParameterSlot> slots = make_slots(backend(), {{3.0, 4.0, 0.0, 0.0}});
    const SushiBLAS::Tensor parameter = slots.front().parameter;
    const std::unique_ptr<Optim::SGD> optimizer = make_sgd(backend(), std::move(slots));

    SA_EXPECT_OK(optimizer->enable_gradient_clip(10.0));
    SA_EXPECT_OK(optimizer->step());
    flush();

    EXPECT_THAT(SaTest::read_tensor(parameter), holds({-3.0, -4.0, 0.0, 0.0}));
    EXPECT_EQ(optimizer->gradient_clip()->applied_scale(), 1.0);
}

/** @brief Tests that the norm is taken over every slot, not slot by slot. */
TEST_F(Medium_GradientClip, TheNormIsTakenOverEverySlot)
{
    std::vector<Optim::ParameterSlot> slots =
        make_slots(backend(), {{3.0, 0.0}, {0.0, 4.0, 0.0}});
    const SushiBLAS::Tensor first = slots[0].parameter;
    const SushiBLAS::Tensor second = slots[1].parameter;
    const std::unique_ptr<Optim::SGD> optimizer = make_sgd(backend(), std::move(slots));

    // Each slot alone has a norm at or below 4; together they have 5.
    SA_EXPECT_OK(optimizer->enable_gradient_clip(2.5));
    SA_EXPECT_OK(optimizer->step());
    flush();

    EXPECT_THAT(SaTest::read_tensor(first), holds({-1.5, 0.0}));
    EXPECT_THAT(SaTest::read_tensor(second), holds({0.0, -2.0, 0.0}));
    EXPECT_EQ(optimizer->gradient_clip()->gradient_count(), 2u);
}

/** @brief Tests that a threshold set between two updates is the one the next update uses. */
TEST_F(Medium_GradientClip, AThresholdSetAtRunTimeIsTheNextUpdates)
{
    std::vector<Optim::ParameterSlot> slots = make_slots(backend(), {{3.0, 4.0}});
    const SushiBLAS::Tensor parameter = slots.front().parameter;
    const std::unique_ptr<Optim::SGD> optimizer = make_sgd(backend(), std::move(slots));

    SA_EXPECT_OK(optimizer->enable_gradient_clip(5.0));
    SA_EXPECT_OK(optimizer->step());
    flush();
    EXPECT_THAT(SaTest::read_tensor(parameter), holds({-3.0, -4.0}));

    optimizer->set_clip_norm(2.5);
    SA_EXPECT_OK(optimizer->step());
    flush();
    EXPECT_THAT(SaTest::read_tensor(parameter), holds({-4.5, -6.0}));
}

/** @brief Tests that the clip norm set at run time is the one the clip reports. */
TEST_F(Small_GradientClip, ASetClipNormIsTheOneTheClipHolds)
{
    const std::unique_ptr<Optim::SGD> optimizer =
        make_sgd(backend, make_slots(backend, {{3.0, 4.0}}));
    SA_EXPECT_OK(optimizer->enable_gradient_clip(5.0));

    optimizer->set_clip_norm(2.5);

    EXPECT_EQ(optimizer->gradient_clip()->max_norm(), 2.5);
}

/** @brief Tests that the momentum kernel of SGD reads the factor as the plain one does. */
TEST_F(Medium_GradientClip, TheMomentumKernelReadsTheFactor)
{
    std::vector<Optim::ParameterSlot> slots = make_slots(backend(), {{3.0, 4.0}});
    const SushiBLAS::Tensor parameter = slots.front().parameter;
    Optim::SGDOptions options;
    options.learning_rate = 1.0;
    options.momentum = 0.5;
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<Optim::SGD> optimizer,
                            Optim::SGD::make(backend(), std::move(slots), options));

    // The first update sets the buffer to the clipped gradient, so momentum adds nothing.
    SA_EXPECT_OK(optimizer->enable_gradient_clip(1.0));
    SA_EXPECT_OK(optimizer->step());
    flush();

    EXPECT_THAT(SaTest::read_tensor(parameter), holds({-0.6, -0.8}));
}

/** @brief Tests that AdamW computes the same factor through the same seam. */
TEST_F(Medium_GradientClip, AdamWReadsTheSameFactor)
{
    SA_ASSERT_OK_AND_ASSIGN(
        const std::unique_ptr<Optim::AdamW> optimizer,
        Optim::AdamW::make(backend(), make_slots(backend(), {{3.0, 4.0}}),
                           Optim::AdamWOptions{}));

    SA_EXPECT_OK(optimizer->enable_gradient_clip(1.0));
    SA_EXPECT_OK(optimizer->step());
    flush();

    EXPECT_NEAR(optimizer->gradient_clip()->applied_scale(), 0.2, K_UPDATE_TOLERANCE);
}

/** @brief Tests that a clipped step applies the clip and an unclipped one does not. */
TEST_P(Medium_GradientClipRow, AClippedStepAppliesTheClipAndAnUnclippedOneDoesNot)
{
    std::vector<Optim::ParameterSlot> slots = make_slots(backend(), {{3.0, 4.0}});
    const SushiBLAS::Tensor parameter = slots.front().parameter;
    const std::unique_ptr<Optim::SGD> optimizer = make_sgd(backend(), std::move(slots));
    if (GetParam().max_norm > 0.0)
    {
        SA_EXPECT_OK(optimizer->enable_gradient_clip(GetParam().max_norm));
    }

    SA_EXPECT_OK(optimizer->step());
    flush();

    EXPECT_THAT(SaTest::read_tensor(parameter), holds(GetParam().expected));
}

INSTANTIATE_TEST_SUITE_P(Rows, Medium_GradientClipRow,
                         ::testing::Values(ClipRow{0.0, {-3.0, -4.0}},
                                           ClipRow{1.0, {-0.6, -0.8}}));

/** @brief Tests that a norm that is not positive and finite is refused. */
TEST_P(Small_GradientClipNorm, TheCheckRefusesANormThatIsNotPositiveAndFinite)
{
    SA_EXPECT_FAILURE(Optim::check_clip_norm(GetParam()), StatusCode::InvalidDescription,
                      "positive and finite");
}

INSTANTIATE_TEST_SUITE_P(Rows, Small_GradientClipNorm,
                         ::testing::Values(0.0, -1.0,
                                           std::numeric_limits<double>::infinity(),
                                           std::numeric_limits<double>::quiet_NaN()));

/** @brief Tests that a positive finite norm is accepted. */
TEST(Small_GradientClipCheck, TheCheckAcceptsAPositiveFiniteNorm)
{
    SA_EXPECT_OK(Optim::check_clip_norm(0.5));
}

/** @brief Tests that an update recorded as separate operations cannot carry a clip. */
TEST(Small_GradientClipCheck, TheCheckRefusesAnUpdateThatDoesNotReplay)
{
    SA_EXPECT_FAILURE(Optim::check_clip_kernel(false), StatusCode::InvalidDescription,
                      "UpdateKernel::Fused");
    SA_EXPECT_OK(Optim::check_clip_kernel(true));
}

/** @brief Tests that a refused clip norm leaves the optimizer without a clip. */
TEST_F(Small_GradientClip, ARefusedClipLeavesNoClip)
{
    const std::unique_ptr<Optim::SGD> optimizer =
        make_sgd(backend, make_slots(backend, {{3.0, 4.0}}));

    SA_EXPECT_FAILURE(optimizer->enable_gradient_clip(0.0), StatusCode::InvalidDescription,
                      "positive and finite");

    EXPECT_EQ(optimizer->gradient_clip(), nullptr);
}

/** @brief Tests that a separate kernel refuses a clip through the optimizer. */
TEST_F(Small_GradientClip, ASeparateKernelRefusesTheClip)
{
    Optim::SGDOptions separate;
    separate.kernel = Optim::UpdateKernel::Separate;
    SA_ASSERT_OK_AND_ASSIGN(
        const std::unique_ptr<Optim::SGD> optimizer,
        Optim::SGD::make(backend, make_slots(backend, {{3.0, 4.0}}), separate));

    SA_EXPECT_FAILURE(optimizer->enable_gradient_clip(1.0), StatusCode::InvalidDescription,
                      "UpdateKernel::Fused");
}

/** @brief Tests that a norm set on an optimizer with no clip stops the process. */
TEST_F(Medium_GradientClipFatal, ANormSetWithoutAClipIsFatal)
{
    const std::unique_ptr<Optim::SGD> optimizer =
        make_sgd(backend, make_slots(backend, {{3.0, 4.0}}));

    SA_EXPECT_FATAL(optimizer->set_clip_norm(1.0), "no gradient clip");
}

/** @brief Tests that a norm no clip can use stops the process. */
TEST_F(Medium_GradientClipFatal, ANormThatIsNotPositiveAndFiniteIsFatal)
{
    const std::unique_ptr<Optim::SGD> optimizer =
        make_sgd(backend, make_slots(backend, {{3.0, 4.0}}));
    SA_EXPECT_OK(optimizer->enable_gradient_clip(1.0));

    SA_EXPECT_FATAL(optimizer->set_clip_norm(0.0), "clip norm", "positive and finite");
}

/** @brief Tests that the key clip_norm enables the clip for each optimizer type. */
TEST_P(Small_GradientClipType, TheKeyClipNormEnablesTheClip)
{
    SA_ASSERT_OK_AND_ASSIGN(
        const std::unique_ptr<Optim::Optimizer> clipped,
        Optim::make_optimizer(backend, GetParam(), make_slots(backend, {{3.0, 4.0}}),
                              attributes(R"({"clip_norm": 0.5})"), "optimizer"));

    ASSERT_NE(clipped->gradient_clip(), nullptr);
    EXPECT_EQ(clipped->gradient_clip()->max_norm(), 0.5);
}

/** @brief Tests that without the key clip_norm no clip is enabled. */
TEST_P(Small_GradientClipType, WithoutTheKeyNoClipIsEnabled)
{
    SA_ASSERT_OK_AND_ASSIGN(
        const std::unique_ptr<Optim::Optimizer> plain,
        Optim::make_optimizer(backend, GetParam(), make_slots(backend, {{3.0, 4.0}}),
                              attributes("{}"), "optimizer"));

    EXPECT_EQ(plain->gradient_clip(), nullptr);
}

/** @brief Tests that a clip_norm of 0 is refused naming the key. */
TEST_P(Small_GradientClipType, AClipNormOfZeroIsRefused)
{
    SA_EXPECT_FAILURE(
        Optim::make_optimizer(backend, GetParam(), make_slots(backend, {{3.0, 4.0}}),
                              attributes(R"({"clip_norm": 0})"), "optimizer"),
        StatusCode::InvalidDescription, "optimizer 'clip_norm'", "positive and finite");
}

INSTANTIATE_TEST_SUITE_P(Rows, Small_GradientClipType,
                         ::testing::Values(std::string_view("sgd"), std::string_view("adamw")));
