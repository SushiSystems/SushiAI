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
 * @file test_gradient_clip.cpp
 * @brief Tests that an update is clipped to a global gradient norm by tasks of its own step.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>
#include <gtest/gtest.h>
#include <SushiBLAS/tensor/tensor.hpp>
#include <SushiAI/config/value.hpp>
#include <SushiAI/allocation/allocation.hpp>
#include <SushiAI/dtype/dtype.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/shape/shape.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/engine_fixture/engine_test.hpp>
#include <SushiAI/engine_fixture/test_graph.hpp>
#include <SushiAI/graph/step_recording.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/optim/adamw.hpp>
#include <SushiAI/optim/gradient_clip.hpp>
#include <SushiAI/optim/optimizer.hpp>
#include <SushiAI/optim/optimizer_registry.hpp>
#include <SushiAI/optim/sgd.hpp>
#include <SushiAI/tensor_values/tensor_values.hpp>

using namespace SushiAI;
// SaTest stays reachable inside a TEST_F body; see tests/core/optim/README.md.
namespace SaTest = SushiAI::Test;

namespace
{
    /** @brief Bounds the absolute error of a float32 value of order one after one update. */
    constexpr double K_UPDATE_TOLERANCE = 1e-6;

    /** @brief Bounds the error of a norm a double holds as it was set. */
    constexpr double K_EXACT_TOLERANCE = 0.0;

    /** @brief Provides an engine and slots of zero parameters with stated gradients. */
    class Medium_GradientClip : public SaTest::EngineTest
    {
        protected:
            /** @brief Returns one slot per entry of @p gradients, each parameter all zeros. */
            std::vector<Optim::ParameterSlot> make_slots(
                const std::vector<std::vector<double>>& gradients)
            {
                std::vector<Optim::ParameterSlot> slots;
                for (std::size_t i = 0; i < gradients.size(); ++i)
                {
                    const Shape shape{static_cast<std::int64_t>(gradients[i].size())};
                    Result<SushiBLAS::Tensor> parameter =
                        allocate_tensor(*engine, shape, DType::Float32);
                    Result<SushiBLAS::Tensor> gradient =
                        allocate_tensor(*engine, shape, DType::Float32);
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
            std::unique_ptr<Optim::SGD> make_sgd(std::vector<Optim::ParameterSlot> slots)
            {
                Optim::SGDOptions options;
                options.learning_rate = 1.0;
                Result<std::unique_ptr<Optim::SGD>> made =
                    Optim::SGD::make(*engine, std::move(slots), options);
                EXPECT_TRUE(made.status().is_ok());
                return std::move(made).value();
            }

            /** @brief Expects @p tensor to hold @p expected, element by element. */
            void expect_values(const SushiBLAS::Tensor& tensor,
                               const std::vector<double>& expected)
            {
                const std::vector<double> found = SaTest::read_tensor(tensor);
                ASSERT_EQ(found.size(), expected.size());
                for (std::size_t i = 0; i < expected.size(); ++i)
                    EXPECT_NEAR(found[i], expected[i], K_UPDATE_TOLERANCE) << "element " << i;
            }
    };
} // namespace

/** @brief Tests that a gradient of norm 5 clipped to 1 moves the parameter by a fifth of it. */
TEST_F(Medium_GradientClip, ANormAboveTheThresholdScalesTheUpdate)
{
    std::vector<Optim::ParameterSlot> slots = make_slots({{3.0, 4.0, 0.0, 0.0}});
    const SushiBLAS::Tensor parameter = slots.front().parameter;
    const std::unique_ptr<Optim::SGD> optimizer = make_sgd(std::move(slots));

    SA_EXPECT_OK(optimizer->enable_gradient_clip(1.0));
    SA_EXPECT_OK(optimizer->step());
    flush();

    expect_values(parameter, {-0.6, -0.8, 0.0, 0.0});
    ASSERT_NE(optimizer->gradient_clip(), nullptr);
    EXPECT_NEAR(optimizer->gradient_clip()->applied_scale(), 0.2, K_UPDATE_TOLERANCE);
}

/** @brief Tests that a gradient whose norm is below the threshold is applied as it is. */
TEST_F(Medium_GradientClip, ANormBelowTheThresholdLeavesTheUpdate)
{
    std::vector<Optim::ParameterSlot> slots = make_slots({{3.0, 4.0, 0.0, 0.0}});
    const SushiBLAS::Tensor parameter = slots.front().parameter;
    const std::unique_ptr<Optim::SGD> optimizer = make_sgd(std::move(slots));

    SA_EXPECT_OK(optimizer->enable_gradient_clip(10.0));
    SA_EXPECT_OK(optimizer->step());
    flush();

    expect_values(parameter, {-3.0, -4.0, 0.0, 0.0});
    EXPECT_NEAR(optimizer->gradient_clip()->applied_scale(), 1.0, K_EXACT_TOLERANCE);
}

/** @brief Tests that the norm is taken over every slot, not slot by slot. */
TEST_F(Medium_GradientClip, TheNormIsTakenOverEverySlot)
{
    std::vector<Optim::ParameterSlot> slots = make_slots({{3.0, 0.0}, {0.0, 4.0, 0.0}});
    const SushiBLAS::Tensor first = slots[0].parameter;
    const SushiBLAS::Tensor second = slots[1].parameter;
    const std::unique_ptr<Optim::SGD> optimizer = make_sgd(std::move(slots));

    // Each slot alone has a norm at or below 4; together they have 5.
    SA_EXPECT_OK(optimizer->enable_gradient_clip(2.5));
    SA_EXPECT_OK(optimizer->step());
    flush();

    expect_values(first, {-1.5, 0.0});
    expect_values(second, {0.0, -2.0, 0.0});
    EXPECT_EQ(optimizer->gradient_clip()->gradient_count(), 2u);
}

/** @brief Tests that a threshold set between two updates is the one the next update uses. */
TEST_F(Medium_GradientClip, AThresholdSetAtRunTimeIsTheNextUpdates)
{
    std::vector<Optim::ParameterSlot> slots = make_slots({{3.0, 4.0}});
    const SushiBLAS::Tensor parameter = slots.front().parameter;
    const std::unique_ptr<Optim::SGD> optimizer = make_sgd(std::move(slots));

    SA_EXPECT_OK(optimizer->enable_gradient_clip(5.0));
    SA_EXPECT_OK(optimizer->step());
    flush();
    expect_values(parameter, {-3.0, -4.0});

    optimizer->set_clip_norm(2.5);
    EXPECT_NEAR(optimizer->gradient_clip()->max_norm(), 2.5, K_EXACT_TOLERANCE);
    SA_EXPECT_OK(optimizer->step());
    flush();
    expect_values(parameter, {-4.5, -6.0});
}

/** @brief Tests that the momentum kernel of SGD reads the factor as the plain one does. */
TEST_F(Medium_GradientClip, TheMomentumKernelReadsTheFactor)
{
    std::vector<Optim::ParameterSlot> slots = make_slots({{3.0, 4.0}});
    const SushiBLAS::Tensor parameter = slots.front().parameter;
    Optim::SGDOptions options;
    options.learning_rate = 1.0;
    options.momentum = 0.5;
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<Optim::SGD> optimizer,
                            Optim::SGD::make(*engine, std::move(slots), options));

    // The first update sets the buffer to the clipped gradient, so momentum adds nothing.
    SA_EXPECT_OK(optimizer->enable_gradient_clip(1.0));
    SA_EXPECT_OK(optimizer->step());
    flush();

    expect_values(parameter, {-0.6, -0.8});
}

/** @brief Tests that AdamW computes the same factor through the same seam. */
TEST_F(Medium_GradientClip, AdamWReadsTheSameFactor)
{
    SA_ASSERT_OK_AND_ASSIGN(
        const std::unique_ptr<Optim::AdamW> optimizer,
        Optim::AdamW::make(*engine, make_slots({{3.0, 4.0}}), Optim::AdamWOptions{}));

    SA_EXPECT_OK(optimizer->enable_gradient_clip(1.0));
    SA_EXPECT_OK(optimizer->step());
    flush();

    EXPECT_NEAR(optimizer->gradient_clip()->applied_scale(), 0.2, K_UPDATE_TOLERANCE);
}

/** @brief Tests that a clip adds one task per gradient and one more, and none when absent. */
TEST_F(Medium_GradientClip, AClipAddsItsTasksAndItsAbsenceAddsNone)
{
    const std::unique_ptr<Optim::SGD> plain = make_sgd(make_slots({{3.0, 0.0}, {0.0, 4.0}}));
    const std::unique_ptr<Optim::SGD> clipped = make_sgd(make_slots({{3.0, 0.0}, {0.0, 4.0}}));
    SA_EXPECT_OK(clipped->enable_gradient_clip(1.0));

    SaTest::TestGraph recording(*engine);
    const Graph::StepRecording step(recording.recorder(), recording.graph_recorder());
    SA_ASSERT_OK_AND_ASSIGN(const std::size_t without, plain->record_into_step(step));
    SA_ASSERT_OK_AND_ASSIGN(const std::size_t with, clipped->record_into_step(step));
    SA_EXPECT_OK(recording.run());
    flush();

    EXPECT_EQ(plain->gradient_clip(), nullptr);
    EXPECT_EQ(without, 2u);
    EXPECT_EQ(with, 5u);
}

/** @brief Tests each description a clip refuses. */
TEST_F(Medium_GradientClip, AClipRefusesWhatItCannotCompute)
{
    const std::unique_ptr<Optim::SGD> optimizer = make_sgd(make_slots({{3.0, 4.0}}));
    SA_EXPECT_FAILURE(optimizer->enable_gradient_clip(0.0), StatusCode::InvalidDescription,
                      "positive and finite");
    SA_EXPECT_FAILURE(optimizer->enable_gradient_clip(-1.0), StatusCode::InvalidDescription,
                      "positive and finite");
    EXPECT_EQ(optimizer->gradient_clip(), nullptr);

    Optim::SGDOptions separate;
    separate.kernel = Optim::UpdateKernel::Separate;
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<Optim::SGD> reference,
                            Optim::SGD::make(*engine, make_slots({{3.0, 4.0}}), separate));
    SA_EXPECT_FAILURE(reference->enable_gradient_clip(1.0), StatusCode::InvalidDescription,
                      "UpdateKernel::Fused");
}

/** @brief Tests that a norm set without a clip, or one no clip can use, stops the process. */
TEST_F(Medium_GradientClip, ANormSetWithoutAClipIsFatal)
{
    const std::unique_ptr<Optim::SGD> optimizer = make_sgd(make_slots({{3.0, 4.0}}));
    SA_EXPECT_FATAL(optimizer->set_clip_norm(1.0), "no gradient clip");

    SA_EXPECT_OK(optimizer->enable_gradient_clip(1.0));
    SA_EXPECT_FATAL(optimizer->set_clip_norm(0.0), "clip norm", "positive and finite");
}

/** @brief Tests that the key clip_norm enables the clip and that its absence enables none. */
TEST_F(Medium_GradientClip, TheKeyClipNormEnablesTheClip)
{
    const auto attributes = [](std::string_view text)
    {
        return Config::parse_json(text, "<test>");
    };

    for (const std::string_view type : {std::string_view("sgd"), std::string_view("adamw")})
    {
        SA_ASSERT_OK_AND_ASSIGN(
            const std::unique_ptr<Optim::Optimizer> clipped,
            Optim::make_optimizer(*engine, type, make_slots({{3.0, 4.0}}),
                                  attributes(R"({"clip_norm": 0.5})"), "optimizer"));
        SA_ASSERT_OK_AND_ASSIGN(
            const std::unique_ptr<Optim::Optimizer> plain,
            Optim::make_optimizer(*engine, type, make_slots({{3.0, 4.0}}), attributes("{}"),
                                  "optimizer"));

        ASSERT_NE(clipped->gradient_clip(), nullptr) << type;
        EXPECT_NEAR(clipped->gradient_clip()->max_norm(), 0.5, K_EXACT_TOLERANCE) << type;
        EXPECT_EQ(plain->gradient_clip(), nullptr) << type;

        SA_EXPECT_FAILURE(
            Optim::make_optimizer(*engine, type, make_slots({{3.0, 4.0}}),
                                  attributes(R"({"clip_norm": 0})"), "optimizer"),
            StatusCode::InvalidDescription, "optimizer 'clip_norm'", "positive and finite");
    }
}
