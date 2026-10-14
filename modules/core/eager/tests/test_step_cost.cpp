/**************************************************************************/
/* test_step_cost.cpp                                                     */
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
 * @file test_step_cost.cpp
 * @brief Measures the host cost of an eager step whose structure held, and of a renumbering.
 * @author Mustafa Garip
 * @see modules/core/eager/README.md
 */

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include <SushiAI/core/status.hpp>
#include <SushiAI/eager/session.hpp>

#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/SushiAI.h>
#include <SushiAI/engine_fixture/engine_test.hpp>

#include "eager_run_support.hpp"

using namespace SushiAI;
using Eager::StepCost;

// SaTest stays reachable inside a TEST_F body; see tests/integration/README.md.
namespace SaTest = SushiAI::Test;
namespace Support = SushiAI::Eager::Support;

namespace
{
    /** @brief Provides the Engine every case runs on. */
    class Unit_EagerStepCost : public SaTest::EngineTest
    {

    };

    /** @brief Holds a step size small enough that fifty layers stay finite. */
    constexpr float K_RATE = 1e-6f;

    /** @brief Holds how many steps run before the measured ones, so every region is recorded. */
    constexpr int K_WARM_STEPS = 3;

    /** @brief Holds how many steps each measurement averages over. */
    constexpr int K_MEASURED_STEPS = 20;

    /** @brief Holds the layer count of the deep model; the design asks for at least 50. */
    constexpr std::size_t K_DEEP_LAYERS = 50;

    /** @brief Sums the costs of several steps. */
    struct CostSum
    {
        std::uint64_t partition_ns = 0; /**< Assigning region keys. */
        std::uint64_t plan_ns = 0;      /**< Finding boundaries and comparing signatures. */
        std::uint64_t compile_ns = 0;   /**< Running the passes over built regions. */
        std::uint64_t total_ns = 0;     /**< The whole flush. */
        std::size_t steps = 0;          /**< How many steps were added. */

        /** @brief Adds the cost of one step. */
        void add(const StepCost& cost) noexcept
        {
            partition_ns += cost.partition_ns;
            plan_ns += cost.plan_ns;
            compile_ns += cost.compile_ns;
            total_ns += cost.total_ns;
            ++steps;
        }
    };

    /** @brief Returns @p nanoseconds over @p steps, in microseconds. */
    [[nodiscard]] double mean_us(std::uint64_t nanoseconds, std::size_t steps)
    {
        return steps == 0 ? 0.0 : static_cast<double>(nanoseconds) / 1000.0
                                      / static_cast<double>(steps);
    }

    /** @brief Prints one row: the three host stages of a step and the whole step beside them. */
    void report(const std::string& label, const CostSum& sum, std::size_t regions)
    {
        const std::uint64_t host = sum.partition_ns + sum.plan_ns + sum.compile_ns;
        std::cout << "[eager step cost] " << label << ": regions " << regions
                  << ", partition " << mean_us(sum.partition_ns, sum.steps) << " us"
                  << ", plan " << mean_us(sum.plan_ns, sum.steps) << " us"
                  << ", compile " << mean_us(sum.compile_ns, sum.steps) << " us"
                  << ", host stages " << mean_us(host, sum.steps) << " us"
                  << ", whole step " << mean_us(sum.total_ns, sum.steps) << " us"
                  << " (mean of " << sum.steps << " steps)" << std::endl;
    }

    /** @brief Returns the widths of a model of @p layers layers, 16 wide inside, 4 results. */
    [[nodiscard]] std::vector<int64_t> widths_of(std::size_t layers)
    {
        std::vector<int64_t> widths(layers + 1, 16);
        widths.back() = 4;
        return widths;
    }

    /**
     * @brief Runs a settled loop and prints the mean host cost of its unchanged steps.
     * @return The failure of a step, or success once every measured step kept every region.
     */
    [[nodiscard]] Status measure_settled(SushiBLAS::Engine& engine, const std::string& label,
                                         const std::vector<int64_t>& widths,
                                         Support::Scoping scoping)
    {
        Support::Loop loop(engine, K_RATE);
        SA_TRY(loop.prepare(engine, widths));

        Support::TraceOptions options;
        options.scoping = scoping;
        for (int step = 0; step < K_WARM_STEPS; ++step)
            SA_TRY(loop.step(options));

        CostSum sum;
        for (int step = 0; step < K_MEASURED_STEPS; ++step)
        {
            SA_TRY(loop.step(options));
            SA_FAIL_IF(loop.session.last_cost().built != 0
                           || loop.session.last_cost().recorded != 0,
                       StatusCode::Incompatible,
                       "{}: a measured step built {} regions and recorded {} again; the loop "
                       "had not settled",
                       label, loop.session.last_cost().built,
                       loop.session.last_cost().recorded);
            sum.add(loop.session.last_cost());
        }

        report(label, sum, loop.executor.region_count());
        return Status();
    }
} // namespace

TEST_F(Unit_EagerStepCost, AnUnchangedStepOfTheTwoLayerModel)
{
    SA_EXPECT_OK(measure_settled(*engine, "two layers, no scope", {6, 4, 3},
                                 Support::Scoping::None));
    SA_EXPECT_OK(measure_settled(*engine, "two layers, one scope per layer", {6, 4, 3},
                                 Support::Scoping::PerLayer));
}

TEST_F(Unit_EagerStepCost, AnUnchangedStepOfAFiftyLayerModel)
{
    SA_EXPECT_OK(measure_settled(*engine, "fifty layers, no scope", widths_of(K_DEEP_LAYERS),
                                 Support::Scoping::None));
    SA_EXPECT_OK(measure_settled(*engine, "fifty layers, one scope per layer",
                                 widths_of(K_DEEP_LAYERS), Support::Scoping::PerLayer));
}

TEST_F(Unit_EagerStepCost, AScopeThatAppearsAndDisappears)
{
    Support::Loop loop(*engine, K_RATE);
    SA_EXPECT_OK(loop.prepare(*engine, widths_of(K_DEEP_LAYERS)));

    Support::TraceOptions options;
    options.scoping = Support::Scoping::PerLayer;
    for (int step = 0; step < K_WARM_STEPS; ++step)
        SA_EXPECT_OK(loop.step(options));

    CostSum appears;
    CostSum disappears;
    std::size_t recorded = 0;
    for (int round = 0; round < K_MEASURED_STEPS / 2; ++round)
    {
        options.extra_scope = true;
        SA_EXPECT_OK(loop.step(options));
        appears.add(loop.session.last_cost());
        recorded += loop.session.last_cost().recorded;

        options.extra_scope = false;
        SA_EXPECT_OK(loop.step(options));
        disappears.add(loop.session.last_cost());
        recorded += loop.session.last_cost().recorded;
    }

    report("fifty layers, a scope appears", appears, loop.executor.region_count());
    report("fifty layers, a scope disappears", disappears, loop.executor.region_count());

    // A renumbering records regions again without compiling them; none means it was not one.
    EXPECT_GT(recorded, 0u);
}
