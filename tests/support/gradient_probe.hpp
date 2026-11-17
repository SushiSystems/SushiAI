/**************************************************************************/
/* gradient_probe.hpp                                                     */
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
 * @file gradient_probe.hpp
 * @brief Declares GradientProbe, the FLOAT64 graph whose loss and gradient a gradient check
 *        evaluates at a point, and the check that compares the two.
 * @author Mustafa Garip
 */

#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>
#include <gtest/gtest.h>
#include <SushiBLAS/tensor/tensor.hpp>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/autograd/backward.hpp>
#include <SushiAI/autograd/gradcheck.hpp>
#include <SushiAI/dtype/dtype.hpp>
#include <SushiAI/execution/execution_backend.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/graph/builder.hpp>
#include <SushiAI/graph/compiled_step.hpp>
#include <SushiAI/graph/fusion.hpp>
#include <SushiAI/graph/fusion_selector.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/graph/tensor_pool.hpp>
#include <SushiAI/seeded_noise/host_noise.hpp>
#include <SushiAI/shape/shape.hpp>
#include <SushiAI/tensor_values/tensor_values.hpp>

namespace SushiAI
{
    namespace Test
    {
        /** @brief Names the leaf a probe differentiates at and the value its loss reads. */
        struct GradientSubject
        {
            Graph::ValueID variable = Graph::K_NO_VALUE; /**< The leaf the gradient is taken at. */
            Graph::ValueID output = Graph::K_NO_VALUE;   /**< The [N, C] log-probabilities. */
        };

        /** @brief Chooses what a probe does besides building, differentiating and compiling. */
        struct GradientProbeOptions
        {
            bool fuse = false;               /**< Whether the fusion pass runs before compiling. */
            bool noise_other_leaves = false; /**< Whether every other leaf is filled from noise. */
            std::uint64_t noise_seed = 0;    /**< Added to a leaf's id to seed that leaf's noise. */

            /** @brief Returns options that run the fusion pass. */
            [[nodiscard]] static GradientProbeOptions fused() noexcept
            {
                GradientProbeOptions options;
                options.fuse = true;
                return options;
            }

            /** @brief Returns options that fill every other leaf from noise seeded with @p seed. */
            [[nodiscard]] static GradientProbeOptions noise_filled(std::uint64_t seed) noexcept
            {
                GradientProbeOptions options;
                options.noise_other_leaves = true;
                options.noise_seed = seed;
                return options;
            }
        };

        /** @brief Selects the constructor whose build function names its own variable. */
        struct BuildsSubject
        {

        };

        /**
         * @class GradientProbe
         * @brief Holds a FLOAT64 graph whose loss and gradient can both be evaluated at a point.
         *
         * The loss is NLL_LOSS of the output the build function returns, against targets of the
         * same shape. The targets start as one-hot rows, row r naming class r mod C. The variable
         * under test is the one leaf the gradient is taken at.
         */
        class GradientProbe
        {
            public:
                /**
                 * @brief Builds the graph over a parameter of @p variable_shape and compiles it.
                 * @param build Called as build(builder, variable); returns the [N, C]
                 *        log-probabilities the loss reads, as a ValueID or a Traced.
                 */
                template<typename BuildFn>
                GradientProbe(SushiAI::ExecutionBackend& backend, Shape variable_shape,
                              BuildFn&& build, GradientProbeOptions options = {})
                    : pool_(backend.engine())
                {
                    const Graph::GraphBuilder builder(graph_);
                    const Graph::ValueID variable =
                        builder.parameter(variable_shape, "x", DType::Float64);

                    // The operands fit, so a refusal is a builder defect: value() stops the test.
                    const Graph::ValueID output = Graph::Traced(build(builder, variable)).value();
                    finish(backend, builder, GradientSubject{variable, output}, options);
                }

                /**
                 * @brief Builds the graph the function describes and compiles it.
                 * @param build Called as build(builder); returns the GradientSubject.
                 */
                template<typename BuildFn>
                GradientProbe(SushiAI::ExecutionBackend& backend, BuildsSubject, BuildFn&& build,
                              GradientProbeOptions options = {})
                    : pool_(backend.engine())
                {
                    const Graph::GraphBuilder builder(graph_);
                    const GradientSubject subject = build(builder);
                    finish(backend, builder, subject, options);
                }

                GradientProbe(const GradientProbe&) = delete;
                GradientProbe& operator=(const GradientProbe&) = delete;
                GradientProbe(GradientProbe&&) = delete;
                GradientProbe& operator=(GradientProbe&&) = delete;
                ~GradientProbe() = default;

                /**
                 * @brief Reports whether the step compiled.
                 * @return False after a refusal, which the constructor already recorded.
                 */
                [[nodiscard]] bool compiled() const noexcept
                {
                    return step_ != nullptr;
                }

                /** @brief Returns the graph the probe built, with its gradient nodes. */
                [[nodiscard]] const Graph::Graph& graph() const noexcept
                {
                    return graph_;
                }

                /** @brief Returns what the fusion pass did, when the probe was asked to run it. */
                [[nodiscard]] const Graph::FusionReport& fusion() const noexcept
                {
                    return fusion_;
                }

                /** @brief Returns what the lowering spelled, so a decline cannot pass unnoticed. */
                [[nodiscard]] SushiRuntime::span<const Graph::FusionSelection> selections()
                    const noexcept
                {
                    if (!compiled())
                        return SushiRuntime::span<const Graph::FusionSelection>();

                    return step_->selections();
                }

                /** @brief Writes the values of a leaf that is not the variable. */
                void fill(Graph::ValueID id, const std::vector<double>& values)
                {
                    write_tensor(pool_.tensor(id), values);
                }

                /** @brief Fills a leaf from the deterministic host stream. */
                void fill_noise(Graph::ValueID id, std::uint64_t seed, double scale = 1.0)
                {
                    fill(id, noise_for(id, seed, scale));
                }

                /** @brief Writes an INT32 leaf's values; the host write records no task. */
                void fill_indices(Graph::ValueID id, const std::vector<std::int32_t>& indices)
                {
                    ASSERT_EQ(pool_.tensor(id).dtype, DType::Int32);
                    fill(id, std::vector<double>(indices.begin(), indices.end()));
                }

                /** @brief Fills the targets with one-hot rows; row r names class r mod classes. */
                void fill_one_hot(int64_t rows, int64_t classes)
                {
                    fill(targets_, one_hot_rows(rows, classes));
                }

                /** @brief Returns the element count of the variable. */
                [[nodiscard]] std::size_t size() const
                {
                    return static_cast<std::size_t>(graph_.value(variable_).shape.num_elements());
                }

                /** @brief Returns the value id of the targets, for a test that shapes them. */
                [[nodiscard]] Graph::ValueID targets() const noexcept
                {
                    return targets_;
                }

                /** @brief Writes @p point into the variable and runs the step once. */
                void run_at(SushiRuntime::span<const double> point)
                {
                    // A refused compile is already a failed expectation; there is no step to run.
                    if (!compiled())
                        return;

                    write_tensor(pool_.tensor(variable_), point);
                    SA_EXPECT_OK(step_->run());
                }

                /** @brief Returns the elements of value @p id after the last run. */
                [[nodiscard]] std::vector<double> read(Graph::ValueID id) const
                {
                    return read_tensor(pool_.tensor(id));
                }

                /** @brief Returns the loss at @p point, or zero when the step did not compile. */
                double value(SushiRuntime::span<const double> point)
                {
                    if (!compiled())
                        return 0.0;

                    run_at(point);
                    return read(loss_)[0];
                }

                /** @brief Writes the analytic gradient at @p point into @p out. */
                void grad(SushiRuntime::span<const double> point, SushiRuntime::span<double> out)
                {
                    if (!compiled())
                        return;

                    run_at(point);

                    const std::vector<double> values = read(gradient_);
                    ASSERT_EQ(values.size(), out.size());
                    std::copy(values.begin(), values.end(), out.begin());
                }

                /** @brief Returns a ScalarFunction bound to this probe. */
                Autograd::ScalarFunction value_fn()
                {
                    return [this](SushiRuntime::span<const double> point) { return value(point); };
                }

                /** @brief Returns a GradientFunction bound to this probe. */
                Autograd::GradientFunction grad_fn()
                {
                    return [this](SushiRuntime::span<const double> point,
                                  SushiRuntime::span<double> out)
                    {
                        grad(point, out);
                    };
                }

            private:
                /** @brief Returns the one-hot rows of a [rows, classes] target. */
                static std::vector<double> one_hot_rows(int64_t rows, int64_t classes)
                {
                    std::vector<double> values(static_cast<std::size_t>(rows * classes), 0.0);
                    for (int64_t row = 0; row < rows; ++row)
                        values[static_cast<std::size_t>(row * classes + row % classes)] = 1.0;

                    return values;
                }

                /** @brief Returns noise for leaf @p id, as many elements as the leaf holds. */
                std::vector<double> noise_for(Graph::ValueID id, std::uint64_t seed, double scale)
                {
                    std::vector<double> values(
                        static_cast<std::size_t>(pool_.tensor(id).num_elements));
                    HostNoise(seed).fill(values, scale);
                    return values;
                }

                /** @brief Fills every leaf but the variable and the targets from noise. */
                void fill_other_leaves(std::uint64_t seed)
                {
                    for (Graph::ValueID id = 0; id < graph_.value_count(); ++id)
                    {
                        const Graph::Lifetime lifetime = graph_.value(id).lifetime;
                        const bool is_leaf = lifetime == Graph::Lifetime::Parameter
                                          || lifetime == Graph::Lifetime::Input;
                        if (!is_leaf || id == variable_ || id == targets_)
                            continue;

                        fill_noise(id, seed + id);
                    }
                }

                /** @brief Adds the loss and the gradient to the graph, then compiles it. */
                void finish(SushiAI::ExecutionBackend& backend, const Graph::GraphBuilder& builder,
                            const GradientSubject& subject, const GradientProbeOptions& options)
                {
                    variable_ = subject.variable;

                    const Shape output_shape = graph_.value(subject.output).shape;
                    targets_ = builder.input(output_shape, "targets", DType::Float64);
                    // The output is a well-formed [N, C], so a refusal is a defect: value() stops.
                    loss_ = builder.nll_loss(subject.output, targets_).value();

                    const std::vector<Graph::ValueID> wrt = {variable_};
                    // The graph is fixed here, so a refusal is a defect: value() stops the test.
                    gradient_ = Autograd::differentiate(
                        graph_, loss_,
                        SushiRuntime::span<const Graph::ValueID>(wrt.data(), wrt.size()))
                        .value().at(0);
                    EXPECT_NE(gradient_, Graph::K_NO_VALUE);

                    if (options.fuse)
                    {
                        const Graph::ValueID observed[] = {loss_, targets_};
                        Graph::FusionOptions fusion_options;
                        fusion_options.observed =
                            SushiRuntime::span<const Graph::ValueID>(observed, 2);
                        fusion_ = Graph::fuse(graph_, fusion_options);
                    }

                    SA_EXPECT_OK(pool_.materialise_leaves(graph_));
                    fill_one_hot(output_shape[0], output_shape[1]);
                    if (options.noise_other_leaves)
                        fill_other_leaves(options.noise_seed);

                    Result<std::unique_ptr<Graph::CompiledStep>> compiled =
                        Graph::CompiledStep::compile(backend, graph_, pool_);
                    SA_EXPECT_OK(compiled);
                    if (compiled.has_value())
                        step_ = std::move(compiled).value();
                }

                /** @brief Holds the forward and gradient nodes. */
                Graph::Graph graph_;

                /** @brief Holds the tensor of each value. */
                Graph::TensorPool pool_;

                /** @brief Holds the compiled step, or null after a refusal. */
                std::unique_ptr<Graph::CompiledStep> step_;

                /** @brief Names the leaf the gradient is taken at. */
                Graph::ValueID variable_ = Graph::K_NO_VALUE;

                /** @brief Names the targets leaf. */
                Graph::ValueID targets_ = Graph::K_NO_VALUE;

                /** @brief Names the one-element loss. */
                Graph::ValueID loss_ = Graph::K_NO_VALUE;

                /** @brief Names the gradient of the loss at the variable. */
                Graph::ValueID gradient_ = Graph::K_NO_VALUE;

                /** @brief Holds what the fusion pass did. */
                Graph::FusionReport fusion_;
        };

        /**
         * @brief Runs the full-Jacobian check of @p probe at @p point and reports a failure.
         * @pre @p point holds at most 200 elements; larger variables use check_directional.
         */
        inline void expect_gradient_matches(GradientProbe& probe, std::vector<double>& point,
                                            const Autograd::GradCheckOptions& options = {})
        {
            ASSERT_LE(point.size(), 200u) << "use the directional mode above 200 inputs";

            const Autograd::GradCheckResult result = Autograd::check_full_jacobian(
                probe.value_fn(), probe.grad_fn(), point, options);

            EXPECT_TRUE(result.passed) << result.to_string();
        }
    } // namespace Test
} // namespace SushiAI
