/**************************************************************************/
/* test_moe.cpp                                                           */
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
 * @file test_moe.cpp
 * @brief Checks the mixture of experts against a host computation and central differences,
 *        and trains a two-block decoder of experts on one batch.
 * @author Mustafa Garip
 */

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>
#include <gtest/gtest.h>
#include <SushiBLAS/engine/engine.hpp>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/autograd/backward.hpp>
#include <SushiAI/autograd/gradcheck.hpp>
#include <SushiAI/config/model_config.hpp>
#include <SushiAI/config/value.hpp>
#include <SushiAI/core/common.hpp>
#include <SushiAI/core/ops.hpp>
#include <SushiAI/core/result.hpp>
#include <SushiAI/core/shape.hpp>
#include <SushiAI/core/status.hpp>
#include <SushiAI/data/dataset.hpp>
#include <SushiAI/data/input.hpp>
#include <SushiAI/data/materialised_dataset.hpp>
#include <SushiAI/data/roles.hpp>
#include <SushiAI/data/target.hpp>
#include <SushiAI/graph/builder.hpp>
#include <SushiAI/graph/compiled_step.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/graph/tensor_pool.hpp>
#include <SushiAI/language/blocks/expert_trace.hpp>
#include <SushiAI/language/blocks/moe_feed_forward.hpp>
#include <SushiAI/language/registration.hpp>
#include <SushiAI/nn/init.hpp>
#include <SushiAI/nn/layer_registry.hpp>
#include <SushiAI/nn/model_graph.hpp>
#include <SushiAI/nn/model_outputs.hpp>
#include <SushiAI/nn/model_state.hpp>
#include <SushiAI/nn/node_registry.hpp>
#include <SushiAI/nn/trace.hpp>
#include <SushiAI/optim/adamw.hpp>
#include <SushiAI/optim/optimizer.hpp>
#include <SushiAI/train/metrics.hpp>
#include <SushiAI/train/objective.hpp>
#include <SushiAI/train/objective_context.hpp>
#include <SushiAI/train/objective_registry.hpp>
#include <SushiAI/train/steps.hpp>
#include <SushiAI/train/trainer.hpp>
#include <SushiAI/train/training_graph.hpp>
#include <common/model_text.hpp>
#include <common/test_status.hpp>
#include <common/test_utils.hpp>

using namespace SushiAI;

// SaTest stays reachable inside a TEST_F body; see tests/integration/README.md.
namespace SaTest = SushiAI::Test;
using Graph::CompiledStep;
using Graph::GraphBuilder;
using Graph::TensorPool;
using Graph::ValueID;
using Language::Blocks::ExpertTrace;
using Language::Blocks::MoEFeedForward;
using SaTest::read_tensor;
using SaTest::write_tensor;

namespace
{
    /** @brief Provides the Engine every case records on. */
    class Integration_Moe : public SaTest::EngineTest
    {

    };

    /** @brief Holds R, the positions of the one sequence the block cases read. */
    constexpr std::size_t K_ROWS = 3;

    /** @brief Holds the width of the sequence the block cases read. */
    constexpr std::size_t K_DIM = 4;

    /** @brief Holds the hidden width of one expert of the block cases. */
    constexpr std::size_t K_HIDDEN = 3;

    /** @brief Holds the experts of the block cases. */
    constexpr std::size_t K_EXPERTS = 3;

    /** @brief Bounds the distance of a FLOAT64 result from the host computation. */
    constexpr double K_HOST_TOLERANCE = 1e-10;

    /** @brief Holds the weight of the balance term in the loss a gradient case checks. */
    constexpr double K_PROBE_BALANCE_WEIGHT = 0.5;

    /** @brief Returns @p count values of the deterministic host stream seeded with @p seed. */
    std::vector<double> noise(std::size_t count, std::uint64_t seed)
    {
        std::vector<double> values(count);
        SaTest::HostNoise(seed).fill(values, 1.0);
        return values;
    }

    /** @brief Returns the tanh form of GELU at @p x, the form K_GELU lowers to. */
    double gelu(double x)
    {
        return 0.5 * x * (1.0 + std::tanh(0.7978845608028654 * (x + 0.044715 * x * x * x)));
    }

    /** @brief Holds the six parameters of a block and the sequence it reads, as host values. */
    struct MixtureValues
    {
        std::vector<double> x;              /**< [K_ROWS, K_DIM]. */
        std::vector<double> router_weight;  /**< [K_EXPERTS, K_DIM]. */
        std::vector<double> router_bias;    /**< [K_EXPERTS]. */
        std::vector<double> expand_weight;  /**< [K_EXPERTS * K_HIDDEN, K_DIM]. */
        std::vector<double> expand_bias;    /**< [K_EXPERTS * K_HIDDEN]. */
        std::vector<double> project_weight; /**< [K_DIM, K_EXPERTS * K_HIDDEN]. */
        std::vector<double> project_bias;   /**< [K_EXPERTS, K_DIM]. */
    };

    /** @brief Holds what the host computes for a mixture. */
    struct HostMixture
    {
        std::vector<double> output; /**< [K_ROWS, K_DIM]. */
        std::vector<double> usage;  /**< [K_EXPERTS], the mean probability of each expert. */
        double balance = 0.0;       /**< K_EXPERTS * sum(usage^2) - 1. */
    };

    /** @brief Returns the softmax of @p scores over the entries @p kept marks. */
    std::vector<double> softmax_over(const std::vector<double>& scores,
                                     const std::vector<bool>& kept)
    {
        double total = 0.0;
        for (std::size_t e = 0; e < scores.size(); ++e)
            total += kept[e] ? std::exp(scores[e]) : 0.0;

        std::vector<double> gate(scores.size(), 0.0);
        for (std::size_t e = 0; e < scores.size(); ++e)
            gate[e] = kept[e] ? std::exp(scores[e]) / total : 0.0;

        return gate;
    }

    /** @brief Marks the entries of @p scores at or above its @p top_k th largest; all for 0. */
    std::vector<bool> kept_scores(const std::vector<double>& scores, std::size_t top_k)
    {
        if (top_k == 0)
            return std::vector<bool>(scores.size(), true);

        std::vector<double> sorted = scores;
        std::sort(sorted.begin(), sorted.end(), std::greater<double>());

        std::vector<bool> kept(scores.size(), false);
        for (std::size_t e = 0; e < scores.size(); ++e)
            kept[e] = scores[e] >= sorted[top_k - 1];

        return kept;
    }

    /** @brief Computes, one expert at a time, what a block of @p values mixes. */
    HostMixture mix_on_host(const MixtureValues& values, std::size_t top_k)
    {
        HostMixture host;
        host.output.assign(K_ROWS * K_DIM, 0.0);
        host.usage.assign(K_EXPERTS, 0.0);

        for (std::size_t r = 0; r < K_ROWS; ++r)
        {
            const double* const row = &values.x[r * K_DIM];
            std::vector<double> scores(K_EXPERTS, 0.0);
            for (std::size_t e = 0; e < K_EXPERTS; ++e)
            {
                scores[e] = values.router_bias[e];
                for (std::size_t d = 0; d < K_DIM; ++d)
                    scores[e] += values.router_weight[e * K_DIM + d] * row[d];
            }

            const std::vector<double> probabilities =
                softmax_over(scores, std::vector<bool>(K_EXPERTS, true));
            const std::vector<double> gate = softmax_over(scores, kept_scores(scores, top_k));

            for (std::size_t e = 0; e < K_EXPERTS; ++e)
            {
                host.usage[e] += probabilities[e] / static_cast<double>(K_ROWS);

                // Expert e is a feed-forward of its own slice of the stacked parameters.
                for (std::size_t d = 0; d < K_DIM; ++d)
                {
                    double projected = values.project_bias[e * K_DIM + d];
                    for (std::size_t j = 0; j < K_HIDDEN; ++j)
                    {
                        const std::size_t unit = e * K_HIDDEN + j;
                        double hidden = values.expand_bias[unit];
                        for (std::size_t k = 0; k < K_DIM; ++k)
                            hidden += values.expand_weight[unit * K_DIM + k] * row[k];

                        projected += values.project_weight[d * K_EXPERTS * K_HIDDEN + unit]
                                   * gelu(hidden);
                    }

                    host.output[r * K_DIM + d] += gate[e] * projected;
                }
            }
        }

        double squares = 0.0;
        for (const double share : host.usage)
            squares += share * share;
        host.balance = static_cast<double>(K_EXPERTS) * squares - 1.0;

        return host;
    }

    /** @brief Returns the values every host case writes, drawn from seeds above @p seed. */
    MixtureValues make_values(std::uint64_t seed)
    {
        MixtureValues values;
        values.x = noise(K_ROWS * K_DIM, seed);
        values.router_weight = noise(K_EXPERTS * K_DIM, seed + 1);
        values.router_bias = noise(K_EXPERTS, seed + 2);
        values.expand_weight = noise(K_EXPERTS * K_HIDDEN * K_DIM, seed + 3);
        values.expand_bias = noise(K_EXPERTS * K_HIDDEN, seed + 4);
        values.project_weight = noise(K_DIM * K_EXPERTS * K_HIDDEN, seed + 5);
        values.project_bias = noise(K_EXPERTS * K_DIM, seed + 6);
        return values;
    }

    /** @brief Returns a block over the extents above that keeps @p top_k experts. */
    MoEFeedForward make_block(std::size_t top_k)
    {
        return MoEFeedForward(static_cast<int64_t>(K_DIM), static_cast<int64_t>(K_HIDDEN),
                              static_cast<int64_t>(K_EXPERTS), static_cast<int64_t>(top_k));
    }

    /** @brief Runs a FLOAT64 block that keeps @p top_k experts and compares with the host. */
    void expect_mixture_matches_host(SushiBLAS::Engine& engine, std::size_t top_k,
                                     std::uint64_t seed)
    {
        Graph::Graph graph;
        const GraphBuilder builder(graph);
        const ValueID x = builder.input(
            Shape{1, static_cast<int64_t>(K_ROWS), static_cast<int64_t>(K_DIM)}, "x",
            DType::Float64);
        MoEFeedForward block = make_block(top_k);
        const NN::Trace trace = NN::Trace::evaluation(builder);
        SA_ASSERT_OK_AND_ASSIGN(const ExpertTrace traced, block.trace_experts(trace, x));
        SA_EXPECT_OK(trace.status());
        ASSERT_FALSE(graph.has_refusal());

        const MixtureValues values = make_values(seed);
        TensorPool pool(engine);
        SA_EXPECT_OK(pool.materialise_leaves(graph));
        write_tensor(pool.tensor(x), values.x);
        write_tensor(pool.tensor(block.router.weight.id_in(graph)), values.router_weight);
        write_tensor(pool.tensor(block.router.bias.id_in(graph)), values.router_bias);
        write_tensor(pool.tensor(block.expand.weight.id_in(graph)), values.expand_weight);
        write_tensor(pool.tensor(block.expand.bias.id_in(graph)), values.expand_bias);
        write_tensor(pool.tensor(block.project.weight.id_in(graph)), values.project_weight);
        write_tensor(pool.tensor(block.project.bias.id_in(graph)), values.project_bias);

        std::unique_ptr<CompiledStep> step;
        SA_ASSERT_OK_AND_ASSIGN(step, CompiledStep::compile(engine, graph, pool));
        SA_EXPECT_OK(step->run());
        SA_EXPECT_OK(step->run());

        const HostMixture host = mix_on_host(values, top_k);
        const std::vector<double> output = read_tensor(pool.tensor(traced.output));
        const std::vector<double> usage = read_tensor(pool.tensor(traced.usage));
        const std::vector<double> balance = read_tensor(pool.tensor(traced.balance));

        ASSERT_EQ(output.size(), host.output.size());
        for (std::size_t i = 0; i < output.size(); ++i)
            EXPECT_NEAR(output[i], host.output[i], K_HOST_TOLERANCE) << "element " << i;

        ASSERT_EQ(usage.size(), host.usage.size());
        for (std::size_t e = 0; e < usage.size(); ++e)
            EXPECT_NEAR(usage[e], host.usage[e], K_HOST_TOLERANCE) << "expert " << e;

        ASSERT_EQ(balance.size(), 1u);
        EXPECT_NEAR(balance[0], host.balance, K_HOST_TOLERANCE);
    }

    /** @brief Names which leaf of a gradient case the gradient is taken at. */
    enum class RoutedVariable : std::uint8_t
    {
        Input,        /**< The [1, K_ROWS, K_DIM] sequence the block reads. */
        RouterWeight, /**< The weight of the router, which every gate and the balance read. */
    };

    /**
     * @brief Holds a FLOAT64 graph of one block whose loss and gradient are evaluated at a point.
     *
     * The loss is the cross-entropy of the block's rows against targets whose row r names
     * class r mod K_DIM, plus K_PROBE_BALANCE_WEIGHT times the balance term.
     */
    class RoutedProbe
    {
        public:
            /** @brief Builds the graph, differentiates it at @p variable and compiles it. */
            RoutedProbe(SushiBLAS::Engine& engine, std::size_t top_k, RoutedVariable variable,
                        std::uint64_t seed)
                : block_(make_block(top_k)), pool_(engine)
            {
                const GraphBuilder builder(graph_);
                const Shape sequence{1, static_cast<int64_t>(K_ROWS),
                                     static_cast<int64_t>(K_DIM)};
                const ValueID x = variable == RoutedVariable::Input
                    ? builder.parameter(sequence, "x", DType::Float64)
                    : builder.input(sequence, "x", DType::Float64);

                const NN::Trace trace = NN::Trace::evaluation(builder);
                const Result<ExpertTrace> traced = block_.trace_experts(trace, x);
                SA_EXPECT_OK(traced);
                if (!traced.has_value())
                    return;

                variable_ = variable == RoutedVariable::Input
                    ? x
                    : block_.router.weight.id_in(graph_);

                const Shape rows{static_cast<int64_t>(K_ROWS), static_cast<int64_t>(K_DIM)};
                targets_ = builder.input(rows, "targets", DType::Float64);
                const ValueID fit = builder.nll_loss(
                    builder.log_softmax(builder.reshape(traced.value().output, rows)), targets_);
                loss_ = builder.add(
                    fit, builder.scale(builder.reshape(traced.value().balance, Shape{1}),
                                       K_PROBE_BALANCE_WEIGHT));

                const std::vector<ValueID> wrt = {variable_};
                gradient_ = Autograd::differentiate(
                    graph_, loss_,
                    SushiRuntime::span<const ValueID>(wrt.data(), wrt.size())).at(0);
                EXPECT_NE(gradient_, Graph::K_NO_VALUE);

                SA_EXPECT_OK(pool_.materialise_leaves(graph_));
                fill_other_leaves(seed);

                Result<std::unique_ptr<CompiledStep>> compiled =
                    CompiledStep::compile(engine, graph_, pool_);
                SA_EXPECT_OK(compiled);
                if (compiled.has_value())
                    step_ = std::move(compiled).value();
            }

            /** @brief Returns the graph the probe built, with its gradient nodes. */
            const Graph::Graph& graph() const noexcept
            {
                return graph_;
            }

            /** @brief Returns how many elements the variable holds; 0 when the trace failed. */
            std::size_t variable_size() const
            {
                if (variable_ == Graph::K_NO_VALUE)
                    return 0;

                return static_cast<std::size_t>(graph_.value(variable_).shape.num_elements());
            }

            /** @brief Returns the loss at @p point, or zero when the step did not compile. */
            double value(SushiRuntime::span<const double> point)
            {
                if (step_ == nullptr)
                    return 0.0;

                write_tensor(pool_.tensor(variable_), point);
                SA_EXPECT_OK(step_->run());
                return read_tensor(pool_.tensor(loss_))[0];
            }

            /** @brief Writes the analytic gradient at @p point into @p out. */
            void gradient(SushiRuntime::span<const double> point, SushiRuntime::span<double> out)
            {
                if (step_ == nullptr)
                    return;

                write_tensor(pool_.tensor(variable_), point);
                SA_EXPECT_OK(step_->run());

                const std::vector<double> values = read_tensor(pool_.tensor(gradient_));
                ASSERT_EQ(values.size(), out.size());
                for (std::size_t i = 0; i < values.size(); ++i)
                    out[i] = values[i];
            }

        private:
            /** @brief Writes the one-hot targets, and noise into each leaf but the variable. */
            void fill_other_leaves(std::uint64_t seed)
            {
                std::vector<double> one_hot(K_ROWS * K_DIM, 0.0);
                for (std::size_t row = 0; row < K_ROWS; ++row)
                    one_hot[row * K_DIM + row % K_DIM] = 1.0;
                write_tensor(pool_.tensor(targets_), one_hot);

                for (ValueID id = 0; id < graph_.value_count(); ++id)
                {
                    const Graph::Lifetime lifetime = graph_.value(id).lifetime;
                    const bool is_leaf = lifetime == Graph::Lifetime::Parameter
                                      || lifetime == Graph::Lifetime::Input;
                    if (!is_leaf || id == variable_ || id == targets_)
                        continue;

                    const std::size_t count =
                        static_cast<std::size_t>(graph_.value(id).shape.num_elements());
                    write_tensor(pool_.tensor(id), noise(count, seed + id));
                }
            }

            /// @brief The block under test, which owns the parameters of the graph.
            MoEFeedForward block_;

            /// @brief The forward and the backward nodes.
            Graph::Graph graph_;

            /// @brief The tensor of every value of the graph.
            TensorPool pool_;

            /// @brief The compiled step; null when the trace or the compilation failed.
            std::unique_ptr<CompiledStep> step_;

            /// @brief The leaf the gradient is taken at.
            ValueID variable_ = Graph::K_NO_VALUE;

            /// @brief The one-hot targets of the cross-entropy.
            ValueID targets_ = Graph::K_NO_VALUE;

            /// @brief The one-element loss.
            ValueID loss_ = Graph::K_NO_VALUE;

            /// @brief The gradient of the loss at the variable.
            ValueID gradient_ = Graph::K_NO_VALUE;
    };

    /** @brief Runs the full-Jacobian check of a block that keeps @p top_k experts. */
    void expect_gradient_matches(SushiBLAS::Engine& engine, std::size_t top_k,
                                 RoutedVariable variable, std::uint64_t seed)
    {
        RoutedProbe probe(engine, top_k, variable, seed);
        ASSERT_NE(probe.variable_size(), 0u);

        std::vector<double> point = noise(probe.variable_size(), seed + 1000);
        const Autograd::GradCheckResult result = Autograd::check_full_jacobian(
            [&probe](SushiRuntime::span<const double> at)
            {
                return probe.value(at);
            },
            [&probe](SushiRuntime::span<const double> at, SushiRuntime::span<double> out)
            {
                probe.gradient(at, out);
            },
            point);

        EXPECT_TRUE(result.passed) << result.to_string();
    }

    /** @brief Holds the rows of the training batch, N. */
    constexpr std::int64_t K_BATCH = 4;

    /** @brief Holds the positions of a row, T, which the model file states as well. */
    constexpr std::int64_t K_LENGTH = 8;

    /** @brief Holds the tokens the decoder scores, V, which the model file states as well. */
    constexpr std::int64_t K_VOCABULARY = 32;

    /** @brief Holds the experts of each block, which the model file states as well. */
    constexpr std::int64_t K_DECODER_EXPERTS = 4;

    /** @brief Holds the updates the one batch is trained for. */
    constexpr std::int64_t K_STEPS = 200;

    /** @brief Holds the learning rate of the AdamW that trains the decoder. */
    constexpr double K_LEARNING_RATE = 3e-3;

    /** @brief Holds the fraction of the first loss the last one must be below. */
    constexpr double K_REMAINING_FRACTION = 0.5;

    /** @brief Holds the smallest share an expert keeps of the batch with the balance on. */
    constexpr double K_SMALLEST_SHARE = 0.10;

    /** @brief Bounds the rounding of a float32 share around the equal share, 1 / experts. */
    constexpr double K_SHARE_TOLERANCE = 1e-5;

    /** @brief Holds a decoder of two blocks of four experts that publishes their ports. */
    constexpr std::string_view K_EXPERT_DECODER = R"({
        "format_version": 2,
        "input": {"shape": [8], "dtype": "int32"},
        "layers": [
            {"type": "embedding", "num_embeddings": 32, "embedding_dim": 16},
            {"type": "position_embedding"},
            {"type": "transformer_block", "heads": 2, "experts": 4, "hidden": 16},
            {"type": "transformer_block", "heads": 2, "experts": 4, "hidden": 16},
            {"type": "language_head", "tie": 0}
        ],
        "outputs": [
            {"node": 4},
            {"node": 2, "port": "balance", "name": "balance2"},
            {"node": 2, "port": "usage", "name": "usage2"},
            {"node": 3, "port": "balance", "name": "balance3"},
            {"node": 3, "port": "usage", "name": "usage3"}
        ]
    })";

    /** @brief Holds the objective that trains the logits alone. */
    constexpr std::string_view K_FIT_ALONE = R"({"output": "output"})";

    /** @brief Holds the objective that adds half the balance term to the fit. */
    constexpr std::string_view K_FIT_AND_BALANCE = R"({"terms": [
        {"type": "next_token", "output": "output"},
        {"type": "expert_balance", "weight": 0.5}
    ]})";

    /** @brief Returns the description of [T] int32 ids per sample. */
    Data::InputDescription token_input()
    {
        return Data::InputDescription{Shape{K_LENGTH}, DType::Int32};
    }

    /** @brief Returns the labels role as [T] int32 ids per sample over K_VOCABULARY tokens. */
    std::vector<Data::TargetDescription> label_roles()
    {
        return {Data::TargetDescription{std::string(Data::K_ROLE_LABELS), Shape{K_LENGTH},
                                        DType::Int32, K_VOCABULARY}};
    }

    /** @brief Returns token i of the stream the batch is cut from: 7 i^2 + 3 i + 1 mod V. */
    std::int32_t token_at(std::int64_t i)
    {
        return static_cast<std::int32_t>((7 * i * i + 3 * i + 1) % K_VOCABULARY);
    }

    /** @brief Returns a dataset of one batch whose label at a position is the next token. */
    Result<Data::MaterialisedDataset> make_dataset(SushiBLAS::Engine& engine)
    {
        SA_TRY_ASSIGN(Data::Batch batch,
                      Data::allocate_batch(engine, K_BATCH, token_input(), label_roles()));

        for (std::int64_t row = 0; row < K_BATCH; ++row)
        {
            for (std::int64_t position = 0; position < K_LENGTH; ++position)
            {
                const std::int64_t at = row * K_LENGTH + position;
                const std::int64_t stream = row * (K_LENGTH + 1) + position;
                batch.inputs.data_as<std::int32_t>()[at] = token_at(stream);
                batch.targets.front().values.data_as<std::int32_t>()[at] = token_at(stream + 1);
            }
        }

        std::vector<Data::Batch> batches;
        batches.push_back(std::move(batch));
        return Data::MaterialisedDataset(std::move(batches), K_BATCH, token_input(),
                                         label_roles(), "tokens");
    }

    /**
     * @brief Trains K_EXPERT_DECODER on the one batch for K_STEPS updates.
     * @param type The objective row; @p attributes its keys, as JSON text.
     * @return The report of the run, or the first failure.
     */
    Result<Train::RunReport> fit_decoder(SushiBLAS::Engine& engine, std::string_view type,
                                         std::string_view attributes)
    {
        NN::NodeRegistry nodes;
        SA_TRY(NN::register_layers(nodes));
        SA_TRY(Language::register_nodes(nodes));
        Train::ObjectiveRegistry objectives;
        SA_TRY(Train::register_objectives(objectives));
        SA_TRY(Language::register_objectives(objectives));

        const Config::ModelSpec spec = SaTest::model_from(K_EXPERT_DECODER);
        SA_TRY_ASSIGN(NN::ModelGraph model, NN::build_model(nodes, spec));

        // The factory of an objective reads the shapes the model publishes.
        Train::ObjectiveContext given;
        given.batch = K_BATCH;
        given.classes = K_VOCABULARY;
        given.input_sample = Shape{K_LENGTH};
        {
            Graph::Graph probe;
            const GraphBuilder builder(probe);
            const NN::Trace trace = NN::Trace::evaluation(builder);
            SA_TRY_ASSIGN(
                const NN::ModelOutputs published,
                model.trace(trace, builder.input(Shape{K_BATCH, K_LENGTH}, "tokens",
                                                 DType::Int32)));
            SA_TRY_ASSIGN(given.outputs, Train::list_output_shapes(probe, published));
        }

        SA_TRY_ASSIGN(const std::unique_ptr<Train::Objective> objective,
                      Train::make_objective(objectives, type, given,
                                            Config::parse_json(attributes, "<test>"),
                                            "objective"));

        Train::GraphOptions graph_options;
        graph_options.dtype = spec.input_dtype;
        SA_TRY_ASSIGN(Train::TrainingGraph built,
                      Train::build_training_graph(model, *objective,
                                                  Shape{K_BATCH, K_LENGTH}, graph_options));

        SA_TRY_ASSIGN(NN::ModelState state, NN::ModelState::create(engine, model));
        TensorPool pool(engine);
        SA_TRY_ASSIGN(std::vector<Optim::ParameterSlot> slots,
                      Train::bind_parameters(built, state, pool));
        SA_TRY(NN::initialize(engine, state));

        Optim::AdamWOptions adamw;
        adamw.learning_rate = K_LEARNING_RATE;
        SA_TRY_ASSIGN(const std::unique_ptr<Optim::AdamW> optimizer,
                      Optim::AdamW::make(engine, slots, adamw));

        Train::TrainerOptions options;
        options.epochs = K_STEPS;
        SA_TRY_ASSIGN(const std::unique_ptr<Train::Trainer> trainer,
                      Train::Trainer::make(engine, built, pool, *objective, *optimizer,
                                           options));

        SA_TRY_ASSIGN(const Data::MaterialisedDataset dataset, make_dataset(engine));
        return trainer->fit(dataset);
    }
} // namespace

/** @brief Tests that a mixture of every expert equals the sum the host computes per expert. */
TEST_F(Integration_Moe, ADenseMixtureEqualsTheHostComputation)
{
    ASSERT_NO_FATAL_FAILURE(expect_mixture_matches_host(*engine, 0, 9100));
}

/** @brief Tests that keeping one and two experts equals the renormalised gate of the host. */
TEST_F(Integration_Moe, AMixtureOfTheTopExpertsEqualsTheHostComputation)
{
    ASSERT_NO_FATAL_FAILURE(expect_mixture_matches_host(*engine, 1, 9200));
    ASSERT_NO_FATAL_FAILURE(expect_mixture_matches_host(*engine, 2, 9300));
}

/** @brief Tests the gradient of the fit and the balance at the weight of the router. */
TEST_F(Integration_Moe, ADenseMixtureDifferentiatesAtItsRouter)
{
    ASSERT_NO_FATAL_FAILURE(
        expect_gradient_matches(*engine, 0, RoutedVariable::RouterWeight, 9400));
}

/** @brief Tests the gradient of the fit and the balance at the sequence the block reads. */
TEST_F(Integration_Moe, ADenseMixtureDifferentiatesAtItsInput)
{
    ASSERT_NO_FATAL_FAILURE(expect_gradient_matches(*engine, 0, RoutedVariable::Input, 9500));
}

/**
 * @brief Tests the gradient through the kept scores of a block that keeps one expert.
 *
 * The mask is constant around the point, so the gradient is that of the kept gate and of
 * the balance, which reads every score.
 */
TEST_F(Integration_Moe, AMixtureOfTheTopExpertDifferentiatesAtItsRouter)
{
    RoutedProbe counted(*engine, 1, RoutedVariable::RouterWeight, 9600);
    std::size_t pool_gradients = 0;
    for (const Graph::Node& node : counted.graph().nodes())
    {
        if (node.op == Ops::K_MAXPOOL2D_BACKWARD.id)
            ++pool_gradients;
    }

    // The comparison is a gradient barrier, so nothing differentiates the row window.
    EXPECT_EQ(pool_gradients, 0u);
    ASSERT_NO_FATAL_FAILURE(
        expect_gradient_matches(*engine, 1, RoutedVariable::RouterWeight, 9600));
}

/** @brief Tests that the fit alone lowers the loss; nothing is asserted about the usage. */
TEST_F(Integration_Moe, ADecoderOfExpertsLowersItsLossWithoutTheBalance)
{
    SA_ASSERT_OK_AND_ASSIGN(const Train::RunReport report,
                            fit_decoder(*engine, "next_token", K_FIT_ALONE));

    EXPECT_EQ(report.steps, static_cast<std::size_t>(K_STEPS));
    EXPECT_EQ(report.compile_count, 1u);
    EXPECT_LT(report.last_loss, K_REMAINING_FRACTION * report.first_loss) << report.to_string();
    EXPECT_EQ(report.last_metrics.find("expert_usage_min"), nullptr);
}

/** @brief Tests that the balance term keeps every expert of both blocks in use. */
TEST_F(Integration_Moe, TheBalanceTermKeepsEveryExpertInUse)
{
    SA_ASSERT_OK_AND_ASSIGN(const Train::RunReport report,
                            fit_decoder(*engine, "sum", K_FIT_AND_BALANCE));

    EXPECT_EQ(report.steps, static_cast<std::size_t>(K_STEPS));
    EXPECT_EQ(report.compile_count, 1u);
    EXPECT_LT(report.last_loss, K_REMAINING_FRACTION * report.first_loss) << report.to_string();

    const Train::Metric* const smallest = report.last_metrics.find("expert_usage_min");
    const Train::Metric* const largest = report.last_metrics.find("expert_usage_max");
    ASSERT_NE(smallest, nullptr);
    ASSERT_NE(largest, nullptr);
    EXPECT_GT(smallest->value, K_SMALLEST_SHARE) << report.to_string();

    // The shares of a block sum to one, so the equal share lies between the extremes.
    const double equal_share = 1.0 / static_cast<double>(K_DECODER_EXPERTS);
    EXPECT_LE(smallest->value, equal_share + K_SHARE_TOLERANCE);
    EXPECT_GE(largest->value, equal_share - K_SHARE_TOLERANCE);
    EXPECT_NE(report.to_string().find("expert_usage_min"), std::string::npos);
}
