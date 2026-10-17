/**************************************************************************/
/* eager_run_support.hpp                                                  */
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
 * @file eager_run_support.hpp
 * @brief Declares the model, the update and the batches the device tests of the eager module share.
 * @author Mustafa Garip
 * @see tests/core/eager/README.md
 */

#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#include <SushiBLAS/blas/blas.hpp>
#include <SushiBLAS/engine/engine.hpp>
#include <SushiBLAS/engine/task_recorder.hpp>

#include <SushiAI/allocation/allocation.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/seam/seam.hpp>
#include <SushiAI/shape/shape.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/eager/partition.hpp>
#include <SushiAI/eager/region_executor.hpp>
#include <SushiAI/eager/scope.hpp>
#include <SushiAI/eager/scope_partitioner.hpp>
#include <SushiAI/eager/session.hpp>
#include <SushiAI/eager/step.hpp>
#include <SushiAI/graph/builder.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/graph/state_id.hpp>
#include <SushiAI/graph/step_recorder.hpp>

#include <SushiAI/SushiAI.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/tensor_values/tensor_values.hpp>

/** @brief Holds what the tests of the eager module share. */
namespace SushiAI::Eager::Support
{
    /** @brief Holds the rows of every batch the device tests feed. */
    inline constexpr int64_t K_ROWS = 8;

    /** @brief Records parameter -= rate * gradient for each pair it holds. */
    class PlainDescent final : public Graph::StepRecorder
    {
        public:
            /** @brief Starts with no pair. */
            explicit PlainDescent(float rate) noexcept : rate_(rate)
            {

            }

            /** @brief Adds one parameter and the tensor its gradient is written into. */
            void add(const SushiBLAS::Tensor& parameter, const SushiBLAS::Tensor& gradient)
            {
                pairs_.emplace_back(parameter, gradient);
            }

            /** @brief Records one axpy per pair and returns how many it recorded. */
            [[nodiscard]] Result<std::size_t> record_into_step(
                SushiBLAS::TaskRecorder& recorder) override
            {
                ++records_;
                return ::SushiAI::Detail::guard_value(::SushiAI::Detail::SeamPhase::Record, [&]()
                {
                    SushiBLAS::BLASOps blas(recorder);
                    for (std::pair<SushiBLAS::Tensor, SushiBLAS::Tensor>& pair : pairs_)
                        blas.axpy(-rate_, pair.second, pair.first);

                    return pairs_.size();
                });
            }

            /** @brief Counts the runs that are about to start. */
            void before_replay() override
            {
                ++before_;
            }

            /** @brief Counts the runs that have finished. */
            void after_replay() override
            {
                ++after_;
            }

            /** @brief Returns how many times the update was recorded. */
            [[nodiscard]] std::size_t record_count() const noexcept
            {
                return records_;
            }

            /** @brief Returns how many runs called after_replay(). */
            [[nodiscard]] std::size_t replay_count() const noexcept
            {
                return after_;
            }

        private:
            /// @brief The step size.
            float rate_;

            /// @brief Each parameter with its gradient.
            std::vector<std::pair<SushiBLAS::Tensor, SushiBLAS::Tensor>> pairs_;

            /// @brief How many times record_into_step() ran.
            std::size_t records_ = 0;

            /// @brief How many times before_replay() ran.
            std::size_t before_ = 0;

            /// @brief How many times after_replay() ran.
            std::size_t after_ = 0;
    };

    /** @brief Holds one linear layer: its two states and their tensors. */
    struct LinearState
    {
        Graph::StateID weight_state = Graph::K_NO_STATE; /**< Identity of the weight. */
        Graph::StateID bias_state = Graph::K_NO_STATE;   /**< Identity of the bias. */
        SushiBLAS::Tensor weight;                        /**< The [inputs, outputs] weight. */
        SushiBLAS::Tensor bias;                          /**< The [outputs] bias. */
    };

    /** @brief Says where trace_model() opens scopes. */
    enum class Scoping : std::uint8_t
    {
        None,            /**< No scope; the step is one region. */
        PerLayer,        /**< One scope "layer" per linear layer and its activation. */
        SplitActivation, /**< A scope closes between the first layer and its activation. */
        EncoderHead,     /**< Scope "encoder" holds every layer but the last, "head" the last. */
    };

    /** @brief Says how trace_model() traces a model. */
    struct TraceOptions
    {
        Scoping scoping = Scoping::None; /**< Where scopes open. */
        bool head_is_relu = false;       /**< Whether the last layer ends in RELU, not TANH. */
        bool extra_scope = false;        /**< Whether a scope "extra" scales the result by 1. */
    };

    /** @brief Fills @p tensor with sin(phase + 0.37 * index) * scale. */
    inline void fill_wave(SushiBLAS::Tensor& tensor, double phase, double scale)
    {
        std::vector<double> values(static_cast<std::size_t>(tensor.num_elements));
        for (std::size_t i = 0; i < values.size(); ++i)
            values[i] = std::sin(phase + 0.37 * static_cast<double>(i)) * scale;

        ::SushiAI::Test::write_tensor(tensor, values);
    }

    /**
     * @brief Returns relu(x * weight + bias) computed on the host, row by row.
     * @param x The [rows, inputs] operand; @p weight is [inputs, outputs], @p bias [outputs].
     */
    [[nodiscard]] inline std::vector<double> linear_relu(const std::vector<double>& x,
                                                         const std::vector<double>& weight,
                                                         const std::vector<double>& bias)
    {
        const std::size_t outputs = bias.size();
        const std::size_t inputs = weight.size() / outputs;
        const std::size_t rows = x.size() / inputs;

        std::vector<double> result(rows * outputs, 0.0);
        for (std::size_t row = 0; row < rows; ++row)
        {
            for (std::size_t column = 0; column < outputs; ++column)
            {
                double sum = bias[column];
                for (std::size_t k = 0; k < inputs; ++k)
                    sum += x[row * inputs + k] * weight[k * outputs + column];

                result[row * outputs + column] = sum > 0.0 ? sum : 0.0;
            }
        }

        return result;
    }

    /**
     * @brief Allocates one linear layer per pair of neighbouring @p widths and fills it.
     * @param widths The input width followed by the width of each layer's result.
     * @return The layers in order, each with two new state identities.
     */
    [[nodiscard]] inline Result<std::vector<LinearState>> make_model(
        SushiBLAS::Engine& engine, const std::vector<int64_t>& widths)
    {
        std::vector<LinearState> layers;
        for (std::size_t i = 0; i + 1 < widths.size(); ++i)
        {
            LinearState layer;
            Graph::StateIdentity weight_identity;
            Graph::StateIdentity bias_identity;
            layer.weight_state = weight_identity.get();
            layer.bias_state = bias_identity.get();

            SA_TRY_ASSIGN(layer.weight,
                          allocate_tensor(engine, Shape{widths[i], widths[i + 1]}));
            SA_TRY_ASSIGN(layer.bias, allocate_tensor(engine, Shape{widths[i + 1]}));
            fill_wave(layer.weight, static_cast<double>(i), 0.4);
            fill_wave(layer.bias, static_cast<double>(i) + 0.5, 0.1);
            layers.push_back(std::move(layer));
        }

        return layers;
    }

    /** @brief Allocates a [K_ROWS, width] tensor filled with a wave of @p phase. */
    [[nodiscard]] inline Result<SushiBLAS::Tensor> make_batch(SushiBLAS::Engine& engine,
                                                              int64_t width, double phase)
    {
        SA_TRY_ASSIGN(SushiBLAS::Tensor batch, allocate_tensor(engine, Shape{K_ROWS, width}));
        fill_wave(batch, phase, 1.0);
        return batch;
    }

    /**
     * @brief Binds every state of @p model as trainable and pairs it with its gradient.
     * @param update Receives one pair per state, weight before bias, layer by layer.
     */
    [[nodiscard]] inline Status bind_model(Session& session, const std::vector<LinearState>& model,
                                           PlainDescent& update)
    {
        for (const LinearState& layer : model)
        {
            SA_TRY(session.bind_state(layer.weight_state, layer.weight, true));
            SA_TRY(session.bind_state(layer.bias_state, layer.bias, true));

            SA_TRY_ASSIGN(const SushiBLAS::Tensor weight_gradient,
                          session.gradient(layer.weight_state));
            SA_TRY_ASSIGN(const SushiBLAS::Tensor bias_gradient,
                          session.gradient(layer.bias_state));
            update.add(layer.weight, weight_gradient);
            update.add(layer.bias, bias_gradient);
        }

        return Status();
    }

    /**
     * @brief Emits one linear layer of @p layer over @p x into @p step.
     * @return The layer's result, before any activation.
     */
    [[nodiscard]] inline Result<Graph::ValueID> trace_linear(Step& step, const LinearState& layer,
                                                             Graph::ValueID x)
    {
        SA_TRY_ASSIGN(const Graph::ValueID weight, step.state(layer.weight_state));
        SA_TRY_ASSIGN(const Graph::ValueID bias, step.state(layer.bias_state));
        return ::SushiAI::Detail::guard_value(::SushiAI::Detail::SeamPhase::Record, [&]()
        {
            return step.builder().bias_add(step.builder().matmul(x, weight), bias);
        });
    }

    /** @brief Emits RELU, or TANH when @p is_relu is false, over @p x into @p step. */
    [[nodiscard]] inline Result<Graph::ValueID> trace_activation(Step& step, Graph::ValueID x,
                                                                 bool is_relu)
    {
        return ::SushiAI::Detail::guard_value(::SushiAI::Detail::SeamPhase::Record, [&]()
        {
            return is_relu ? step.builder().relu(x) : step.builder().tanh(x);
        });
    }

    /**
     * @brief Traces @p model over @p inputs and the NLL loss against @p targets into @p step.
     * @param model At least one layer; every layer but the last ends in RELU.
     * @return The one-element loss.
     */
    [[nodiscard]] inline Result<Graph::ValueID> trace_model(
        Step& step, const std::vector<LinearState>& model, const SushiBLAS::Tensor& inputs,
        const SushiBLAS::Tensor& targets, const TraceOptions& options = {})
    {
        SA_TRY_ASSIGN(Graph::ValueID hidden, step.input(inputs, "inputs"));

        for (std::size_t i = 0; i < model.size(); ++i)
        {
            const bool last = i + 1 == model.size();
            const bool is_relu = !last || options.head_is_relu;

            if (options.scoping == Scoping::None)
            {
                SA_TRY_ASSIGN(const Graph::ValueID projected, trace_linear(step, model[i], hidden));
                SA_TRY_ASSIGN(hidden, trace_activation(step, projected, is_relu));
                continue;
            }

            if (options.scoping == Scoping::SplitActivation && i == 0)
            {
                Graph::ValueID projected = Graph::K_NO_VALUE;
                {
                    const Scope scope(step.scopes(), "linear");
                    SA_TRY_ASSIGN(projected, trace_linear(step, model[i], hidden));
                }

                const Scope scope(step.scopes(), "activation");
                SA_TRY_ASSIGN(hidden, trace_activation(step, projected, is_relu));
                continue;
            }

            const char* name = "layer";
            if (options.scoping == Scoping::EncoderHead)
                name = last ? "head" : "encoder";
            else if (options.scoping == Scoping::SplitActivation)
                name = "rest";

            const Scope scope(step.scopes(), name);
            SA_TRY_ASSIGN(const Graph::ValueID projected, trace_linear(step, model[i], hidden));
            SA_TRY_ASSIGN(hidden, trace_activation(step, projected, is_relu));
        }

        if (options.extra_scope)
        {
            const Scope scope(step.scopes(), "extra");
            SA_TRY_ASSIGN(hidden, ::SushiAI::Detail::guard_value(
                                      ::SushiAI::Detail::SeamPhase::Record, [&]()
            {
                return step.builder().scale(hidden, 1.0);
            }));
        }

        SA_TRY_ASSIGN(const Graph::ValueID one_hot, step.input(targets, "targets"));
        return ::SushiAI::Detail::guard_value(::SushiAI::Detail::SeamPhase::Record, [&]()
        {
            return step.builder().nll_loss(hidden, one_hot);
        });
    }

    /** @brief Owns a session over one model, its update and one batch, for a training loop. */
    struct Loop
    {
        /** @brief Builds the executor and the session on @p engine with a step size of @p rate. */
        Loop(SushiBLAS::Engine& engine, float rate, SessionOptions options = {})
            : executor(engine), session(engine, executor, partitioner, options), update(rate)
        {

        }

        /**
         * @brief Allocates the model and one batch and binds every state to the session.
         * @param widths The input width followed by the width of each layer's result.
         */
        [[nodiscard]] Status prepare(SushiBLAS::Engine& engine, const std::vector<int64_t>& widths)
        {
            SA_TRY_ASSIGN(model, make_model(engine, widths));
            SA_TRY_ASSIGN(inputs, make_batch(engine, widths.front(), 0.25));
            SA_TRY_ASSIGN(targets, make_batch(engine, widths.back(), 1.75));
            return bind_model(session, model, update);
        }

        /**
         * @brief Traces, differentiates, updates and reads one step.
         * @return The bytes of the one-element loss.
         */
        [[nodiscard]] Result<std::vector<unsigned char>> step_bytes(
            const TraceOptions& options = {})
        {
            Step step = session.begin_step();
            SA_TRY_ASSIGN(const Graph::ValueID loss,
                          trace_model(step, model, inputs, targets, options));
            SA_TRY(step.backward(loss));
            SA_TRY(step.apply(update));
            SA_TRY_ASSIGN(const SushiBLAS::Tensor* const tensor, step.read(loss));
            return ::SushiAI::Test::read_tensor_bytes(*tensor);
        }

        /** @brief Runs one step as step_bytes() does and returns the loss as a double. */
        [[nodiscard]] Result<double> step(const TraceOptions& options = {})
        {
            Step step = session.begin_step();
            SA_TRY_ASSIGN(const Graph::ValueID loss,
                          trace_model(step, model, inputs, targets, options));
            SA_TRY(step.backward(loss));
            SA_TRY(step.apply(update));
            return step.read_scalar(loss);
        }

        /** @brief Returns how often the region named @p name was lowered into a recorder. */
        [[nodiscard]] std::size_t emit_count(const char* name, Direction direction) const
        {
            return executor.emit_count(session.region_names().find(name, direction));
        }

        ScopePartitioner partitioner;   /**< Cuts each step where a scope opens or closes. */
        RegionExecutor executor;        /**< Runs each step. */
        Session session;                /**< The session under test. */
        PlainDescent update;            /**< The update applied behind each step. */
        std::vector<LinearState> model; /**< The layers, bound to the session. */
        SushiBLAS::Tensor inputs;       /**< The [K_ROWS, first width] batch. */
        SushiBLAS::Tensor targets;      /**< The [K_ROWS, last width] targets. */
    };

} // namespace SushiAI::Eager::Support
