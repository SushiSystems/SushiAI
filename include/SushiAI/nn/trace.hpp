/**************************************************************************/
/* trace.hpp                                                              */
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
 * @file trace.hpp
 * @brief Declares Trace, the buffer-update sinks and the three forms a layer is traced in.
 * @author Mustafa Garip
 */

#pragma once

#include <cstdint>
#include <type_traits>
#include <utility>
#include <vector>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/core/check.hpp>
#include <SushiAI/core/result.hpp>
#include <SushiAI/core/shape.hpp>
#include <SushiAI/core/status.hpp>
#include <SushiAI/graph/builder.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/graph/state_id.hpp>
#include <SushiAI/nn/buffer.hpp>
#include <SushiAI/nn/module.hpp>

namespace SushiAI
{
    namespace NN
    {
        /** @brief Names what a traced layer emits: its training form or its evaluation form. */
        enum class TraceMode : std::uint8_t
        {
            Training,   /**< Batch statistics; state updates are requested. */
            Evaluation, /**< Stored statistics; requesting an update fails. */
        };

        /** @brief Describes target = (1 - momentum) * target + momentum * source_scale * source. */
        struct BufferUpdate
        {
            /** @brief Names the buffer the update writes, in every graph. */
            Graph::StateId target = Graph::K_NO_STATE;

            /** @brief Names the value of the traced graph the update reads. */
            Graph::ValueId source = Graph::K_NO_VALUE;

            /** @brief Weighs the source against the buffer's current content. */
            double momentum = 0.0;

            /** @brief Scales the source before it is blended in. */
            double source_scale = 1.0;
        };

        /** @brief Receives the buffer updates a training trace requests. */
        class BufferUpdateSink
        {
            public:
                /** @brief Releases the sink through its base. */
                virtual ~BufferUpdateSink() = default;

                /** @brief Takes one update, in the order the trace requests them. */
                virtual void add(const BufferUpdate& update) = 0;
        };

        /** @brief Collects the updates of one trace, in the order they were requested. */
        class BufferUpdateList final : public BufferUpdateSink
        {
            public:
                /** @brief Appends @p update and its source. */
                void add(const BufferUpdate& update) override;

                /**
                 * @brief Returns every update, in the order it was added.
                 * @return A view that the next add() invalidates.
                 */
                [[nodiscard]] SushiRuntime::span<const BufferUpdate> updates() const noexcept;

                /**
                 * @brief Lists the source value of every update, in the order of updates().
                 * @return A view that the next add() invalidates.
                 */
                [[nodiscard]] SushiRuntime::span<const Graph::ValueId> sources() const noexcept;

            private:
                /** @brief Holds the updates in the order they were added. */
                std::vector<BufferUpdate> updates_;

                /** @brief Holds updates_[i].source at index i. */
                std::vector<Graph::ValueId> sources_;
        };

        /** @brief Returns the sink that drops every update, for a forward-only pass. */
        [[nodiscard]] BufferUpdateSink& discard_updates() noexcept;

        /**
         * @brief Carries what a layer needs to trace itself: the builder, the mode, the sink.
         *
         * A trace refers to its builder and its sink and owns neither. A layer form returns a
         * value id, so a layer that refuses its input calls fail() and returns its input; the
         * caller reads status() after the layer returns.
         */
        class Trace final
        {
            public:
                /**
                 * @brief Returns a trace in Training mode that hands its updates to @p updates.
                 * @param builder Must outlive the trace.
                 * @param updates Must outlive the trace.
                 */
                [[nodiscard]] static Trace training(const Graph::GraphBuilder& builder,
                                                    BufferUpdateSink& updates) noexcept;

                /**
                 * @brief Returns a trace in Evaluation mode, which has no sink.
                 * @param builder Must outlive the trace.
                 */
                [[nodiscard]] static Trace evaluation(const Graph::GraphBuilder& builder) noexcept;

                /** @brief Releases nothing; a trace owns neither its builder nor its sink. */
                ~Trace() = default;

                /** @brief Forbids a copy, whose failure the callers of the original never read. */
                Trace(const Trace&) = delete;

                /** @brief Forbids assignment from another trace, for the reason the copy is. */
                Trace& operator=(const Trace&) = delete;

                /** @brief Forbids a move; a factory's result is built in place. */
                Trace(Trace&&) = delete;

                /** @brief Forbids move assignment, for the reason the copy is. */
                Trace& operator=(Trace&&) = delete;

                /** @brief Returns the builder the traced nodes go to. */
                [[nodiscard]] const Graph::GraphBuilder& builder() const noexcept;

                /** @brief Returns the mode this trace runs in. */
                [[nodiscard]] TraceMode mode() const noexcept;

                /**
                 * @brief Records why a layer refused its input, and marks the graph refused.
                 * @param failure A failed Status; a success is fatal. The first one is kept.
                 */
                void fail(Status failure) const noexcept;

                /** @brief Returns the first failure fail() was given, or a success. */
                [[nodiscard]] const Status& status() const noexcept;

                /**
                 * @brief Returns the value @p traced holds, or records its failure with fail().
                 * @param fallback Returned on a failure; a refusing layer passes its input.
                 */
                [[nodiscard]] Graph::ValueId value_or_fail(const Result<Graph::ValueId>& traced,
                                                           Graph::ValueId fallback) const noexcept;

                /**
                 * @brief Hands one update of @p target to the sink.
                 * @param target A defined buffer; an undefined one is fatal.
                 * @param source A value of builder().graph(); anything else is fatal.
                 * @return InvalidDescription in Evaluation mode, where nothing is recorded.
                 */
                [[nodiscard]] Status update(Buffer& target, Graph::ValueId source,
                                            double momentum, double source_scale) const;

            private:
                /** @brief Binds the three parts; @p updates is null in Evaluation mode. */
                Trace(const Graph::GraphBuilder& builder, BufferUpdateSink* updates,
                      TraceMode mode) noexcept;

                /** @brief Points at the builder the traced nodes go to. */
                const Graph::GraphBuilder* builder_;

                /** @brief Points at the sink of a Training trace; null in Evaluation mode. */
                BufferUpdateSink* updates_;

                /** @brief Holds the first failure; mutable, as a layer sees a const trace. */
                mutable Status failure_;

                /** @brief Holds the mode the factory stated. */
                TraceMode mode_;
        };

        namespace Detail
        {
            /** @brief Detects a static fields() on T. */
            template<typename T, typename = void>
            struct HasFields : std::false_type
            {

            };

            /** @brief Holds for a T that offers fields(). */
            template<typename T>
            struct HasFields<T, std::void_t<decltype(T::fields())>> : std::true_type
            {

            };

            /** @brief Detects forward(const Trace&, ValueId) returning ValueId on T. */
            template<typename T, typename = void>
            struct HasTraceForward : std::false_type
            {

            };

            /** @brief Holds for a T whose forward takes a trace. */
            template<typename T>
            struct HasTraceForward<T, std::enable_if_t<std::is_same<
                decltype(std::declval<T&>().forward(
                    std::declval<const Trace&>(),
                    std::declval<Graph::ValueId>())),
                Graph::ValueId>::value>>
                : std::true_type
            {

            };

            /** @brief Detects trace_training(const Trace&, ValueId) returning ValueId on T. */
            template<typename T, typename = void>
            struct HasTraceTraining : std::false_type
            {

            };

            /** @brief Holds for a T that offers the training form. */
            template<typename T>
            struct HasTraceTraining<T, std::enable_if_t<std::is_same<
                decltype(std::declval<T&>().trace_training(
                    std::declval<const Trace&>(),
                    std::declval<Graph::ValueId>())),
                Graph::ValueId>::value>>
                : std::true_type
            {

            };

            /** @brief Detects trace_evaluation(const Trace&, ValueId) returning ValueId on T. */
            template<typename T, typename = void>
            struct HasTraceEvaluation : std::false_type
            {

            };

            /** @brief Holds for a T that offers the evaluation form. */
            template<typename T>
            struct HasTraceEvaluation<T, std::enable_if_t<std::is_same<
                decltype(std::declval<T&>().trace_evaluation(
                    std::declval<const Trace&>(),
                    std::declval<Graph::ValueId>())),
                Graph::ValueId>::value>>
                : std::true_type
            {

            };

            /** @brief Detects output_shape(const Shape&) returning Shape on T. */
            template<typename T, typename = void>
            struct HasOutputShapeOfInput : std::false_type
            {

            };

            /** @brief Holds for a T that maps a per-sample input shape to its output shape. */
            template<typename T>
            struct HasOutputShapeOfInput<T, std::enable_if_t<std::is_same<
                decltype(std::declval<T&>().output_shape(std::declval<const Shape&>())),
                Shape>::value>>
                : std::true_type
            {

            };

            /** @brief Holds when T has exactly one of trace_training and trace_evaluation. */
            template<typename T>
            inline constexpr bool HasHalfModalPair =
                HasTraceTraining<T>::value != HasTraceEvaluation<T>::value;

            /** @brief Holds when T has a modal member and a forward of either kind as well. */
            template<typename T>
            inline constexpr bool HasModalMemberAndForward =
                (HasTraceTraining<T>::value || HasTraceEvaluation<T>::value) &&
                (IsModule<T>::value || HasTraceForward<T>::value);

        } // namespace Detail

        /** @brief Holds when T has forward(const Trace&, ValueId) -> ValueId and fields(). */
        template<typename T>
        inline constexpr bool TraceModule =
            Detail::HasFields<T>::value && Detail::HasTraceForward<T>::value;

        /**
         * @brief Holds when T has trace_training and trace_evaluation, fields(), and no forward.
         *
         * Both members are (const Trace&, ValueId) -> ValueId.
         */
        template<typename T>
        inline constexpr bool ModalModule =
            Detail::HasFields<T>::value && Detail::HasTraceTraining<T>::value &&
            Detail::HasTraceEvaluation<T>::value && !Module<T> && !TraceModule<T>;

        /** @brief Holds when trace_layer() can trace T in one of the three forms. */
        template<typename T>
        inline constexpr bool Traceable = Module<T> || TraceModule<T> || ModalModule<T>;

        /**
         * @brief Traces one layer in whichever of the three forms it has.
         *
         * A TraceModule is given the trace, a ModalModule the member of the trace's mode, and
         * a Module the trace's builder; a type that is both of the outer two is a TraceModule.
         * @return The id of the layer's result; meaningless once trace.status() holds a failure.
         */
        template<typename L>
        Graph::ValueId trace_layer(L& layer, const Trace& trace, Graph::ValueId x)
        {
            static_assert(!Detail::HasHalfModalPair<L>,
                          "A layer has one of trace_training and trace_evaluation and lacks "
                          "the other. A layer whose emission depends on the mode states both, "
                          "each as (const Trace&, Graph::ValueId) returning Graph::ValueId. "
                          "Check the layer named in the instantiation below, not this line.");
            static_assert(!Detail::HasModalMemberAndForward<L>,
                          "A layer has trace_training or trace_evaluation and a forward as "
                          "well, so which of them traces it is unstated. A modal layer has "
                          "the pair and no forward. Check the layer named in the instantiation "
                          "below, not this line.");
            static_assert(Traceable<L>,
                          "A traced layer needs a static fields() and one of three forms: "
                          "forward(const Graph::GraphBuilder&, Graph::ValueId), "
                          "forward(const Trace&, Graph::ValueId), or the pair trace_training "
                          "and trace_evaluation over (const Trace&, Graph::ValueId), each "
                          "returning exactly Graph::ValueId. Check the layer named in the "
                          "instantiation below, not this line.");

            if constexpr (TraceModule<L>)
            {
                return layer.forward(trace, x);
            }
            else if constexpr (ModalModule<L>)
            {
                switch (trace.mode())
                {
                    case TraceMode::Training:
                        return layer.trace_training(trace, x);
                    case TraceMode::Evaluation:
                        return layer.trace_evaluation(trace, x);
                }

                SA_FATAL("a trace holds the mode {}, which names no layer form",
                         static_cast<int>(trace.mode()));
            }
            else
            {
                return layer.forward(trace.builder(), x);
            }
        }

    } // namespace NN
} // namespace SushiAI
