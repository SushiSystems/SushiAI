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
 * @brief Declares Trace, its sinks and the three forms a layer is traced in.
 * @author Mustafa Garip
 */

#pragma once

#include <cstdint>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/failure/check.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/shape/shape.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/graph/builder.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/graph/state_id.hpp>
#include <SushiAI/nn/buffer.hpp>
#include <SushiAI/nn/host_table.hpp>
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
            Graph::StateID target = Graph::K_NO_STATE;

            /** @brief Names the value of the traced graph the update reads. */
            Graph::ValueID source = Graph::K_NO_VALUE;

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
                [[nodiscard]] SushiRuntime::span<const Graph::ValueID> sources() const noexcept;

            private:
                /** @brief Holds the updates in the order they were added. */
                std::vector<BufferUpdate> updates_;

                /** @brief Holds updates_[i].source at index i. */
                std::vector<Graph::ValueID> sources_;
        };

        /** @brief Returns the sink that drops every update, for a forward-only pass. */
        [[nodiscard]] BufferUpdateSink& discard_updates() noexcept;

        /** @brief Names one leaf a step is fed beside the model input. */
        struct FedLeaf
        {
            /** @brief Names the leaf to whoever feeds it; it outlives the graph. */
            std::string_view name;

            /** @brief Names the leaf in the traced graph. */
            Graph::ValueID value = Graph::K_NO_VALUE;
        };

        /** @brief Receives the leaves a carrying trace declares beside the model input. */
        class FedLeafSink
        {
            public:
                /** @brief Releases the sink through its base. */
                virtual ~FedLeafSink() = default;

                /** @brief Returns the leaf declared under @p name, or K_NO_VALUE for none. */
                [[nodiscard]] virtual Graph::ValueID find(std::string_view name) const noexcept = 0;

                /** @brief Takes one leaf, in the order the trace declares them. */
                virtual void add(const FedLeaf& leaf) = 0;
        };

        /** @brief Collects the fed leaves of one trace, in the order they were declared. */
        class FedLeafList final : public FedLeafSink
        {
            public:
                /** @brief Returns the leaf declared under @p name, or K_NO_VALUE for none. */
                [[nodiscard]] Graph::ValueID find(std::string_view name) const noexcept override;

                /** @brief Appends @p leaf. */
                void add(const FedLeaf& leaf) override;

                /**
                 * @brief Returns every leaf, in the order it was added.
                 * @return A view that the next add() invalidates.
                 */
                [[nodiscard]] SushiRuntime::span<const FedLeaf> leaves() const noexcept;

            private:
                /** @brief Holds the leaves in the order they were added. */
                std::vector<FedLeaf> leaves_;
        };

        /** @brief Describes leaf = next, applied after each run of a step. */
        struct CarriedValue
        {
            /** @brief Names the leaf a run reads, which holds what the run before left. */
            Graph::ValueID leaf = Graph::K_NO_VALUE;

            /** @brief Names the value of the traced graph the leaf takes after a run. */
            Graph::ValueID next = Graph::K_NO_VALUE;

            /** @brief Holds what every element of the leaf is before the first run. */
            double initial_value = 0.0;
        };

        /** @brief Receives the values a carrying trace keeps from a run to the next. */
        class CarriedValueSink
        {
            public:
                /** @brief Releases the sink through its base. */
                virtual ~CarriedValueSink() = default;

                /** @brief Takes one carried value, in the order the trace requests them. */
                virtual void add(const CarriedValue& carried) = 0;
        };

        /** @brief Collects the carried values of one trace, in the order they were requested. */
        class CarriedValueList final : public CarriedValueSink
        {
            public:
                /** @brief Appends @p carried and its next value. */
                void add(const CarriedValue& carried) override;

                /**
                 * @brief Returns every carried value, in the order it was added.
                 * @return A view that the next add() invalidates.
                 */
                [[nodiscard]] SushiRuntime::span<const CarriedValue> values() const noexcept;

                /**
                 * @brief Lists the next value of every entry, in the order of values().
                 * @return A view that the next add() invalidates.
                 */
                [[nodiscard]] SushiRuntime::span<const Graph::ValueID> sources() const noexcept;

            private:
                /** @brief Holds the carried values in the order they were added. */
                std::vector<CarriedValue> values_;

                /** @brief Holds values_[i].next at index i. */
                std::vector<Graph::ValueID> sources_;
        };

        /**
         * @brief Carries what a layer needs to trace itself: the builder, the mode, the sinks.
         *
         * A trace refers to its builder and its sinks and owns none of them. A layer form returns a
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
                 * @param seed What the random sites of the trace derive their draws from.
                 */
                [[nodiscard]] static Trace training(const Graph::GraphBuilder& builder,
                                                    BufferUpdateSink& updates,
                                                    std::uint64_t seed = 0) noexcept;

                /**
                 * @brief Returns a trace in Evaluation mode, which has no sink.
                 * @param builder Must outlive the trace.
                 */
                [[nodiscard]] static Trace evaluation(const Graph::GraphBuilder& builder) noexcept;

                /**
                 * @brief Returns a trace in Evaluation mode whose step carries state between
                 *        runs and is fed the leaves a layer declares.
                 * @param builder Must outlive the trace, as must @p fed and @p carried.
                 */
                [[nodiscard]] static Trace carrying(const Graph::GraphBuilder& builder,
                                                    FedLeafSink& fed,
                                                    CarriedValueSink& carried) noexcept;

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

                /** @brief Returns the seed training() was given, or 0 for any other trace. */
                [[nodiscard]] std::uint64_t seed() const noexcept;

                /**
                 * @brief Counts one more site that draws at each run of the traced step.
                 * @return The order of the site among those claimed so far, from 0.
                 */
                [[nodiscard]] std::uint64_t claim_random_site() const noexcept;

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
                [[nodiscard]] Graph::ValueID value_or_fail(const Result<Graph::ValueID>& traced,
                                                           Graph::ValueID fallback) const noexcept;

                /**
                 * @brief Returns the id @p traced holds, or records its failure with fail().
                 * @param fallback Returned on a failure; a refusing layer passes its input.
                 */
                [[nodiscard]] Graph::ValueID value_or_fail(const Graph::Traced& traced,
                                                           Graph::ValueID fallback) const noexcept;

                /**
                 * @brief Hands one update of @p target to the sink.
                 * @param target A defined buffer; an undefined one is fatal.
                 * @param source A value of builder().graph(); anything else is fatal.
                 * @return InvalidDescription in Evaluation mode, where nothing is recorded.
                 */
                [[nodiscard]] Status update(Buffer& target, Graph::ValueID source,
                                            double momentum, double source_scale) const noexcept;

                /** @brief Reports whether carrying() built this trace. */
                [[nodiscard]] bool is_carrying() const noexcept;

                /**
                 * @brief Declares the leaf a run is fed under @p name, once per trace.
                 * @param name Names the leaf to the feeder; it must outlive the graph.
                 * @return The leaf, the one an earlier call declared under @p name, or
                 *         InvalidDescription when that one differs in shape or type, or the
                 *         trace is not a carrying one.
                 */
                [[nodiscard]] Result<Graph::ValueID> feed(std::string_view name, Shape shape,
                                                          DType dtype) const noexcept;

                /**
                 * @brief Declares the leaf of a value the step carries between runs.
                 * @param debug_name A label for diagnostics; it must outlive the graph.
                 * @return The leaf, which carry() must be given once, or InvalidDescription
                 *         when the trace is not a carrying one.
                 */
                [[nodiscard]] Result<Graph::ValueID> carried_leaf(Shape shape,
                                                                  std::string_view debug_name,
                                                                  DType dtype) const noexcept;

                /**
                 * @brief Asks that @p leaf hold @p next after each run.
                 * @param leaf What carried_leaf() returned; any other value is fatal.
                 * @param next A value of builder().graph() shaped and typed as @p leaf is.
                 * @param initial_value What @p leaf holds before the first run and after a reset.
                 * @return InvalidDescription when the two differ in shape or type, or the
                 *         trace is not a carrying one.
                 */
                [[nodiscard]] Status carry(Graph::ValueID leaf, Graph::ValueID next,
                                           double initial_value = 0.0) const noexcept;

                /**
                 * @brief Names the sink host_table() hands its tables to.
                 * @param tables Must outlive the trace; the step built from it fills them.
                 */
                void collect_tables(HostTableSink& tables) noexcept;

                /**
                 * @brief Declares a leaf the step fills once with @p values, no model state.
                 * @param debug_name A label for diagnostics; it must outlive the graph.
                 * @param values One number per element of @p shape, the last axis fastest.
                 * @return The leaf, or InvalidDescription for a trace no collect_tables()
                 *         named a sink for, a type that is not float32 or float64, or a count
                 *         of values that is not the elements of @p shape.
                 */
                [[nodiscard]] Result<Graph::ValueID> host_table(Shape shape,
                                                                std::string_view debug_name,
                                                                DType dtype,
                                                                std::vector<double> values) const noexcept;

            private:
                /** @brief Binds the parts; every sink a form does not have is null. */
                Trace(const Graph::GraphBuilder& builder, BufferUpdateSink* updates,
                      FedLeafSink* fed, CarriedValueSink* carried, TraceMode mode,
                      std::uint64_t seed) noexcept;

                /** @brief Points at the builder the traced nodes go to. */
                const Graph::GraphBuilder* builder_;

                /** @brief Points at the sink of a Training trace; null in Evaluation mode. */
                BufferUpdateSink* updates_;

                /** @brief Points at the sink of the fed leaves; null unless carrying. */
                FedLeafSink* fed_;

                /** @brief Points at the sink of the carried values; null unless carrying. */
                CarriedValueSink* carried_;

                /** @brief Holds the first failure; mutable, as a layer sees a const trace. */
                mutable Status failure_;

                /** @brief Holds the mode the factory stated. */
                TraceMode mode_;

                /** @brief Holds the seed the factory stated. */
                std::uint64_t seed_;

                /** @brief Counts the claimed random sites; mutable, as failure_ is. */
                mutable std::uint64_t random_sites_ = 0;

                /** @brief Points at the sink of the host tables; null until one is named. */
                HostTableSink* tables_ = nullptr;
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

            /** @brief Detects forward(const Trace&, ValueID) returning ValueID on T. */
            template<typename T, typename = void>
            struct HasTraceForward : std::false_type
            {

            };

            /** @brief Holds for a T whose forward takes a trace. */
            template<typename T>
            struct HasTraceForward<T, std::enable_if_t<std::is_same<
                decltype(std::declval<T&>().forward(
                    std::declval<const Trace&>(),
                    std::declval<Graph::ValueID>())),
                Graph::ValueID>::value>>
                : std::true_type
            {

            };

            /** @brief Detects trace_training(const Trace&, ValueID) returning ValueID on T. */
            template<typename T, typename = void>
            struct HasTraceTraining : std::false_type
            {

            };

            /** @brief Holds for a T that offers the training form. */
            template<typename T>
            struct HasTraceTraining<T, std::enable_if_t<std::is_same<
                decltype(std::declval<T&>().trace_training(
                    std::declval<const Trace&>(),
                    std::declval<Graph::ValueID>())),
                Graph::ValueID>::value>>
                : std::true_type
            {

            };

            /** @brief Detects trace_evaluation(const Trace&, ValueID) returning ValueID on T. */
            template<typename T, typename = void>
            struct HasTraceEvaluation : std::false_type
            {

            };

            /** @brief Holds for a T that offers the evaluation form. */
            template<typename T>
            struct HasTraceEvaluation<T, std::enable_if_t<std::is_same<
                decltype(std::declval<T&>().trace_evaluation(
                    std::declval<const Trace&>(),
                    std::declval<Graph::ValueID>())),
                Graph::ValueID>::value>>
                : std::true_type
            {

            };

            /** @brief Detects forward_incremental(const Trace&, ValueID, const Shape&) on T. */
            template<typename T, typename = void>
            struct HasIncrementalForward : std::false_type
            {

            };

            /** @brief Holds for a T that offers the incremental form, returning ValueID. */
            template<typename T>
            struct HasIncrementalForward<T, std::enable_if_t<std::is_same<
                decltype(std::declval<T&>().forward_incremental(
                    std::declval<const Trace&>(),
                    std::declval<Graph::ValueID>(),
                    std::declval<const Shape&>())),
                Graph::ValueID>::value>>
                : std::true_type
            {

            };

            /** @brief Detects output_shape(const Shape&) returning Result<Shape> on T. */
            template<typename T, typename = void>
            struct HasOutputShapeOfInput : std::false_type
            {

            };

            /** @brief Holds for a T that maps a per-sample input shape to its output shape. */
            template<typename T>
            struct HasOutputShapeOfInput<T, std::enable_if_t<std::is_same<
                decltype(std::declval<T&>().output_shape(std::declval<const Shape&>())),
                Result<Shape>>::value>>
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

        /** @brief Holds when T has forward(const Trace&, ValueID) -> ValueID and fields(). */
        template<typename T>
        inline constexpr bool TraceModule =
            Detail::HasFields<T>::value && Detail::HasTraceForward<T>::value;

        /**
         * @brief Holds when T has trace_training and trace_evaluation, fields(), and no forward.
         *
         * Both members are (const Trace&, ValueID) -> ValueID.
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
        Graph::ValueID trace_layer(L& layer, const Trace& trace, Graph::ValueID x) noexcept
        {
            static_assert(!Detail::HasHalfModalPair<L>,
                          "A layer has one of trace_training and trace_evaluation and lacks "
                          "the other. A layer whose emission depends on the mode states both, "
                          "each as (const Trace&, Graph::ValueID) returning Graph::ValueID. "
                          "Check the layer named in the instantiation below, not this line.");
            static_assert(!Detail::HasModalMemberAndForward<L>,
                          "A layer has trace_training or trace_evaluation and a forward as "
                          "well, so which of them traces it is unstated. A modal layer has "
                          "the pair and no forward. Check the layer named in the instantiation "
                          "below, not this line.");
            static_assert(Traceable<L>,
                          "A traced layer needs a static fields() and one of three forms: "
                          "forward(const Graph::GraphBuilder&, const Graph::Traced&) returning "
                          "exactly Graph::Traced, forward(const Trace&, Graph::ValueID), or the "
                          "pair trace_training and trace_evaluation over (const Trace&, "
                          "Graph::ValueID), the last two returning exactly Graph::ValueID. "
                          "Check the layer named in the instantiation below, not this line.");

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
                return trace.value_or_fail(layer.forward(trace.builder(), x), x);
            }
        }

        /**
         * @brief Holds when T has forward_incremental(const Trace&, ValueID, const Shape&)
         *        returning ValueID beside one of the three forms.
         */
        template<typename T>
        inline constexpr bool IncrementalModule =
            Traceable<T> && Detail::HasIncrementalForward<T>::value;

        /**
         * @brief Traces the incremental form of one layer in a carrying trace.
         * @param x The newest slice of the value the layer reads in its ordinary form.
         * @param sample The per-sample shape of that value in the ordinary form.
         * @return The id of the layer's result; meaningless once trace.status() holds a failure.
         */
        template<typename L>
        Graph::ValueID trace_layer_incremental(L& layer, const Trace& trace, Graph::ValueID x,
                                               const Shape& sample) noexcept
        {
            static_assert(IncrementalModule<L>,
                          "A layer traced one slice at a time needs forward_incremental("
                          "const Trace&, Graph::ValueID, const Shape&) returning exactly "
                          "Graph::ValueID beside its ordinary form. Check the layer named in "
                          "the instantiation below, not this line.");

            return layer.forward_incremental(trace, x, sample);
        }

    } // namespace NN
} // namespace SushiAI
