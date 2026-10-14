/**************************************************************************/
/* test_trace.cpp                                                         */
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
 * @file test_trace.cpp
 * @brief Tests Trace, the buffer-update sinks, the three layer forms and trace_layer.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>
#include <gtest/gtest.h>
#include <SushiAI/core/common.hpp>
#include <SushiAI/core/shape.hpp>
#include <SushiAI/core/status.hpp>
#include <SushiAI/graph/builder.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/graph/state_id.hpp>
#include <SushiAI/nn/buffer.hpp>
#include <SushiAI/nn/layers.hpp>
#include <SushiAI/nn/module.hpp>
#include <SushiAI/nn/trace.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/failure_expectations/throw_expectations.hpp>

using namespace SushiAI;
using NN::Buffer;
using NN::BufferUpdate;
using NN::BufferUpdateList;
using NN::Linear;
using NN::ReLU;
using NN::Sequential;
using NN::Trace;
using NN::TraceMode;

namespace
{
    /** @brief Holds the absolute tolerance of a double stored and read back unchanged. */
    constexpr double K_EXACT_TOLERANCE = 1e-15;

    /** @brief Counts which of its two forms a trace asked for; stands in for a modal leaf. */
    struct ModalStub
    {
        /** @brief Counts the calls of trace_training. */
        int training_calls = 0;

        /** @brief Counts the calls of trace_evaluation. */
        int evaluation_calls = 0;

        /** @brief Lists no state. */
        static constexpr auto fields()
        {
            return std::tuple<>{};
        }

        /** @brief Emits one ReLU and counts the call. */
        Graph::ValueID trace_training(const Trace& trace, Graph::ValueID x)
        {
            ++training_calls;
            return trace.builder().relu(x);
        }

        /** @brief Emits nothing, returns its input and counts the call. */
        Graph::ValueID trace_evaluation(const Trace&, Graph::ValueID x)
        {
            ++evaluation_calls;
            return x;
        }
    };

    /** @brief Holds a modal child and passes the trace on; stands in for a composite block. */
    struct CompositeStub
    {
        /** @brief Holds the modal child the trace is passed to. */
        ModalStub inner{};

        /** @brief Lists the child. */
        static constexpr auto fields()
        {
            return std::tuple{NN::child("inner", &CompositeStub::inner)};
        }

        /** @brief Traces the child in the form the trace's mode names. */
        Graph::ValueID forward(const Trace& trace, Graph::ValueID x)
        {
            return NN::trace_layer(inner, trace, x);
        }
    };

    /** @brief Takes its trace by value, which a layer form must not. */
    struct ByValueTraceStub
    {
        /** @brief Lists no state. */
        static constexpr auto fields()
        {
            return std::tuple<>{};
        }

        /** @brief Returns its input. */
        Graph::ValueID forward(Trace, Graph::ValueID x)
        {
            return x;
        }
    };

    /** @brief States the training form only, which trace_layer refuses to compile. */
    struct TrainingOnlyStub
    {
        /** @brief Lists no state. */
        static constexpr auto fields()
        {
            return std::tuple<>{};
        }

        /** @brief Returns its input. */
        Graph::ValueID trace_training(const Trace&, Graph::ValueID x)
        {
            return x;
        }
    };

    /** @brief States the modal pair and a builder forward, which trace_layer refuses. */
    struct ModalWithForwardStub
    {
        /** @brief Lists no state. */
        static constexpr auto fields()
        {
            return std::tuple<>{};
        }

        /** @brief Returns its input. */
        Graph::ValueID forward(const Graph::GraphBuilder&, Graph::ValueID x)
        {
            return x;
        }

        /** @brief Returns its input. */
        Graph::ValueID trace_training(const Trace&, Graph::ValueID x)
        {
            return x;
        }

        /** @brief Returns its input. */
        Graph::ValueID trace_evaluation(const Trace&, Graph::ValueID x)
        {
            return x;
        }
    };

    /** @brief Offers both forwards and counts which one a caller picked. */
    struct BothForwardsStub
    {
        /** @brief Counts the calls of the builder forward. */
        int builder_calls = 0;

        /** @brief Counts the calls of the trace forward. */
        int trace_calls = 0;

        /** @brief Lists no state. */
        static constexpr auto fields()
        {
            return std::tuple<>{};
        }

        /** @brief Returns its input and counts the call. */
        Graph::ValueID forward(const Graph::GraphBuilder&, Graph::ValueID x)
        {
            ++builder_calls;
            return x;
        }

        /** @brief Returns its input and counts the call. */
        Graph::ValueID forward(const Trace&, Graph::ValueID x)
        {
            ++trace_calls;
            return x;
        }
    };

    /** @brief Has the modal pair and no fields(), so no form fits it. */
    struct FieldlessStub
    {
        /** @brief Returns its input. */
        Graph::ValueID trace_training(const Trace&, Graph::ValueID x)
        {
            return x;
        }

        /** @brief Returns its input. */
        Graph::ValueID trace_evaluation(const Trace&, Graph::ValueID x)
        {
            return x;
        }
    };

    /** @brief Refuses every input: records one failure on the trace and returns its input. */
    struct RefusingStub
    {
        /** @brief Lists no state. */
        static constexpr auto fields()
        {
            return std::tuple<>{};
        }

        /** @brief Records an InvalidDescription that names the stub and emits nothing. */
        Graph::ValueID forward(const Trace& trace, Graph::ValueID x)
        {
            trace.fail(Status::from_message(StatusCode::InvalidDescription, []()
            {
                return std::string("RefusingStub refuses its input");
            }));
            return x;
        }
    };

    /** @brief Returns a failure whose message is @p text. */
    Status failure_saying(std::string_view text)
    {
        return Status::from_message(StatusCode::InvalidDescription, [text]()
        {
            return std::string(text);
        });
    }

    /** @brief Returns a defined [3] buffer that no graph has read. */
    Buffer defined_buffer()
    {
        Buffer buffer;
        const Status defined = buffer.define(Shape{3}, 0.0);
        static_cast<void>(defined);
        return buffer;
    }

    /** @brief Requests an update of a buffer that define() never shaped, for a death test. */
    void update_undefined_buffer()
    {
        Graph::Graph graph;
        const Graph::GraphBuilder builder(graph);
        const Graph::ValueID x = builder.input(Shape{2, 3}, "x");

        Buffer buffer;
        BufferUpdateList list;
        const Status ignored = Trace::training(builder, list).update(buffer, x, 0.1, 1.0);
        static_cast<void>(ignored);
    }

    /** @brief Requests an update that reads @p source of a one-value graph, for a death test. */
    void update_from(Graph::ValueID source)
    {
        Graph::Graph graph;
        const Graph::GraphBuilder builder(graph);
        static_cast<void>(builder.input(Shape{2, 3}, "x"));

        Buffer buffer = defined_buffer();
        BufferUpdateList list;
        const Status ignored = Trace::training(builder, list).update(buffer, source, 0.1, 1.0);
        static_cast<void>(ignored);
    }

    /** @brief Collects the parameter names of @p model in walk order. */
    template<typename M>
    std::vector<std::string> parameter_names(M& model)
    {
        std::vector<std::string> names;
        NN::for_each_parameter(model, [&names](const std::string& name, NN::Parameter&)
        {
            names.push_back(name);
        });
        return names;
    }

} // namespace

static_assert(NN::Module<Linear> && !NN::TraceModule<Linear> && !NN::ModalModule<Linear>,
              "a builder-only leaf has the Module form alone");
static_assert(NN::ModalModule<ModalStub> && !NN::Module<ModalStub> &&
                  !NN::TraceModule<ModalStub>,
              "a leaf with the pair and no forward has the modal form alone");
static_assert(NN::TraceModule<CompositeStub> && !NN::Module<CompositeStub> &&
                  !NN::ModalModule<CompositeStub>,
              "a composite whose forward takes a trace has the trace form alone");
static_assert(NN::Traceable<Linear> && NN::Traceable<ModalStub> &&
                  NN::Traceable<CompositeStub>,
              "each of the three forms is traceable");

static_assert(NN::Module<Sequential<Linear, ReLU>> && NN::TraceModule<Sequential<Linear, ReLU>>,
              "a Sequential of Modules keeps its builder forward and gains the trace one");
static_assert(!NN::Module<Sequential<Linear, ModalStub>>,
              "a Sequential holding a modal layer must not offer a builder forward");
static_assert(NN::TraceModule<Sequential<Linear, ModalStub>>,
              "a Sequential holding a modal layer is traced through a trace");
static_assert(!NN::Module<Sequential<Sequential<Linear, ModalStub>, ReLU>> &&
                  NN::TraceModule<Sequential<Sequential<Linear, ModalStub>, ReLU>>,
              "a modal layer two levels down still removes the builder forward");
static_assert(!NN::Module<Sequential<CompositeStub>> &&
                  NN::TraceModule<Sequential<CompositeStub>>,
              "a Sequential holding a trace-only composite is trace-only");
static_assert(NN::ModuleContainer<Sequential<Linear, ModalStub>>,
              "a Sequential holding a modal layer is still walked as a container");
static_assert(NN::Module<BothForwardsStub> && NN::TraceModule<BothForwardsStub> &&
                  !NN::ModalModule<BothForwardsStub>,
              "a layer with both forwards has the Module form and the trace form");

// trace_layer() and Sequential assert these two are false, so neither stub can be traced.
static_assert(NN::Detail::HasHalfModalPair<TrainingOnlyStub>,
              "a layer with trace_training and no trace_evaluation must be detected");
static_assert(NN::Detail::HasModalMemberAndForward<ModalWithForwardStub>,
              "a layer with the modal pair and a forward must be detected");
static_assert(!NN::Detail::HasHalfModalPair<ModalStub> &&
                  !NN::Detail::HasModalMemberAndForward<ModalStub>,
              "a well-formed modal layer trips neither check");
static_assert(!NN::Detail::HasHalfModalPair<Linear> &&
                  !NN::Detail::HasModalMemberAndForward<Linear>,
              "a builder-only layer trips neither check");
static_assert(!NN::Traceable<TrainingOnlyStub> && !NN::Traceable<FieldlessStub> &&
                  !NN::Traceable<int>,
              "half a pair, a missing fields() and a non-layer are not traceable");
static_assert(!NN::ModalModule<ModalWithForwardStub>,
              "a layer with a forward is not modal, whatever else it has");

static_assert(std::is_final<Trace>::value, "Trace is not a base class");
static_assert(!std::is_copy_constructible<Trace>::value &&
                  !std::is_move_constructible<Trace>::value,
              "a second trace would hold a failure the caller of the first never reads");
static_assert(!NN::TraceModule<ByValueTraceStub> && !NN::Traceable<ByValueTraceStub>,
              "a forward that takes its trace by value is not a layer form");
static_assert(!std::is_default_constructible<Trace>::value,
              "a trace comes from one of its two factories");
static_assert(!std::is_convertible<Graph::GraphBuilder, Trace>::value,
              "a builder must not turn into a trace of an unstated mode");
static_assert(std::has_virtual_destructor<NN::BufferUpdateSink>::value,
              "a sink is released through its base");

/** @brief Tests that each factory reports its own mode and the builder it was given. */
TEST(Unit_Trace, AFactoryStatesItsModeAndKeepsItsBuilder)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    BufferUpdateList list;

    const Trace training = Trace::training(builder, list);
    const Trace evaluation = Trace::evaluation(builder);

    EXPECT_EQ(training.mode(), TraceMode::Training);
    EXPECT_EQ(evaluation.mode(), TraceMode::Evaluation);
    EXPECT_EQ(&training.builder(), &builder);
    EXPECT_EQ(&evaluation.builder(), &builder);
}

/** @brief Tests that update under an evaluation trace fails as an invalid description. */
TEST(Unit_Trace, UpdateUnderAnEvaluationTraceFails)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueID x = builder.input(Shape{2, 3}, "x");
    Buffer buffer = defined_buffer();

    const Trace trace = Trace::evaluation(builder);

    SA_EXPECT_FAILURE(trace.update(buffer, x, 0.1, 1.0), StatusCode::InvalidDescription,
                      "evaluation trace");
}

/** @brief Tests that a new list holds no update and no source. */
TEST(Unit_Trace, ANewListIsEmpty)
{
    const BufferUpdateList list;

    EXPECT_EQ(list.updates().size(), 0u);
    EXPECT_EQ(list.sources().size(), 0u);
}

/** @brief Tests that each update under a list adds one update and one source, in order. */
TEST(Unit_Trace, EachUpdateAddsOneUpdateAndOneSource)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueID x = builder.input(Shape{2, 3}, "x");
    const Graph::ValueID y = builder.relu(x);
    Buffer mean = defined_buffer();
    Buffer variance = defined_buffer();
    ASSERT_NE(mean.state(), variance.state());

    BufferUpdateList list;
    const Trace trace = Trace::training(builder, list);

    SA_EXPECT_OK(trace.update(mean, x, 0.1, 1.0));
    ASSERT_EQ(list.updates().size(), 1u);
    ASSERT_EQ(list.sources().size(), 1u);

    SA_EXPECT_OK(trace.update(variance, y, 0.25, 1.5));
    ASSERT_EQ(list.updates().size(), 2u);
    ASSERT_EQ(list.sources().size(), 2u);

    // The four numbers are stored as given, so they compare exactly.
    EXPECT_EQ(list.updates()[0].target, mean.state());
    EXPECT_EQ(list.updates()[0].source, x);
    EXPECT_NEAR(list.updates()[0].momentum, 0.1, K_EXACT_TOLERANCE);
    EXPECT_NEAR(list.updates()[0].source_scale, 1.0, K_EXACT_TOLERANCE);
    EXPECT_EQ(list.updates()[1].target, variance.state());
    EXPECT_EQ(list.updates()[1].source, y);
    EXPECT_NEAR(list.updates()[1].momentum, 0.25, K_EXACT_TOLERANCE);
    EXPECT_NEAR(list.updates()[1].source_scale, 1.5, K_EXACT_TOLERANCE);
    EXPECT_EQ(list.sources()[0], x);
    EXPECT_EQ(list.sources()[1], y);
}

/** @brief Tests that an update adds no value to the traced graph. */
TEST(Unit_Trace, AnUpdateAddsNothingToTheGraph)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueID x = builder.input(Shape{2, 3}, "x");
    Buffer buffer = defined_buffer();
    BufferUpdateList list;
    const std::size_t before = graph.value_count();

    SA_EXPECT_OK(Trace::training(builder, list).update(buffer, x, 0.1, 1.0));

    EXPECT_EQ(graph.value_count(), before);
    EXPECT_EQ(buffer.id_in(graph), Graph::K_NO_VALUE);
}

/** @brief Tests that a recorded target still names its buffer after the buffer is moved. */
TEST(Unit_Trace, ARecordedTargetSurvivesAMoveOfItsBuffer)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueID x = builder.input(Shape{2, 3}, "x");
    Buffer buffer = defined_buffer();
    BufferUpdateList list;

    SA_EXPECT_OK(Trace::training(builder, list).update(buffer, x, 0.1, 1.0));
    const Buffer moved = std::move(buffer);

    ASSERT_EQ(list.updates().size(), 1u);
    EXPECT_EQ(list.updates()[0].target, moved.state());
    EXPECT_NE(list.updates()[0].target, Graph::K_NO_STATE);
}

/**
 * @brief Tests that the discarding sink accepts an update in Training mode, adds no value to
 *        the graph and is one object.
 */
TEST(Unit_Trace, TheDiscardingSinkAcceptsAnUpdateAndIsOneObject)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueID x = builder.input(Shape{2, 3}, "x");
    Buffer buffer = defined_buffer();
    const std::size_t before = graph.value_count();

    const Trace trace = Trace::training(builder, NN::discard_updates());

    EXPECT_EQ(trace.mode(), TraceMode::Training);
    SA_EXPECT_OK(trace.update(buffer, x, 0.1, 1.0));
    EXPECT_EQ(graph.value_count(), before);
    EXPECT_EQ(&NN::discard_updates(), &NN::discard_updates());
}

/** @brief Tests that an update of a buffer define() never shaped stops the process. */
TEST(Unit_TraceDeathTest, AnUpdateOfAnUndefinedBufferIsFatal)
{
    SA_EXPECT_FATAL(update_undefined_buffer(), "define() has not given a shape");
}

/** @brief Tests that an update of a buffer no graph has read names the buffer's identity. */
TEST(Unit_Trace, AnUpdateOfABufferThatWasNeverReadIsRecorded)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueID x = builder.input(Shape{2, 3}, "x");
    Buffer buffer = defined_buffer();
    BufferUpdateList list;
    const std::size_t before = graph.value_count();

    SA_EXPECT_OK(Trace::training(builder, list).update(buffer, x, 0.1, 1.0));

    ASSERT_EQ(list.updates().size(), 1u);
    EXPECT_NE(list.updates()[0].target, Graph::K_NO_STATE);
    EXPECT_EQ(list.updates()[0].target, buffer.state());
    EXPECT_EQ(graph.value_count(), before);
    EXPECT_EQ(buffer.id_in(graph), Graph::K_NO_VALUE);
}

/** @brief Tests that a new trace holds no failure and leaves its graph unrefused. */
TEST(Unit_Trace, ANewTraceHoldsNoFailure)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);

    const Trace trace = Trace::evaluation(builder);

    SA_EXPECT_OK(trace.status());
    EXPECT_FALSE(graph.has_refusal());
}

/** @brief Tests that fail() keeps the first failure it is given and marks the graph refused. */
TEST(Unit_Trace, FailKeepsTheFirstFailureAndMarksTheGraphRefused)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Trace trace = Trace::evaluation(builder);

    trace.fail(failure_saying("the first reason"));
    trace.fail(failure_saying("the second reason"));

    SA_EXPECT_FAILURE(trace.status(), StatusCode::InvalidDescription, "the first reason");
    EXPECT_EQ(trace.status().message().find("second"), std::string_view::npos);
    EXPECT_TRUE(graph.has_refusal());
}

/** @brief Tests that value_or_fail() returns a held value and records nothing. */
TEST(Unit_Trace, ValueOrFailReturnsAHeldValue)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueID x = builder.input(Shape{2, 3}, "x");
    const Graph::ValueID y = builder.relu(x);
    const Trace trace = Trace::evaluation(builder);

    EXPECT_EQ(trace.value_or_fail(Result<Graph::ValueID>(y), x), y);
    SA_EXPECT_OK(trace.status());
}

/** @brief Tests that value_or_fail() returns the fallback for a failure and records it. */
TEST(Unit_Trace, ValueOrFailRecordsAFailureAndReturnsTheFallback)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueID x = builder.input(Shape{2, 3}, "x");
    const Trace trace = Trace::evaluation(builder);

    const Result<Graph::ValueID> refused(failure_saying("the layer does not fit"));

    EXPECT_EQ(trace.value_or_fail(refused, x), x);
    SA_EXPECT_FAILURE(trace.status(), StatusCode::InvalidDescription, "does not fit");
}

/** @brief Tests that fail() given a success stops the process. */
TEST(Unit_TraceDeathTest, FailGivenASuccessIsFatal)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Trace trace = Trace::evaluation(builder);

    SA_EXPECT_FATAL(trace.fail(Status()), "Trace::fail was given a success");
}

/** @brief Tests that Sequential throws the failure a child recorded and traces no later child. */
TEST(Unit_Trace, SequentialThrowsTheFailureAChildRecorded)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueID x = builder.input(Shape{2, 4}, "x");
    Sequential model{RefusingStub{}, ModalStub{}};
    const Trace trace = Trace::evaluation(builder);

    SA_EXPECT_ERROR(static_cast<void>(model.forward(trace, x)), "RefusingStub refuses");
    EXPECT_EQ(model.layer<1>().evaluation_calls, 0);
}

/** @brief Tests that Sequential folds an input shape through output_shape() of every child. */
TEST(Unit_Trace, SequentialFoldsAnInputShapeThroughItsChildren)
{
    const Sequential model{Linear{4, 8}, ReLU{}, Linear{8, 3}};

    EXPECT_EQ(model.output_shape(Shape{4}), (Shape{3}));
    SA_EXPECT_ERROR(static_cast<void>(model.output_shape(Shape{5})), "Linear(4, 8)", "[5]");
}

/** @brief Tests that an update whose source is NO_VALUE stops the process. */
TEST(Unit_TraceDeathTest, AnUpdateWithoutASourceIsFatal)
{
    SA_EXPECT_FATAL(update_from(Graph::K_NO_VALUE), "a buffer update reads value");
}

/** @brief Tests that an update whose source the graph does not hold stops the process. */
TEST(Unit_TraceDeathTest, AnUpdateFromAValueOutsideTheGraphIsFatal)
{
    SA_EXPECT_FATAL(update_from(Graph::ValueID{7}), "reads value 7", "holds 1 values");
}

/** @brief Tests that trace_layer calls the training form alone under a training trace. */
TEST(Unit_Trace, TraceLayerCallsTheTrainingFormInTrainingMode)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueID x = builder.input(Shape{2, 3}, "x");
    ModalStub stub;

    const Graph::ValueID y =
        NN::trace_layer(stub, Trace::training(builder, NN::discard_updates()), x);

    EXPECT_EQ(stub.training_calls, 1);
    EXPECT_EQ(stub.evaluation_calls, 0);
    EXPECT_NE(y, x);
}

/** @brief Tests that trace_layer calls the evaluation form alone under an evaluation trace. */
TEST(Unit_Trace, TraceLayerCallsTheEvaluationFormInEvaluationMode)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueID x = builder.input(Shape{2, 3}, "x");
    ModalStub stub;

    const Graph::ValueID y = NN::trace_layer(stub, Trace::evaluation(builder), x);

    EXPECT_EQ(stub.training_calls, 0);
    EXPECT_EQ(stub.evaluation_calls, 1);
    EXPECT_EQ(y, x);
}

/** @brief Tests that trace_layer traces a Module exactly as its builder forward does. */
TEST(Unit_Trace, TraceLayerTracesAModuleThroughTheBuilder)
{
    Graph::Graph traced_graph;
    const Graph::GraphBuilder traced_builder(traced_graph);
    Linear traced{4, 3};
    const Graph::ValueID traced_out = NN::trace_layer(
        traced, Trace::evaluation(traced_builder), traced_builder.input(Shape{2, 4}, "x"));

    Graph::Graph built_graph;
    const Graph::GraphBuilder built_builder(built_graph);
    Linear built{4, 3};
    const Graph::ValueID built_out =
        built.forward(built_builder, built_builder.input(Shape{2, 4}, "x"));

    EXPECT_EQ(traced_out, built_out);
    EXPECT_EQ(traced_graph.value_count(), built_graph.value_count());
    EXPECT_EQ(traced_graph.value(traced_out).shape, (Shape{2, 3}));
}

/** @brief Tests that trace_layer hands the trace to a composite, which reaches its child. */
TEST(Unit_Trace, TraceLayerPassesTheTraceToATraceModule)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueID x = builder.input(Shape{2, 3}, "x");
    CompositeStub composite;

    static_cast<void>(NN::trace_layer(composite, Trace::evaluation(builder), x));
    static_cast<void>(
        NN::trace_layer(composite, Trace::training(builder, NN::discard_updates()), x));
    static_cast<void>(
        NN::trace_layer(composite, Trace::training(builder, NN::discard_updates()), x));

    EXPECT_EQ(composite.inner.evaluation_calls, 1);
    EXPECT_EQ(composite.inner.training_calls, 2);
}

/** @brief Tests that trace_layer gives a layer with both forwards the trace, not the builder. */
TEST(Unit_Trace, TraceLayerPrefersTheTraceForwardOverTheBuilderForward)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueID x = builder.input(Shape{2, 3}, "x");
    BothForwardsStub stub;

    const Graph::ValueID y = NN::trace_layer(stub, Trace::evaluation(builder), x);

    EXPECT_EQ(stub.trace_calls, 1);
    EXPECT_EQ(stub.builder_calls, 0);
    EXPECT_EQ(y, x);
}

/** @brief Tests that a modal layer nested two Sequentials down is traced in the trace's mode. */
TEST(Unit_Trace, ANestedSequentialTracesItsModalChildInTheTracesMode)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueID x = builder.input(Shape{2, 4}, "x");
    Sequential outer{Sequential{Linear{4, 3}, ModalStub{}}, ReLU{}};

    static_cast<void>(outer.forward(Trace::evaluation(builder), x));
    EXPECT_EQ(outer.layer<0>().layer<1>().evaluation_calls, 1);
    EXPECT_EQ(outer.layer<0>().layer<1>().training_calls, 0);

    static_cast<void>(outer.forward(Trace::training(builder, NN::discard_updates()), x));
    EXPECT_EQ(outer.layer<0>().layer<1>().evaluation_calls, 1);
    EXPECT_EQ(outer.layer<0>().layer<1>().training_calls, 1);
}

/** @brief Tests that Sequential's trace forward gives each child the form of the mode. */
TEST(Unit_Trace, SequentialTracesAModalChildInTheTracesMode)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueID x = builder.input(Shape{2, 4}, "x");
    Sequential model{Linear{4, 3}, ModalStub{}};

    const Graph::ValueID evaluated = model.forward(Trace::evaluation(builder), x);
    EXPECT_EQ(model.layer<1>().evaluation_calls, 1);
    EXPECT_EQ(model.layer<1>().training_calls, 0);
    EXPECT_EQ(graph.value(evaluated).shape, (Shape{2, 3}));

    const Graph::ValueID trained =
        model.forward(Trace::training(builder, NN::discard_updates()), x);
    EXPECT_EQ(model.layer<1>().evaluation_calls, 1);
    EXPECT_EQ(model.layer<1>().training_calls, 1);
    EXPECT_EQ(graph.value(trained).shape, (Shape{2, 3}));
    EXPECT_EQ(parameter_names(model), (std::vector<std::string>{"0.weight", "0.bias"}));
}

/** @brief Tests that a Sequential of Modules traces the same graph through either forward. */
TEST(Unit_Trace, SequentialOfModulesTracesTheSameThroughBothForwards)
{
    Graph::Graph traced_graph;
    const Graph::GraphBuilder traced_builder(traced_graph);
    Sequential traced{Linear{4, 8}, ReLU{}, Linear{8, 3}};
    const Graph::ValueID traced_out = traced.forward(
        Trace::evaluation(traced_builder), traced_builder.input(Shape{2, 4}, "x"));

    Graph::Graph built_graph;
    const Graph::GraphBuilder built_builder(built_graph);
    Sequential built{Linear{4, 8}, ReLU{}, Linear{8, 3}};
    const Graph::ValueID built_out =
        built.forward(built_builder, built_builder.input(Shape{2, 4}, "x"));

    EXPECT_EQ(traced_out, built_out);
    EXPECT_EQ(traced_graph.value_count(), built_graph.value_count());
    EXPECT_EQ(parameter_names(traced), parameter_names(built));
}

/** @brief Tests that an evaluation trace is not a carrying one and refuses both declarations. */
TEST(Unit_Trace, AnEvaluationTraceRefusesAFedLeafAndACarriedValue)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueID x = builder.input(Shape{2, 3}, "x");
    const std::size_t nodes = graph.node_count();

    const Trace trace = Trace::evaluation(builder);

    EXPECT_FALSE(trace.is_carrying());
    SA_EXPECT_FAILURE(trace.feed("position", Shape{1, 1}, DType::Int32),
                      StatusCode::InvalidDescription, "'position'", "carrying trace");
    SA_EXPECT_FAILURE(trace.carried_leaf(Shape{2, 3}, "kept", DType::Float32),
                      StatusCode::InvalidDescription, "'kept'", "carrying trace");
    SA_EXPECT_FAILURE(trace.carry(x, x), StatusCode::InvalidDescription, "carrying trace");
    EXPECT_EQ(graph.node_count(), nodes);
}

/** @brief Tests that a training trace refuses a fed leaf, as its step feeds the model input. */
TEST(Unit_Trace, ATrainingTraceRefusesAFedLeaf)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    BufferUpdateList updates;

    const Trace trace = Trace::training(builder, updates);

    EXPECT_FALSE(trace.is_carrying());
    SA_EXPECT_FAILURE(trace.feed("position", Shape{1, 1}, DType::Int32),
                      StatusCode::InvalidDescription, "'position'");
}

/** @brief Tests that a carrying trace is in Evaluation mode and refuses a buffer update. */
TEST(Unit_Trace, ACarryingTraceIsAnEvaluationTraceThatRefusesABufferUpdate)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueID x = builder.input(Shape{2, 3}, "x");
    Buffer buffer = defined_buffer();
    NN::FedLeafList fed;
    NN::CarriedValueList carried;

    const Trace trace = Trace::carrying(builder, fed, carried);

    EXPECT_TRUE(trace.is_carrying());
    EXPECT_EQ(trace.mode(), TraceMode::Evaluation);
    SA_EXPECT_FAILURE(trace.update(buffer, x, 0.1, 1.0), StatusCode::InvalidDescription,
                      "evaluation trace");
    EXPECT_TRUE(carried.values().empty());
}

/** @brief Tests that a name declares one input leaf however often a layer asks for it. */
TEST(Unit_Trace, AFedLeafIsDeclaredOncePerName)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    NN::FedLeafList fed;
    NN::CarriedValueList carried;
    const Trace trace = Trace::carrying(builder, fed, carried);

    SA_ASSERT_OK_AND_ASSIGN(const Graph::ValueID first,
                            trace.feed("position", Shape{1, 1}, DType::Int32));
    const std::size_t nodes = graph.node_count();
    SA_ASSERT_OK_AND_ASSIGN(const Graph::ValueID again,
                            trace.feed("position", Shape{1, 1}, DType::Int32));
    SA_ASSERT_OK_AND_ASSIGN(const Graph::ValueID mask,
                            trace.feed("mask", Shape{1, 4}, DType::Float32));

    EXPECT_EQ(first, again);
    EXPECT_NE(first, mask);
    EXPECT_EQ(graph.node_count(), nodes + 1u);
    EXPECT_EQ(graph.value(first).lifetime, Graph::Lifetime::Input);
    EXPECT_EQ(graph.value(first).dtype, DType::Int32);
    ASSERT_EQ(fed.leaves().size(), 2u);
    EXPECT_EQ(fed.leaves()[0].name, "position");
    EXPECT_EQ(fed.leaves()[0].value, first);
    EXPECT_EQ(fed.leaves()[1].name, "mask");
    EXPECT_EQ(fed.find("mask"), mask);
    EXPECT_EQ(fed.find("absent"), Graph::K_NO_VALUE);
}

/** @brief Tests that a name declared again with another shape or type is refused. */
TEST(Unit_Trace, AFedLeafDeclaredAgainAsAnotherLeafIsRefused)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    NN::FedLeafList fed;
    NN::CarriedValueList carried;
    const Trace trace = Trace::carrying(builder, fed, carried);
    SA_ASSERT_OK_AND_ASSIGN(const Graph::ValueID first,
                            trace.feed("mask", Shape{1, 4}, DType::Float32));
    static_cast<void>(first);

    SA_EXPECT_FAILURE(trace.feed("mask", Shape{1, 5}, DType::Float32),
                      StatusCode::InvalidDescription, "'mask'", "[1, 4]", "[1, 5]");
    SA_EXPECT_FAILURE(trace.feed("mask", Shape{1, 4}, DType::Float64),
                      StatusCode::InvalidDescription, "'mask'");
    EXPECT_EQ(fed.leaves().size(), 1u);
}

/** @brief Tests that carry records the leaf, the value it takes and its initial value. */
TEST(Unit_Trace, CarryRecordsTheLeafItsNextValueAndItsInitialValue)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    NN::FedLeafList fed;
    NN::CarriedValueList carried;
    const Trace trace = Trace::carrying(builder, fed, carried);

    SA_ASSERT_OK_AND_ASSIGN(const Graph::ValueID leaf,
                            trace.carried_leaf(Shape{4, 3}, "kept", DType::Float32));
    const Graph::ValueID next = builder.scale(leaf, 2.0);
    SA_EXPECT_OK(trace.carry(leaf, next, 0.5));

    EXPECT_EQ(graph.value(leaf).lifetime, Graph::Lifetime::Input);
    EXPECT_TRUE(fed.leaves().empty());
    ASSERT_EQ(carried.values().size(), 1u);
    EXPECT_EQ(carried.values()[0].leaf, leaf);
    EXPECT_EQ(carried.values()[0].next, next);
    EXPECT_NEAR(carried.values()[0].initial_value, 0.5, K_EXACT_TOLERANCE);
    ASSERT_EQ(carried.sources().size(), 1u);
    EXPECT_EQ(carried.sources()[0], next);
}

/** @brief Tests that a next value of another shape than its leaf is refused and not recorded. */
TEST(Unit_Trace, CarryRefusesANextValueOfAnotherShape)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    NN::FedLeafList fed;
    NN::CarriedValueList carried;
    const Trace trace = Trace::carrying(builder, fed, carried);
    SA_ASSERT_OK_AND_ASSIGN(const Graph::ValueID leaf,
                            trace.carried_leaf(Shape{4, 3}, "kept", DType::Float32));
    const Graph::ValueID other = builder.reshape(leaf, Shape{3, 4});

    SA_EXPECT_FAILURE(trace.carry(leaf, other), StatusCode::InvalidDescription, "[4, 3]",
                      "[3, 4]");
    EXPECT_TRUE(carried.values().empty());
}

/** @brief Tests that a leaf carried_leaf() did not declare stops the process. */
TEST(Unit_TraceDeathTest, CarryingAValueThatIsNoCarriedLeafIsFatal)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    NN::FedLeafList fed;
    NN::CarriedValueList carried;
    const Trace trace = Trace::carrying(builder, fed, carried);
    const Graph::ValueID x = builder.input(Shape{4, 3}, "x");
    const Graph::ValueID computed = builder.scale(x, 2.0);

    SA_EXPECT_FATAL(static_cast<void>(trace.carry(computed, computed)), "did not declare");
}
