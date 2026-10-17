/**************************************************************************/
/* step.cpp                                                               */
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
 * @file step.cpp
 * @brief Implements Step, the trace of one step and the points at which it runs.
 * @author Mustafa Garip
 * @see modules/core/eager/README.md
 */

#include <SushiAI/eager/step.hpp>

#include <utility>
#include <vector>

#include <SushiRuntime/core/span.hpp>

#include <SushiAI/autograd/backward.hpp>
#include <SushiAI/dtype/dtype.hpp>
#include <SushiAI/ops/ops.hpp>
#include <SushiAI/seam/seam.hpp>
#include <SushiAI/shape/shape.hpp>

#include "tensor_shape.hpp"

namespace SushiAI
{
    namespace Eager
    {
        Step::Step(Session& session)
            : session_(&session), builder_(trace_.graph), scopes_(trace_.graph)
        {

        }

        Step::~Step()
        {
            session_->close_step();
        }

        Result<Graph::ValueID> Step::state(Graph::StateID state)
        {
            const Session::BoundState* const bound = session_->find_state(state);
            SA_FAIL_IF(bound == nullptr, StatusCode::InvalidDescription,
                       "State {} is not bound to the session; call Session::bind_state() first",
                       static_cast<std::uint64_t>(state));

            const Graph::ValueID traced = trace_.graph.state_leaf(state);
            if (traced != Graph::K_NO_VALUE)
                return traced;

            return ::SushiAI::Detail::guard_value(::SushiAI::Detail::SeamPhase::Record, [&]()
            {
                const Graph::ValueID leaf =
                    bound->gradient != K_NO_STORAGE
                        ? builder_.parameter(bound->shape, "state", bound->dtype)
                        : builder_.buffer(bound->shape, "state", bound->dtype);
                trace_.graph.bind_state_leaf(state, leaf);
                return leaf;
            });
        }

        Result<Graph::ValueID> Step::input(const SushiBLAS::Tensor& batch, std::string_view name)
        {
            SA_FAIL_IF(!batch.storage || batch.num_elements == 0, StatusCode::InvalidDescription,
                       "The tensor fed as '{}' holds no element", name);

            SA_TRY_ASSIGN(const Graph::ValueID leaf,
                          ::SushiAI::Detail::guard_value(::SushiAI::Detail::SeamPhase::Record,
                                                         [&]()
            {
                return builder_.input(shape_of(batch), name, batch.dtype);
            }));

            SA_TRY(feed(leaf, batch));
            return leaf;
        }

        Status Step::feed(Graph::ValueID leaf, const SushiBLAS::Tensor& source)
        {
            SA_FAIL_IF(stage_ == Stage::Ran, StatusCode::Unsupported,
                       "The step has run; a leaf is fed before the step runs");
            SA_FAIL_IF(leaf >= trace_.graph.value_count(), StatusCode::InvalidDescription,
                       "Value {} does not exist in this step, which holds {} values", leaf,
                       trace_.graph.value_count());

            const Graph::NodeID producer = trace_.graph.value(leaf).producer;
            SA_FAIL_IF(producer != Graph::K_NO_NODE
                           && trace_.graph.node(producer).op != Ops::K_LEAF.id,
                       StatusCode::InvalidDescription,
                       "Value {} is the result of {} and cannot be fed; only a leaf can",
                       leaf, Ops::name_of(trace_.graph.node(producer).op));
            SA_FAIL_IF(!source.storage, StatusCode::InvalidDescription,
                       "The tensor fed into value {} holds no element", leaf);

            return ::SushiAI::Detail::guard(::SushiAI::Detail::SeamPhase::Allocate, [&]()
            {
                trace_.feeds.push_back(Feed{leaf, source});
                return Status();
            });
        }

        Status Step::keep(Graph::ValueID value)
        {
            SA_FAIL_IF(stage_ == Stage::Ran, StatusCode::Unsupported,
                       "The step has run; a value is kept before the step runs");
            SA_FAIL_IF(value >= trace_.graph.value_count(), StatusCode::InvalidDescription,
                       "Value {} does not exist in this step, which holds {} values", value,
                       trace_.graph.value_count());

            return ::SushiAI::Detail::guard(::SushiAI::Detail::SeamPhase::Allocate, [&]()
            {
                trace_.observed.push_back(value);
                return Status();
            });
        }

        Status Step::backward(Graph::ValueID loss)
        {
            SA_FAIL_IF(stage_ != Stage::Tracing, StatusCode::InvalidDescription,
                       "backward() is called once, before the step runs");
            SA_TRY(keep(loss));

            Graph::Graph& graph = trace_.graph;
            std::vector<Graph::ValueID> wrt;
            SA_TRY(::SushiAI::Detail::guard(::SushiAI::Detail::SeamPhase::Record, [&]()
            {
                for (const std::pair<Graph::StateID, Graph::ValueID>& leaf :
                     graph.state_leaves())
                {
                    const Session::BoundState* const bound = session_->find_state(leaf.first);
                    if (bound == nullptr || bound->gradient == K_NO_STORAGE)
                        continue;

                    wrt.push_back(leaf.second);
                    trace_.gradients.push_back(StateGradient{leaf.first, Graph::K_NO_VALUE});
                }

                // The segments cover the forward nodes only, so they are cut before any other.
                trace_.segments = scopes_.seal();

                const std::vector<Graph::ValueID> gradients = Autograd::differentiate(
                    graph, loss, SushiRuntime::span<const Graph::ValueID>(wrt.data(), wrt.size()),
                    Graph::K_NO_VALUE, &trace_.provenance);
                for (std::size_t i = 0; i < gradients.size(); ++i)
                    trace_.gradients[i].value = gradients[i];

                return Status();
            }));

            stage_ = Stage::Differentiated;
            closed_at_ = graph.node_count();
            return Status();
        }

        Status Step::apply(Graph::StepRecorder& update)
        {
            SA_FAIL_IF(stage_ == Stage::Ran, StatusCode::Unsupported,
                       "The step has run; an update is applied by the run, not after it");

            trace_.trailing = &update;
            return flush();
        }

        Status Step::require_no_late_node() const
        {
            SA_FAIL_IF(trace_.graph.node_count() != closed_at_, StatusCode::Unsupported,
                       "{} nodes were traced after the step was differentiated or ran. A step "
                       "runs once; tracing on after a read needs the phases of the eager "
                       "design, which are not built",
                       trace_.graph.node_count() - closed_at_);

            return Status();
        }

        Status Step::flush()
        {
            if (stage_ == Stage::Ran)
                return Status();

            if (stage_ == Stage::Tracing)
            {
                SA_TRY(::SushiAI::Detail::guard(::SushiAI::Detail::SeamPhase::Record, [&]()
                {
                    trace_.segments = scopes_.seal();
                    trace_.provenance.forward_node_count = trace_.graph.node_count();
                    trace_.provenance.origin.clear();
                    return Status();
                }));
                closed_at_ = trace_.graph.node_count();
            }

            SA_TRY(require_no_late_node());
            SA_TRY(session_->run(trace_));
            stage_ = Stage::Ran;
            return Status();
        }

        Result<const SushiBLAS::Tensor*> Step::read(Graph::ValueID value)
        {
            if (stage_ != Stage::Ran)
            {
                SA_TRY(keep(value));
                SA_TRY(flush());
            }

            SA_TRY(require_no_late_node());
            return session_->tensor_of(value);
        }

        Result<double> Step::read_scalar(Graph::ValueID value)
        {
            SA_TRY_ASSIGN(const SushiBLAS::Tensor* const tensor, read(value));
            SA_FAIL_IF(tensor->num_elements != 1, StatusCode::InvalidDescription,
                       "Value {} holds {} elements; read_scalar() reads a value of one", value,
                       tensor->num_elements);

            SA_FAIL_IF(tensor->dtype != DType::Float32 && tensor->dtype != DType::Float64,
                       StatusCode::InvalidDescription,
                       "Value {} is {}; read_scalar() reads float32 and float64", value,
                       dtype_name(tensor->dtype));

            return tensor->dtype == DType::Float32
                       ? static_cast<double>(tensor->data_as<float>()[0])
                       : tensor->data_as<double>()[0];
        }

        Status Step::finish()
        {
            return flush();
        }

    } // namespace Eager
} // namespace SushiAI
