/**************************************************************************/
/* context.cpp                                                            */
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
 * @file context.cpp
 * @brief Implements the LoweringContext accessors, arity guards and fusion selection.
 * @author Mustafa Garip
 */

#include <SushiAI/graph/lowering/context.hpp>

#include <cstddef>
#include <string_view>
#include <SushiAI/core/check.hpp>
#include <SushiAI/core/ops.hpp>

namespace SushiAI
{
    namespace Graph
    {
        LoweringContext::LoweringContext(SushiBLAS::TaskRecorder& recorder,
                                         SushiBLAS::RngStream& stream, TensorPool& pool,
                                         const Graph& graph, NodeId id,
                                         const FusionSelector& selector,
                                         std::vector<FusionSelection>& selections,
                                         std::size_t& emitted) noexcept
            : recorder_(recorder),
              stream_(stream),
              pool_(pool),
              graph_(graph),
              id_(id),
              selector_(selector),
              selections_(selections),
              emitted_(emitted)
        {
        }

        SushiBLAS::TaskRecorder& LoweringContext::recorder() const noexcept
        {
            return recorder_;
        }

        SushiBLAS::RngStream& LoweringContext::stream() const noexcept
        {
            return stream_;
        }

        const Graph& LoweringContext::graph() const noexcept
        {
            return graph_;
        }

        NodeId LoweringContext::id() const noexcept
        {
            return id_;
        }

        const Node& LoweringContext::node() const
        {
            return graph_.node(id_);
        }

        std::string_view LoweringContext::name() const
        {
            return Ops::name_of(node().op);
        }

        void LoweringContext::require_arity(std::size_t inputs, std::size_t outputs) const
        {
            const Node& current = node();
            SA_FATAL_IF(current.inputs.size() != inputs || current.outputs.size() != outputs,
                        "{} (node {}) takes {} operand(s) and {} result(s); it has {} and {}",
                        name(), id_, inputs, outputs, current.inputs.size(),
                        current.outputs.size());
        }

        void LoweringContext::require_arity_at_least(std::size_t inputs,
                                                     std::size_t outputs) const
        {
            const Node& current = node();
            SA_FATAL_IF(current.inputs.size() < inputs || current.outputs.size() < outputs,
                        "{} (node {}) takes at least {} operand(s) and {} result(s); it has {} "
                        "and {}",
                        name(), id_, inputs, outputs, current.inputs.size(),
                        current.outputs.size());
        }

        void LoweringContext::require_variadic_arity(std::size_t inputs,
                                                     std::size_t outputs) const
        {
            const Node& current = node();
            SA_FATAL_IF(current.inputs.size() < inputs || current.outputs.size() != outputs,
                        "{} (node {}) takes at least {} operand(s) and exactly {} result(s); it "
                        "has {} and {}",
                        name(), id_, inputs, outputs, current.inputs.size(),
                        current.outputs.size());
        }

        void LoweringContext::check_input(std::size_t index) const
        {
            SA_FATAL_IF(index >= node().inputs.size(),
                        "{} expects at least {} operands but was given {}",
                        name(), index + 1, node().inputs.size());
        }

        void LoweringContext::check_output(std::size_t index) const
        {
            SA_FATAL_IF(index >= node().outputs.size(),
                        "{} expects at least {} results but declares {}",
                        name(), index + 1, node().outputs.size());
        }

        const Value& LoweringContext::input_value(std::size_t index) const
        {
            check_input(index);
            return graph_.value(node().inputs[index]);
        }

        const Value& LoweringContext::output_value(std::size_t index) const
        {
            check_output(index);
            return graph_.value(node().outputs[index]);
        }

        const SushiBLAS::Tensor& LoweringContext::input(std::size_t index) const
        {
            check_input(index);
            return pool_.tensor(node().inputs[index]);
        }

        SushiBLAS::Tensor& LoweringContext::mutable_input(std::size_t index) const
        {
            check_input(index);
            return pool_.tensor(node().inputs[index]);
        }

        Result<SushiBLAS::Tensor&> LoweringContext::output(std::size_t index) const
        {
            check_output(index);
            const ValueId target = node().outputs[index];
            return pool_.materialise(target, graph_.value(target));
        }

        bool LoweringContext::output_is_bound(std::size_t index) const
        {
            check_output(index);
            return pool_.is_bound(node().outputs[index]);
        }

        void LoweringContext::alias_output(std::size_t result_index,
                                           std::size_t operand_index) const
        {
            check_output(result_index);
            check_input(operand_index);
            pool_.alias(node().outputs[result_index], node().inputs[operand_index]);
        }

        Result<SushiBLAS::Tensor&> LoweringContext::scratch(int slot, const Shape& shape,
                                                            DType dtype) const
        {
            return pool_.scratch(id_, slot, shape, dtype);
        }

        Result<SushiBLAS::Tensor&> LoweringContext::ones(const Shape& shape, DType dtype) const
        {
            return pool_.ones(shape, dtype);
        }

        void LoweringContext::count(std::size_t tasks) const noexcept
        {
            emitted_ += tasks;
        }

        std::size_t LoweringContext::emitted() const noexcept
        {
            return emitted_;
        }

        FusionChoice LoweringContext::select(const FusionQuery& query) const
        {
            const FusionChoice choice = selector_.choose(query);

            // A GEMM epilogue narrowed to FusedForm::GemmBias is still a spelling of the match.
            const bool narrowed_within_family =
                is_gemm_epilogue(query.matched) && choice.form == FusedForm::GemmBias;

            SA_FATAL_IF(choice.form != FusedForm::None && choice.form != query.matched
                        && !narrowed_within_family,
                        "A fusion selector answered {} to a {} node, which is not a spelling of "
                        "it. A selector may take the whole match, narrow a GEMM epilogue to "
                        "gemm+bias, or take nothing at all.",
                        fused_form_name(choice.form), fused_form_name(query.matched));

            selections_.push_back(FusionSelection{query.matched, choice.form, choice.reason, 0});
            return choice;
        }

        void LoweringContext::close_selection(std::size_t emitted_before) const noexcept
        {
            selections_.back().tasks = emitted_ - emitted_before;
        }
    } // namespace Graph
} // namespace SushiAI
