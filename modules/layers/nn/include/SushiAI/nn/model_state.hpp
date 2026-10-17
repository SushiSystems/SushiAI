/**************************************************************************/
/* model_state.hpp                                                        */
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
 * @file model_state.hpp
 * @brief Declares ModelState, the tensors a model owns across the graphs it is traced into.
 * @author Mustafa Garip
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>
#include <SushiBLAS/engine/engine.hpp>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/dtype/dtype.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/seam/seam.hpp>
#include <SushiAI/shape/shape.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/graph/state_id.hpp>
#include <SushiAI/graph/tensor_pool.hpp>
#include <SushiAI/nn/buffer.hpp>
#include <SushiAI/nn/module.hpp>

namespace SushiAI
{
    namespace NN
    {
        /** @brief Names what an entry of a ModelState is. */
        enum class StateKind : std::uint8_t
        {
            Parameter, /**< A learnable tensor, declared by a Parameter. */
            Buffer,    /**< A running tensor, defined by a Buffer. */
        };

        /** @brief Describes one tensor a ModelState owns. */
        struct StateEntry
        {
            /** @brief The dotted name the model walk gave the tensor. */
            std::string name;

            /** @brief Whether the tensor is a parameter or a buffer. */
            StateKind kind = StateKind::Parameter;

            /** @brief The identity every graph of the model binds the tensor under. */
            Graph::StateID state = Graph::K_NO_STATE;

            /** @brief The extents of the tensor. */
            Shape shape;

            /** @brief The element type of the tensor. */
            DType dtype = DType::Float32;

            /** @brief The fill the tensor starts from; a buffer holds its initial value. */
            ParameterInit init;

            /** @brief The inputs feeding one output unit, as the parameter declared them. */
            int64_t fan_in = 0;

            /** @brief The outputs one input unit feeds, as the parameter declared them. */
            int64_t fan_out = 0;
        };

        namespace Detail
        {
            /** @brief Describes a declared parameter as a state entry named @p name. */
            [[nodiscard]] StateEntry describe_parameter(const std::string& name,
                                                        const Parameter& parameter);

            /** @brief Describes a defined buffer as a state entry named @p name. */
            [[nodiscard]] StateEntry describe_buffer(const std::string& name,
                                                     const Buffer& buffer);

        } // namespace Detail

        /**
         * @brief Owns one tensor per declared parameter and defined buffer of a model.
         *
         * Every tensor is allocated once. attach() binds the storage of a graph's state leaves
         * to a pool, so each pool reads and writes the same numbers as the state.
         */
        class ModelState final
        {
            public:
                /**
                 * @brief Walks @p model once and returns a state with one tensor per entry.
                 * @param engine The engine whose context allocates the tensors.
                 * @param model A model traced at least once; an untraced one gives no entries.
                 * @param prefix Dotted prefix joined to every entry name; empty for none.
                 * @return The state, or the failure from_entries() returns.
                 */
                template<typename M>
                [[nodiscard]] static Result<ModelState> create(SushiBLAS::Engine& engine, M& model,
                                                               std::string_view prefix = {});

                /**
                 * @brief Allocates one tensor per entry, in the order given.
                 * @param engine The engine whose context allocates the tensors.
                 * @param entries The entries; an empty list gives an empty state.
                 * @pre No entry has K_NO_STATE; such an entry stops the process.
                 * @return The state, InvalidDescription naming the entry that repeats a name or
                 *         an identity, or the failure of an allocation.
                 */
                [[nodiscard]] static Result<ModelState> from_entries(
                    SushiBLAS::Engine& engine, std::vector<StateEntry> entries);

                /** @brief Takes the tensors and the indexes of @p other. */
                ModelState(ModelState&& other) noexcept;

                /** @brief Takes the tensors and the indexes of @p other. */
                ModelState& operator=(ModelState&& other) noexcept;

                /** @brief Deleted: a state owns its storage and is not copied. */
                ModelState(const ModelState&) = delete;

                /** @brief Deleted: a state owns its storage and is not copied. */
                ModelState& operator=(const ModelState&) = delete;

                /** @brief Releases the tensors this state owns. */
                ~ModelState();

                /**
                 * @brief Binds the storage of every state leaf of @p graph to @p pool.
                 * @param graph A graph traced from the model this state was created from.
                 * @param pool The pool to bind into; it must hold no leaf of @p graph yet.
                 * @return Incompatible when a leaf has no entry, or differs in shape or dtype.
                 */
                [[nodiscard]] Status attach(const Graph::Graph& graph,
                                            Graph::TensorPool& pool) const;

                /** @brief Returns every entry, in the order the walk or from_entries gave them. */
                [[nodiscard]] SushiRuntime::span<const StateEntry> entries() const noexcept;

                /**
                 * @brief Returns the tensor of the entry named @p name.
                 * @return The tensor this state owns, or InvalidDescription naming @p name.
                 */
                [[nodiscard]] Result<SushiBLAS::Tensor*> tensor(std::string_view name);

                /**
                 * @brief Returns the tensor of the entry with identity @p state.
                 * @return The tensor this state owns, or InvalidDescription for an unknown state.
                 */
                [[nodiscard]] Result<SushiBLAS::Tensor*> tensor(Graph::StateID state);

            private:
                /** @brief Starts empty; only the factories and moves build a state. */
                ModelState() = default;

                /** @brief The entries, in the order they were given. */
                std::vector<StateEntry> entries_;

                /** @brief The tensors, one per entry at the same position. */
                std::vector<SushiBLAS::Tensor> tensors_;

                /** @brief Maps each entry name to its position in entries_. */
                std::unordered_map<std::string, std::size_t> by_name_;

                /** @brief Maps each entry identity to its position in entries_. */
                std::unordered_map<Graph::StateID, std::size_t> by_state_;

                /**
                 * @brief Fills by_name_ and by_state_ from @p entries, refusing a repetition.
                 * @pre No entry has K_NO_STATE; such an entry is fatal.
                 */
                [[nodiscard]] Status index_entries(const std::vector<StateEntry>& entries);

                /**
                 * @brief Writes into @p positions the position of the entry each leaf of @p graph
                 *        binds, in the order of graph.state_leaves().
                 * @return Incompatible when a leaf has no entry, or differs in shape or dtype.
                 */
                [[nodiscard]] Status match_leaves(const Graph::Graph& graph,
                                                  std::vector<std::size_t>& positions) const;
        };

        template<typename M>
        Result<ModelState> ModelState::create(SushiBLAS::Engine& engine, M& model,
                                              std::string_view prefix)
        {
            std::vector<StateEntry> entries;
            auto on_parameter = [&entries](const std::string& name, Parameter& parameter)
            {
                entries.push_back(Detail::describe_parameter(name, parameter));
            };
            auto on_buffer = [&entries](const std::string& name, Buffer& buffer)
            {
                entries.push_back(Detail::describe_buffer(name, buffer));
            };
            const StateVisitor visitor(on_parameter, on_buffer);

            SA_TRY(SushiAI::Detail::guard(SushiAI::Detail::SeamPhase::Allocate, [&]() -> Status
            {
                walk_state(model, prefix, visitor);
                return Status();
            }));

            return from_entries(engine, std::move(entries));
        }

    } // namespace NN
} // namespace SushiAI
