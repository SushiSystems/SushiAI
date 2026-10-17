/**************************************************************************/
/* module_list.hpp                                                        */
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
 * @file module_list.hpp
 * @brief Declares ModuleList, a run-time count of same-typed children named by index.
 * @author Mustafa Garip
 */

#pragma once

#include <cstddef>
#include <type_traits>
#include <utility>
#include <vector>
#include <SushiAI/failure/check.hpp>

namespace SushiAI
{
    namespace NN
    {
        /**
         * @brief Holds a run-time count of same-typed children, named by index.
         * @tparam T The child type; walkable by the state walk and nothrow move constructible.
         */
        template<typename T>
        class ModuleList final
        {
            public:
                static_assert(std::is_nothrow_move_constructible<T>::value,
                              "ModuleList's child must be nothrow move constructible, so "
                              "growing the list keeps each child's state identity.");

                /** @brief Appends @p child; its name is the index it lands on. */
                void push_back(T child)
                {
                    children_.push_back(std::move(child));
                }

                /** @brief Returns how many children the list holds. */
                [[nodiscard]] std::size_t size() const noexcept
                {
                    return children_.size();
                }

                /**
                 * @brief Returns the child at @p index.
                 * @pre index < size().
                 */
                [[nodiscard]] T& operator[](std::size_t index)
                {
                    SA_FATAL_IF(index >= children_.size(),
                                "ModuleList index {} is out of range; the list holds {} children",
                                index, children_.size());
                    return children_[index];
                }

                /**
                 * @brief Returns the child at @p index, read-only.
                 * @pre index < size().
                 */
                [[nodiscard]] const T& operator[](std::size_t index) const
                {
                    SA_FATAL_IF(index >= children_.size(),
                                "ModuleList index {} is out of range; the list holds {} children",
                                index, children_.size());
                    return children_[index];
                }

                /** @brief Returns an iterator to the first child. */
                [[nodiscard]] auto begin() noexcept
                {
                    return children_.begin();
                }

                /** @brief Returns the iterator past the last child. */
                [[nodiscard]] auto end() noexcept
                {
                    return children_.end();
                }

                /** @brief Returns a read-only iterator to the first child. */
                [[nodiscard]] auto begin() const noexcept
                {
                    return children_.begin();
                }

                /** @brief Returns the read-only iterator past the last child. */
                [[nodiscard]] auto end() const noexcept
                {
                    return children_.end();
                }

                /**
                 * @brief Applies @p visitor to each child with its index, in index order.
                 * @param visitor Called as visitor(std::size_t index, T& child).
                 */
                template<typename Fn>
                void visit_children(Fn&& visitor)
                {
                    for (std::size_t index = 0; index < children_.size(); ++index)
                        visitor(index, children_[index]);
                }

            private:
                /** @brief Holds the children in the order they were appended. */
                std::vector<T> children_{};
        };

    } // namespace NN
} // namespace SushiAI
