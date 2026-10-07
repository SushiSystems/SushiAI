/**************************************************************************/
/* scope.hpp                                                              */
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
 * @file scope.hpp
 * @brief Declares Scope, the guard that opens a scope on a ScopeTrace and closes it.
 * @author Mustafa Garip
 * @see include/SushiAI/eager/README.md
 */

#pragma once

#include <string_view>

#include <SushiAI/eager/scope_trace.hpp>

namespace SushiAI
{
    namespace Eager
    {
        /**
         * @brief Opens a named scope for its own lifetime.
         * @see include/SushiAI/eager/README.md
         */
        class Scope final
        {
            public:
                /**
                 * @brief Opens @p name on @p trace.
                 * @param trace The trace to mark; must outlive this object.
                 * @param name Scope name; see ScopeTrace::open().
                 */
                Scope(ScopeTrace& trace, std::string_view name) : trace_(&trace)
                {
                    trace_->open(name);
                }

                /** @brief Closes the scope this object opened. */
                ~Scope()
                {
                    trace_->close();
                }

                Scope(const Scope&) = delete;
                Scope& operator=(const Scope&) = delete;
                Scope(Scope&&) = delete;
                Scope& operator=(Scope&&) = delete;

            private:
                /// @brief The trace this scope is open on.
                ScopeTrace* trace_;
        };

    } // namespace Eager
} // namespace SushiAI
