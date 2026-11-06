/**************************************************************************/
/* lazy_shared.hpp                                                        */
/**************************************************************************/
/*                          This file is part of:                         */
/*                                 SushiAI                                */
/*                 https://github.com/SushiSystems/SushiAI                */
/*                         https://sushisystems.io                        */
/**************************************************************************/
/* Copyright (c) 2026-present Mustafa Garip & Sushi Systems               */
/*                                                                        */
/* Licensed under the Apache License, Version 2.0 (the "License"); you    */
/* may not use this file except in compliance with the License. You may   */
/* obtain a copy of the License at                                        */
/*                                                                        */
/*     https://www.apache.org/licenses/LICENSE-2.0                        */
/*                                                                        */
/* Unless required by applicable law or agreed to in writing, software    */
/* distributed under the License is distributed on an "AS IS" BASIS,      */
/* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or        */
/* implied. See the License for the specific language governing           */
/* permissions and limitations under the License.                         */
/**************************************************************************/

/**
 * @file lazy_shared.hpp
 * @brief Declares LazyShared, the holder that creates one value on first use through a factory.
 * @author Mustafa Garip
 */

#pragma once

#include <functional>
#include <optional>
#include <utility>
#include <SushiAI/failure/check.hpp>
#include <SushiAI/failure/result.hpp>

namespace SushiAI
{
    namespace Test
    {
        /**
         * @brief Holds one value that a factory creates the first time it is asked for.
         * @tparam T The held type; moved out of the factory's Result.
         */
        template <typename T>
        class LazyShared final
        {
            public:
                /** @brief Names the call that creates the value or says why it cannot. */
                using Factory = std::function<Result<T>()>;

                /** @brief Keeps @p factory, which is not called until acquire(). */
                explicit LazyShared(Factory factory) : factory_(std::move(factory))
                {

                }

                /**
                 * @brief Replaces the factory.
                 * @pre acquire() has not created the value yet.
                 */
                void use_factory(Factory factory)
                {
                    SA_FATAL_IF(value_.has_value(),
                                "The factory cannot change once the value exists");
                    factory_ = std::move(factory);
                }

                /**
                 * @brief Returns the value, creating it on the first call.
                 * @return The same value on every call, or the failure of the factory, in which
                 *         case the next call asks the factory again.
                 */
                [[nodiscard]] Result<T&> acquire()
                {
                    if (!value_)
                    {
                        Result<T> created = factory_();
                        if (!created.has_value())
                            return created.status();
                        value_.emplace(std::move(created).value());
                    }
                    return *value_;
                }

                /** @brief Destroys the value, if one exists; the factory stays. */
                void release() noexcept
                {
                    value_.reset();
                }

            private:
                /// @brief Creates the value.
                Factory factory_;

                /// @brief Holds the value from the first acquire() to release().
                std::optional<T> value_;
        };

    } // namespace Test
} // namespace SushiAI
