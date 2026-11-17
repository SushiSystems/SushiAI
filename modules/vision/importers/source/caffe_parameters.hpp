/**************************************************************************/
/* caffe_parameters.hpp                                                   */
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
 * @file caffe_parameters.hpp
 * @brief Declares the reader of one parameter message of a Caffe layer, such as convolution_param.
 * @author Mustafa Garip
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/interchange/caffe_net.hpp>
#include <SushiAI/io/protobuf_text.hpp>

namespace SushiAI
{
    namespace Vision::Importers
    {
        /** @brief Holds the units of the Caffe translation that no caller names. */
        namespace Detail::Caffe
        {
            /** @brief Holds one setting of a 2-D window along its two axes. */
            struct WindowPair
            {
                std::int64_t height = 0; /**< The value along the height axis. */
                std::int64_t width = 0;  /**< The value along the width axis. */
            };

            /**
             * @brief Reads the fields of one parameter message of a layer.
             *
             * A layer that holds no such message reads as a message with no fields, so every
             * read returns its fallback. Each failure names the layer, and the line of the field.
             */
            class ParameterReader final
            {
                public:
                    /**
                     * @brief Binds the reader to the message @p message of @p layer.
                     * @param layer Must outlive the reader.
                     * @param message The field name of the message, such as "pooling_param".
                     */
                    ParameterReader(const Interchange::CaffeLayer& layer,
                                    std::string_view message) noexcept;

                    /** @brief Returns whether the message states the field @p key. */
                    [[nodiscard]] bool has(std::string_view key) const noexcept;

                    /**
                     * @brief Returns the whole number @p key states, or @p fallback when absent.
                     * @return InvalidDescription when the value is no whole number.
                     */
                    [[nodiscard]] Result<std::int64_t> read_integer(std::string_view key,
                                                                    std::int64_t fallback) const;

                    /**
                     * @brief Returns the number @p key states, or @p fallback when absent.
                     * @return InvalidDescription when the value is no finite number.
                     */
                    [[nodiscard]] Result<double> read_real(std::string_view key,
                                                           double fallback) const;

                    /**
                     * @brief Returns the bool @p key states, or @p fallback when absent.
                     * @return InvalidDescription when the value is neither true nor false.
                     */
                    [[nodiscard]] Result<bool> read_bool(std::string_view key,
                                                         bool fallback) const;

                    /**
                     * @brief Returns the bare word or string @p key states, or @p fallback.
                     * @return InvalidDescription when the value is a number or a message.
                     */
                    [[nodiscard]] Result<std::string> read_word(std::string_view key,
                                                                std::string_view fallback) const;

                    /**
                     * @brief Returns every whole number the repeated field @p key states.
                     * @return InvalidDescription when one is no whole number.
                     */
                    [[nodiscard]] Result<std::vector<std::int64_t>> read_integers(
                        std::string_view key) const;

                    /**
                     * @brief Returns every number the repeated field @p key states.
                     * @return InvalidDescription when one is no finite number.
                     */
                    [[nodiscard]] Result<std::vector<double>> read_reals(
                        std::string_view key) const;

                    /**
                     * @brief Returns every whole number of the repeated field @p key inside the
                     *        message @p inner, which the message states.
                     * @return InvalidDescription when one is no whole number.
                     */
                    [[nodiscard]] Result<std::vector<std::int64_t>> read_nested_integers(
                        std::string_view inner, std::string_view key) const;

                    /**
                     * @brief Returns a window setting as Caffe spells it, over @p fallback.
                     *
                     * The setting is the repeated @p list_key, one entry for both axes or two
                     * for the height and the width, or the pair @p height_key and @p width_key.
                     * @return InvalidDescription for both spellings at once, for one of the pair
                     *         without the other, or for more than two entries.
                     */
                    [[nodiscard]] Result<WindowPair> read_window(std::string_view list_key,
                                                                 std::string_view height_key,
                                                                 std::string_view width_key,
                                                                 WindowPair fallback) const;

                private:
                    /** @brief Returns InvalidDescription unless @p field is a number. */
                    [[nodiscard]] Status expect_number(const IO::TextField& field) const;

                    /** @brief Parses the finite number @p field holds. */
                    [[nodiscard]] Result<double> parse_real(const IO::TextField& field) const;

                    /** @brief Parses the whole number @p field holds. */
                    [[nodiscard]] Result<std::int64_t> parse_integer(
                        const IO::TextField& field) const;

                    /** @brief Holds the layer the message belongs to. */
                    const Interchange::CaffeLayer* layer_;

                    /** @brief Holds the message, or null when the layer states none. */
                    const IO::TextField* message_;
            };
        } // namespace Detail::Caffe
    } // namespace Vision::Importers
} // namespace SushiAI
