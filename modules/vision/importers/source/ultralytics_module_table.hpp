/**************************************************************************/
/* ultralytics_module_table.hpp                                           */
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
 * @file ultralytics_module_table.hpp
 * @brief Declares the table of Ultralytics module translators and what each works with.
 * @author Mustafa Garip
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <string_view>
#include <vector>
#include <SushiAI/config/value.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/vision/importers/ultralytics_checkpoint.hpp>
#include <SushiAI/vision/importers/ultralytics_tensors.hpp>

namespace SushiAI
{
    namespace Vision::Importers
    {
        /** @brief Holds the units of the Ultralytics translation that no caller names. */
        namespace Detail
        {
            /** @brief Holds the compound scale a model file states for its size. */
            struct ModuleScale
            {
                std::string name;   /**< The scale's letter, such as "n"; empty for none. */
                double depth = 1.0; /**< The factor of every repeat count. */
                double width = 1.0; /**< The factor of every channel count. */
                std::int64_t max_channels = std::numeric_limits<std::int64_t>::max(); /**< Cap. */
            };

            /** @brief Holds one entry of the backbone or the head, its names resolved. */
            struct ModuleEntry
            {
                std::int64_t index = 0;          /**< Its row index. */
                std::vector<std::int64_t> from;  /**< The rows it reads, as the file states. */
                bool listed_from = false;        /**< Whether the file states a list. */
                std::int64_t repeats = 1;        /**< Its repeat count before the scale. */
                std::string module;              /**< The module name, such as "C3k2". */
                std::vector<Config::Value> args; /**< Its arguments, "nc" and "None" resolved. */
            };

            /** @brief Holds a translation under way: the rows so far and each row's channels. */
            struct ModuleProgress
            {
                ModuleScale scale;                  /**< The scale of the model. */
                std::int64_t classes = 0;           /**< The class count, "nc". */
                std::int64_t reg_max = 16;          /**< The box bins, "reg_max". */
                bool end2end = false;               /**< Whether the head is one-to-one too. */
                bool legacy = true;                 /**< False once a C3k2 row is read. */
                std::int64_t input_channels = 3;    /**< The channels of the model input. */
                std::vector<std::int64_t> channels; /**< The output channels of each row. */
                std::vector<Config::Value> layers;  /**< The rows of the model file. */
                std::vector<std::string> types;     /**< The row types, in order of first use. */
                std::int64_t image_side = 640;      /**< The side of the square input. */
                Config::Value anchors;              /**< The "anchors" of the file, in pixels. */
                std::vector<Config::Value> outputs; /**< What the model publishes. */
                Config::Value task;                 /**< The task section the head wrote. */
                TensorRules tensor_rules;           /**< How the checkpoint's names map. */
                SushiRuntime::span<const UltralyticsModuleClass> modules; /**< Pickled classes. */
            };

            /** @brief Returns the class pickled at @p path, or an empty view when none is. */
            [[nodiscard]] std::string_view find_module_class(const ModuleProgress& progress,
                                                             std::string_view path) noexcept;

            /**
             * @brief Translates one entry, appends its rows and pushes its channels.
             * @return Success after one channel count was pushed and at least one row appended.
             */
            using ModuleTranslator = Status (*)(const ModuleEntry& entry, ModuleProgress& progress);

            /** @brief Holds one row of the table: an Ultralytics module and its translator. */
            struct ModuleTableRow
            {
                std::string_view module;     /**< The module name the file spells. */
                ModuleTranslator translate;  /**< The translator of that module. */
            };

            /** @brief Returns the translator row of @p module, or null when none holds it. */
            [[nodiscard]] const ModuleTableRow* find_module_translator(
                std::string_view module) noexcept;

            /** @brief Translates a Conv entry into a conv row. */
            [[nodiscard]] Status translate_conv(const ModuleEntry& entry, ModuleProgress& progress);

            /** @brief Translates a C3k2 entry into a c3k2 row. */
            [[nodiscard]] Status translate_c3k2(const ModuleEntry& entry, ModuleProgress& progress);

            /** @brief Translates an SPPF entry into an sppf row. */
            [[nodiscard]] Status translate_sppf(const ModuleEntry& entry, ModuleProgress& progress);

            /** @brief Translates a C2PSA entry into a c2psa row. */
            [[nodiscard]] Status translate_c2psa(const ModuleEntry& entry, ModuleProgress& progress);

            /** @brief Translates an nn.Upsample entry into an upsample row. */
            [[nodiscard]] Status translate_upsample(const ModuleEntry& entry,
                                                    ModuleProgress& progress);

            /** @brief Translates a Concat entry into a concat row. */
            [[nodiscard]] Status translate_concat(const ModuleEntry& entry,
                                                  ModuleProgress& progress);

            /** @brief Translates a C3 entry into a c3 row. */
            [[nodiscard]] Status translate_c3(const ModuleEntry& entry, ModuleProgress& progress);

            /** @brief Translates a C2f entry into a c2f row. */
            [[nodiscard]] Status translate_c2f(const ModuleEntry& entry, ModuleProgress& progress);

            /**
             * @brief Translates a Detect entry: an anchor-free detect row, or with "anchors" the
             *        three convolutions and the anchor task of the anchor-based head.
             */
            [[nodiscard]] Status translate_detect(const ModuleEntry& entry,
                                                  ModuleProgress& progress);

            /**
             * @brief Returns the channels of the row @p reference of @p entry names.
             * @return InvalidDescription when it names no earlier row.
             */
            [[nodiscard]] Result<std::int64_t> source_channels(const ModuleEntry& entry,
                                                               const ModuleProgress& progress,
                                                               std::int64_t reference);

            /** @brief Returns the width of @p stated channels under the scale, as Ultralytics. */
            [[nodiscard]] std::int64_t scaled_channels(const ModuleProgress& progress,
                                                       std::int64_t stated) noexcept;

            /** @brief Returns the repeat count of @p entry under the scale, as Ultralytics. */
            [[nodiscard]] std::int64_t scaled_repeats(const ModuleEntry& entry,
                                                      const ModuleProgress& progress) noexcept;

            /**
             * @brief Returns argument @p position of @p entry as a whole number, or @p fallback
             *        when the entry has fewer arguments.
             * @return InvalidDescription naming the entry when it is no whole number.
             */
            [[nodiscard]] Result<std::int64_t> argument_int(const ModuleEntry& entry,
                                                            std::size_t position,
                                                            std::int64_t fallback);

            /** @brief Returns argument @p position as a number, or @p fallback when absent. */
            [[nodiscard]] Result<double> argument_real(const ModuleEntry& entry,
                                                       std::size_t position, double fallback);

            /** @brief Returns argument @p position as a bool, or @p fallback when absent. */
            [[nodiscard]] Result<bool> argument_bool(const ModuleEntry& entry,
                                                     std::size_t position, bool fallback);

            /** @brief Returns Unsupported when @p entry has more than @p most arguments. */
            [[nodiscard]] Status reject_extra_arguments(const ModuleEntry& entry,
                                                        std::size_t most);

            /**
             * @brief Appends a row of @p type holding @p keys and @p values, and the "from" of
             *        @p entry when it reads other than the row above.
             */
            void append_row(ModuleProgress& progress, const ModuleEntry& entry,
                            std::string_view type, std::vector<std::string> keys,
                            std::vector<Config::Value> values);
        } // namespace Detail
    } // namespace Vision::Importers
} // namespace SushiAI
