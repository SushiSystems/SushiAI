/**************************************************************************/
/* ultralytics_model.cpp                                                  */
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
 * @file ultralytics_model.cpp
 * @brief Defines the translation of an Ultralytics model description through the module table.
 * @author Mustafa Garip
 */

#include <SushiAI/vision/importers/ultralytics_model.hpp>

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <string>
#include <string_view>
#include <utility>
#include <vector>
#include <SushiAI/config/model_config.hpp>
#include <SushiAI/config/value.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/failure/status.hpp>
#include "ultralytics_module_table.hpp"

namespace SushiAI
{
    namespace Vision::Importers
    {
        namespace
        {
            /** @brief Holds the normalisation epsilon of every Ultralytics block. */
            constexpr double K_BN_EPSILON = 0.001;

            /** @brief Holds the normalisation momentum of every Ultralytics block. */
            constexpr double K_BN_MOMENTUM = 0.03;

            /** @brief Lists the row types whose blocks hold a normalisation. */
            constexpr std::string_view K_NORMALISED_TYPES[] = {"conv", "c3", "c2f", "c3k2", "sppf",
                                                               "c2psa", "detect"};

            /** @brief Returns the integer @p key of @p root, or @p fallback when absent. */
            Result<std::int64_t> read_int_or(const Config::Value& root, std::string_view key,
                                             std::int64_t fallback, std::string_view origin)
            {
                if (!root.has(key) || root.at_or(key, Config::Value()).is_null())
                    return fallback;
                SA_TRY_ASSIGN(const Config::Value& value, root.at(key, origin));
                return value.as_int(key);
            }

            /** @brief Reads the scale @p root names from its "scales", as Ultralytics does. */
            Result<Detail::ModuleScale> read_scale(const Config::Value& root,
                                                   std::string_view origin)
            {
                Detail::ModuleScale scale;
                if (!root.has("scales"))
                {
                    // Files before the scales table state the two factors at the top level.
                    SA_TRY_ASSIGN(scale.depth,
                                  root.at_or("depth_multiple", Config::Value::real(1.0))
                                      .as_real("depth_multiple"));
                    SA_TRY_ASSIGN(scale.width,
                                  root.at_or("width_multiple", Config::Value::real(1.0))
                                      .as_real("width_multiple"));
                    return scale;
                }
                SA_TRY_ASSIGN(const Config::Value& scales, root.at("scales", origin));
                SA_FAIL_IF(scales.kind() != Config::Kind::Object || scales.keys().empty(),
                           StatusCode::InvalidDescription,
                           "{}: the model states 'scales' that are no table of sizes", origin);
                scale.name = scales.keys().front();
                const Config::Value absent;
                const Config::Value& stated = root.at_or("scale", absent);
                if (stated.kind() == Config::Kind::String)
                {
                    SA_TRY_ASSIGN(const std::string& name, stated.as_string("scale"));
                    scale.name = name;
                }
                SA_TRY_ASSIGN(const Config::Value& factors, scales.at(scale.name, origin));
                SA_TRY_ASSIGN(const std::vector<Config::Value>& values,
                              factors.as_array("scales"));
                SA_FAIL_IF(values.size() != 3, StatusCode::InvalidDescription,
                           "{}: the scale '{}' states {} factors, not depth, width and the "
                           "channel cap",
                           origin, scale.name, values.size());
                SA_TRY_ASSIGN(scale.depth, values[0].as_real("depth"));
                SA_TRY_ASSIGN(scale.width, values[1].as_real("width"));
                SA_TRY_ASSIGN(scale.max_channels, values[2].as_int("max_channels"));
                return scale;
            }

            /** @brief Returns @p value with the names Ultralytics evaluates replaced. */
            Config::Value resolve_argument(const Config::Value& value, std::int64_t classes,
                                           const Config::Value& anchors)
            {
                if (value.kind() != Config::Kind::String)
                    return value;
                const std::string& text = value.as_string("argument").value();
                if (text == "nc")
                    return Config::Value::integer(classes);
                if (text == "anchors")
                    return anchors;
                if (text == "None")
                    return Config::Value();
                if (text == "True" || text == "False")
                    return Config::Value::boolean(text == "True");
                return value;
            }

            /** @brief Reads entry @p index of the backbone and the head into a module entry. */
            Result<Detail::ModuleEntry> read_entry(const Config::Value& value, std::int64_t index,
                                                   const Detail::ModuleProgress& progress,
                                                   std::string_view origin)
            {
                SA_TRY_ASSIGN(const std::vector<Config::Value>& parts, value.as_array(origin));
                SA_FAIL_IF(parts.size() != 4, StatusCode::InvalidDescription,
                           "{}: entry {} holds {} parts, not from, repeats, module and "
                           "arguments",
                           origin, index, parts.size());
                Detail::ModuleEntry entry;
                entry.index = index;
                if (parts[0].kind() == Config::Kind::Array)
                {
                    entry.listed_from = true;
                    for (const Config::Value& source : parts[0].as_array("from").value())
                    {
                        SA_TRY_ASSIGN(const std::int64_t reference, source.as_int("from"));
                        entry.from.push_back(reference);
                    }
                }
                else
                {
                    SA_TRY_ASSIGN(const std::int64_t reference, parts[0].as_int("from"));
                    entry.from.push_back(reference);
                }
                SA_TRY_ASSIGN(entry.repeats, parts[1].as_int("repeats"));
                SA_TRY_ASSIGN(const std::string& module, parts[2].as_string("module"));
                entry.module = module;
                SA_TRY_ASSIGN(const std::vector<Config::Value>& args, parts[3].as_array("args"));
                for (const Config::Value& argument : args)
                    entry.args.push_back(
                        resolve_argument(argument, progress.classes, progress.anchors));
                return entry;
            }

            /** @brief Reads the backbone and the head of @p root, in row order. */
            Result<std::vector<Detail::ModuleEntry>> read_entries(
                const Config::Value& root, const Detail::ModuleProgress& progress,
                std::string_view origin)
            {
                std::vector<Detail::ModuleEntry> entries;
                for (const std::string_view part : {std::string_view("backbone"),
                                                    std::string_view("head")})
                {
                    SA_TRY_ASSIGN(const Config::Value& list, root.at(part, origin));
                    SA_TRY_ASSIGN(const std::vector<Config::Value>& rows, list.as_array(part));
                    for (const Config::Value& row : rows)
                    {
                        SA_TRY_ASSIGN(Detail::ModuleEntry entry,
                                      read_entry(row, static_cast<std::int64_t>(entries.size()),
                                                 progress, origin));
                        entries.push_back(std::move(entry));
                    }
                }
                return entries;
            }

            /** @brief Returns Unsupported listing every module no translator reads, or success. */
            Status reject_unknown_modules(const std::vector<Detail::ModuleEntry>& entries,
                                          std::string_view origin)
            {
                std::string missing;
                for (const Detail::ModuleEntry& entry : entries)
                {
                    if (Detail::find_module_translator(entry.module) != nullptr)
                        continue;
                    missing += missing.empty() ? "" : ", ";
                    missing += entry.module + " (entry " + std::to_string(entry.index) + ")";
                }
                SA_FAIL_IF(!missing.empty(), StatusCode::Unsupported,
                           "{}: no translator reads these modules: {}", origin, missing);
                return Status();
            }

            /** @brief Returns the "defaults" object: the normalisation of each type used. */
            Config::Value write_defaults(const std::vector<std::string>& types)
            {
                std::vector<std::string> keys;
                std::vector<Config::Value> values;
                for (const std::string& type : types)
                {
                    bool normalised = false;
                    for (const std::string_view candidate : K_NORMALISED_TYPES)
                        normalised = normalised || candidate == type;
                    if (!normalised)
                        continue;
                    keys.push_back(type);
                    values.push_back(Config::Value::object(
                        {"bn_epsilon", "bn_momentum"},
                        {Config::Value::real(K_BN_EPSILON), Config::Value::real(K_BN_MOMENTUM)}));
                }
                return Config::Value::object(std::move(keys), std::move(values));
            }

        } // namespace

        Result<UltralyticsTranslation> describe_ultralytics_model(
            const Config::Value& architecture, UltralyticsInputSize image_size,
            SushiRuntime::span<const UltralyticsModuleClass> modules,
            std::string_view origin) noexcept
        {
            SA_FAIL_IF(architecture.kind() != Config::Kind::Object, StatusCode::InvalidDescription,
                       "{}: the model description is not a table", origin);
            SA_FAIL_IF(image_size.height < 1 || image_size.width < 1,
                       StatusCode::InvalidDescription,
                       "{}: the input size {}x{} has a side that is not positive", origin,
                       image_size.height, image_size.width);
            Detail::ModuleProgress progress;
            progress.image_size = image_size;
            progress.modules = modules;
            SA_TRY_ASSIGN(progress.classes, read_int_or(architecture, "nc", 0, origin));
            SA_FAIL_IF(progress.classes < 1, StatusCode::InvalidDescription,
                       "{}: the model states no class count 'nc'", origin);
            SA_TRY_ASSIGN(progress.reg_max, read_int_or(architecture, "reg_max", 16, origin));
            const Config::Value absent;
            const Config::Value& end2end = architecture.at_or("end2end", absent);
            progress.end2end =
                end2end.kind() == Config::Kind::Bool && end2end.as_bool("end2end").value();
            SA_TRY_ASSIGN(const std::int64_t ch, read_int_or(architecture, "ch", 3, origin));
            SA_TRY_ASSIGN(progress.input_channels,
                          read_int_or(architecture, "channels", ch, origin));
            progress.anchors = architecture.at_or("anchors", Config::Value());
            SA_TRY_ASSIGN(progress.scale, read_scale(architecture, origin));
            SA_TRY_ASSIGN(const std::vector<Detail::ModuleEntry> entries,
                          read_entries(architecture, progress, origin));
            SA_TRY(reject_unknown_modules(entries, origin));

            for (const Detail::ModuleEntry& entry : entries)
            {
                SA_FAIL_IF(!progress.outputs.empty(), StatusCode::Unsupported,
                           "{}: entry {} ({}) follows the Detect head, and the head must be last",
                           origin, entry.index, entry.module);
                const Status translated =
                    Detail::find_module_translator(entry.module)->translate(entry, progress);
                SA_TRY(SA_ANNOTATE(translated, "{}", origin));
            }
            SA_FAIL_IF(progress.outputs.empty(), StatusCode::Unsupported,
                       "{}: the model has no Detect entry, and only detectors are translated",
                       origin);

            const Config::Value input = Config::Value::object(
                {"shape"}, {Config::Value::array({Config::Value::integer(image_size.height),
                                                  Config::Value::integer(image_size.width),
                                                  Config::Value::integer(progress.input_channels)})});
            UltralyticsTranslation out;
            out.document = Config::Value::object(
                {"format_version", "input", "defaults", "layers", "outputs", "task"},
                {Config::Value::integer(Config::K_MODEL_VERSION), input,
                 write_defaults(progress.types), Config::Value::array(std::move(progress.layers)),
                 Config::Value::array(std::move(progress.outputs)), std::move(progress.task)});
            out.tensor_rules = std::move(progress.tensor_rules);
            return out;
        }

        Result<Config::ModelSpec> translate_ultralytics_model(
            const Config::Value& architecture, UltralyticsInputSize image_size,
            SushiRuntime::span<const UltralyticsModuleClass> modules,
            std::string_view origin) noexcept
        {
            SA_TRY_ASSIGN(const UltralyticsTranslation translation,
                          describe_ultralytics_model(architecture, image_size, modules, origin));
            return Config::parse_model(translation.document, origin);
        }
    } // namespace Vision::Importers
} // namespace SushiAI
