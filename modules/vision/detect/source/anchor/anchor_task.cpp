/**************************************************************************/
/* anchor_task.cpp                                                        */
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
 * @file anchor_task.cpp
 * @brief Implements the reader and the writer of an anchor task's section of a model file.
 * @author Mustafa Garip
 * @see modules/vision/detect/README.md
 */

#include <SushiAI/vision/detect/anchor/anchor_task.hpp>

#include <array>
#include <cmath>
#include <cstddef>
#include <iterator>
#include <utility>

#include <SushiAI/config/section_reader.hpp>
#include <SushiAI/core/status.hpp>

namespace SushiAI
{
    namespace Vision::Detect::Anchor
    {
        namespace
        {
            using Config::Kind;
            using Config::Value;

            /** @brief Names the section in a diagnostic; every refusal starts with it. */
            constexpr std::string_view K_TASK = "task";

            /** @brief Lists every key a task section of this type may carry. */
            constexpr std::string_view K_TASK_KEYS[] = {"type", "classes", "anchors", "heads",
                                                      "suppression"};

            /** @brief Lists every key an entry of "heads" may carry. */
            constexpr std::string_view K_HEAD_KEYS[] = {"output", "anchors", "coding",
                                                      "scale_xy"};

            /** @brief Lists every key "suppression" may carry. */
            constexpr std::string_view K_SUPPRESSION_KEYS[] = {"criterion", "distance_exponent"};

            /** @brief Pairs one head coding with the string a task section states for it. */
            struct CodingName
            {
                HeadCoding value;      /**< The enumerator. */
                std::string_view name; /**< Its string under "coding". */
            };

            /** @brief Pairs one suppression criterion with its string and what it reads. */
            struct CriterionName
            {
                SuppressionCriterion value; /**< The enumerator. */
                std::string_view name;      /**< Its string under "criterion". */
                bool reads_exponent;        /**< Whether the measure reads distance_exponent. */
            };

            /** @brief Lists every head coding; the reader and the writer both look it up. */
            constexpr CodingName K_CODING_NAMES[] = {{HeadCoding::Logit, "logit"},
                                                     {HeadCoding::Activated, "activated"}};

            /** @brief Lists every criterion; the reader and the writer both look it up. */
            constexpr CriterionName K_CRITERION_NAMES[] = {
                {SuppressionCriterion::IoU, "iou", false},
                {SuppressionCriterion::DistanceIoU, "distance_iou", true}};

            /** @brief Returns the row of @p table for the enumerator @p value, or null. */
            template<typename Row, std::size_t N, typename Enumerator>
            const Row* find_by_value(const Row (&table)[N], Enumerator value) noexcept
            {
                for (const Row& row : table)
                    if (row.value == value)
                        return &row;
                return nullptr;
            }

            /** @brief Returns the row of @p table whose string is @p name, or null. */
            template<typename Row, std::size_t N>
            const Row* find_by_name(const Row (&table)[N], std::string_view name) noexcept
            {
                for (const Row& row : table)
                    if (row.name == name)
                        return &row;
                return nullptr;
            }

            using Config::entry_of;
            using Config::KeySpan;
            using Config::member_of;
            using Config::read_float32;
            using Config::SectionReader;

            /** @brief Returns the strings of @p table, quoted and separated by commas. */
            template<typename Row, std::size_t N>
            std::string list_names(const Row (&table)[N])
            {
                std::array<std::string_view, N> names{};
                for (std::size_t index = 0; index < N; ++index)
                    names[index] = table[index].name;
                return Config::list_quoted(Config::KeySpan(names.data(), names.size()));
            }

            /** @brief Reports whether @p value is a finite number above zero. */
            bool is_positive(float value) noexcept
            {
                return std::isfinite(value) && value > 0.0f;
            }

            /** @brief Reads "anchors" of @p task into @p anchors, one [width, height] each. */
            Status read_anchors(const Value& task, std::vector<AnchorSize>& anchors)
            {
                const std::string where = member_of(K_TASK, "anchors");
                SA_TRY_ASSIGN(const Value* const list,
                              SectionReader(task, K_TASK).find_of("anchors", Kind::Array));

                const std::vector<Value>& entries = list->as_array(where);
                anchors.reserve(entries.size());
                for (std::size_t i = 0; i < entries.size(); ++i)
                {
                    const std::string entry = entry_of(where, i);
                    SA_FAIL_IF(entries[i].kind() != Kind::Array, StatusCode::InvalidDescription,
                               "{} must be [width, height], but it is {}", entry,
                               Config::kind_name(entries[i].kind()));

                    const std::vector<Value>& sides = entries[i].as_array(entry);
                    SA_FAIL_IF(sides.size() != 2, StatusCode::InvalidDescription,
                               "{} must be [width, height], but it lists {} values", entry,
                               sides.size());

                    AnchorSize anchor;
                    SA_TRY_ASSIGN(anchor.width, read_float32(sides[0], entry + " width"));
                    SA_TRY_ASSIGN(anchor.height, read_float32(sides[1], entry + " height"));
                    anchors.push_back(anchor);
                }
                return Status();
            }

            /** @brief Reads the "anchors" of a head, named @p where, into @p mask. */
            Status read_mask(const Value& head, std::string_view where,
                             std::vector<std::int64_t>& mask)
            {
                const std::string mask_where = member_of(where, "anchors");
                SA_TRY_ASSIGN(const Value* const list,
                              SectionReader(head, where).find_of("anchors", Kind::Array));

                const std::vector<Value>& indices = list->as_array(mask_where);
                mask.reserve(indices.size());
                for (std::size_t slot = 0; slot < indices.size(); ++slot)
                {
                    SA_FAIL_IF(indices[slot].kind() != Kind::Int, StatusCode::InvalidDescription,
                               "{} must be a whole number, but it is {}",
                               entry_of(mask_where, slot),
                               Config::kind_name(indices[slot].kind()));
                    mask.push_back(indices[slot].as_int(mask_where));
                }
                return Status();
            }

            /**
             * @brief Reads one entry of "heads", named @p where, into @p head and @p mask.
             * @param num_classes The "classes" of the task, which the head's options take.
             */
            Status read_head(const Value& entry, std::string_view where,
                             std::int64_t num_classes, AnchorTaskHead& head,
                             std::vector<std::int64_t>& mask)
            {
                const SectionReader reader(entry, where);
                SA_TRY(reader.check_object());
                SA_TRY(reader.check_keys(KeySpan(K_HEAD_KEYS, std::size(K_HEAD_KEYS))));

                SA_TRY_ASSIGN(const Value* const output, reader.find_of("output", Kind::String));
                head.output = output->as_string(where);

                SA_TRY(read_mask(entry, where, mask));

                SA_TRY_ASSIGN(const Value* const coding, reader.find_of("coding", Kind::String));
                const std::string& stated = coding->as_string(where);
                const CodingName* const row = find_by_name(K_CODING_NAMES, stated);
                SA_FAIL_IF(row == nullptr, StatusCode::InvalidDescription,
                           "{} is '{}'; it is one of {}", member_of(where, "coding"), stated,
                           list_names(K_CODING_NAMES));

                SA_TRY_ASSIGN(const Value* const scale, reader.find("scale_xy"));
                SA_TRY_ASSIGN(head.options.scale_xy,
                              read_float32(*scale, reader.member("scale_xy")));

                head.options.coding = row->value;
                head.options.num_classes = num_classes;
                return Status();
            }

            /** @brief Reads "heads" of @p task into the heads and the masks of @p result. */
            Status read_heads(const Value& task, AnchorTask& result)
            {
                const std::string where = member_of(K_TASK, "heads");
                SA_TRY_ASSIGN(const Value* const list,
                              SectionReader(task, K_TASK).find_of("heads", Kind::Array));

                const std::vector<Value>& entries = list->as_array(where);
                result.heads.resize(entries.size());
                result.layout.head_anchors.resize(entries.size());
                for (std::size_t k = 0; k < entries.size(); ++k)
                    SA_TRY(read_head(entries[k], entry_of(where, k), result.num_classes,
                                     result.heads[k], result.layout.head_anchors[k]));
                return Status();
            }

            /** @brief Reads "suppression" of @p task into the criterion and its exponent. */
            Status read_suppression(const Value& task, AnchorTask& result)
            {
                const std::string where = member_of(K_TASK, "suppression");
                SA_TRY_ASSIGN(const Value* const suppression,
                              SectionReader(task, K_TASK).find_of("suppression", Kind::Object));

                const SectionReader reader(*suppression, where);
                SA_TRY(reader.check_keys(
                    KeySpan(K_SUPPRESSION_KEYS, std::size(K_SUPPRESSION_KEYS))));

                SA_TRY_ASSIGN(const Value* const criterion,
                              reader.find_of("criterion", Kind::String));
                const std::string& stated = criterion->as_string(where);
                const CriterionName* const row = find_by_name(K_CRITERION_NAMES, stated);
                SA_FAIL_IF(row == nullptr, StatusCode::InvalidDescription,
                           "{} is '{}'; it is one of {}", member_of(where, "criterion"), stated,
                           list_names(K_CRITERION_NAMES));
                result.criterion = row->value;

                if (!row->reads_exponent)
                {
                    SA_FAIL_IF(suppression->has("distance_exponent"),
                               StatusCode::InvalidDescription,
                               "{} is stated, but the criterion '{}' does not read it",
                               member_of(where, "distance_exponent"), stated);
                    return Status();
                }

                SA_TRY_ASSIGN(const Value* const exponent, reader.find("distance_exponent"));
                SA_TRY_ASSIGN(result.distance_exponent,
                              read_float32(*exponent, reader.member("distance_exponent")));
                return Status();
            }

            /** @brief Checks head @p index of @p task against the task's classes and anchors. */
            Status validate_task_head(const AnchorTask& task, std::size_t index)
            {
                const std::string where = entry_of(member_of(K_TASK, "heads"), index);
                const AnchorTaskHead& head = task.heads[index];
                const std::vector<std::int64_t>& mask = task.layout.head_anchors[index];
                const std::int64_t anchors = static_cast<std::int64_t>(task.layout.anchors.size());

                SA_FAIL_IF(head.output.empty(), StatusCode::InvalidDescription,
                           "{} is an empty string; it names the model output the head reads",
                           member_of(where, "output"));
                SA_FAIL_IF(mask.empty(), StatusCode::InvalidDescription,
                           "{} lists no anchor", member_of(where, "anchors"));

                for (std::size_t slot = 0; slot < mask.size(); ++slot)
                {
                    SA_FAIL_IF(mask[slot] < 0 || mask[slot] >= anchors,
                               StatusCode::InvalidDescription,
                               "{} names anchor {}; the task has {} anchors",
                               entry_of(member_of(where, "anchors"), slot), mask[slot], anchors);
                }

                SA_FAIL_IF(head.options.num_classes != task.num_classes,
                           StatusCode::InvalidDescription,
                           "{} holds {} classes, but 'classes' of the task is {}", where,
                           head.options.num_classes, task.num_classes);
                SA_FAIL_IF(!is_positive(head.options.scale_xy), StatusCode::InvalidDescription,
                           "{} must be finite and above zero; got {}",
                           member_of(where, "scale_xy"), head.options.scale_xy);
                return Status();
            }

            /** @brief Checks the classes, the anchors and every head of @p task. */
            Status validate_anchor_task(const AnchorTask& task)
            {
                SA_FAIL_IF(task.num_classes < 1, StatusCode::InvalidDescription,
                           "{} must be at least 1; got {}", member_of(K_TASK, "classes"),
                           task.num_classes);

                const std::string anchors_where = member_of(K_TASK, "anchors");
                SA_FAIL_IF(task.layout.anchors.empty(), StatusCode::InvalidDescription,
                           "{} lists no anchor", anchors_where);
                for (std::size_t i = 0; i < task.layout.anchors.size(); ++i)
                {
                    const AnchorSize& anchor = task.layout.anchors[i];
                    SA_FAIL_IF(!is_positive(anchor.width) || !is_positive(anchor.height),
                               StatusCode::InvalidDescription,
                               "{} must have finite sides above zero; got {} by {}",
                               entry_of(anchors_where, i), anchor.width, anchor.height);
                }

                const std::string heads_where = member_of(K_TASK, "heads");
                SA_FAIL_IF(task.heads.empty(), StatusCode::InvalidDescription,
                           "{} lists no head", heads_where);
                SA_FAIL_IF(task.heads.size() != task.layout.head_anchors.size(),
                           StatusCode::InvalidDescription,
                           "{} holds {} heads, but the layout of the task holds {} masks",
                           heads_where, task.heads.size(), task.layout.head_anchors.size());

                for (std::size_t k = 0; k < task.heads.size(); ++k)
                    SA_TRY(validate_task_head(task, k));

                // validate_head reads the whole layout, so every head is named by its key first.
                for (std::size_t k = 0; k < task.heads.size(); ++k)
                    SA_TRY(validate_head(task.heads[k].options, task.layout,
                                         static_cast<std::int64_t>(k)));
                return Status();
            }

            /** @brief Builds "anchors": each anchor as [width, height] in double. */
            Value anchors_value(const std::vector<AnchorSize>& anchors)
            {
                std::vector<Value> entries;
                entries.reserve(anchors.size());
                for (const AnchorSize& anchor : anchors)
                {
                    entries.push_back(
                        Value::array({Value::real(static_cast<double>(anchor.width)),
                                      Value::real(static_cast<double>(anchor.height))}));
                }
                return Value::array(std::move(entries));
            }

            /** @brief Builds entry @p index of "heads" from @p head and its @p mask. */
            Result<Value> head_value(const AnchorTaskHead& head,
                                     const std::vector<std::int64_t>& mask, std::size_t index)
            {
                const CodingName* const coding = find_by_value(K_CODING_NAMES, head.options.coding);
                SA_FAIL_IF(coding == nullptr, StatusCode::InvalidDescription,
                           "{} holds the value {}, which is no head coding; the codings are {}",
                           member_of(entry_of(member_of(K_TASK, "heads"), index), "coding"),
                           static_cast<std::int64_t>(head.options.coding),
                           list_names(K_CODING_NAMES));

                std::vector<Value> indices;
                indices.reserve(mask.size());
                for (std::int64_t anchor : mask)
                    indices.push_back(Value::integer(anchor));

                return Value::object(
                    {"output", "anchors", "coding", "scale_xy"},
                    {Value::string(head.output), Value::array(std::move(indices)),
                     Value::string(std::string(coding->name)),
                     Value::real(static_cast<double>(head.options.scale_xy))});
            }

            /** @brief Builds "heads" from the heads and the masks of @p task. */
            Result<Value> heads_value(const AnchorTask& task)
            {
                std::vector<Value> entries;
                entries.reserve(task.heads.size());
                for (std::size_t k = 0; k < task.heads.size(); ++k)
                {
                    SA_TRY_ASSIGN(Value entry,
                                  head_value(task.heads[k], task.layout.head_anchors[k], k));
                    entries.push_back(std::move(entry));
                }
                return Value::array(std::move(entries));
            }

            /** @brief Builds "suppression": the criterion, and the exponent when it is read. */
            Result<Value> suppression_value(const AnchorTask& task)
            {
                const std::string where = member_of(K_TASK, "suppression");
                const CriterionName* const criterion =
                    find_by_value(K_CRITERION_NAMES, task.criterion);
                SA_FAIL_IF(criterion == nullptr, StatusCode::InvalidDescription,
                           "{} holds the value {}, which is no criterion; the criteria are {}",
                           member_of(where, "criterion"),
                           static_cast<std::int64_t>(task.criterion),
                           list_names(K_CRITERION_NAMES));

                if (!criterion->reads_exponent)
                {
                    return Value::object({"criterion"},
                                         {Value::string(std::string(criterion->name))});
                }

                SA_FAIL_IF(!std::isfinite(task.distance_exponent), StatusCode::InvalidDescription,
                           "{} must be finite; got {}", member_of(where, "distance_exponent"),
                           task.distance_exponent);
                return Value::object(
                    {"criterion", "distance_exponent"},
                    {Value::string(std::string(criterion->name)),
                     Value::real(static_cast<double>(task.distance_exponent))});
            }
        } // namespace

        Result<AnchorTask> read_anchor_task(const Config::Value& task)
        {
            const Config::SectionReader reader(task, K_TASK);
            SA_TRY(reader.check_object());

            SA_TRY_ASSIGN(const Value* const type,
                          reader.find_of("type", Kind::String));
            SA_FAIL_IF(type->as_string(K_TASK) != K_ANCHOR_TASK_TYPE,
                       StatusCode::InvalidDescription,
                       "{} is '{}'; an anchor task states '{}'", member_of(K_TASK, "type"),
                       type->as_string(K_TASK), K_ANCHOR_TASK_TYPE);
            SA_TRY(reader.check_keys(Config::KeySpan(K_TASK_KEYS, std::size(K_TASK_KEYS))));

            AnchorTask result;
            SA_TRY_ASSIGN(const Value* const classes,
                          reader.find_of("classes", Kind::Int));
            result.num_classes = classes->as_int(K_TASK);

            SA_TRY(read_anchors(task, result.layout.anchors));
            SA_TRY(read_heads(task, result));
            SA_TRY(read_suppression(task, result));
            SA_TRY(validate_anchor_task(result));
            return Result<AnchorTask>(std::move(result));
        }

        Result<Config::Value> write_anchor_task(const AnchorTask& task)
        {
            SA_TRY(validate_anchor_task(task));

            SA_TRY_ASSIGN(Value heads, heads_value(task));
            SA_TRY_ASSIGN(Value suppression, suppression_value(task));

            return Value::object(
                {"type", "classes", "anchors", "heads", "suppression"},
                {Value::string(std::string(K_ANCHOR_TASK_TYPE)),
                 Value::integer(task.num_classes), anchors_value(task.layout.anchors),
                 std::move(heads), std::move(suppression)});
        }

    } // namespace Vision::Detect::Anchor
} // namespace SushiAI
