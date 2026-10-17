/**************************************************************************/
/* coco_json.cpp                                                          */
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
 * @file coco_json.cpp
 * @brief Implements the reader of a dataset kept as one annotation JSON and an image folder.
 * @author Mustafa Garip
 */

#include <SushiAI/vision/datasets/coco_json.hpp>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <string>
#include <string_view>
#include <system_error>
#include <unordered_map>
#include <utility>
#include <vector>


#include <SushiAI/config/json_events.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/vision/imaging/boxes.hpp>
#include <SushiAI/vision/imaging/polygon.hpp>

#include "disk.hpp"

namespace SushiAI::Vision::Datasets
{
    namespace
    {
        /** @brief Names this reader in a message. */
        constexpr std::string_view K_READER = "read_coco_json";

        /** @brief Names the folder of root that holds the annotation files. */
        constexpr std::string_view K_ANNOTATION_FOLDER = "annotations";

        /** @brief Holds what an annotation file's name opens with, before the split. */
        constexpr std::string_view K_ANNOTATION_PREFIX = "instances_";

        /** @brief Holds the extension of an annotation file. */
        constexpr std::string_view K_JSON_EXTENSION = ".json";

        /** @brief Counts the numbers of a box: x, y, width, height. */
        constexpr std::size_t K_BOX_NUMBERS = 4;

        /** @brief Counts the numbers of the shortest outline: three pairs. */
        constexpr std::size_t K_MIN_POLYGON_NUMBERS = 6;

        /** @brief Counts the lists of a file the reader keeps. */
        constexpr std::size_t K_LIST_COUNT = 3;

        /** @brief Names a member of the root of a file. */
        enum class ListKind : std::uint8_t
        {
            Images,      /**< The list "images". */
            Categories,  /**< The list "categories". */
            Annotations, /**< The list "annotations". */
            Other,       /**< Any other member, which the reader passes over. */
        };

        /** @brief Holds the key of each kept list, at the value of its ListKind. */
        constexpr std::array<std::string_view, K_LIST_COUNT> K_LIST_KEYS = {
            "images", "categories", "annotations"};

        /** @brief Names a member of an entry of a list. */
        enum class Member : std::uint8_t
        {
            Other,        /**< A member the reader passes over. */
            Id,           /**< "id" of an image or of a category. */
            FileName,     /**< "file_name" of an image. */
            Width,        /**< "width" of an image. */
            Height,       /**< "height" of an image. */
            Name,         /**< "name" of a category. */
            ImageID,      /**< "image_id" of an annotation. */
            CategoryID,   /**< "category_id" of an annotation. */
            Crowd,        /**< "iscrowd" of an annotation. */
            Box,          /**< "bbox" of an annotation. */
            Segmentation, /**< "segmentation" of an annotation. */
        };

        /** @brief Pairs the key of a member with its name. */
        struct MemberKey
        {
            std::string_view key; /**< Holds the key as the file spells it. */
            Member member;        /**< Names the member. */
        };

        /** @brief Lists the members kept of an entry of "images". */
        constexpr MemberKey K_IMAGE_MEMBERS[] = {{"id", Member::Id},
                                                 {"file_name", Member::FileName},
                                                 {"width", Member::Width},
                                                 {"height", Member::Height}};

        /** @brief Lists the members kept of an entry of "categories". */
        constexpr MemberKey K_CATEGORY_MEMBERS[] = {{"id", Member::Id}, {"name", Member::Name}};

        /** @brief Lists the members kept of an entry of "annotations". */
        constexpr MemberKey K_ANNOTATION_MEMBERS[] = {{"image_id", Member::ImageID},
                                                      {"category_id", Member::CategoryID},
                                                      {"iscrowd", Member::Crowd},
                                                      {"bbox", Member::Box},
                                                      {"segmentation", Member::Segmentation}};

        /** @brief Returns the member @p key names in an entry of @p list. */
        [[nodiscard]] Member member_of(ListKind list, std::string_view key) noexcept
        {
            const MemberKey* first = nullptr;
            std::size_t count = 0;

            if (list == ListKind::Images)
            {
                first = K_IMAGE_MEMBERS;
                count = std::size(K_IMAGE_MEMBERS);
            }
            else if (list == ListKind::Categories)
            {
                first = K_CATEGORY_MEMBERS;
                count = std::size(K_CATEGORY_MEMBERS);
            }
            else if (list == ListKind::Annotations)
            {
                first = K_ANNOTATION_MEMBERS;
                count = std::size(K_ANNOTATION_MEMBERS);
            }

            for (std::size_t row = 0; row < count; ++row)
            {
                if (first[row].key == key)
                    return first[row].member;
            }

            return Member::Other;
        }

        /** @brief Names what a scalar of the file is. */
        enum class ScalarKind : std::uint8_t
        {
            Whole,  /**< A whole number. */
            Real,   /**< A number with a fraction or an exponent. */
            String, /**< A string. */
            Null,   /**< A null. */
            Other,  /**< A boolean, or an object or a list where a scalar is read. */
        };

        /** @brief Holds one scalar of the file. */
        struct Scalar
        {
            ScalarKind kind = ScalarKind::Other; /**< Says which field below is set. */
            std::int64_t whole = 0;              /**< Holds a whole number. */
            double real = 0.0;                   /**< Holds a number of either kind. */
            std::string_view text;               /**< Holds a string, for the event only. */
        };

        /** @brief Reports whether a scalar is a number of either kind. */
        [[nodiscard]] bool is_number(const Scalar& scalar) noexcept
        {
            return scalar.kind == ScalarKind::Whole || scalar.kind == ScalarKind::Real;
        }

        /** @brief Holds what the reader keeps of one entry of "images". */
        struct RawImage
        {
            std::int64_t id = 0;     /**< Holds "id". */
            std::int64_t width = 0;  /**< Holds "width". */
            std::int64_t height = 0; /**< Holds "height". */
            std::string name;        /**< Holds "file_name". */
            bool has_id = false;     /**< Says "id" is a whole number. */
            bool has_name = false;   /**< Says "file_name" is a string. */
            bool has_width = false;  /**< Says "width" is a whole number. */
            bool has_height = false; /**< Says "height" is a whole number. */
        };

        /** @brief Holds what the reader keeps of one entry of "categories". */
        struct RawCategory
        {
            std::int64_t id = 0;   /**< Holds "id". */
            std::string name;      /**< Holds "name". */
            bool has_id = false;   /**< Says "id" is a whole number. */
            bool has_name = false; /**< Says "name" is a string. */
        };

        /** @brief Names what "segmentation" of an annotation holds. */
        enum class Outline : std::uint8_t
        {
            Absent,    /**< No member, or a null. */
            RunLength, /**< An object: a run-length mask. */
            NotList,   /**< A scalar that is no null. */
            Parts,     /**< A list whose every part is at least three pairs of numbers. */
            BadPart,   /**< A list with a part that is not. */
        };

        /** @brief Holds what the reader keeps of one entry of "annotations". */
        struct RawAnnotation
        {
            std::int64_t image_id = 0;               /**< Holds "image_id". */
            std::int64_t category_id = 0;            /**< Holds "category_id". */
            std::array<double, K_BOX_NUMBERS> box{}; /**< Holds "bbox". */
            Imaging::Polygon outline;                /**< Holds the outline, once resolved. */
            bool has_image_id = false;               /**< Says "image_id" is a whole number. */
            bool has_category_id = false;            /**< Says "category_id" is one. */
            bool is_crowd = false;                   /**< Says "iscrowd" is a nonzero whole. */
            bool is_box = false;                     /**< Says "bbox" is four numbers. */
            Outline segmentation = Outline::Absent;  /**< Says what "segmentation" holds. */
        };

        /** @brief Holds the width and the height of one image, in pixels. */
        struct Extent
        {
            double width = 0.0;  /**< Holds the width. */
            double height = 0.0; /**< Holds the height. */
        };

        /** @brief Holds the parts of an annotation met before the entry of its image. */
        struct PendingOutline
        {
            std::size_t annotation = 0;         /**< Holds the position of the annotation. */
            std::vector<double> numbers;        /**< Holds the numbers of every part, joined. */
            std::vector<std::size_t> part_ends; /**< Holds where each part ends in numbers. */
        };

        /** @brief Holds what one pass over a file keeps of it. */
        struct RawFile
        {
            std::vector<RawImage> images;              /**< Holds "images", in file order. */
            std::vector<RawCategory> categories;       /**< Holds "categories", likewise. */
            std::vector<RawAnnotation> annotations;    /**< Holds "annotations", likewise. */
            std::vector<PendingOutline> pending;       /**< Holds outlines not yet resolved. */
            std::array<bool, K_LIST_COUNT> has_list{}; /**< Says each key holds a list. */
        };

        /** @brief Reports whether an annotation is left out before its outline is read. */
        [[nodiscard]] bool is_left_out(const RawAnnotation& annotation) noexcept
        {
            return annotation.is_crowd || annotation.box[2] <= 0.0 || annotation.box[3] <= 0.0;
        }

        /** @brief Returns twice the area an outline encloses, by the sum over its edges. */
        [[nodiscard]] double doubled_area(const Imaging::Polygon& outline) noexcept
        {
            double sum = 0.0;

            for (std::size_t vertex = 0; vertex < outline.size(); ++vertex)
            {
                const Imaging::Point& a = outline[vertex];
                const Imaging::Point& b = outline[(vertex + 1) % outline.size()];
                sum += static_cast<double>(a.x) * b.y - static_cast<double>(b.x) * a.y;
            }

            return sum < 0.0 ? -sum : sum;
        }

        /**
         * @brief Returns the part of the largest area, in fractions of the image.
         * @param numbers Holds the pixel coordinates of every part, joined.
         * @param part_ends Holds where each part ends in @p numbers; the first part of the
         *                  largest area wins.
         */
        [[nodiscard]] Imaging::Polygon largest_outline(const std::vector<double>& numbers,
                                                       const std::vector<std::size_t>& part_ends,
                                                       const Extent& extent)
        {
            Imaging::Polygon largest;
            double largest_area = -1.0;
            std::size_t begin = 0;

            for (const std::size_t end : part_ends)
            {
                Imaging::Polygon outline;
                outline.reserve((end - begin) / 2);
                for (std::size_t number = begin; number < end; number += 2)
                {
                    outline.push_back(
                        Imaging::Point{static_cast<float>(numbers[number] / extent.width),
                                       static_cast<float>(numbers[number + 1] / extent.height)});
                }

                const double area = doubled_area(outline);
                if (area > largest_area)
                {
                    largest_area = area;
                    largest = std::move(outline);
                }

                begin = end;
            }

            return largest;
        }

        /**
         * @class COCOEvents
         * @brief Keeps, of the events of one file, the members the reader reads.
         * @pre The RawFile outlives the handler.
         */
        class COCOEvents final : public Config::JSONEvents
        {
            public:
                /** @brief Binds the record the events fill and says whether outlines are kept. */
                COCOEvents(RawFile& file, COCOSegmentation segmentation) noexcept
                    : file_(&file), keeps_outlines_(segmentation == COCOSegmentation::Polygons)
                {

                }

                bool begin_object() override
                {
                    return open(Container::Object);
                }

                bool end_object() override
                {
                    return close();
                }

                bool begin_array() override
                {
                    return open(Container::Array);
                }

                bool end_array() override
                {
                    return close();
                }

                bool key(std::string_view text) override
                {
                    if (skipped_ > 0)
                        return true;

                    if (state_ == State::Entry)
                        member_ = member_of(list_, text);

                    return state_ != State::Root || name_list(text);
                }

                bool null() override
                {
                    Scalar scalar;
                    scalar.kind = ScalarKind::Null;
                    return take(scalar);
                }

                bool boolean(bool) override
                {
                    return take(Scalar{});
                }

                bool whole(std::int64_t value) override
                {
                    Scalar scalar;
                    scalar.kind = ScalarKind::Whole;
                    scalar.whole = value;
                    scalar.real = static_cast<double>(value);
                    return take(scalar);
                }

                bool real(double value) override
                {
                    Scalar scalar;
                    scalar.kind = ScalarKind::Real;
                    scalar.real = value;
                    return take(scalar);
                }

                bool string(std::string_view text) override
                {
                    Scalar scalar;
                    scalar.kind = ScalarKind::String;
                    scalar.text = text;
                    return take(scalar);
                }

                /** @brief Returns the key of a kept list the root states twice, or nothing. */
                [[nodiscard]] std::string_view repeated_list() const noexcept
                {
                    return repeated_;
                }

            private:
                /** @brief Names where in the document the next event falls. */
                enum class State : std::uint8_t
                {
                    Document, /**< Outside the root. */
                    Root,     /**< In the root object. */
                    List,     /**< In one of the three lists. */
                    Entry,    /**< In an entry of a list that is an object. */
                    Box,      /**< In "bbox" of an annotation. */
                    Parts,    /**< In "segmentation" of an annotation. */
                    Part,     /**< In one part of "segmentation". */
                };

                /** @brief Names what an opening event opens. */
                enum class Container : std::uint8_t
                {
                    Object, /**< An object. */
                    Array,  /**< A list. */
                };

                /** @brief Passes over what opens, with everything in it. */
                void skip() noexcept
                {
                    skipped_ = 1;
                }

                /** @brief Sets the list the next root value belongs to; false when restated. */
                [[nodiscard]] bool name_list(std::string_view key) noexcept
                {
                    list_ = ListKind::Other;
                    for (std::size_t kind = 0; kind < K_LIST_COUNT; ++kind)
                    {
                        if (K_LIST_KEYS[kind] != key)
                            continue;

                        if (is_stated_[kind])
                        {
                            repeated_ = K_LIST_KEYS[kind];
                            return false;
                        }

                        is_stated_[kind] = true;
                        list_ = static_cast<ListKind>(kind);
                    }

                    return true;
                }

                /** @brief Appends an entry with no member to the list being read. */
                void add_entry()
                {
                    if (list_ == ListKind::Images)
                        file_->images.emplace_back();
                    else if (list_ == ListKind::Categories)
                        file_->categories.emplace_back();
                    else
                        file_->annotations.emplace_back();
                }

                /** @brief Handles the opening of an object or of a list. */
                [[nodiscard]] bool open(Container container)
                {
                    if (skipped_ > 0)
                    {
                        ++skipped_;
                        return true;
                    }

                    switch (state_)
                    {
                        case State::Document:
                            if (container == Container::Object)
                                state_ = State::Root;
                            else
                                skip();
                            break;
                        case State::Root:
                            open_list(container);
                            break;
                        case State::List:
                            add_entry();
                            member_ = Member::Other;
                            if (container == Container::Object)
                                state_ = State::Entry;
                            else
                                skip();
                            break;
                        case State::Entry:
                            open_member(container);
                            break;
                        case State::Box:
                            is_box_ = false;
                            skip();
                            break;
                        case State::Parts:
                            open_part(container);
                            break;
                        case State::Part:
                            is_part_ = false;
                            skip();
                            break;
                    }

                    return true;
                }

                /** @brief Enters a kept list, and passes over any other value of the root. */
                void open_list(Container container) noexcept
                {
                    if (list_ == ListKind::Other || container != Container::Array)
                    {
                        skip();
                        return;
                    }

                    file_->has_list[static_cast<std::size_t>(list_)] = true;
                    state_ = State::List;
                }

                /** @brief Enters "bbox" or "segmentation", and passes over any other member. */
                void open_member(Container container)
                {
                    const bool in_annotation = list_ == ListKind::Annotations;

                    if (in_annotation && member_ == Member::Box && container == Container::Array)
                    {
                        state_ = State::Box;
                        box_count_ = 0;
                        is_box_ = true;
                        return;
                    }

                    if (in_annotation && member_ == Member::Segmentation && keeps_outlines_ &&
                        container == Container::Array)
                    {
                        state_ = State::Parts;
                        numbers_.clear();
                        part_ends_.clear();
                        file_->annotations.back().segmentation = Outline::Parts;
                        return;
                    }

                    set_member(Scalar{});
                    if (in_annotation && member_ == Member::Segmentation && keeps_outlines_)
                        file_->annotations.back().segmentation = Outline::RunLength;

                    skip();
                }

                /** @brief Enters a part that is a list; any other part spoils the member. */
                void open_part(Container container) noexcept
                {
                    if (container != Container::Array)
                    {
                        file_->annotations.back().segmentation = Outline::BadPart;
                        skip();
                        return;
                    }

                    state_ = State::Part;
                    part_begin_ = numbers_.size();
                    is_part_ = true;
                }

                /** @brief Handles the closing of an object or of a list. */
                [[nodiscard]] bool close()
                {
                    if (skipped_ > 0)
                    {
                        --skipped_;
                        return true;
                    }

                    switch (state_)
                    {
                        case State::Document:
                            break;
                        case State::Root:
                            state_ = State::Document;
                            break;
                        case State::List:
                            state_ = State::Root;
                            break;
                        case State::Entry:
                            close_entry();
                            state_ = State::List;
                            break;
                        case State::Box:
                            file_->annotations.back().is_box =
                                is_box_ && box_count_ == K_BOX_NUMBERS;
                            state_ = State::Entry;
                            break;
                        case State::Parts:
                            state_ = State::Entry;
                            break;
                        case State::Part:
                            close_part();
                            state_ = State::Parts;
                            break;
                    }

                    return true;
                }

                /** @brief Keeps a part of at least three pairs of numbers; spoils the member. */
                void close_part()
                {
                    const std::size_t count = numbers_.size() - part_begin_;
                    if (is_part_ && count >= K_MIN_POLYGON_NUMBERS && count % 2 == 0)
                    {
                        part_ends_.push_back(numbers_.size());
                        return;
                    }

                    file_->annotations.back().segmentation = Outline::BadPart;
                }

                /** @brief Finishes an entry: the size of an image, the outline of an object. */
                void close_entry()
                {
                    if (!keeps_outlines_)
                        return;

                    if (list_ == ListKind::Images)
                    {
                        const RawImage& image = file_->images.back();
                        if (image.has_id && image.has_width && image.has_height &&
                            image.width >= 1 && image.height >= 1)
                        {
                            extents_.emplace(image.id,
                                             Extent{static_cast<double>(image.width),
                                                    static_cast<double>(image.height)});
                        }
                    }
                    else if (list_ == ListKind::Annotations)
                    {
                        keep_outline();
                    }
                }

                /** @brief Resolves the outline of the last annotation, or keeps its parts. */
                void keep_outline()
                {
                    RawAnnotation& annotation = file_->annotations.back();
                    const bool is_kept = annotation.segmentation == Outline::Parts &&
                                         annotation.has_image_id && annotation.is_box &&
                                         !is_left_out(annotation);
                    if (!is_kept)
                        return;

                    const auto extent = extents_.find(annotation.image_id);
                    if (extent != extents_.end())
                    {
                        annotation.outline = largest_outline(numbers_, part_ends_,
                                                             extent->second);
                        return;
                    }

                    file_->pending.push_back(
                        PendingOutline{file_->annotations.size() - 1, numbers_, part_ends_});
                }

                /** @brief Handles a scalar, wherever it falls. */
                [[nodiscard]] bool take(const Scalar& scalar)
                {
                    if (skipped_ > 0)
                        return true;

                    switch (state_)
                    {
                        case State::Document:
                        case State::Root:
                            break;
                        case State::List:
                            add_entry();
                            break;
                        case State::Entry:
                            set_member(scalar);
                            break;
                        case State::Box:
                            take_box_number(scalar);
                            break;
                        case State::Parts:
                            file_->annotations.back().segmentation = Outline::BadPart;
                            break;
                        case State::Part:
                            is_part_ = is_part_ && is_number(scalar);
                            numbers_.push_back(scalar.real);
                            break;
                    }

                    return true;
                }

                /** @brief Counts one entry of "bbox" and keeps it when it is a number. */
                void take_box_number(const Scalar& scalar) noexcept
                {
                    is_box_ = is_box_ && is_number(scalar);
                    if (box_count_ < K_BOX_NUMBERS)
                        file_->annotations.back().box[box_count_] = scalar.real;

                    ++box_count_;
                }

                /** @brief Sets the member named last from a scalar; the last value wins. */
                void set_member(const Scalar& scalar)
                {
                    if (list_ == ListKind::Images)
                        set_image_member(file_->images.back(), scalar);
                    else if (list_ == ListKind::Categories)
                        set_category_member(file_->categories.back(), scalar);
                    else
                        set_annotation_member(file_->annotations.back(), scalar);
                }

                /** @brief Sets a member of an image. */
                void set_image_member(RawImage& image, const Scalar& scalar) const
                {
                    const bool is_whole = scalar.kind == ScalarKind::Whole;

                    if (member_ == Member::Id)
                    {
                        image.has_id = is_whole;
                        image.id = scalar.whole;
                    }
                    else if (member_ == Member::Width)
                    {
                        image.has_width = is_whole;
                        image.width = scalar.whole;
                    }
                    else if (member_ == Member::Height)
                    {
                        image.has_height = is_whole;
                        image.height = scalar.whole;
                    }
                    else if (member_ == Member::FileName)
                    {
                        image.has_name = scalar.kind == ScalarKind::String;
                        image.name.assign(scalar.text.data(), scalar.text.size());
                    }
                }

                /** @brief Sets a member of a category. */
                void set_category_member(RawCategory& category, const Scalar& scalar) const
                {
                    if (member_ == Member::Id)
                    {
                        category.has_id = scalar.kind == ScalarKind::Whole;
                        category.id = scalar.whole;
                    }
                    else if (member_ == Member::Name)
                    {
                        category.has_name = scalar.kind == ScalarKind::String;
                        category.name.assign(scalar.text.data(), scalar.text.size());
                    }
                }

                /** @brief Sets a member of an annotation that a scalar can hold. */
                void set_annotation_member(RawAnnotation& annotation, const Scalar& scalar) const
                {
                    const bool is_whole = scalar.kind == ScalarKind::Whole;

                    if (member_ == Member::ImageID)
                    {
                        annotation.has_image_id = is_whole;
                        annotation.image_id = scalar.whole;
                    }
                    else if (member_ == Member::CategoryID)
                    {
                        annotation.has_category_id = is_whole;
                        annotation.category_id = scalar.whole;
                    }
                    else if (member_ == Member::Crowd)
                    {
                        annotation.is_crowd = is_whole && scalar.whole != 0;
                    }
                    else if (member_ == Member::Box)
                    {
                        annotation.is_box = false;
                    }
                    else if (member_ == Member::Segmentation && keeps_outlines_)
                    {
                        annotation.segmentation = scalar.kind == ScalarKind::Null
                                                      ? Outline::Absent
                                                      : Outline::NotList;
                    }
                }

                /// @brief The record the events fill.
                RawFile* file_ = nullptr;

                /// @brief Whether "segmentation" is read.
                bool keeps_outlines_ = false;

                /// @brief Where the next event falls.
                State state_ = State::Document;

                /// @brief How many containers being passed over are open.
                std::size_t skipped_ = 0;

                /// @brief The list the root key named last.
                ListKind list_ = ListKind::Other;

                /// @brief The member the entry key named last.
                Member member_ = Member::Other;

                /// @brief Whether the root has stated the key of each kept list.
                std::array<bool, K_LIST_COUNT> is_stated_{};

                /// @brief The key of a kept list the root states twice, or nothing.
                std::string_view repeated_;

                /// @brief How many entries of the "bbox" being read have passed.
                std::size_t box_count_ = 0;

                /// @brief Whether every entry of the "bbox" being read is a number.
                bool is_box_ = false;

                /// @brief The numbers of the parts of the "segmentation" being read.
                std::vector<double> numbers_;

                /// @brief Where each kept part ends in numbers_.
                std::vector<std::size_t> part_ends_;

                /// @brief Where the part being read begins in numbers_.
                std::size_t part_begin_ = 0;

                /// @brief Whether every entry of the part being read is a number.
                bool is_part_ = false;

                /// @brief The size of each image met so far, by its id; the first id wins.
                std::unordered_map<std::int64_t, Extent> extents_;
        };

        /** @brief Names one entry of one of the three lists of a file, for a message. */
        struct Entry
        {
            std::string_view list;   /**< Says "image", "category" or "annotation". */
            std::size_t index;       /**< Holds the entry's position in its list. */
            const std::string& file; /**< Names the file. */
        };

        /** @brief Pairs an id with the position, in its list, of the entry that has it. */
        struct IDEntry
        {
            std::int64_t id;      /**< Holds the id the file states. */
            std::size_t position; /**< Holds the position of the entry in its list. */
        };

        /** @brief Returns InvalidFile, with the index, unless a member is a whole number. */
        [[nodiscard]] Status check_whole(bool is_whole, const Entry& entry, std::string_view key)
        {
            SA_FAIL_IF(!is_whole, StatusCode::InvalidFile,
                       "'{}': {}: {} {} has no whole number '{}'", entry.file, K_READER,
                       entry.list, entry.index, key);
            return Status();
        }

        /** @brief Returns InvalidFile, with the index, unless a member is a string. */
        [[nodiscard]] Status check_text(bool is_text, const Entry& entry, std::string_view key)
        {
            SA_FAIL_IF(!is_text, StatusCode::InvalidFile, "'{}': {}: {} {} has no string '{}'",
                       entry.file, K_READER, entry.list, entry.index, key);
            return Status();
        }

        /**
         * @brief Returns the ids of a list, sorted by id.
         * @tparam Record Has the fields id and has_id.
         * @return InvalidFile, with both indices, for an id two entries share.
         */
        template <typename Record>
        [[nodiscard]] Result<std::vector<IDEntry>> index_ids(const std::vector<Record>& entries,
                                                             std::string_view list,
                                                             const std::string& file)
        {
            std::vector<IDEntry> ids;
            ids.reserve(entries.size());

            for (std::size_t index = 0; index < entries.size(); ++index)
            {
                SA_TRY(check_whole(entries[index].has_id, Entry{list, index, file}, "id"));
                ids.push_back(IDEntry{entries[index].id, index});
            }

            std::sort(ids.begin(), ids.end(), [](const IDEntry& a, const IDEntry& b)
            {
                return a.id != b.id ? a.id < b.id : a.position < b.position;
            });

            for (std::size_t entry = 1; entry < ids.size(); ++entry)
            {
                SA_FAIL_IF(ids[entry].id == ids[entry - 1].id, StatusCode::InvalidFile,
                           "'{}': {}: {} {} and {} {} share the id {}", file, K_READER, list,
                           ids[entry - 1].position, list, ids[entry].position, ids[entry].id);
            }

            return Result<std::vector<IDEntry>>(std::move(ids));
        }

        /**
         * @brief Returns the rank, among sorted ids, of the id a member of an entry holds.
         * @param named Says which list the ids are of, as the file spells it.
         * @return InvalidFile, with the index of the entry, for an id the list lacks.
         */
        [[nodiscard]] Result<std::size_t> rank_of(bool is_whole, std::int64_t id,
                                                  const Entry& entry, std::string_view key,
                                                  const std::vector<IDEntry>& ids,
                                                  std::string_view named)
        {
            SA_TRY(check_whole(is_whole, entry, key));

            const auto found = std::lower_bound(ids.begin(), ids.end(), id,
                                                [](const IDEntry& candidate, std::int64_t wanted)
            {
                return candidate.id < wanted;
            });

            SA_FAIL_IF(found == ids.end() || found->id != id, StatusCode::InvalidFile,
                       "'{}': {}: {} {} has the '{}' {}, which '{}' lacks", entry.file, K_READER,
                       entry.list, entry.index, key, id, named);
            return static_cast<std::size_t>(found - ids.begin());
        }

        /** @brief Returns InvalidFile for a "segmentation" the reader keeps no outline of. */
        [[nodiscard]] Status check_outline(Outline segmentation, const Entry& entry)
        {
            SA_FAIL_IF(segmentation == Outline::RunLength, StatusCode::InvalidFile,
                       "'{}': {}: {} {} holds a run-length mask; the reader keeps polygons only",
                       entry.file, K_READER, entry.list, entry.index);
            SA_FAIL_IF(segmentation == Outline::NotList, StatusCode::InvalidFile,
                       "'{}': {}: {} {} has a 'segmentation' that is no list of polygons",
                       entry.file, K_READER, entry.list, entry.index);
            SA_FAIL_IF(segmentation == Outline::BadPart, StatusCode::InvalidFile,
                       "'{}': {}: {} {} has a 'segmentation' part that is not at least 3 pairs "
                       "of numbers",
                       entry.file, K_READER, entry.list, entry.index);
            return Status();
        }

        /**
         * @brief Appends one sample per entry of "images" and returns the size of each.
         * @return InvalidFile, with the index, for an entry with no name or no positive size.
         */
        [[nodiscard]] Result<std::vector<Extent>> read_images(
            const std::vector<RawImage>& images, const std::filesystem::path& folder,
            const std::string& file, SampleList& list)
        {
            std::vector<Extent> extents;
            extents.reserve(images.size());
            list.files.reserve(images.size());
            list.samples.reserve(images.size());

            for (std::size_t index = 0; index < images.size(); ++index)
            {
                const RawImage& image = images[index];
                const Entry entry{"image", index, file};

                SA_TRY(check_text(image.has_name, entry, "file_name"));
                SA_TRY(check_whole(image.has_width, entry, "width"));
                SA_TRY(check_whole(image.has_height, entry, "height"));
                SA_FAIL_IF(image.width < 1 || image.height < 1, StatusCode::InvalidFile,
                           "'{}': {}: image {} is {} by {} pixels", file, K_READER, index,
                           image.height, image.width);

                Sample sample;
                sample.file = static_cast<std::uint32_t>(index);
                list.files.push_back(folder / std::filesystem::u8path(image.name));
                list.samples.push_back(std::move(sample));
                extents.push_back(Extent{static_cast<double>(image.width),
                                         static_cast<double>(image.height)});
            }

            return Result<std::vector<Extent>>(std::move(extents));
        }

        /** @brief Returns the names of the categories in the order of their ids. */
        [[nodiscard]] Result<std::vector<std::string>> read_class_names(
            const std::vector<RawCategory>& categories, const std::vector<IDEntry>& ids,
            const std::string& file)
        {
            std::vector<std::string> names;
            names.reserve(ids.size());

            for (const IDEntry& id : ids)
            {
                const RawCategory& category = categories[id.position];

                SA_TRY(check_text(category.has_name, Entry{"category", id.position, file},
                                  "name"));
                names.push_back(category.name);
            }

            return Result<std::vector<std::string>>(std::move(names));
        }

        /** @brief Holds what add_objects resolves the annotations of a file against. */
        struct Resolved
        {
            const std::vector<IDEntry>& image_ids;    /**< Holds the image ids, sorted. */
            const std::vector<IDEntry>& category_ids; /**< Holds the category ids, sorted. */
            const std::vector<Extent>& extents;       /**< Holds the size of each image. */
            const std::string& file;                  /**< Names the file. */
        };

        /**
         * @brief Adds every annotation that is kept to the sample of its image.
         * @param raw Its outlines are moved out.
         * @return InvalidFile, with the index, for the first annotation at fault.
         */
        [[nodiscard]] Status add_objects(RawFile& raw, const Resolved& resolved,
                                         COCOSegmentation segmentation, SampleList& list)
        {
            // raw.pending is in annotation order, so one cursor pairs it with the loop.
            std::size_t pending = 0;

            for (std::size_t index = 0; index < raw.annotations.size(); ++index)
            {
                RawAnnotation& annotation = raw.annotations[index];
                const Entry entry{"annotation", index, resolved.file};
                const PendingOutline* waiting = nullptr;
                if (pending < raw.pending.size() && raw.pending[pending].annotation == index)
                    waiting = &raw.pending[pending++];

                SA_TRY_ASSIGN(const std::size_t image_rank,
                              rank_of(annotation.has_image_id, annotation.image_id, entry,
                                      "image_id", resolved.image_ids, "images"));
                SA_TRY_ASSIGN(const std::size_t class_index,
                              rank_of(annotation.has_category_id, annotation.category_id, entry,
                                      "category_id", resolved.category_ids, "categories"));

                if (annotation.is_crowd)
                    continue;

                SA_FAIL_IF(!annotation.is_box, StatusCode::InvalidFile,
                           "'{}': {}: {} {} has no 'bbox' of {} numbers", entry.file, K_READER,
                           entry.list, entry.index, K_BOX_NUMBERS);
                if (is_left_out(annotation))
                    continue;

                const std::size_t image = resolved.image_ids[image_rank].position;
                const Extent& extent = resolved.extents[image];
                const std::array<double, K_BOX_NUMBERS>& box = annotation.box;

                Imaging::Polygon polygon;
                if (segmentation == COCOSegmentation::Polygons)
                {
                    SA_TRY(check_outline(annotation.segmentation, entry));
                    polygon = waiting == nullptr
                                  ? std::move(annotation.outline)
                                  : largest_outline(waiting->numbers, waiting->part_ends, extent);
                }

                Imaging::NormalizedBox fraction;
                fraction.x_center = static_cast<float>((box[0] + box[2] / 2.0) / extent.width);
                fraction.y_center = static_cast<float>((box[1] + box[3] / 2.0) / extent.height);
                fraction.width = static_cast<float>(box[2] / extent.width);
                fraction.height = static_cast<float>(box[3] / extent.height);

                list.samples[image].objects.add(static_cast<std::int32_t>(class_index), fraction,
                                                std::move(polygon));
            }

            return Status();
        }

        /**
         * @brief Checks what one pass kept of a file and builds its list.
         * @param raw Its outlines are moved out.
         * @return InvalidFile for a missing list, an empty one and the first entry at fault.
         */
        [[nodiscard]] Result<SampleList> build_list(RawFile& raw,
                                                    const std::filesystem::path& folder,
                                                    const std::string& file,
                                                    COCOSegmentation segmentation)
        {
            for (std::size_t kind = 0; kind < K_LIST_COUNT; ++kind)
            {
                SA_FAIL_IF(!raw.has_list[kind], StatusCode::InvalidFile,
                           "'{}': {}: the file has no list '{}'", file, K_READER,
                           K_LIST_KEYS[kind]);
            }

            SA_FAIL_IF(raw.images.empty() || raw.categories.empty(), StatusCode::InvalidFile,
                       "'{}': {}: the file lists {} images and {} categories", file, K_READER,
                       raw.images.size(), raw.categories.size());

            SA_TRY_ASSIGN(const std::vector<IDEntry> image_ids,
                          index_ids(raw.images, "image", file));
            SA_TRY_ASSIGN(const std::vector<IDEntry> category_ids,
                          index_ids(raw.categories, "category", file));

            SampleList list;
            list.class_count = static_cast<std::int32_t>(raw.categories.size());
            SA_TRY_ASSIGN(list.class_names,
                          read_class_names(raw.categories, category_ids, file));
            SA_TRY_ASSIGN(const std::vector<Extent> extents,
                          read_images(raw.images, folder, file, list));

            SA_TRY(add_objects(raw, Resolved{image_ids, category_ids, extents, file},
                               segmentation, list));
            return Result<SampleList>(std::move(list));
        }

        /** @brief Reads one file into a list, in one pass over its events. */
        [[nodiscard]] Result<SampleList> read_file(const std::filesystem::path& path,
                                                   const std::filesystem::path& folder,
                                                   COCOSegmentation segmentation)
        {
            const std::string file = Detail::display_of(path);

            SA_TRY(Detail::size_of_file(path, K_READER));

            RawFile raw;
            COCOEvents events(raw, segmentation);
            const Status read = Config::read_json_events(path, events);
            if (!read.is_ok())
                return Detail::boundary_failure(read.code(), K_READER, path, read.message());

            SA_FAIL_IF(!events.repeated_list().empty(), StatusCode::InvalidFile,
                       "'{}': {}: the file states '{}' twice", file, K_READER,
                       events.repeated_list());

            return build_list(raw, folder, file, segmentation);
        }

        /** @brief Returns a path as it is when absolute, and below @p root otherwise. */
        [[nodiscard]] std::filesystem::path below(const std::filesystem::path& root,
                                                  const std::filesystem::path& path)
        {
            return path.is_relative() ? root / path : path;
        }
    } // namespace

    Result<SampleList> read_coco_json(const COCOJSONOptions& options)
    {
        return Detail::at_boundary(K_READER, [&]() -> Result<SampleList>
        {
            const std::filesystem::path file =
                options.annotation_file.empty()
                    ? options.root / K_ANNOTATION_FOLDER /
                          (std::string(K_ANNOTATION_PREFIX) + options.split +
                           std::string(K_JSON_EXTENSION))
                    : below(options.root, options.annotation_file);
            const std::filesystem::path folder =
                options.image_folder.empty() ? options.root / options.split
                                             : below(options.root, options.image_folder);

            std::error_code ignored;
            SA_FAIL_IF(!std::filesystem::is_directory(folder, ignored), StatusCode::FileNotFound,
                       "'{}': {}: no such image folder", Detail::display_of(folder), K_READER);

            return Detail::at_boundary(K_READER, [&]()
            {
                return read_file(file, folder, options.segmentation);
            }, file);
        }, options.root);
    }
} // namespace SushiAI::Vision::Datasets
