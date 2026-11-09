/**************************************************************************/
/* test_coco_json.cpp                                                     */
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
 * @file test_coco_json.cpp
 * @brief Tests the reader of a dataset kept as one annotation JSON and an image folder.
 * @author Mustafa Garip
 */
#include <cstddef>
#include <filesystem>
#include <sstream>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include "datasets_test_support.hpp"
#include <SushiAI/failure/result.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/vision/datasets/byte_source.hpp>
#include <SushiAI/vision/datasets/coco_json.hpp>
#include <SushiAI/vision/datasets/sample.hpp>
#include <SushiAI/vision/imaging/boxes.hpp>
#include <support/tolerance.hpp>

using namespace SushiAI::Vision::Datasets;
using SushiAI::StatusCode;
using SushiAI::Vision::Imaging::NormalizedBox;

namespace
{
    /**
     * @brief Holds the absolute difference a fraction may have from its quotient: 32 times the
     *        float unit roundoff, the smallest power of two not tighter than the earlier 1e-6f.
     */
    constexpr float K_FRACTION_TOLERANCE =
        static_cast<float>(32.0 * TestSupport::unit_roundoff<float>());

    /** @brief Holds two images of 200 by 100 pixels, the second listed first by id. */
    constexpr const char* K_IMAGES =
        R"("images": [{"id": 9, "file_name": "b.jpg", "width": 200, "height": 100},
                      {"id": 4, "file_name": "sub/a.jpg", "width": 200, "height": 100}])";

    /** @brief Holds three categories whose ids are sparse and not in order. */
    constexpr const char* K_CATEGORIES =
        R"("categories": [{"id": 90, "name": "brush"}, {"id": 1, "name": "person"},
                          {"id": 17, "name": "cat"}])";

    /** @brief Returns the document of the two images, the three categories and the annotations. */
    std::string document_of(const std::string& annotations)
    {
        return std::string("{") + K_IMAGES + ", " + K_CATEGORIES + ", \"annotations\": [" +
               annotations + "]}";
    }

    /** @brief Returns a source holding the image folder train and the annotation text. */
    MemorySource source_with(const std::string& text)
    {
        MemorySource source;
        source.add_directory("set/train");
        source.add_file("set/annotations/instances_train.json", text);
        return source;
    }

    /** @brief Returns a source of the split "train" holding the annotations of the document. */
    MemorySource split_of(const std::string& annotations)
    {
        return source_with(document_of(annotations));
    }

    /** @brief Returns options that read the split "train" of the folder set. */
    COCOJSONOptions options_of()
    {
        COCOJSONOptions options;
        options.root = "set";
        return options;
    }

    /** @brief Returns the path of a file of a list below a directory, with forward slashes. */
    std::string relative_file(const SampleList& list, std::size_t sample,
                              const std::filesystem::path& base)
    {
        return list.files[list.samples[sample].file].lexically_relative(base).generic_string();
    }

    /** @brief Returns one line per sample: its file, then each object with its box and outline. */
    std::vector<std::string> description_of(const SampleList& list,
                                            const std::filesystem::path& base)
    {
        std::vector<std::string> lines;
        for (std::size_t sample = 0; sample < list.samples.size(); ++sample)
        {
            const auto& objects = list.samples[sample].objects;
            std::ostringstream line;
            line.precision(9);
            line << relative_file(list, sample, base);
            for (std::size_t object = 0; object < objects.size(); ++object)
            {
                const NormalizedBox& box = objects.boxes()[object];
                line << " | class " << objects.classes()[object] << " box " << box.x_center << ' '
                     << box.y_center << ' ' << box.width << ' ' << box.height << " outline";
                for (const auto& vertex : objects.polygons()[object])
                    line << ' ' << vertex.x << ',' << vertex.y;
            }
            lines.push_back(line.str());
        }

        return lines;
    }

    /** @brief Holds the annotations of one document and where the message points. */
    struct AnnotationRow
    {
        std::string name;         /**< The row's name. */
        std::string annotations;  /**< The text inside the annotation list. */
        std::string cause;        /**< The text the message holds. */
    };

    /** @brief Holds a whole document and the text the message holds. */
    struct DocumentRow
    {
        std::string name;      /**< The row's name. */
        std::string document;  /**< The whole annotation file. */
        std::string cause;     /**< The text the message holds. */
    };

    /** @brief Runs read_coco_json over each annotation of an unknown image or category. */
    class Small_DatasetsCocoJsonUnknown : public ::testing::TestWithParam<AnnotationRow>
    {
    };

    /** @brief Runs read_coco_json over each box that is not four numbers. */
    class Small_DatasetsCocoJsonBox : public ::testing::TestWithParam<AnnotationRow>
    {
    };

    /** @brief Runs read_coco_json over each document with an invalid image or category list. */
    class Small_DatasetsCocoJsonLists : public ::testing::TestWithParam<DocumentRow>
    {
    };

    /** @brief Names a test after its annotation row. */
    std::string annotation_row_name(const ::testing::TestParamInfo<AnnotationRow>& info)
    {
        return info.param.name;
    }

    /** @brief Names a test after its document row. */
    std::string document_row_name(const ::testing::TestParamInfo<DocumentRow>& info)
    {
        return info.param.name;
    }
} // namespace

/** @brief Tests that a box becomes a fraction of its image. */
TEST(Small_DatasetsCocoJson, ABoxBecomesAFractionOfItsImage)
{
    const MemorySource source = split_of(
        R"({"id": 1, "image_id": 4, "category_id": 90, "bbox": [10, 20, 50, 20]})");

    SA_ASSERT_OK_AND_ASSIGN(const SampleList list, read_coco_json(options_of(), source));

    ASSERT_EQ(list.samples.size(), 2u);
    ASSERT_EQ(list.samples[1].objects.size(), 1u);
    const NormalizedBox& box = list.samples[1].objects.boxes()[0];
    EXPECT_NEAR(box.x_center, 35.0f / 200.0f, K_FRACTION_TOLERANCE);
    EXPECT_NEAR(box.y_center, 30.0f / 100.0f, K_FRACTION_TOLERANCE);
    EXPECT_NEAR(box.width, 50.0f / 200.0f, K_FRACTION_TOLERANCE);
    EXPECT_NEAR(box.height, 20.0f / 100.0f, K_FRACTION_TOLERANCE);
}

/** @brief Tests that category ids become dense class indices in id order. */
TEST(Small_DatasetsCocoJson, CategoryIdsBecomeDenseClassesInIdOrder)
{
    const MemorySource source = split_of(
        R"({"id": 1, "image_id": 4, "category_id": 90, "bbox": [10, 20, 50, 20]},
           {"id": 2, "image_id": 4, "category_id": 1, "bbox": [0, 0, 200, 100.0]})");

    SA_ASSERT_OK_AND_ASSIGN(const SampleList list, read_coco_json(options_of(), source));

    EXPECT_EQ(list.class_names, (std::vector<std::string>{"person", "cat", "brush"}));
    EXPECT_EQ(list.class_count, 3);
    const Sample& sample = list.samples[1];
    EXPECT_FALSE(sample.class_index.has_value());
    ASSERT_EQ(sample.objects.size(), 2u);
    EXPECT_EQ(sample.objects.classes()[0], 2);
    EXPECT_EQ(sample.objects.classes()[1], 0);
}

/** @brief Tests that a crowd annotation and a box of no width are left out. */
TEST(Small_DatasetsCocoJson, ACrowdAnnotationAndABoxOfNoWidthAreLeftOut)
{
    const MemorySource source = split_of(
        R"({"id": 1, "image_id": 4, "category_id": 90, "bbox": [10, 20, 50, 20]},
           {"id": 3, "image_id": 4, "category_id": 17, "bbox": [1, 1, 5, 5], "iscrowd": 1},
           {"id": 4, "image_id": 4, "category_id": 17, "bbox": [1, 1, 0, 5]})");

    SA_ASSERT_OK_AND_ASSIGN(const SampleList list, read_coco_json(options_of(), source));

    EXPECT_TRUE(list.samples[0].objects.is_empty());
    ASSERT_EQ(list.samples[1].objects.size(), 1u);
    EXPECT_EQ(list.samples[1].objects.classes()[0], 2);
    EXPECT_TRUE(list.samples[1].objects.polygons()[0].empty());
}

/** @brief Tests that each image resolves to its file in the image folder, in id order. */
TEST(Small_DatasetsCocoJson, ImageNamesResolveInTheImageFolder)
{
    const MemorySource source = split_of("");

    SA_ASSERT_OK_AND_ASSIGN(const SampleList list, read_coco_json(options_of(), source));

    ASSERT_EQ(list.samples.size(), 2u);
    EXPECT_EQ(relative_file(list, 0, "set"), "train/b.jpg");
    EXPECT_EQ(relative_file(list, 1, "set"), "train/sub/a.jpg");
}

/** @brief Tests that the file and the image folder may be named. */
TEST(Small_DatasetsCocoJson, TheFileAndTheFolderMayBeNamed)
{
    MemorySource source;
    source.add_directory("set/pictures");
    source.add_file("set/labels.json", document_of(""));
    COCOJSONOptions options = options_of();
    options.annotation_file = "labels.json";
    options.image_folder = "pictures";

    SA_ASSERT_OK_AND_ASSIGN(const SampleList list, read_coco_json(options, source));

    EXPECT_EQ(relative_file(list, 0, "set"), "pictures/b.jpg");
}

/** @brief Tests that a polygon is kept only when polygons are asked for. */
TEST(Small_DatasetsCocoJson, APolygonIsKeptOnlyWhenAskedFor)
{
    const MemorySource source = split_of(
        R"({"id": 1, "image_id": 9, "category_id": 1, "bbox": [0, 0, 100, 50],
            "segmentation": [[0, 0, 2, 0, 2, 2], [0, 0, 100, 0, 100, 50, 0, 50]]})");
    COCOJSONOptions options = options_of();

    SA_ASSERT_OK_AND_ASSIGN(const SampleList ignored, read_coco_json(options, source));
    EXPECT_TRUE(ignored.samples[0].objects.polygons()[0].empty());

    options.segmentation = COCOSegmentation::Polygons;
    SA_ASSERT_OK_AND_ASSIGN(const SampleList kept, read_coco_json(options, source));
    EXPECT_EQ(kept.samples[0].objects.polygons()[0].size(), 4u);
}

/** @brief Tests that the largest part of a multi-part polygon is taken. */
TEST(Small_DatasetsCocoJson, TheLargestPartOfAPolygonIsTaken)
{
    const MemorySource source = split_of(
        R"({"id": 1, "image_id": 9, "category_id": 1, "bbox": [0, 0, 100, 50],
            "segmentation": [[0, 0, 2, 0, 2, 2], [0, 0, 100, 0, 100, 50, 0, 50]]})");
    COCOJSONOptions options = options_of();
    options.segmentation = COCOSegmentation::Polygons;

    SA_ASSERT_OK_AND_ASSIGN(const SampleList list, read_coco_json(options, source));

    ASSERT_EQ(list.samples[0].objects.size(), 1u);
    const auto& polygon = list.samples[0].objects.polygons()[0];
    ASSERT_EQ(polygon.size(), 4u);
    EXPECT_NEAR(polygon[2].x, 100.0f / 200.0f, K_FRACTION_TOLERANCE);
    EXPECT_NEAR(polygon[2].y, 50.0f / 100.0f, K_FRACTION_TOLERANCE);
}

/** @brief Tests that a run-length mask is accepted when polygons are not asked for. */
TEST(Small_DatasetsCocoJson, ARunLengthMaskIsAcceptedByDefault)
{
    const MemorySource source = split_of(
        R"({"id": 1, "image_id": 9, "category_id": 1, "bbox": [0, 0, 10, 10]},
           {"id": 2, "image_id": 9, "category_id": 1, "bbox": [0, 0, 10, 10],
            "segmentation": {"counts": "abc", "size": [100, 200]}})");

    SA_ASSERT_OK_AND_ASSIGN(const SampleList list, read_coco_json(options_of(), source));

    EXPECT_EQ(list.samples[0].objects.size(), 2u);
}

/** @brief Tests that a run-length mask fails by its index when polygons are asked for. */
TEST(Small_DatasetsCocoJson, ARunLengthMaskIsRefusedWhenPolygonsAreAskedFor)
{
    const MemorySource source = split_of(
        R"({"id": 1, "image_id": 9, "category_id": 1, "bbox": [0, 0, 10, 10]},
           {"id": 2, "image_id": 9, "category_id": 1, "bbox": [0, 0, 10, 10],
            "segmentation": {"counts": "abc", "size": [100, 200]}})");
    COCOJSONOptions options = options_of();
    options.segmentation = COCOSegmentation::Polygons;

    SA_EXPECT_FAILURE(read_coco_json(options, source), StatusCode::InvalidFile,
                      "instances_train.json", "read_coco_json",
                      "annotation 1 holds a run-length mask");
}

/** @brief Tests that a polygon part of an odd or short number list fails by its index. */
TEST(Small_DatasetsCocoJson, AMalformedPolygonIsRefused)
{
    const MemorySource source = split_of(
        R"({"id": 1, "image_id": 9, "category_id": 1, "bbox": [0, 0, 10, 10],
            "segmentation": [[0, 0, 2, 0, 2, 2, 5]]})");
    COCOJSONOptions options = options_of();
    options.segmentation = COCOSegmentation::Polygons;

    SA_EXPECT_FAILURE(read_coco_json(options, source), StatusCode::InvalidFile, "annotation 0",
                      "'segmentation' part", "at least 3 pairs");
}

/** @brief Tests that an annotation of an unknown image or category fails by its index. */
TEST_P(Small_DatasetsCocoJsonUnknown, AnUnknownImageOrCategoryNamesTheAnnotation)
{
    const MemorySource source = split_of(GetParam().annotations);

    SA_EXPECT_FAILURE(read_coco_json(options_of(), source), StatusCode::InvalidFile,
                      "instances_train.json", "read_coco_json", GetParam().cause);
}

INSTANTIATE_TEST_SUITE_P(
    Rows, Small_DatasetsCocoJsonUnknown,
    ::testing::Values(
        AnnotationRow{"Image",
                      R"({"id": 1, "image_id": 9, "category_id": 1, "bbox": [0, 0, 10, 10]},
                         {"id": 2, "image_id": 5, "category_id": 1, "bbox": [0, 0, 10, 10]})",
                      "annotation 1 has the 'image_id' 5, which 'images' lacks"},
        AnnotationRow{"Category",
                      R"({"id": 1, "image_id": 9, "category_id": 2, "bbox": [0, 0, 10, 10],
                          "iscrowd": 1})",
                      "annotation 0 has the 'category_id' 2, which 'categories' lacks"}),
    annotation_row_name);

/** @brief Tests that a box that is not four numbers fails by the index of its annotation. */
TEST_P(Small_DatasetsCocoJsonBox, AMalformedBoxNamesTheAnnotation)
{
    const MemorySource source = split_of(GetParam().annotations);

    SA_EXPECT_FAILURE(read_coco_json(options_of(), source), StatusCode::InvalidFile,
                      GetParam().cause);
}

INSTANTIATE_TEST_SUITE_P(
    Rows, Small_DatasetsCocoJsonBox,
    ::testing::Values(
        AnnotationRow{"ThreeNumbers",
                      R"({"id": 1, "image_id": 9, "category_id": 1, "bbox": [0, 0, 10]})",
                      "annotation 0 has no 'bbox' of 4 numbers"},
        AnnotationRow{"ANonNumber",
                      R"({"id": 1, "image_id": 9, "category_id": 1, "bbox": [0, 0, 10, "x"]})",
                      "annotation 0 has no 'bbox' of 4 numbers"}),
    annotation_row_name);

/** @brief Tests that an image id that is not a whole number fails by the annotation's index. */
TEST(Small_DatasetsCocoJson, AnImageIdThatIsNotAWholeNumberNamesTheAnnotation)
{
    const MemorySource source =
        split_of(R"({"id": 1, "image_id": "9", "category_id": 1})");

    SA_EXPECT_FAILURE(read_coco_json(options_of(), source), StatusCode::InvalidFile,
                      "annotation 0 has no whole number 'image_id'");
}

/** @brief Tests that a repeated category id, an image of no size and a nameless image fail. */
TEST_P(Small_DatasetsCocoJsonLists, TheImagesAndTheCategoriesAreChecked)
{
    const MemorySource source = source_with(GetParam().document);

    SA_EXPECT_FAILURE(read_coco_json(options_of(), source), StatusCode::InvalidFile,
                      GetParam().cause);
}

INSTANTIATE_TEST_SUITE_P(
    Rows, Small_DatasetsCocoJsonLists,
    ::testing::Values(
        DocumentRow{"TwoCategoriesOfOneId",
                    R"({"images": [{"id": 1, "file_name": "a.jpg", "width": 2, "height": 2}],
                        "categories": [{"id": 3, "name": "a"}, {"id": 7, "name": "b"},
                                       {"id": 3, "name": "c"}],
                        "annotations": []})",
                    "category 0 and category 2 share the id 3"},
        DocumentRow{"AnImageOfZeroWidth",
                    R"({"images": [{"id": 1, "file_name": "a.jpg", "width": 2, "height": 2},
                                   {"id": 2, "file_name": "b.jpg", "width": 0, "height": 2}],
                        "categories": [{"id": 3, "name": "a"}], "annotations": []})",
                    "image 1 is 2 by 0 pixels"},
        DocumentRow{"AnImageWithoutAName",
                    R"({"images": [{"id": 1, "width": 2, "height": 2}],
                        "categories": [{"id": 3, "name": "a"}], "annotations": []})",
                    "image 0 has no string 'file_name'"}),
    document_row_name);

/** @brief Tests that an absent image folder is reported with its name. */
TEST(Small_DatasetsCocoJson, AnAbsentImageFolderIsNotFound)
{
    const MemorySource source;

    SA_EXPECT_FAILURE(read_coco_json(options_of(), source), StatusCode::FileNotFound, "train",
                      "read_coco_json", "no such image folder");
}

/** @brief Tests that an absent annotation file is reported with its name. */
TEST(Small_DatasetsCocoJson, AnAbsentAnnotationFileIsNotFound)
{
    MemorySource source;
    source.add_directory("set/train");

    SA_EXPECT_FAILURE(read_coco_json(options_of(), source), StatusCode::FileNotFound,
                      "annotations/instances_train.json", "read_coco_json", "no such file");
}

/** @brief Tests that text that is no JSON document is refused. */
TEST(Small_DatasetsCocoJson, TextThatIsNoJsonIsRefused)
{
    const MemorySource source = source_with("{\"images\": [");

    SA_EXPECT_FAILURE(read_coco_json(options_of(), source), StatusCode::InvalidFile,
                      "instances_train.json", "read_coco_json", "not valid JSON");
}

/** @brief Tests that a document without one of its three lists is refused. */
TEST(Small_DatasetsCocoJson, ADocumentWithoutAListIsRefused)
{
    const MemorySource source = source_with(std::string("{") + K_IMAGES + ", \"annotations\": []}");

    SA_EXPECT_FAILURE(read_coco_json(options_of(), source), StatusCode::InvalidFile,
                      "the file has no list 'categories'");
}

/** @brief Tests that annotations met before their images and categories still resolve. */
TEST(Small_DatasetsCocoJson, TheListsMayComeInAnyOrder)
{
    const std::string annotations =
        R"("annotations": [
               {"segmentation": [[0, 0, 2, 0, 2, 2], [0, 0, 100, 0, 100, 50, 0, 50]],
                "bbox": [10, 20, 50, 20], "category_id": 90, "image_id": 4, "id": 1},
               {"id": 2, "image_id": 9, "category_id": 1, "bbox": [0, 0, 200, 100]}])";
    const MemorySource in_order =
        source_with(std::string("{") + K_IMAGES + ", " + K_CATEGORIES + ", " + annotations + "}");
    const MemorySource reversed = source_with(
        std::string("{\"info\": {\"year\": 2017, \"notes\": [1, [2]]}, ") + annotations + ", " +
        K_CATEGORIES + ", " + K_IMAGES + "}");
    COCOJSONOptions options = options_of();
    options.segmentation = COCOSegmentation::Polygons;

    SA_ASSERT_OK_AND_ASSIGN(const SampleList expected, read_coco_json(options, in_order));
    SA_ASSERT_OK_AND_ASSIGN(const SampleList list, read_coco_json(options, reversed));

    EXPECT_EQ(list.class_names, expected.class_names);
    EXPECT_EQ(description_of(list, "set"), description_of(expected, "set"));
}

/** @brief Tests that an annotation listed before its image and category still reaches both. */
TEST(Small_DatasetsCocoJson, AnAnnotationListedBeforeItsImageStillReachesIt)
{
    const MemorySource source = source_with(
        std::string("{") + K_CATEGORIES + ", " + K_IMAGES +
        R"(, "annotations": [
               {"segmentation": [[0, 0, 2, 0, 2, 2], [0, 0, 100, 0, 100, 50, 0, 50]],
                "bbox": [10, 20, 50, 20], "category_id": 90, "image_id": 4, "id": 1}]})");
    COCOJSONOptions options = options_of();
    options.segmentation = COCOSegmentation::Polygons;

    SA_ASSERT_OK_AND_ASSIGN(const SampleList list, read_coco_json(options, source));

    EXPECT_EQ(list.samples[1].objects.classes()[0], 2);
    EXPECT_EQ(list.samples[1].objects.polygons()[0].size(), 4u);
    EXPECT_TRUE(list.samples[0].objects.is_empty());
}

/** @brief Tests that a list the file states twice is refused by its key. */
TEST(Small_DatasetsCocoJson, AListStatedTwiceIsRefused)
{
    const MemorySource source = source_with(std::string("{") + K_IMAGES + ", " + K_CATEGORIES +
                                            ", \"annotations\": [], " + K_IMAGES + "}");

    SA_EXPECT_FAILURE(read_coco_json(options_of(), source), StatusCode::InvalidFile,
                      "instances_train.json", "read_coco_json", "states 'images' twice");
}

/** @brief Tests that the file-system form reads a set written to disk. */
TEST(Medium_DatasetsCocoJson, ASetOnDiskIsRead)
{
    const Support::ScratchDirectory scratch("sushiai_test_datasets_coco_disk");
    scratch.make_directory("train");
    scratch.write_text("annotations/instances_train.json",
                       document_of(R"({"id": 1, "image_id": 4, "category_id": 90,
                                       "bbox": [10, 20, 50, 20]})"));
    COCOJSONOptions options;
    options.root = scratch.path();

    SA_ASSERT_OK_AND_ASSIGN(const SampleList list, read_coco_json(options));

    EXPECT_EQ(relative_file(list, 1, scratch.path()), "train/sub/a.jpg");
    EXPECT_EQ(list.samples[1].objects.size(), 1u);
}
