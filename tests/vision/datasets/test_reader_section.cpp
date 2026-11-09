/**************************************************************************/
/* test_reader_section.cpp                                                */
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
 * @file test_reader_section.cpp
 * @brief Tests the reader table, the task table and the dataset a data section builds of them.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <string_view>
#include <vector>
#include <gtest/gtest.h>
#include <SushiRuntime/core/span.hpp>
#include "datasets_test_support.hpp"
#include <SushiAI/failure/result.hpp>
#include <SushiAI/config/value.hpp>
#include <SushiAI/shape/shape.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/data/dataset.hpp>
#include <SushiAI/data/epoch.hpp>
#include <SushiAI/data/epoch_permutation.hpp>
#include <SushiAI/data/roles.hpp>
#include <SushiAI/data/target.hpp>
#include <SushiAI/data/threaded_loader.hpp>
#include <SushiAI/execution/host_execution_backend.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/model_text/model_text.hpp>
#include <SushiAI/vision/datasets/binary_collator.hpp>
#include <SushiAI/vision/datasets/collator.hpp>
#include <SushiAI/vision/datasets/fit_sample.hpp>
#include <SushiAI/vision/datasets/reader_section.hpp>
#include <SushiAI/vision/datasets/reader_table.hpp>
#include <SushiAI/vision/datasets/sample.hpp>
#include <SushiAI/vision/datasets/segmentation_collator.hpp>
#include <SushiAI/vision/datasets/task_table.hpp>
#include <SushiAI/vision/imaging/image.hpp>
#include <SushiAI/vision/imaging/raw_cache.hpp>
#include <support/tolerance.hpp>

using namespace SushiAI;
using namespace SushiAI::Vision::Datasets;
using Vision::Imaging::ImageSize;
namespace SaTest = SushiAI::Test;

namespace
{
    /**
     * @brief Holds the absolute tolerance of a byte divided by 255 in float: 32 times the float
     *        unit roundoff, the smallest power of two not tighter than the earlier 1e-6f.
     */
    constexpr float K_SCALE_TOLERANCE =
        static_cast<float>(32.0 * TestSupport::unit_roundoff<float>());

    /** @brief Holds a root no test creates, for a section a refusal decides before any read. */
    constexpr std::string_view K_ABSENT_ROOT = "sushiai_absent_section_root";

    /** @brief Holds the seed of the shuffled section. */
    constexpr std::uint64_t K_SHUFFLE_SEED = 7;

    /** @brief Counts the images of the numbered split the shuffled section reads. */
    constexpr std::size_t K_NUMBERED_IMAGES = 5;

    /** @brief Provides a host engine and the sections these tests build datasets from. */
    class SectionBench : public ::testing::Test
    {
        protected:
            /**
             * @brief Parses a section that opens with @p root and ends with @p members.
             * @param members JSON members, each led by a comma; may be empty.
             */
            static Config::Value section_over(std::string_view root, std::string_view members)
            {
                const std::string text = R"({"root": ")" + std::string(root) + "\""
                                       + std::string(members) + "}";
                return SaTest::attributes_from(text);
            }

            /** @brief Parses a section whose root is the scratch directory. */
            static Config::Value section_of(const Support::ScratchDirectory& scratch,
                                            std::string_view members)
            {
                return section_over(scratch.path().generic_string(), members);
            }

            /** @brief Builds the dataset of a section called "data" on the host engine. */
            Result<std::unique_ptr<Data::Dataset>> build(const Config::Value& section)
            {
                return make_reader_dataset(backend.engine(), section, "data");
            }

            /** @brief Writes two 2 by 4 white images; the second has one object of class 2. */
            static void write_yolo_split(const Support::ScratchDirectory& scratch)
            {
                for (const char* name : {"images/train/a.saim", "images/train/b.saim"})
                    Support::write_raw_image(scratch, name,
                                             Support::uniform_image(ImageSize{2, 4}, 3, 255));

                scratch.write_text("labels/train/b.txt", "2 0.5 0.5 0.5 1.0\n");
            }

            /** @brief Writes K_NUMBERED_IMAGES white images, image i with one object of class i. */
            static void write_numbered_split(const Support::ScratchDirectory& scratch)
            {
                for (std::size_t i = 0; i < K_NUMBERED_IMAGES; ++i)
                {
                    const std::string stem = std::to_string(i);
                    Support::write_raw_image(scratch, ("images/train/" + stem + ".saim").c_str(),
                                             Support::uniform_image(ImageSize{2, 4}, 3, 255));
                    scratch.write_text(("labels/train/" + stem + ".txt").c_str(),
                                       stem + " 0.5 0.5 0.5 1.0\n");
                }
            }

            /** @brief Writes one image file under each of two class folders of "train". */
            static void write_folder_split(const Support::ScratchDirectory& scratch)
            {
                scratch.write("train/cat/a.jpg", {1});
                scratch.write("train/dog/b.jpg", {1});
            }

            /** @brief Returns the inputs of batch @p index, which must read. */
            static std::vector<float> inputs_of(Data::Dataset& dataset, std::size_t index)
            {
                Data::Batch batch;
                EXPECT_TRUE(SaTest::status_is_ok(dataset.read(index, batch))) << index;

                const float* const first = batch.inputs.data_as<float>();
                return std::vector<float>(first, first + batch.inputs.num_elements);
            }

            /** @brief Returns the inputs of every batch of a dataset, in batch order. */
            static std::vector<std::vector<float>> inputs_of_every_batch(Data::Dataset& dataset)
            {
                std::vector<std::vector<float>> inputs;
                for (std::size_t index = 0; index < dataset.batch_count(); ++index)
                    inputs.push_back(inputs_of(dataset, index));

                return inputs;
            }

            /** @brief Returns the class that slot 0 of batch @p index holds, which must read. */
            static std::int32_t first_class_of(Data::Dataset& dataset, std::size_t index)
            {
                Data::Batch batch;
                EXPECT_TRUE(SaTest::status_is_ok(dataset.read(index, batch))) << index;

                const Data::BatchTarget* const classes =
                    Data::find_target(batch.targets, Data::K_ROLE_CLASSES);
                EXPECT_NE(classes, nullptr);
                return classes == nullptr ? -1 : classes->values.data_as<std::int32_t>()[0];
            }

            /** @brief Holds the host engine the datasets are built on. */
            HostExecutionBackend backend;
    };

    /** @brief Builds datasets from sections that need no file, because a refusal comes first. */
    class Small_DatasetsReaderSection : public SectionBench
    {
    };

    /** @brief Builds datasets from sections whose reader lists files written for the test. */
    class Medium_DatasetsReaderSection : public SectionBench
    {
    };

    /** @brief Returns the names of the rows of a table, in order. */
    template <typename Entry>
    std::vector<std::string_view> names_of(SushiRuntime::span<const Entry> rows)
    {
        std::vector<std::string_view> names;
        for (const Entry& row : rows)
            names.push_back(row.name);

        return names;
    }

    /** @brief Returns the order Data::EpochPermutation gives a seed, an epoch and a count. */
    std::vector<std::int32_t> order_of(std::uint64_t seed, std::int64_t epoch,
                                       std::uint64_t count)
    {
        const Data::EpochPermutation permutation(seed, epoch, count);

        std::vector<std::int32_t> order;
        for (std::uint64_t position = 0; position < count; ++position)
            order.push_back(static_cast<std::int32_t>(permutation.at(position)));

        return order;
    }

    /** @brief Returns how many raw image files a folder holds. */
    std::size_t raw_images_in(const std::filesystem::path& folder)
    {
        std::size_t count = 0;
        for (const std::filesystem::directory_entry& entry :
             std::filesystem::directory_iterator(folder))
            count += entry.path().extension() == ".saim" ? 1 : 0;

        return count;
    }

    /** @brief Expects a status to name each needle. */
    ::testing::AssertionResult names_each(const Status& status,
                                          const std::vector<std::string_view>& needles)
    {
        for (std::string_view needle : needles)
        {
            if (status.message().find(needle) == std::string_view::npos)
                return ::testing::AssertionFailure()
                       << "the diagnostic does not name \"" << needle << "\".\n"
                       << "  message: " << status.message();
        }

        return ::testing::AssertionSuccess();
    }

    /** @brief Holds a reader, the keys it reads and nothing else the row states. */
    struct KeysRow
    {
        std::string_view reader;              /**< Names the row of the reader table. */
        std::vector<std::string_view> keys;   /**< Lists the keys the row reads, in order. */
    };

    /** @brief Returns the keys each row of the reader table reads. */
    const std::vector<KeysRow>& keys_rows()
    {
        static const std::vector<KeysRow> rows = {
            {"yolo_txt", {}},
            {"image_folder", {}},
            {"idx", {}},
            {"cifar", {}},
            {"coco_json", {"annotations", "images", "polygons"}},
            {"voc_xml", {"class_names", "keep_difficult"}},
            {"mask_png", {"images", "masks", "image_suffix", "mask_suffix"}},
            {"csv", {"file", "features", "targets"}},
        };
        return rows;
    }

    /** @brief Names a test after the reader of its row. */
    std::string keys_row_name(const ::testing::TestParamInfo<KeysRow>& info)
    {
        return std::string(info.param.reader);
    }

    /** @brief Checks the keys one row of the reader table lists. */
    class Small_DatasetsReaderTableKeys : public ::testing::TestWithParam<KeysRow>
    {
    };

    /** @brief Holds a reader, a section its keys have the wrong kind in, and what it names. */
    struct KeyKindRow
    {
        std::string_view name;    /**< Names the test of the row. */
        std::string_view reader;  /**< Names the row of the reader table. */
        std::string_view section; /**< Holds the JSON of the section. */
        std::string_view key;     /**< Names the key in the diagnostic. */
        std::string_view kind;    /**< Names the kind the key should have. */
    };

    /** @brief Returns the refusals of a key of the wrong kind. */
    const std::vector<KeyKindRow>& key_kind_rows()
    {
        static const std::vector<KeyKindRow> rows = {
            {"CSVFeaturesAsAString", "csv", R"({"features": "c"})", "data 'features'", "array"},
            {"COCOPolygonsAsANumber", "coco_json", R"({"polygons": 3})", "data 'polygons'",
             "boolean"},
        };
        return rows;
    }

    /** @brief Names a test after its row. */
    std::string key_kind_row_name(const ::testing::TestParamInfo<KeyKindRow>& info)
    {
        return std::string(info.param.name);
    }

    /** @brief Checks that a key of the wrong kind is refused before any file is read. */
    class Small_DatasetsReaderTableKeyKinds : public ::testing::TestWithParam<KeyKindRow>
    {
    };

    /** @brief Holds a task and the fit and the keys its row states. */
    struct TaskRow
    {
        std::string_view task;                 /**< Names the row of the task table. */
        SampleFit fit;                         /**< Gives the fit the row states. */
        std::vector<std::string_view> keys;    /**< Lists the keys the row reads, in order. */
    };

    /** @brief Returns the fit and the keys each task states. */
    const std::vector<TaskRow>& task_rows()
    {
        static const std::vector<TaskRow> rows = {
            {"detection", SampleFit::Letterbox, {"max_objects"}},
            {"classification", SampleFit::Resize, {}},
            {"binary", SampleFit::Resize, {"positive"}},
            {"multi_label", SampleFit::Resize, {}},
            {"regression", SampleFit::Resize, {}},
            {"segmentation", SampleFit::Letterbox, {"ignore_index", "background"}},
        };
        return rows;
    }

    /** @brief Names a test after the task of its row. */
    std::string task_row_name(const ::testing::TestParamInfo<TaskRow>& info)
    {
        return std::string(info.param.task);
    }

    /** @brief Checks the fit and the keys of one task. */
    class Small_DatasetsTaskTableStates : public ::testing::TestWithParam<TaskRow>
    {
    };

    /** @brief Names a test after the index of the row of the task table it runs over. */
    std::string task_index_name(const ::testing::TestParamInfo<std::size_t>& info)
    {
        return std::string(task_table()[info.param].name);
    }

    /** @brief Runs a check over the index of each row of the task table. */
    class Small_DatasetsTaskTableCollators : public ::testing::TestWithParam<std::size_t>
    {
    };

    /** @brief Holds a "positive" section of the binary row that builds a collator. */
    class Small_DatasetsTaskTablePositive : public ::testing::TestWithParam<std::string_view>
    {
    };

    /** @brief Names a test after the section text of its row, which holds no usable character. */
    std::string positive_row_name(const ::testing::TestParamInfo<std::string_view>& info)
    {
        return "Row" + std::to_string(info.index);
    }

    /** @brief Holds a binary section the row refuses and the diagnostic it gives. */
    struct PositiveRefusal
    {
        std::string_view name;     /**< Names the test of the row. */
        std::string_view section;  /**< Holds the JSON of the section. */
        std::int32_t classes;      /**< Counts the classes the samples have. */
        std::string_view first;    /**< Names the first piece of the diagnostic. */
        std::string_view second;   /**< Names the second piece of the diagnostic; may be empty. */
    };

    /** @brief Returns the refusals of the binary row. */
    const std::vector<PositiveRefusal>& positive_refusals()
    {
        static const std::vector<PositiveRefusal> rows = {
            {"AClassNameTheListLacks", R"({"positive": "bird"})", 2, "data 'positive' is 'bird'",
             "'cat', 'dog'"},
            {"AnIndexOutsideTheClasses", R"({"positive": 2})", 2, "the positive class is 2", ""},
            {"ACountOtherThanTwo", R"({})", 3, "data 'task' is 'binary'", "the samples have 3"},
        };
        return rows;
    }

    /** @brief Names a test after its row. */
    std::string positive_refusal_name(const ::testing::TestParamInfo<PositiveRefusal>& info)
    {
        return std::string(info.param.name);
    }

    /** @brief Checks the refusals of the binary row. */
    class Small_DatasetsTaskTablePositiveRefusals
        : public ::testing::TestWithParam<PositiveRefusal>
    {
    };

    /** @brief Holds an ignore index the segmentation row refuses and the diagnostic it gives. */
    struct IgnoreRefusal
    {
        std::string_view name;     /**< Names the test of the row. */
        std::string_view section;  /**< Holds the JSON of the section. */
        std::string_view first;    /**< Names the first piece of the diagnostic. */
        std::string_view second;   /**< Names the second piece of the diagnostic; may be empty. */
    };

    /** @brief Returns the refusals of the segmentation row. */
    const std::vector<IgnoreRefusal>& ignore_refusals()
    {
        static const std::vector<IgnoreRefusal> rows = {
            {"AnIndexThatIsAClass", R"({"ignore_index": 1})",
             "the ignore index 1 is one of the 3", ""},
            {"AValueOf33Bits", R"({"ignore_index": 4294967296})",
             "data 'ignore_index' is 4294967296", "32 bits"},
        };
        return rows;
    }

    /** @brief Names a test after its row. */
    std::string ignore_refusal_name(const ::testing::TestParamInfo<IgnoreRefusal>& info)
    {
        return std::string(info.param.name);
    }

    /** @brief Checks the refusals of the segmentation row. */
    class Small_DatasetsTaskTableIgnoreRefusals : public ::testing::TestWithParam<IgnoreRefusal>
    {
    };

    /** @brief Holds a section a refusal decides on its values, and what the diagnostic names. */
    struct SectionRefusal
    {
        std::string_view name;                  /**< Names the test of the row. */
        std::string_view members;               /**< Holds the members after the root. */
        std::vector<std::string_view> needles;  /**< Lists what the diagnostic must name. */
    };

    /** @brief Returns the sections refused before a reader lists a file. */
    const std::vector<SectionRefusal>& section_refusals()
    {
        static const std::vector<SectionRefusal> rows = {
            {"AFitOutsideTheList",
             R"(, "reader": "yolo_txt", "task": "detection", "split": "train",
                "input": [4, 4], "max_objects": 2, "classes": 3, "fit": "crop")",
             {"data 'fit' is 'crop'", "'resize', 'letterbox'"}},
            {"AnUnknownReader",
             R"(, "reader": "tfrecord", "task": "detection", "input": [4, 4],
                "max_objects": 2)",
             {"data 'reader'", "'tfrecord'", "'yolo_txt'", "'image_folder'", "'idx'", "'cifar'",
              "'coco_json'", "'voc_xml'", "'mask_png'", "'csv'"}},
            {"AnUnknownTask",
             R"(, "reader": "yolo_txt", "task": "pose", "input": [4, 4])",
             {"data 'task'", "'pose'", "'detection'", "'classification'", "'binary'",
              "'multi_label'", "'regression'", "'segmentation'"}},
            {"AKeyNoSectionReads",
             R"(, "reader": "yolo_txt", "task": "detection", "input": [4, 4],
                "max_objects": 2, "batch_sizes": 4)",
             {"data", "'batch_sizes'", "'batch_size'"}},
            {"AKeyOfAnotherTask",
             R"(, "reader": "image_folder", "task": "classification", "input": [4, 4],
                "max_objects": 2)",
             {"data", "'max_objects'"}},
            {"AKeyOfAnotherReader",
             R"(, "reader": "yolo_txt", "task": "detection", "input": [4, 4],
                "max_objects": 2, "features": ["a"])",
             {"data", "'features'", "'max_objects'"}},
            {"PrefetchWithoutWorkers",
             R"(, "reader": "yolo_txt", "task": "detection", "batch_size": 2, "input": [4, 4],
                "max_objects": 2, "classes": 3, "prefetch": 4)",
             {"data 'prefetch'", "data 'workers' is 0"}},
            {"ACacheSideWithoutACache",
             R"(, "reader": "yolo_txt", "task": "detection", "batch_size": 2, "input": [4, 4],
                "max_objects": 2, "classes": 3, "cache_side": 320)",
             {"data 'cache_side'", "data 'cache'"}},
            {"ShuffleOnASectionOfRows",
             R"(, "reader": "csv", "task": "regression", "batch_size": 1,
                "file": "rows/set.csv", "features": ["a"], "targets": ["b"], "shuffle": true)",
             {"data 'shuffle'", "'regression'", "rows of numbers"}},
            {"AnInputOfThreeEntries",
             R"(, "reader": "image_folder", "task": "classification", "input": [4, 4, 3])",
             {"data 'input'", "3 entries", "[height, width]"}},
            {"AnInputExtentOfZero",
             R"(, "reader": "image_folder", "task": "classification", "input": [4, 0])",
             {"data 'input[1]' is 0"}},
            {"ZeroChannels",
             R"(, "reader": "image_folder", "task": "classification", "input": [4, 4],
                "channels": 0)",
             {"data 'channels' is 0"}},
            {"AnUnknownPipeline",
             R"(, "reader": "yolo_txt", "task": "detection", "batch_size": 2, "input": [4, 4],
                "max_objects": 2, "classes": 3, "pipeline": "mixup")",
             {"data 'pipeline' is 'mixup'", "'none'", "'default'"}},
            {"TheDefaultPipelineOnANonSquareInput",
             R"(, "reader": "yolo_txt", "task": "detection", "batch_size": 2, "input": [4, 6],
                "max_objects": 2, "classes": 3, "pipeline": "default")",
             {"square", "4 by 6"}},
        };
        return rows;
    }

    /** @brief Names a test after its row. */
    std::string section_refusal_name(const ::testing::TestParamInfo<SectionRefusal>& info)
    {
        return std::string(info.param.name);
    }

    /** @brief Builds each section the refusals decide on values, over a root no test creates. */
    class Small_DatasetsReaderSectionRefusals
        : public SectionBench,
          public ::testing::WithParamInterface<SectionRefusal>
    {
    };
} // namespace

/** @brief Tests that the reader table holds the eight readers of the module, by name. */
TEST(Small_DatasetsReaderTable, ItListsTheEightReaders)
{
    const std::vector<std::string_view> expected = {"yolo_txt", "image_folder", "idx", "cifar",
                                                    "coco_json", "voc_xml", "mask_png", "csv"};

    EXPECT_EQ(names_of(reader_table()), expected);
}

/** @brief Tests that a lookup of a reader no row names finds none. */
TEST(Small_DatasetsReaderTable, ALookupOfAReaderTheTableLacksFindsNone)
{
    EXPECT_EQ(find_reader("tfrecord"), nullptr);
}

/** @brief Tests that a row lists the keys it reads of the section, and the first four none. */
TEST_P(Small_DatasetsReaderTableKeys, ARowListsTheKeysItReads)
{
    const ReaderEntry* const row = find_reader(GetParam().reader);
    ASSERT_NE(row, nullptr);

    EXPECT_EQ(std::vector<std::string_view>(row->keys.begin(), row->keys.end()),
              GetParam().keys);
}

INSTANTIATE_TEST_SUITE_P(Rows, Small_DatasetsReaderTableKeys, ::testing::ValuesIn(keys_rows()),
                         keys_row_name);

/** @brief Tests that a row names a key of the wrong kind before it reads any file. */
TEST_P(Small_DatasetsReaderTableKeyKinds, ARowNamesAKeyOfTheWrongKind)
{
    const KeyKindRow& row = GetParam();

    ReaderRequest request;
    request.root = K_ABSENT_ROOT;
    request.section = SaTest::attributes_from(row.section);

    const ReaderEntry* const reader = find_reader(row.reader);
    ASSERT_NE(reader, nullptr);
    SA_EXPECT_FAILURE(reader->read(request), StatusCode::InvalidDescription, row.key, row.kind);
}

INSTANTIATE_TEST_SUITE_P(Rows, Small_DatasetsReaderTableKeyKinds,
                         ::testing::ValuesIn(key_kind_rows()), key_kind_row_name);

/** @brief Tests that the csv row reads its keys of the request's section. */
TEST(Medium_DatasetsReaderTableFiles, TheCSVRowReadsItsKeysOfTheSection)
{
    const Support::ScratchDirectory scratch("sushiai_test_datasets_table_keys");
    scratch.write_text("rows/set.csv", "a,b,c\n1,2,3\n");

    ReaderRequest request;
    request.root = scratch.path();
    request.section = SaTest::attributes_from(
        R"({"file": "rows/set.csv", "features": ["c"], "targets": ["a", "b"]})");

    const ReaderEntry* const csv = find_reader("csv");
    ASSERT_NE(csv, nullptr);
    SA_ASSERT_OK_AND_ASSIGN(const SampleList list, csv->read(request));
    EXPECT_EQ(list.table.values, (std::vector<float>{3.0f, 1.0f, 2.0f}));
}

/** @brief Tests that the mask_png row refuses the class count 0 it is given. */
TEST(Small_DatasetsReaderTable, TheMaskPNGRowRefusesAClassCountOfZero)
{
    ReaderRequest request;
    request.root = K_ABSENT_ROOT;
    request.section = SaTest::attributes_from(R"({"images": "pictures", "masks": "labels"})");

    const ReaderEntry* const row = find_reader("mask_png");
    ASSERT_NE(row, nullptr);
    SA_EXPECT_FAILURE(row->read(request), StatusCode::InvalidDescription, "read_mask_png",
                      "the class count is 0");
}

/** @brief Tests that the mask_png row passes the class count and the folders it is given. */
TEST(Medium_DatasetsReaderTableFiles, TheMaskPNGRowPassesTheClassCountAndTheFolders)
{
    const Support::ScratchDirectory scratch("sushiai_test_datasets_table_masks");
    scratch.write("pictures/train/a.jpg", {1});
    scratch.write("labels/train/a.png", {1});

    ReaderRequest request;
    request.root = scratch.path();
    request.class_count = 5;
    request.section = SaTest::attributes_from(R"({"images": "pictures", "masks": "labels"})");

    const ReaderEntry* const row = find_reader("mask_png");
    ASSERT_NE(row, nullptr);
    SA_ASSERT_OK_AND_ASSIGN(const SampleList list, row->read(request));
    EXPECT_EQ(list.class_count, 5);
    EXPECT_EQ(list.samples.size(), 1u);
}

namespace
{
    /** @brief Writes the annotation of one image whose one object is a difficult dog. */
    void write_difficult_dog(const Support::ScratchDirectory& scratch)
    {
        scratch.write_text("ImageSets/Main/train.txt", "a\n");
        scratch.write_text(
            "Annotations/a.xml",
            "<annotation><size><width>8</width><height>8</height></size><object><name>dog</name>"
            "<difficult>1</difficult><bndbox><xmin>1</xmin><ymin>1</ymin><xmax>4</xmax>"
            "<ymax>4</ymax></bndbox></object></annotation>");
    }
} // namespace

/** @brief Tests that the voc_xml row passes the class list and drops a difficult object. */
TEST(Medium_DatasetsReaderTableFiles, TheVOCXMLRowPassesTheClassList)
{
    const Support::ScratchDirectory scratch("sushiai_test_datasets_table_voc_classes");
    write_difficult_dog(scratch);

    ReaderRequest request;
    request.root = scratch.path();
    request.section = SaTest::attributes_from(R"({"class_names": ["cat", "dog"]})");

    const ReaderEntry* const row = find_reader("voc_xml");
    ASSERT_NE(row, nullptr);
    SA_ASSERT_OK_AND_ASSIGN(const SampleList dropped, row->read(request));
    EXPECT_EQ(dropped.class_count, 2);
    EXPECT_TRUE(dropped.samples[0].objects.is_empty());
}

/** @brief Tests that the voc_xml row passes the flag that keeps a difficult object. */
TEST(Medium_DatasetsReaderTableFiles, TheVOCXMLRowPassesTheDifficultFlag)
{
    const Support::ScratchDirectory scratch("sushiai_test_datasets_table_voc_flag");
    write_difficult_dog(scratch);

    ReaderRequest request;
    request.root = scratch.path();
    request.section = SaTest::attributes_from(
        R"({"class_names": ["cat", "dog"], "keep_difficult": true})");

    const ReaderEntry* const row = find_reader("voc_xml");
    ASSERT_NE(row, nullptr);
    SA_ASSERT_OK_AND_ASSIGN(const SampleList kept, row->read(request));
    ASSERT_EQ(kept.samples[0].objects.size(), 1u);
    EXPECT_EQ(kept.samples[0].objects.classes()[0], 1);
}

/** @brief Tests that the cifar row refuses a class count no CIFAR labels have. */
TEST(Small_DatasetsReaderTable, TheCIFARRowRefusesAnotherClassCount)
{
    ReaderRequest request;
    request.root = "sushiai_absent_cifar";
    request.class_count = 7;

    const ReaderEntry* const cifar = find_reader("cifar");
    ASSERT_NE(cifar, nullptr);
    SA_EXPECT_FAILURE(cifar->read(request), StatusCode::InvalidDescription, "7 classes",
                      "CIFAR-10", "coarse");
}

/** @brief Tests that a task row states the fit and the keys of its task. */
TEST_P(Small_DatasetsTaskTableStates, ARowStatesItsFitAndItsKeys)
{
    const TaskEntry* const row = find_task(GetParam().task);
    ASSERT_NE(row, nullptr);

    EXPECT_EQ(row->fit, GetParam().fit);
    EXPECT_EQ(std::vector<std::string_view>(row->keys.begin(), row->keys.end()),
              GetParam().keys);
}

INSTANTIATE_TEST_SUITE_P(Rows, Small_DatasetsTaskTableStates, ::testing::ValuesIn(task_rows()),
                         task_row_name);

/** @brief Tests that a task row builds the collator of its task from one request. */
TEST_P(Small_DatasetsTaskTableCollators, ARowBuildsTheCollatorOfItsTask)
{
    const TaskEntry& row = task_table()[GetParam()];
    const Config::Value section = SaTest::attributes_from(R"({"max_objects": 5})");

    // One list serves every row: two named classes, and a table of two features and a target.
    SampleList list;
    list.class_names = {"cat", "dog"};
    list.class_count = 2;
    list.table.feature_count = 2;
    list.table.target_count = 1;

    SA_ASSERT_OK_AND_ASSIGN(
        const std::unique_ptr<const Collator> collator,
        row.make_collator(section, "data", CollatorRequest{list, 2, ImageSize{4, 4}}));
    EXPECT_EQ(collator->task(), row.name);
}

INSTANTIATE_TEST_SUITE_P(Rows, Small_DatasetsTaskTableCollators,
                         ::testing::Range<std::size_t>(0, task_table().size()), task_index_name);

/** @brief Tests that the binary row reads "positive" as a class name or as an index. */
TEST_P(Small_DatasetsTaskTablePositive, TheBinaryRowReadsThePositiveClass)
{
    SampleList list;
    list.class_names = {"cat", "dog"};
    list.class_count = 2;

    const TaskEntry* const binary = find_task(K_TASK_BINARY);
    ASSERT_NE(binary, nullptr);

    const Config::Value section = SaTest::attributes_from(GetParam());
    SA_ASSERT_OK_AND_ASSIGN(
        const std::unique_ptr<const Collator> collator,
        binary->make_collator(section, "data", CollatorRequest{list, 2, ImageSize{}}));
    EXPECT_EQ(collator->task(), K_TASK_BINARY);
}

INSTANTIATE_TEST_SUITE_P(Rows, Small_DatasetsTaskTablePositive,
                         ::testing::Values(std::string_view(R"({})"),
                                           std::string_view(R"({"positive": "cat"})"),
                                           std::string_view(R"({"positive": 0})")),
                         positive_row_name);

/** @brief Tests that the binary row refuses a positive class its samples lack. */
TEST_P(Small_DatasetsTaskTablePositiveRefusals, TheBinaryRowRefusesAPositiveClassItLacks)
{
    const PositiveRefusal& row = GetParam();

    SampleList list;
    list.class_names = {"cat", "dog"};
    list.class_count = 2;

    const TaskEntry* const binary = find_task(K_TASK_BINARY);
    ASSERT_NE(binary, nullptr);

    SA_EXPECT_FAILURE(binary->make_collator(SaTest::attributes_from(row.section), "data",
                                            CollatorRequest{list, row.classes, ImageSize{}}),
                      StatusCode::InvalidDescription, row.first, row.second);
}

INSTANTIATE_TEST_SUITE_P(Rows, Small_DatasetsTaskTablePositiveRefusals,
                         ::testing::ValuesIn(positive_refusals()), positive_refusal_name);

/** @brief Tests that the segmentation row reads "ignore_index" into the shape of its target. */
TEST(Small_DatasetsTaskTable, TheSegmentationRowReadsTheIgnoreIndex)
{
    const SampleList list;
    const TaskEntry* const segmentation = find_task(K_TASK_SEGMENTATION);
    ASSERT_NE(segmentation, nullptr);

    SA_ASSERT_OK_AND_ASSIGN(
        const std::unique_ptr<const Collator> collator,
        segmentation->make_collator(SaTest::attributes_from(R"({"ignore_index": -1})"), "data",
                                    CollatorRequest{list, 3, ImageSize{4, 4}}));
    EXPECT_EQ(collator->targets()[0].sample_shape, (Shape{4, 4}));
}

/** @brief Tests that the segmentation row refuses an ignore index that is a class. */
TEST_P(Small_DatasetsTaskTableIgnoreRefusals, TheSegmentationRowRefusesAnIgnoreIndex)
{
    const SampleList list;
    const TaskEntry* const segmentation = find_task(K_TASK_SEGMENTATION);
    ASSERT_NE(segmentation, nullptr);

    SA_EXPECT_FAILURE(segmentation->make_collator(SaTest::attributes_from(GetParam().section),
                                                  "data",
                                                  CollatorRequest{list, 3, ImageSize{4, 4}}),
                      StatusCode::InvalidDescription, GetParam().first, GetParam().second);
}

INSTANTIATE_TEST_SUITE_P(Rows, Small_DatasetsTaskTableIgnoreRefusals,
                         ::testing::ValuesIn(ignore_refusals()), ignore_refusal_name);

/** @brief Tests that a section is refused on its values before a reader lists a file. */
TEST_P(Small_DatasetsReaderSectionRefusals, ASectionIsRefusedBeforeAnyFileIsRead)
{
    const SectionRefusal& row = GetParam();

    const Result<std::unique_ptr<Data::Dataset>> outcome =
        build(section_over(K_ABSENT_ROOT, row.members));

    EXPECT_EQ(outcome.status().code(), StatusCode::InvalidDescription);
    EXPECT_TRUE(names_each(outcome.status(), row.needles));
}

INSTANTIATE_TEST_SUITE_P(Rows, Small_DatasetsReaderSectionRefusals,
                         ::testing::ValuesIn(section_refusals()), section_refusal_name);

/** @brief Tests that a detection section over a YOLO split yields letterboxed batches. */
TEST_F(Medium_DatasetsReaderSection, ADetectionSectionBuildsLetterboxedBatches)
{
    const Support::ScratchDirectory scratch("sushiai_test_datasets_section_detection");
    write_yolo_split(scratch);

    SA_ASSERT_OK_AND_ASSIGN(
        const std::unique_ptr<Data::Dataset> dataset,
        build(section_of(scratch, R"(, "reader": "yolo_txt", "task": "detection",
            "split": "train", "batch_size": 2, "input": [4, 4], "max_objects": 2,
            "classes": 3)")));

    EXPECT_EQ(dataset->name(), "yolo_txt");
    EXPECT_EQ(dataset->batch_size(), 2);
    EXPECT_EQ(dataset->batch_count(), 1u);
    EXPECT_EQ(dataset->sample_shape(), (Shape{4, 4, 3}));

    const Data::TargetDescription* const boxes =
        Data::find_target(dataset->targets(), Data::K_ROLE_BOXES);
    const Data::TargetDescription* const classes =
        Data::find_target(dataset->targets(), Data::K_ROLE_CLASSES);
    ASSERT_NE(boxes, nullptr);
    ASSERT_NE(classes, nullptr);
    EXPECT_EQ(boxes->sample_shape, (Shape{2, 4}));
    EXPECT_EQ(classes->classes, 3);

    Data::Batch batch;
    ASSERT_TRUE(SaTest::status_is_ok(dataset->read(0, batch)));

    // Row 0 of each 4 by 4 by 3 input is padding of 114, which only a letterbox leaves.
    EXPECT_NEAR(batch.inputs.data_as<float>()[0], 114.0f / 255.0f, K_SCALE_TOLERANCE);

    const Data::BatchTarget* const valid = Data::find_target(batch.targets, Data::K_ROLE_VALID);
    const Data::BatchTarget* const read = Data::find_target(batch.targets, Data::K_ROLE_CLASSES);
    ASSERT_NE(valid, nullptr);
    ASSERT_NE(read, nullptr);
    EXPECT_EQ(valid->values.data_as<float>()[0], 0.0f);
    EXPECT_EQ(valid->values.data_as<float>()[2], 1.0f);
    EXPECT_EQ(read->values.data_as<std::int32_t>()[2], 2);
}

/** @brief Tests that a detection section stating "fit": "resize" stretches instead of padding. */
TEST_F(Medium_DatasetsReaderSection, ADetectionSectionStatingResizeStretchesTheImages)
{
    const Support::ScratchDirectory scratch("sushiai_test_datasets_section_resize");
    write_yolo_split(scratch);

    SA_ASSERT_OK_AND_ASSIGN(
        const std::unique_ptr<Data::Dataset> dataset,
        build(section_of(scratch, R"(, "reader": "yolo_txt", "task": "detection",
            "split": "train", "batch_size": 2, "input": [4, 4], "max_objects": 2,
            "classes": 3, "fit": "resize")")));

    Data::Batch batch;
    ASSERT_TRUE(SaTest::status_is_ok(dataset->read(0, batch)));

    // A uniform 255 image stretched to 4 by 4 leaves no padding, so row 0 is 1 as well.
    EXPECT_NEAR(batch.inputs.data_as<float>()[0], 1.0f, K_SCALE_TOLERANCE);
}

/** @brief Tests that a classification section takes the class count of its folders. */
TEST_F(Medium_DatasetsReaderSection, AClassificationSectionTakesTheClassesOfTheList)
{
    const Support::ScratchDirectory scratch("sushiai_test_datasets_section_folder");
    write_folder_split(scratch);

    SA_ASSERT_OK_AND_ASSIGN(
        const std::unique_ptr<Data::Dataset> dataset,
        build(section_of(scratch, R"(, "reader": "image_folder", "task": "classification",
            "batch_size": 2, "input": [2, 3], "channels": 1)")));

    EXPECT_EQ(dataset->sample_shape(), (Shape{2, 3, 1}));
    ASSERT_EQ(dataset->targets().size(), 1u);
    EXPECT_EQ(dataset->targets()[0].role, Data::K_ROLE_LABELS);
    EXPECT_EQ(dataset->targets()[0].classes, 2);
}

/** @brief Tests that a section may state more classes than its samples name. */
TEST_F(Medium_DatasetsReaderSection, TheClassCountMayExceedTheList)
{
    const Support::ScratchDirectory scratch("sushiai_test_datasets_section_classes_above");
    write_folder_split(scratch);

    SA_ASSERT_OK_AND_ASSIGN(
        const std::unique_ptr<Data::Dataset> dataset,
        build(section_of(scratch, R"(, "reader": "image_folder", "task": "classification",
            "batch_size": 2, "input": [2, 2], "classes": 5)")));

    EXPECT_EQ(dataset->targets()[0].classes, 5);
}

/** @brief Tests that a section may not state fewer classes than its samples name. */
TEST_F(Medium_DatasetsReaderSection, TheClassCountMayNotFallBelowTheList)
{
    const Support::ScratchDirectory scratch("sushiai_test_datasets_section_classes_below");
    write_folder_split(scratch);

    SA_EXPECT_FAILURE(
        build(section_of(scratch, R"(, "reader": "image_folder", "task": "classification",
            "batch_size": 2, "input": [2, 2], "classes": 1)")),
        StatusCode::InvalidDescription, "data 'classes' is 1", "name 2 classes");
}

/** @brief Tests that a detection section without its object slots is refused by key. */
TEST_F(Medium_DatasetsReaderSection, ADetectionSectionNeedsMaxObjects)
{
    const Support::ScratchDirectory scratch("sushiai_test_datasets_section_slots");
    write_yolo_split(scratch);

    SA_EXPECT_FAILURE(build(section_of(scratch, R"(, "reader": "yolo_txt", "task": "detection",
            "batch_size": 2, "input": [4, 4], "classes": 3)")),
        StatusCode::InvalidDescription, "data", "'max_objects'");
}

/** @brief Tests that a regression section over a CSV file yields rows and their targets. */
TEST_F(Medium_DatasetsReaderSection, ARegressionSectionReadsTheRowsOfACSVFile)
{
    const Support::ScratchDirectory scratch("sushiai_test_datasets_section_regression");
    scratch.write_text("train.csv", "a,b,y\n1,2,3\n4,5,6\n7,8,9\n");

    SA_ASSERT_OK_AND_ASSIGN(
        const std::unique_ptr<Data::Dataset> dataset,
        build(section_of(scratch, R"(, "reader": "csv", "task": "regression",
            "batch_size": 2, "targets": ["y"])")));

    EXPECT_EQ(dataset->name(), "csv");
    EXPECT_EQ(dataset->batch_count(), 1u);
    EXPECT_EQ(dataset->sample_shape(), Shape{2});
    ASSERT_EQ(dataset->targets().size(), 1u);
    EXPECT_EQ(dataset->targets()[0].role, Data::K_ROLE_VALUES);
    EXPECT_EQ(dataset->targets()[0].sample_shape, Shape{1});

    Data::Batch batch;
    ASSERT_TRUE(SaTest::status_is_ok(dataset->read(0, batch)));

    // The cells are whole numbers copied into the batch, so they compare exactly.
    const float* inputs = batch.inputs.data_as<float>();
    const float* values = batch.targets[0].values.data_as<float>();
    EXPECT_EQ(std::vector<float>(inputs, inputs + 4),
              (std::vector<float>{1.0f, 2.0f, 4.0f, 5.0f}));
    EXPECT_EQ(std::vector<float>(values, values + 2), (std::vector<float>{3.0f, 6.0f}));
}

/** @brief Tests that a segmentation section letterboxes a mask and pads with the ignore index. */
TEST_F(Medium_DatasetsReaderSection, ASegmentationSectionFitsTheMaskOfEachImage)
{
    const Support::ScratchDirectory scratch("sushiai_test_datasets_section_segmentation");
    Support::write_raw_image(scratch, "images/train/a.saim",
                             Support::uniform_image(ImageSize{2, 4}, 3, 255));
    Support::write_raw_image(scratch, "masks/train/a.png",
                             Support::gray_image(ImageSize{2, 4}, {0, 1, 2, 0, 1, 1, 2, 2}));

    SA_ASSERT_OK_AND_ASSIGN(
        const std::unique_ptr<Data::Dataset> dataset,
        build(section_of(scratch, R"(, "reader": "mask_png", "task": "segmentation",
            "batch_size": 1, "input": [4, 4], "classes": 3, "ignore_index": -1)")));

    EXPECT_EQ(dataset->sample_shape(), (Shape{4, 4, 3}));
    ASSERT_EQ(dataset->targets().size(), 1u);
    EXPECT_EQ(dataset->targets()[0].role, Data::K_ROLE_MASK);
    EXPECT_EQ(dataset->targets()[0].sample_shape, (Shape{4, 4}));
    EXPECT_EQ(dataset->targets()[0].classes, 3);

    Data::Batch batch;
    ASSERT_TRUE(SaTest::status_is_ok(dataset->read(0, batch)));

    // The 2 by 4 mask sits in rows 1 and 2, as its image does; the padding is the ignore index.
    const std::int32_t* mask = batch.targets[0].values.data_as<std::int32_t>();
    EXPECT_EQ(std::vector<std::int32_t>(mask, mask + 16),
              (std::vector<std::int32_t>{-1, -1, -1, -1, 0, 1, 2, 0, 1, 1, 2, 2, -1, -1, -1, -1}));
}

/** @brief Tests that a task of images over rows of numbers is refused. */
TEST_F(Medium_DatasetsReaderSection, AnImageTaskOverARowReaderIsRefused)
{
    const Support::ScratchDirectory scratch("sushiai_test_datasets_section_kind_rows");
    scratch.write_text("train.csv", "a,y\n1,2\n");

    SA_EXPECT_FAILURE(
        build(section_of(scratch, R"(, "reader": "csv", "task": "classification",
            "input": [2, 2])")),
        StatusCode::InvalidDescription, "data 'task' is 'classification'", "reads images",
        "the reader 'csv' lists rows of numbers");
}

/** @brief Tests that a task of rows of numbers over an image reader is refused. */
TEST_F(Medium_DatasetsReaderSection, ARowTaskOverAnImageReaderIsRefused)
{
    const Support::ScratchDirectory scratch("sushiai_test_datasets_section_kind_images");
    write_folder_split(scratch);

    SA_EXPECT_FAILURE(
        build(section_of(scratch, R"(, "reader": "image_folder", "task": "regression")")),
        StatusCode::InvalidDescription, "data 'task' is 'regression'", "reads rows of numbers",
        "the reader 'image_folder' lists none");
}

/** @brief Tests that "cache" writes one raw image file per sample. */
TEST_F(Medium_DatasetsReaderSection, ACacheHoldsOneRawImagePerSample)
{
    const Support::ScratchDirectory scratch("sushiai_test_datasets_section_cache_files");
    write_yolo_split(scratch);
    const std::filesystem::path cache = scratch.path() / "cache";

    SA_ASSERT_OK_AND_ASSIGN(
        const std::unique_ptr<Data::Dataset> dataset,
        build(section_of(scratch, R"(, "reader": "yolo_txt", "task": "detection",
            "batch_size": 2, "input": [4, 4], "max_objects": 2, "classes": 3,
            "cache": ")" + cache.generic_string() + "\"")));

    EXPECT_EQ(raw_images_in(cache), 2u);
}

/** @brief Tests that batches are read from the cache once the sources are gone. */
TEST_F(Medium_DatasetsReaderSection, ACacheServesTheBatchesWhenTheSourcesAreGone)
{
    const Support::ScratchDirectory scratch("sushiai_test_datasets_section_cache_read");
    write_yolo_split(scratch);
    const std::filesystem::path cache = scratch.path() / "cache";

    SA_ASSERT_OK_AND_ASSIGN(
        const std::unique_ptr<Data::Dataset> dataset,
        build(section_of(scratch, R"(, "reader": "yolo_txt", "task": "detection",
            "batch_size": 2, "input": [4, 4], "max_objects": 2, "classes": 3,
            "cache": ")" + cache.generic_string() + "\"")));
    std::filesystem::remove_all(scratch.path() / "images");

    Data::Batch batch;
    ASSERT_TRUE(SaTest::status_is_ok(dataset->read(0, batch)));
    EXPECT_NEAR(batch.inputs.data_as<float>()[12], 1.0f, K_SCALE_TOLERANCE);
}

/** @brief Tests that "shuffle" and "seed" read the batches in the order of that seed. */
TEST_F(Medium_DatasetsReaderSection, AShuffledSectionReadsTheBatchesInTheOrderOfItsSeed)
{
    const Support::ScratchDirectory scratch("sushiai_test_datasets_section_shuffle");
    write_numbered_split(scratch);

    SA_ASSERT_OK_AND_ASSIGN(
        const std::unique_ptr<Data::Dataset> dataset,
        build(section_of(scratch, R"(, "reader": "yolo_txt", "task": "detection",
            "batch_size": 1, "input": [4, 4], "max_objects": 1, "classes": 5,
            "shuffle": true, "seed": 7)")));
    Data::begin_epoch(*dataset, 1);

    std::vector<std::int32_t> read;
    for (std::size_t index = 0; index < dataset->batch_count(); ++index)
        read.push_back(first_class_of(*dataset, index));

    EXPECT_EQ(read, order_of(K_SHUFFLE_SEED, 1, K_NUMBERED_IMAGES));
}

/** @brief Tests that "pipeline": "default" changes the inputs the "none" pipeline reads. */
TEST_F(Medium_DatasetsReaderSection, ThePipelineKeyChangesTheInputsOfABatch)
{
    const Support::ScratchDirectory scratch("sushiai_test_datasets_section_pipeline");
    write_yolo_split(scratch);
    const std::string members = R"(, "reader": "yolo_txt", "task": "detection",
        "batch_size": 2, "input": [16, 16], "max_objects": 8, "classes": 3, "pipeline": ")";

    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<Data::Dataset> fitted,
                            build(section_of(scratch, members + "none\"")));
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<Data::Dataset> augmented,
                            build(section_of(scratch, members + "default\"")));

    EXPECT_EQ(augmented->sample_shape(), (Shape{16, 16, 3}));
    EXPECT_NE(inputs_of(*augmented, 0), inputs_of(*fitted, 0));
}

/** @brief Tests that "workers" puts the dataset behind a loader. */
TEST_F(Medium_DatasetsReaderSection, WorkersPutTheDatasetBehindALoader)
{
    const Support::ScratchDirectory scratch("sushiai_test_datasets_section_workers_kind");
    write_yolo_split(scratch);

    const std::string_view members = R"(, "reader": "yolo_txt", "task": "detection",
        "batch_size": 1, "input": [4, 4], "max_objects": 2, "classes": 3)";

    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<Data::Dataset> direct,
                            build(section_of(scratch, members)));
    SA_ASSERT_OK_AND_ASSIGN(
        const std::unique_ptr<Data::Dataset> threaded,
        build(section_of(scratch, std::string(members) + R"(, "workers": 2, "prefetch": 2)")));

    EXPECT_NE(dynamic_cast<const Data::ThreadedLoader*>(threaded.get()), nullptr);
    EXPECT_EQ(dynamic_cast<const Data::ThreadedLoader*>(direct.get()), nullptr);
}

/** @brief Tests that the workers read the same batches the dataset behind them reads. */
TEST_F(Medium_DatasetsReaderSection, WorkersReadTheSameBatchesAhead)
{
    const Support::ScratchDirectory scratch("sushiai_test_datasets_section_workers");
    write_yolo_split(scratch);

    const std::string_view members = R"(, "reader": "yolo_txt", "task": "detection",
        "batch_size": 1, "input": [4, 4], "max_objects": 2, "classes": 3)";

    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<Data::Dataset> direct,
                            build(section_of(scratch, members)));
    SA_ASSERT_OK_AND_ASSIGN(
        const std::unique_ptr<Data::Dataset> threaded,
        build(section_of(scratch, std::string(members) + R"(, "workers": 2, "prefetch": 2)")));

    ASSERT_EQ(threaded->batch_count(), direct->batch_count());
    EXPECT_EQ(inputs_of_every_batch(*threaded), inputs_of_every_batch(*direct));
}

/** @brief Tests that a segmentation section over polygons paints them into the mask. */
TEST_F(Medium_DatasetsReaderSection, ASegmentationSectionPaintsThePolygonsOfALabelFile)
{
    const Support::ScratchDirectory scratch("sushiai_test_datasets_section_polygons");
    Support::write_raw_image(scratch, "images/train/a.saim",
                             Support::uniform_image(ImageSize{4, 4}, 3, 255));
    scratch.write_text("labels/train/a.txt", "1 0.0 0.0 0.5 0.0 0.5 0.5 0.0 0.5\n");

    SA_ASSERT_OK_AND_ASSIGN(
        const std::unique_ptr<Data::Dataset> dataset,
        build(section_of(scratch, R"(, "reader": "yolo_txt", "task": "segmentation",
            "batch_size": 1, "input": [4, 4], "classes": 3, "ignore_index": -1,
            "background": 2)")));

    Data::Batch batch;
    ASSERT_TRUE(SaTest::status_is_ok(dataset->read(0, batch)));

    // The polygon is the top left quarter; every other pixel takes the background.
    const std::int32_t* mask = batch.targets[0].values.data_as<std::int32_t>();
    EXPECT_EQ(std::vector<std::int32_t>(mask, mask + 16),
              (std::vector<std::int32_t>{1, 1, 2, 2, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2}));
}
