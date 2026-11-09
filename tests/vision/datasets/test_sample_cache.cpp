/**************************************************************************/
/* test_sample_cache.cpp                                                  */
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
 * @file test_sample_cache.cpp
 * @brief Tests the raw cache of a sample list: its names, its writer, its use and what is stale.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include "datasets_test_support.hpp"
#include <SushiAI/failure/status.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/vision/codecs/decode_options.hpp>
#include <SushiAI/vision/datasets/byte_sink.hpp>
#include <SushiAI/vision/datasets/byte_source.hpp>
#include <SushiAI/vision/datasets/load_sample_image.hpp>
#include <SushiAI/vision/datasets/sample.hpp>
#include <SushiAI/vision/datasets/sample_cache.hpp>
#include <SushiAI/vision/imaging/image.hpp>
#include <SushiAI/vision/imaging/raw_cache.hpp>

using namespace SushiAI;
using namespace SushiAI::Vision::Datasets;
using Vision::Imaging::Image;
using Vision::Imaging::ImageSize;

namespace
{
    /** @brief Returns an RGB image whose bytes rise from a first value. */
    Image rising_image(ImageSize size, std::uint8_t first)
    {
        Image image(size, 3);
        for (std::size_t i = 0; i < image.byte_count(); ++i)
            image.data()[i] = static_cast<std::uint8_t>(first + i);

        return image;
    }

    /** @brief Returns a list of one sample per file, each file being one image. */
    SampleList list_of(const std::vector<std::filesystem::path>& files)
    {
        SampleList list;
        for (const std::filesystem::path& file : files)
        {
            Sample sample;
            sample.file = static_cast<std::uint32_t>(list.files.size());
            list.files.push_back(file);
            list.samples.push_back(sample);
        }

        return list;
    }

    /** @brief Returns the options of a cache below a scratch directory. */
    SampleCacheOptions options_of(const Support::ScratchDirectory& scratch)
    {
        SampleCacheOptions options;
        options.directory = scratch.path() / "cache";
        return options;
    }

    /** @brief Returns the byte count of the raw image file of an RGB image of a size. */
    std::uint64_t raw_bytes(ImageSize size)
    {
        return Vision::Imaging::K_RAW_IMAGE_HEADER_BYTES +
               static_cast<std::uint64_t>(size.height) * size.width * 3;
    }
    /** @brief Returns the bytes of an image, row by row. */
    std::vector<std::uint8_t> bytes_of(const Image& image)
    {
        return std::vector<std::uint8_t>(image.data(), image.data() + image.byte_count());
    }

    /** @brief Returns a source that holds the four bytes "abcd" as images/a.jpg. */
    MemorySource memory_with_one_image()
    {
        MemorySource files;
        files.add_file("images/a.jpg", "abcd");
        return files;
    }

    /** @brief Holds the options and the list of a write_sample_cache the module refuses. */
    struct CacheRefusal
    {
        /** @brief Names the row. */
        const char* name;

        /** @brief Holds the directory of the cache, or none. */
        const char* directory;

        /** @brief Holds the longest side. */
        std::int32_t longest_side;

        /** @brief Tells whether the one sample names a mask. */
        bool with_mask;

        /** @brief Holds a text the refusal carries. */
        const char* message;

        /** @brief Holds another text the refusal carries, or none. */
        const char* detail;
    };

    /** @brief Holds one refused write per row. */
    class Small_DatasetsSampleCacheRefusal : public ::testing::TestWithParam<CacheRefusal>
    {
    };
} // namespace

/** @brief Tests that the first pass writes every image and the second writes none. */
TEST(Medium_DatasetsSampleCache, TheFirstPassWritesAndTheSecondReuses)
{
    const Support::ScratchDirectory scratch("sushiai_test_datasets_cache_passes");
    const SampleList list = list_of({
        Support::write_raw_image(scratch, "images/a.jpg", rising_image(ImageSize{2, 3}, 0)),
        Support::write_raw_image(scratch, "images/b.jpg", rising_image(ImageSize{4, 2}, 9)),
    });
    const SampleCacheOptions options = options_of(scratch);

    SA_ASSERT_OK_AND_ASSIGN(const SampleCacheReport first, write_sample_cache(list, options));
    EXPECT_EQ(first.written, 2u);
    EXPECT_EQ(first.reused, 0u);
    EXPECT_EQ(first.bytes_written, raw_bytes(ImageSize{2, 3}) + raw_bytes(ImageSize{4, 2}));

    SA_ASSERT_OK_AND_ASSIGN(const SampleCacheReport second, write_sample_cache(list, options));
    EXPECT_EQ(second.written, 0u);
    EXPECT_EQ(second.reused, 2u);
    EXPECT_EQ(second.bytes_written, 0u);
}

/** @brief Tests that a list pointed at the cache loads the pixels of its images from it. */
TEST(Medium_DatasetsSampleCache, AListPointedAtTheCacheLoadsTheSamePixels)
{
    const Support::ScratchDirectory scratch("sushiai_test_datasets_cache_use");
    const Image original = rising_image(ImageSize{2, 3}, 40);
    const SampleList list =
        list_of({Support::write_raw_image(scratch, "images/a.jpg", original)});
    const SampleCacheOptions options = options_of(scratch);

    SA_EXPECT_OK(write_sample_cache(list, options));
    SA_ASSERT_OK_AND_ASSIGN(const SampleList cached, use_sample_cache(list, options));

    ASSERT_EQ(cached.files.size(), 1u);
    EXPECT_EQ(cached.files[0].parent_path(), options.directory);
    EXPECT_EQ(cached.files[0].extension(), ".saim");
    EXPECT_EQ(cached.files[0].filename().string().rfind("a-", 0), 0u);

    Image loaded;
    SA_EXPECT_OK(load_sample_image(cached, 0, Vision::Codecs::DecodeOptions{}, loaded));
    ASSERT_EQ(loaded.size(), original.size());
    ASSERT_EQ(loaded.channels(), 3);
    EXPECT_EQ(bytes_of(loaded), bytes_of(original));
}

/** @brief Tests that a longest side reduces a larger image and keeps a smaller one. */
TEST(Medium_DatasetsSampleCache, ALongestSideReducesOnlyWhatExceedsIt)
{
    const Support::ScratchDirectory scratch("sushiai_test_datasets_cache_side");
    const SampleList list = list_of({
        Support::write_raw_image(scratch, "images/wide.jpg",
                                 Support::uniform_image(ImageSize{4, 8}, 3, 200)),
        Support::write_raw_image(scratch, "images/small.jpg",
                                 Support::uniform_image(ImageSize{2, 2}, 3, 100)),
    });

    SampleCacheOptions options = options_of(scratch);
    options.longest_side = 4;

    SA_EXPECT_OK(write_sample_cache(list, options));
    SA_ASSERT_OK_AND_ASSIGN(const SampleList cached, use_sample_cache(list, options));

    Image loaded;
    SA_EXPECT_OK(load_sample_image(cached, 0, Vision::Codecs::DecodeOptions{}, loaded));
    EXPECT_EQ(loaded.height(), 2);
    EXPECT_EQ(loaded.width(), 4);
    EXPECT_EQ(loaded.data()[0], 200);

    SA_EXPECT_OK(load_sample_image(cached, 1, Vision::Codecs::DecodeOptions{}, loaded));
    EXPECT_EQ(loaded.height(), 2);
    EXPECT_EQ(loaded.width(), 2);
}

/** @brief Tests that an image that changed has no cache file until it is cached again. */
TEST(Medium_DatasetsSampleCache, AnImageThatChangedIsStale)
{
    const Support::ScratchDirectory scratch("sushiai_test_datasets_cache_stale");
    const SampleList list = list_of({Support::write_raw_image(
        scratch, "images/a.jpg", Support::uniform_image(ImageSize{2, 2}, 3, 1))});
    const SampleCacheOptions options = options_of(scratch);

    SA_EXPECT_OK(write_sample_cache(list, options));

    // Another byte count is another cache file, whatever the two modification times are.
    Support::write_raw_image(scratch, "images/a.jpg",
                             Support::uniform_image(ImageSize{2, 3}, 3, 2));
    SA_EXPECT_FAILURE(use_sample_cache(list, options), StatusCode::FileNotFound,
                      "images/a.jpg", "sample cache", "changed since");

    SA_ASSERT_OK_AND_ASSIGN(const SampleCacheReport again, write_sample_cache(list, options));
    EXPECT_EQ(again.written, 1u);
    EXPECT_EQ(again.reused, 0u);

    SA_ASSERT_OK_AND_ASSIGN(const SampleList cached, use_sample_cache(list, options));
    Image loaded;
    SA_EXPECT_OK(load_sample_image(cached, 0, Vision::Codecs::DecodeOptions{}, loaded));
    EXPECT_EQ(loaded.width(), 3);
    EXPECT_EQ(loaded.data()[0], 2);
}

/** @brief Tests that a file of stored pixels and a mask keep their paths and are not cached. */
TEST(Medium_DatasetsSampleCache, StoredPixelsAndMasksAreLeftAlone)
{
    const Support::ScratchDirectory scratch("sushiai_test_datasets_cache_alone");

    SampleList list;
    list.files = {
        scratch.write("pixels.bin", Support::rising_bytes(4, 0)),
        Support::write_raw_image(scratch, "images/a.jpg",
                                 Support::uniform_image(ImageSize{2, 2}, 3, 7)),
        Support::write_raw_image(scratch, "masks/a.png",
                                 Support::uniform_image(ImageSize{2, 2}, 1, 0)),
    };

    Sample stored;
    stored.file = 0;
    stored.stored = StoredPixels{0, ImageSize{2, 2}, 1, PlaneLayout::Interleaved};
    Sample masked;
    masked.file = 1;
    masked.mask = 2;
    list.samples = {stored, masked};

    const SampleCacheOptions options = options_of(scratch);
    SA_ASSERT_OK_AND_ASSIGN(const SampleCacheReport report, write_sample_cache(list, options));
    EXPECT_EQ(report.written, 1u);

    SA_ASSERT_OK_AND_ASSIGN(const SampleList cached, use_sample_cache(list, options));
    EXPECT_EQ(cached.files[0], list.files[0]);
    EXPECT_NE(cached.files[1], list.files[1]);
    EXPECT_EQ(cached.files[2], list.files[2]);
}

/** @brief Tests that the name of a cache file follows the source and the options alone. */
TEST(Small_DatasetsSampleCache, ANameFollowsTheSourceAndTheOptions)
{
    const MemorySource files = memory_with_one_image();
    const std::filesystem::path source = "images/a.jpg";

    SampleCacheOptions options;
    options.directory = "cache";
    SA_ASSERT_OK_AND_ASSIGN(const std::filesystem::path first,
                            sample_cache_path(source, options, &files));
    SA_ASSERT_OK_AND_ASSIGN(const std::filesystem::path again,
                            sample_cache_path(source, options, &files));
    EXPECT_EQ(first, again);
    EXPECT_EQ(first.parent_path(), options.directory);

    // "a-", sixteen digits of the hash and ".saim".
    EXPECT_EQ(first.filename().string().size(), 2u + 16u + 5u);

    options.longest_side = 320;
    SA_ASSERT_OK_AND_ASSIGN(const std::filesystem::path reduced,
                            sample_cache_path(source, options, &files));
    EXPECT_NE(first, reduced);

    SA_EXPECT_FAILURE(sample_cache_path("images/gone.jpg", options, &files),
                      StatusCode::FileNotFound, "gone.jpg");
}

/**
 * @brief Tests that the digits of a cache name are the 64-bit FNV-1a hash of the key text, so a
 *        cache written before the hash moved into its own module is still found.
 *
 * A memory source gives every file the stamp 0, so the key of images/a.jpg of four bytes at a
 * longest side of 320 is "images/a.jpg|4|0|320|0". Its hash, ec970685015c5c41, was computed in
 * Python from the offset basis 14695981039346656037 and the prime 1099511628211.
 */
TEST(Small_DatasetsSampleCache, TheNameIsTheFNV1aHashOfTheKeyText)
{
    const MemorySource files = memory_with_one_image();

    SampleCacheOptions options;
    options.directory = "cache";
    options.longest_side = 320;
    SA_ASSERT_OK_AND_ASSIGN(const std::filesystem::path path,
                            sample_cache_path("images/a.jpg", options, &files));

    EXPECT_EQ(path.filename().string(), "a-ec970685015c5c41.saim");
}

/** @brief Tests that a list is pointed at the cache file of its image when the source holds one. */
TEST(Small_DatasetsSampleCache, AListIsPointedAtTheCacheFileTheSourceHolds)
{
    MemorySource files = memory_with_one_image();
    SampleCacheOptions options;
    options.directory = "cache";
    SA_ASSERT_OK_AND_ASSIGN(const std::filesystem::path target,
                            sample_cache_path("images/a.jpg", options, &files));
    files.add_file(target, "cached");

    SA_ASSERT_OK_AND_ASSIGN(const SampleList cached,
                            use_sample_cache(list_of({"images/a.jpg"}), options, &files));

    ASSERT_EQ(cached.files.size(), 1u);
    EXPECT_EQ(cached.files[0], target);
}

/** @brief Tests that a list whose image has no cache file in the source is refused. */
TEST(Small_DatasetsSampleCache, AnImageWithoutACacheFileIsRefused)
{
    const MemorySource files = memory_with_one_image();
    SampleCacheOptions options;
    options.directory = "cache";

    SA_EXPECT_FAILURE(use_sample_cache(list_of({"images/a.jpg"}), options, &files),
                      StatusCode::FileNotFound, "images/a.jpg", "sample cache", "changed since");
}

/** @brief Tests that a pass over images whose cache files exist writes none and reuses each. */
TEST(Small_DatasetsSampleCache, ACacheFileTheSourceHoldsIsReused)
{
    MemorySource files = memory_with_one_image();
    SampleCacheOptions options;
    options.directory = "cache";
    SA_ASSERT_OK_AND_ASSIGN(const std::filesystem::path target,
                            sample_cache_path("images/a.jpg", options, &files));
    files.add_file(target, "cached");

    SA_ASSERT_OK_AND_ASSIGN(const SampleCacheReport report,
                            write_sample_cache(list_of({"images/a.jpg"}), options, DatasetStorage{&files}));

    EXPECT_EQ(report.written, 0u);
    EXPECT_EQ(report.reused, 1u);
    EXPECT_EQ(report.bytes_written, 0u);
}

/** @brief Tests that a write refuses a cache with no directory, a side out of range or over masks. */
TEST_P(Small_DatasetsSampleCacheRefusal, RefusesOptionsItCannotServe)
{
    const CacheRefusal& row = GetParam();
    SampleList list = list_of({"images/a.jpg", "masks/a.png"});
    list.samples.resize(1);
    if (row.with_mask)
        list.samples[0].mask = 1;

    SampleCacheOptions options;
    options.directory = row.directory;
    options.longest_side = row.longest_side;

    SA_EXPECT_FAILURE(write_sample_cache(list, options), StatusCode::InvalidDescription,
                      row.message, row.detail);
}

INSTANTIATE_TEST_SUITE_P(
    Rows, Small_DatasetsSampleCacheRefusal,
    ::testing::Values(CacheRefusal{"NoDirectory", "", 0, false, "no directory", ""},
                      CacheRefusal{"ANegativeLongestSide", "cache", -1, false,
                                   "a longest side of -1", ""},
                      CacheRefusal{"ALongestSideOverMasks", "cache", 1, true,
                                   "a longest side of 1", "masks"}),
    [](const ::testing::TestParamInfo<CacheRefusal>& info) { return std::string(info.param.name); });
