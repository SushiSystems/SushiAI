/**************************************************************************/
/* weight_fixtures.hpp                                                    */
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
 * @file weight_fixtures.hpp
 * @brief Declares the in-memory weight files of one small model, in two containers, for the io tests.
 * @author Mustafa Garip
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>
#include "archive_bytes.hpp"

namespace SushiAI
{
    namespace Test
    {
        /** @brief Returns the 6 elements of the weight `fc.weight`, of shape [2, 3]. */
        inline std::vector<float> fixture_weight()
        {
            return {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
        }

        /** @brief Returns the 2 elements of the bias `fc.bias`, of shape [2]. */
        inline std::vector<float> fixture_bias()
        {
            return {0.5f, -0.5f};
        }

        /** @brief Returns the bytes of @p values, each as 4 little-endian bytes. */
        inline std::string float_bytes(const std::vector<float>& values)
        {
            std::string out;
            for (const float value : values)
            {
                std::uint32_t bits = 0;
                std::memcpy(&bits, &value, sizeof(bits));
                put_le(out, bits, 4);
            }
            return out;
        }

        /** @brief Returns the pickle text of BINUNICODE @p text, which is under 256 bytes. */
        inline std::string unicode_pickle(const std::string& text)
        {
            std::string out = "X";
            put_le(out, text.size(), 4);
            return out + text;
        }

        /** @brief Returns the pickle of one tensor value of @p storage_type over storage @p key. */
        inline std::string tensor_pickle(const std::string& storage_type, char key,
                                         const std::string& size, const std::string& stride,
                                         char count)
        {
            // The pickle of _rebuild_tensor_v2: storage persistent id, offset 0, size, stride.
            std::string out = "ctorch._utils\n_rebuild_tensor_v2\n((";
            out += unicode_pickle("storage");
            out += "ctorch\n" + storage_type + "\n";
            out += unicode_pickle(std::string(1, key));
            out += unicode_pickle("cpu");
            out += "K" + std::string(1, count) + "tQK" + std::string(1, '\0');
            out += size + stride + std::string(1, '\x89');
            out += "ccollections\nOrderedDict\n)RtR";
            return out;
        }

        /**
         * @brief Returns a PyTorch file, as `torch.save` writes one, whose root is
         *        {"state_dict": {"net.fc.weight", "net.fc.bias", "net.bn.num_batches_tracked"}}.
         *
         * The weight is fixture_weight() as a [2, 3] float32 tensor, the bias fixture_bias() as a
         * [2] float32 tensor and the counter one int64 element, 5, as a [1] tensor.
         */
        inline std::vector<std::byte> torch_fixture()
        {
            const std::string pickle =
                std::string("\x80\x02}") + unicode_pickle("state_dict") + "}" +
                unicode_pickle("net.fc.weight") +
                tensor_pickle("FloatStorage", '0', std::string("K\x02K\x03\x86", 5),
                              std::string("K\x03K\x01\x86", 5), '\x06') +
                "s" + unicode_pickle("net.fc.bias") +
                tensor_pickle("FloatStorage", '1', std::string("K\x02\x85", 3),
                              std::string("K\x01\x85", 3), '\x02') +
                "s" + unicode_pickle("net.bn.num_batches_tracked") +
                tensor_pickle("LongStorage", '2', std::string("K\x01\x85", 3),
                              std::string("K\x01\x85", 3), '\x01') +
                "ss.";
            std::string counter;
            put_le(counter, 5, 8);
            return zip_of({
                {"a/data.pkl", pickle},
                {"a/byteorder", "little"},
                {"a/data/0", float_bytes(fixture_weight())},
                {"a/data/1", float_bytes(fixture_bias())},
                {"a/data/2", counter},
            });
        }

        /**
         * @brief Returns a safetensors file of "net.fc.weight" and "net.fc.bias", the tensors of
         *        torch_fixture() but for the counter, which the container cannot state.
         */
        inline std::vector<std::byte> safetensors_fixture()
        {
            const std::string header =
                R"({"net.fc.weight":{"dtype":"F32","shape":[2,3],"data_offsets":[0,24]},)"
                R"("net.fc.bias":{"dtype":"F32","shape":[2],"data_offsets":[24,32]}})";
            std::string file;
            put_le(file, header.size(), 8);
            file += header;
            file += float_bytes(fixture_weight());
            file += float_bytes(fixture_bias());
            return bytes_of(file);
        }

    } // namespace Test
} // namespace SushiAI
