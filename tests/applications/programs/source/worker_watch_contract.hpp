/**************************************************************************/
/* worker_watch_contract.hpp                                              */
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
 * @file worker_watch_contract.hpp
 * @brief Declares WorkerWatchContract, the promises every WorkerWatch keeps.
 * @author Mustafa Garip
 */

#pragma once

#include <algorithm>
#include <chrono>
#include <string>
#include <vector>
#include <gtest/gtest.h>

#include "master_cluster.hpp"

namespace SushiAI
{
    namespace Test
    {
        /**
         * @brief Holds the promises of the WorkerWatch port.
         *
         * Each function takes a watch over exactly two attached workers, named "left" and
         * "right", so the in-memory adapter and the real one run the same cases.
         */
        class WorkerWatchContract
        {
            public:
                /** @brief Checks that the watch lists both attached workers by name. */
                static void expect_lists_the_attached_workers(Apps::WorkerWatch& watch)
                {
                    std::vector<std::string> names = watch.attached_names();
                    std::sort(names.begin(), names.end());

                    ASSERT_EQ(names.size(), 2u);
                    EXPECT_EQ(names[0], "left");
                    EXPECT_EQ(names[1], "right");
                }

                /** @brief Checks that a count already attached is reached at once. */
                static void expect_a_count_already_attached_is_reached(Apps::WorkerWatch& watch)
                {
                    EXPECT_TRUE(watch.wait_for_count(2, std::chrono::milliseconds(20)));
                }

                /** @brief Checks that a count above the attached ones is not reached. */
                static void expect_a_count_above_the_attached_is_not_reached(
                    Apps::WorkerWatch& watch)
                {
                    EXPECT_FALSE(watch.wait_for_count(3, std::chrono::milliseconds(20)));
                }

                /** @brief Checks that a wait that fails lets its whole bound pass on the clock. */
                static void expect_a_failed_wait_passes_its_bound_on_the_clock(
                    Apps::WorkerWatch& watch)
                {
                    const std::chrono::milliseconds bound(20);
                    const auto before = watch.now();

                    EXPECT_FALSE(watch.wait_for_count(3, bound));

                    EXPECT_GE(watch.now() - before, bound);
                }

                /** @brief Checks that the clock never goes back. */
                static void expect_the_clock_never_goes_back(Apps::WorkerWatch& watch)
                {
                    const auto first = watch.now();

                    const auto second = watch.now();

                    EXPECT_GE(second, first);
                }
        };
    } // namespace Test
} // namespace SushiAI
