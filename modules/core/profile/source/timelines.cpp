/**************************************************************************/
/* timelines.cpp                                                          */
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
 * @file timelines.cpp
 * @brief Defines mark_period, timeline_name and TimelineBinding over one atomic slot per timeline.
 * @author Mustafa Garip
 * @see modules/core/profile/README.md
 */

#include <SushiAI/profile/timelines.hpp>

#include <array>
#include <atomic>
#include <cstddef>
#include <string_view>

#include <SushiAI/profile/profile_config.hpp>
#include <SushiAI/profile/timeline_binding.hpp>

#if SA_ENABLE_PROFILER
#include <SushiProfiler/periods/period_timeline.hpp>
#include <SushiProfiler/session/session.hpp>

#include <SushiAI/failure/check.hpp>
#endif

namespace SushiAI
{
    namespace Profile
    {
        namespace
        {
            /** @brief Holds how many timelines there are; the last enumerator plus one. */
            constexpr std::size_t K_TIMELINE_COUNT = 2;

            /** @brief Holds the names of the timelines, indexed by Timeline. */
            constexpr std::array<std::string_view, K_TIMELINE_COUNT> K_TIMELINE_NAMES = {
                "train.step",
                "inference.batch",
            };

            /** @brief Returns the slot index of @p timeline. */
            constexpr std::size_t slot_of(Timeline timeline) noexcept
            {
                return static_cast<std::size_t>(timeline);
            }
        } // namespace

        std::string_view timeline_name(Timeline timeline) noexcept
        {
            return K_TIMELINE_NAMES[slot_of(timeline)];
        }

#if SA_ENABLE_PROFILER
        namespace
        {
            /** @brief Holds the bound timeline of each Timeline, or null while unbound. */
            std::array<std::atomic<SushiProfiler::PeriodTimeline*>, K_TIMELINE_COUNT>
                bound_timelines{};
        } // namespace

        void mark_period(Timeline timeline) noexcept
        {
            SushiProfiler::PeriodTimeline* bound =
                bound_timelines[slot_of(timeline)].load(std::memory_order_acquire);
            if (bound != nullptr)
                static_cast<void>(bound->mark());
        }

        TimelineBinding::TimelineBinding(SushiProfiler::Session& session)
        {
            for (std::size_t slot = 0; slot < K_TIMELINE_COUNT; ++slot)
            {
                SA_ASSERT(bound_timelines[slot].load(std::memory_order_acquire) == nullptr);
                bound_timelines[slot].store(&session.timeline(K_TIMELINE_NAMES[slot]),
                                            std::memory_order_release);
            }
        }

        TimelineBinding::~TimelineBinding()
        {
            for (std::atomic<SushiProfiler::PeriodTimeline*>& slot : bound_timelines)
                slot.store(nullptr, std::memory_order_release);
        }
#else
        void mark_period(Timeline) noexcept
        {

        }

        TimelineBinding::TimelineBinding(SushiProfiler::Session&)
        {

        }

        TimelineBinding::~TimelineBinding()
        {

        }
#endif
    } // namespace Profile
} // namespace SushiAI
