/**************************************************************************/
/* test_ops.cpp                                                           */
/**************************************************************************/
/*                          This file is part of:                         */
/*                                 SushiAI                                */
/**************************************************************************/

// Device-free. The uniqueness proof already runs as a static_assert inside
// ops.hpp; these tests cover what that assert cannot - that the table is
// actually complete, and that the reverse lookup behaves.

#include <set>
#include <string_view>
#include <gtest/gtest.h>
#include <SushiAI/core/ops.hpp>

namespace Ops = SushiAI::Ops;

TEST(Unit_Ops, EveryIdentityHashesDistinctly)
{
    std::set<std::uint32_t> seen;
    for (const Ops::OpName& entry : Ops::ALL)
        EXPECT_TRUE(seen.insert(entry.id.value).second)
            << "duplicate OpID for " << entry.name;

    EXPECT_EQ(seen.size(), Ops::ALL.size());
}

TEST(Unit_Ops, EveryNameCarriesTheLibraryPrefix)
{
    // The prefix is what keeps a SushiAI identity from colliding with
    // SushiBLAS's own (blas.lvl3.gemm and friends) in the profiler's report.
    //
    // rfind(prefix, 0) rather than starts_with(), which is C++20: it searches
    // backwards from position 0, so it can only match at position 0, which is
    // the same question asked with the vocabulary C++17 has.
    constexpr std::string_view PREFIX = "sushiai.";
    for (const Ops::OpName& entry : Ops::ALL)
        EXPECT_EQ(entry.name.rfind(PREFIX, 0), 0u) << entry.name;
}

TEST(Unit_Ops, NameOfRoundTripsEveryIdentity)
{
    for (const Ops::OpName& entry : Ops::ALL)
        EXPECT_EQ(Ops::name_of(entry.id), entry.name);
}

TEST(Unit_Ops, NameOfReportsUnknownForAForeignIdentity)
{
    const SushiAI::OpID foreign(0xDEADBEEFu);
    EXPECT_EQ(Ops::name_of(foreign), "sushiai.unknown");
}

TEST(Unit_Ops, TheTableContainsEveryDeclaredIdentity)
{
    // A new op declared but left out of ALL would escape the uniqueness
    // static_assert entirely, so the membership is checked by name here.
    const std::array expected = {
        std::string_view("sushiai.leaf"),
        std::string_view("sushiai.accumulate"),
        std::string_view("sushiai.cast"),
        std::string_view("sushiai.matmul"),
        std::string_view("sushiai.bias_add"),
        std::string_view("sushiai.add"),
        std::string_view("sushiai.sub"),
        std::string_view("sushiai.mul"),
        std::string_view("sushiai.scale"),
        std::string_view("sushiai.dynamic_scale"),
        std::string_view("sushiai.exp"),
        std::string_view("sushiai.zero"),
        std::string_view("sushiai.relu"),
        std::string_view("sushiai.gelu"),
        std::string_view("sushiai.sigmoid"),
        std::string_view("sushiai.tanh"),
        std::string_view("sushiai.relu_backward"),
        std::string_view("sushiai.gelu_backward"),
        std::string_view("sushiai.sigmoid_backward"),
        std::string_view("sushiai.tanh_backward"),
        std::string_view("sushiai.log_softmax"),
        std::string_view("sushiai.nll_loss"),
        std::string_view("sushiai.sum_rows"),
        std::string_view("sushiai.optim.adamw"),
        std::string_view("sushiai.optim.sgd"),
        std::string_view("sushiai.fused.gemm_bias"),
        std::string_view("sushiai.fused.gemm_bias_relu"),
        std::string_view("sushiai.fused.gemm_bias_sigmoid"),
        std::string_view("sushiai.fused.gemm_bias_tanh"),
        std::string_view("sushiai.fused.exp_sub_scale"),
    };

    EXPECT_EQ(Ops::ALL.size(), expected.size());

    for (std::string_view name : expected)
    {
        bool found = false;
        for (const Ops::OpName& entry : Ops::ALL)
            if (entry.name == name) { found = true; break; }
        EXPECT_TRUE(found) << name << " is missing from Ops::ALL";
    }
}
