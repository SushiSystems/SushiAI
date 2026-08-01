#include <cstring>
#include <gtest/gtest.h>
#include <SushiAI/version.hpp>

TEST(Unit_Version, MatchesEncodedMacro)
{
    EXPECT_EQ(SUSHIAI_VERSION, SUSHIAI_VERSION_ENCODE(SUSHIAI_VERSION_MAJOR,
                                                       SUSHIAI_VERSION_MINOR,
                                                       SUSHIAI_VERSION_PATCH));
}

TEST(Unit_Version, RuntimeStringMatchesCompileTimeMacro)
{
    EXPECT_STREQ(SushiAI::version_string(), SUSHIAI_VERSION_STRING);
}
