# tests/core/graph/lowering/test_all.cpp


Suites: `Small_LoweringTable`, `Small_UnloweredIdentity`.

Tests the table that maps each operation identity to its lowering. The lowering is eight domain
tables under `modules/core/graph/include/SushiAI/graph/lowering/`, which `all.hpp` joins into `K_LOWERINGS`. The
identities that have no row are listed in `K_UNLOWERED`, in `unlowered.hpp`.

The file needs no device. It reads the constants and the constexpr helpers of `all.hpp`. The
`static_assert`s in `all.hpp` prove the same counts at build time. The row count of each domain
table is asserted by the test file of that domain.

## EveryIdentityIsLoweredOrListedAsUnlowered

Each entry of `Ops::K_ALL` has exactly one row, or it is listed in `K_UNLOWERED` and has none, and
never both. The rows plus the unlowered identities add up to the operation table's size, so no row
names an identity the operation table lacks and no identity has two rows.

## TheUnloweredIdentitiesHaveNoRow

One row per identity of `K_UNLOWERED`: `find_lowering` returns no function for it and no row of
`K_LOWERINGS` names it.
