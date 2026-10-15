# tests/core/graph/lowering/test_all.cpp


Suites: `Unit_LoweringTable`.

Tests the table that maps each operation identity to its lowering. The lowering is eight domain
tables under `modules/core/graph/include/SushiAI/graph/lowering/`, which `all.hpp` joins into `K_LOWERINGS`. The
identities that have no row are listed in `K_UNLOWERED`, in `unlowered.hpp`.

The file needs no device. It reads the constants and the constexpr helpers of `all.hpp`. The
`static_assert`s in `all.hpp` prove the same counts at build time. These cases name the identity
or the domain whose count is wrong, so a change to one table reads as one line in the report.

## EveryIdentityIsLoweredOrListedAsUnlowered

Each entry of `Ops::K_ALL` has exactly one row, or it is listed in `K_UNLOWERED` and has none.

## TheUnloweredIdentitiesHaveNoRow

`K_UNLOWERED` holds four identities, and none of them has a row.

## NoOperationHasTwoRows

No identity appears in two rows of `K_LOWERINGS`.

## TheCombinedTableCoversTheOperationTable

Every identity is accounted for, and the rows plus the unlowered identities add up to the
operation table's size.

## EachDomainTableHasItsRowCount

The domain tables hold 17, 14, 6, 3, 4, 15, 6 and 6 rows, in the order elementwise, activation,
linalg, loss, shape, spatial, norm and fused. The joined table holds 71.
