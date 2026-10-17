# tests/core/failure/test_error.cpp


Suites: `Small_Error`.

Tests the `Error` type and the `SA_THROW_IF` macro of
`modules/core/failure/include/SushiAI/failure/error.hpp`. No case needs a device.

## throw_when

`throw_when` runs `SA_THROW_IF` on its flag with the message "extent {} is not positive" and
counts the calls that pass the check, so a case can tell a throw from a pass.

## EveryElementTypeHasItsWidthAndItsName

`dtype_size` gives 2, 4, 8, 8, 16, 4 and 8 bytes for half, float32, float64, complex32,
complex64, int32 and int64, and `dtype_name` gives each its lower-case name.

## OnlyTheRealFloatingTypesAreRealFloats

`is_real_float` is true for half, float32 and float64 and false for the two complex and the two
integer types. A `static_assert` on float32 pins that it is usable at compile time.

## TheMaximumRankIsTheRankSushiBLASStores

`K_MAX_RANK` equals `SushiBLAS::Core::K_MAX_TENSOR_RANK`.

## AnErrorIsARuntimeErrorWithItsMessage

`Error` derives from `std::runtime_error`, and a thrown one is caught as `std::runtime_error`
with its message unchanged.

## ThrowIfThrowsAFormattedErrorOnlyWhenItsConditionHolds

A false condition passes and the call count rises to 1. A true condition throws an `Error` whose
message reads "extent -2 is not positive", and the count stays at 1.
