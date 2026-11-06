# tests/core/engine_fixture/test_lazy_shared.cpp


Suites: `Small_LazyShared`.

Tests that `LazyShared` creates its value once, on first use, through the factory it was given.
The factory of every case is a host function that counts its calls, so no case opens a device.

## TheFactoryIsNotCalledBeforeTheFirstAcquire

A `LazyShared` that was made and never asked has called its factory zero times.

## EveryAcquireReturnsTheSameValueFromOneCall

Two `acquire()` calls return the same address, and the factory ran once.

## AFailedFactoryGivesItsFailureAndIsAskedAgain

A factory that fails gives `acquire()` its `StatusCode::Unsupported` failure with its message,
twice, and the factory ran twice.

## ReleaseMakesTheNextAcquireCreateAgain

After `release()`, `acquire()` returns a value the factory created a second time.

## ANewFactoryReplacesTheOneKept

`use_factory()` before the first `acquire()` makes the value come from the new factory, and the
first factory is never called.
