# tests/vision/training/test_run_progress.cpp


Suites: `Small_RunProgress`.

Tests `report_run_progress` and `notify_guarded` of `run_progress.hpp`, which hand a progress report
to a detection run's callback and turn a throw of the callback into a failure. The progress type
is the test's own, one integer. No case needs a device.

### AnEmptyCallbackSucceeds

An empty callback succeeds without a call, through both functions.

### TheCallbackReceivesTheProgressOnce

A callback handed a progress of 7 is called once and sees 7.

### AThrowOfTheCallbackBecomesAFailure

A callback that throws `std::runtime_error("callback refused update")` makes the call return a
failure whose message holds that text.

### AThrowThatIsNoStdExceptionBecomesAFailure

A callback that throws an integer makes the call fail with `DeviceFailed` and the message "an
exception of an unknown type".
