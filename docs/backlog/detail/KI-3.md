# KI-3 `gradcheck` does not check the fp16 path of a mixed-precision graph

The row of this entry is in `docs/backlog/LIBRARY_BEHAVIOUR.md`. This file holds what did not fit the row.

- Origin: own
- Where: `modules/core/autograd/include/SushiAI/autograd/gradcheck.hpp`
- Cause: it takes its point in FLOAT64
- Recognise: a wrong fp16 rule passes the check
- Workaround: check the rule in FLOAT64
- Fix: none, a finite-difference check needs FLOAT64 precision
