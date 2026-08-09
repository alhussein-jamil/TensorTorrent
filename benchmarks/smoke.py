"""Fast smoke entry — light suites only (fit / budget / hetero).

Heavy suites: ``python -m benchmarks.public --suite deepmlp|crossover|transformer``
"""

from __future__ import annotations

from benchmarks.public import main

if __name__ == "__main__":
    raise SystemExit(main(["--smoke", "--suite", "all"]))
