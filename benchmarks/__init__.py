"""Public benchmark suite for TensorTorrent.

Run::

    python -m benchmarks.public --smoke
    python -m benchmarks.public --suite all
    python -m benchmarks.smoke

Results are written under ``benchmarks/results/<timestamp>/`` (gitignored).
Frozen public evidence lives under ``benchmarks/published/<date>/``.
"""

__all__ = ["__version__"]

__version__ = "0.3.1"
