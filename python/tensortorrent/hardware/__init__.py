"""Hardware discovery and profiling helpers.

Import concrete modules directly (e.g. ``tensortorrent.hardware.discovery``)
when avoiding circular imports with execution backends.
"""

from tensortorrent.hardware.fingerprint import machine_fingerprint

__all__ = ["machine_fingerprint"]
