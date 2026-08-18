# Releasing TensorTorrent

Releases are tag-driven. Python package metadata, the Cargo workspace version, `tensortorrent.__version__`, and the changelog entry must agree before a tag is pushed.

## Pre-release checklist

1. Update the version in:
   - `pyproject.toml`
   - workspace/package Cargo metadata as required by the repository
   - `python/tensortorrent/__init__.py`
2. Add the matching section to `CHANGELOG.md`.
3. Run the normal quality gate:

   ```bash
   make check
   make native-gate
   ```

4. Validate version consistency:

   ```bash
   uv run python tools/check_version.py --tag vX.Y.Z
   ```

5. For backend/runtime changes, complete the relevant target-hardware validation before publishing.

## Tag

```bash
git tag -a vX.Y.Z -m "TensorTorrent X.Y.Z"
git push origin main vX.Y.Z
```

The release workflow builds artifacts, creates the GitHub release from the changelog, and publishes to PyPI through the configured trusted-publisher environment.

## PyPI trusted publisher

Expected project configuration:

| Field | Value |
| --- | --- |
| Project | `tensortorrent` |
| GitHub owner | `alhussein-jamil` |
| Repository | `TensorTorrent` |
| Workflow | `release.yml` |
| Environment | `pypi` |

No long-lived PyPI token should be required when trusted publishing is configured correctly.

## Wheel matrix

The release workflow builds:

| Host | Architecture | Python |
| --- | --- | --- |
| Linux | x86-64 | 3.10–3.13 |
| Linux | AArch64 | 3.12–3.13 |
| macOS | Apple Silicon (AArch64) | 3.10–3.13 |
| macOS | Intel (x86-64) | 3.12 |

The release workflow is the source of truth for the exact matrix used by a given tag. Other CPython versions can build from the sdist when Rust is available.

## Post-release verification

Install the published version into a clean environment and verify the native extension:

```bash
python -m pip install "tensortorrent==X.Y.Z"
tensortorrent doctor
```

For an accelerator release, repeat the target-host validation with the published wheel rather than relying only on a source checkout.
