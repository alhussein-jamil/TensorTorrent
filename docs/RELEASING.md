# Releasing StreamCompiler

StreamCompiler uses semantic versions and tags releases as `vMAJOR.MINOR.PATCH`.
The Python package, Rust workspace, and public Python API carry the same version.
CI rejects drift between them and rejects a release tag without a matching
changelog section.

## Prepare a release

1. Update the version in `pyproject.toml`, `Cargo.toml`, and
   `python/streamcompiler/__init__.py`.
2. Move the relevant entries from `Unreleased` in `CHANGELOG.md` to a section
   named for the version, for example `## 0.2.0`.
3. Run the complete local gate:

   ```bash
   make check
   uv run python tools/check_version.py --tag v0.2.0
   ```

4. Commit the release changes and create an annotated tag:

   ```bash
   git tag -a v0.2.0 -m "StreamCompiler 0.2.0"
   ```

5. Push the commit and tag, wait for CI, then create the GitHub release from the
   tag using the matching changelog section as its notes.

Release publication is currently manual. CI validates tags and builds wheel and
source artifacts, but it does not publish packages or GitHub releases.
