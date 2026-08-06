# Releasing

Versions are SemVer (`vMAJOR.MINOR.PATCH`). Python, Cargo workspace, and
`__version__` must match; the tag must have a `CHANGELOG.md` section.

CI: PRs + pushes to `main`. Release: push a version tag → wheels, GitHub
Release (notes from CHANGELOG), PyPI.

```
tag vX.Y.Z → validate → wheels → GitHub Release + PyPI
```

## Checklist

1. Bump version in `pyproject.toml`, `Cargo.toml`, `python/tensortorrent/__init__.py`.
2. Add `## X.Y.Z` to `CHANGELOG.md`.
3. `make check && uv run python tools/check_version.py --tag vX.Y.Z`
4. Commit, then:

   ```bash
   git tag -a vX.Y.Z -m "TensorTorrent X.Y.Z"
   git push origin main vX.Y.Z
   ```

5. Watch `Actions → release`.

## PyPI trusted publisher (once)

https://pypi.org/manage/account/publishing/

| Field | Value |
|-------|-------|
| Project | `tensortorrent` |
| Owner | `alhussein-jamil` |
| Repo | `TensorTorrent` |
| Workflow | `release.yml` |
| Environment | `pypi` |

Repo Settings → Environments → `pypi` (no secrets).

## Wheel matrix

| Target  | Python             | Notes |
|---------|--------------------|-------|
| x86_64  | 3.10–3.13          | sdist on 3.12 |
| aarch64 | 3.12, 3.13         | `ubuntu-24.04-arm` |

## After release

- https://github.com/alhussein-jamil/TensorTorrent/releases
- `pip install tensortorrent==X.Y.Z && tensortorrent doctor`
