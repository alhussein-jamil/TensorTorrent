# Releasing TensorTorrent

TensorTorrent uses semantic versions and tags releases as `vMAJOR.MINOR.PATCH`.
The Python package, Rust workspace, and public Python API carry the same version.
CI rejects drift between them and rejects a release tag without a matching
changelog section.

Release publication is **fully automated** via the `release.yml` GitHub Actions
workflow. Pushing a tag triggers wheel builds, a GitHub Release, and a PyPI
publish — no manual uploads needed.

---

## Release flow overview

```
git tag v0.3.0 → push tag
        │
        ▼
release.yml
  ├── validate          check_version.py verifies tag matches pyproject.toml
  ├── wheels            maturin builds manylinux wheels (3.10, 3.11, 3.12)
  │                     and sdist; aarch64 on native ubuntu-24.04-arm
  ├── publish-github    gh release create with --generate-notes
  └── publish-pypi      pypa/gh-action-pypi-publish via OIDC trusted publishing
```

---

## Pre-release checklist

1. **Bump the version** in all three places (they must match exactly):
   - `pyproject.toml` → `version = "X.Y.Z"`
   - `Cargo.toml` → `version = "X.Y.Z"` (workspace root and any crates that
     re-export it)
   - `python/tensortorrent/__init__.py` → `__version__ = "X.Y.Z"`

2. **Update CHANGELOG.md** — move relevant entries from `Unreleased` to a new
   section named for the version, e.g. `## 0.3.0`.

3. **Run the local gate** to catch drift before pushing:

   ```bash
   make check
   uv run python tools/check_version.py --tag v0.3.0
   make audit          # cargo audit + pip-audit
   make coverage       # pytest --cov-fail-under=70
   ```

4. **Commit** the release changes:

   ```bash
   git add pyproject.toml Cargo.toml Cargo.lock python/tensortorrent/__init__.py CHANGELOG.md
   git commit -m "chore: release v0.3.0"
   ```

5. **Create and push an annotated tag**:

   ```bash
   git tag -a v0.3.0 -m "TensorTorrent 0.3.0"
   git push origin main v0.3.0
   ```

6. **Watch the workflow** at `Actions → release`. All jobs except `publish-pypi`
   run unconditionally. `publish-pypi` requires the one-time PyPI setup below.

---

## One-time PyPI trusted-publisher setup

TensorTorrent uses PyPI's OIDC trusted-publisher mechanism — no long-lived API
token is stored in GitHub secrets. This is a one-time configuration step per
project.

1. Log into PyPI and open <https://pypi.org/manage/account/publishing/>.
2. Under **"Add a new pending publisher"** fill in:

   | Field               | Value                 |
   |---------------------|-----------------------|
   | PyPI project name   | `tensortorrent`       |
   | GitHub owner        | `alhussein-jamil`     |
   | Repository          | `TensorTorrent`       |
   | Workflow name       | `release.yml`         |
   | Environment name    | `pypi`                |

   Also ensure a GitHub Actions environment named `pypi` exists under
   Settings → Environments (no secrets needed for OIDC). This repo already
   has that environment.

3. Save. On the first tag push the project is auto-created on PyPI and the
   wheel is published using the short-lived OIDC token — no password needed.

Until this is configured the `publish-pypi` job fails with an OIDC error; all
other jobs (validate, wheels, publish-github) succeed normally and the GitHub
Release is still created.

---

## Wheel matrix

| Target  | Python versions      | Blocking? | Notes |
|---------|----------------------|-----------|-------|
| x86_64  | 3.10, 3.11, 3.12     | Yes       | manylinux auto; sdist built once on 3.12 leg |
| aarch64 | 3.12                 | Yes       | native `ubuntu-24.04-arm` runner (same as CI) |

---

## aarch64 builds

The aarch64 leg runs on GitHub-hosted `ubuntu-24.04-arm` (native ARM64), not
QEMU. It is part of the required `wheels` matrix: a failure blocks the release.

---

## Post-release checks

After the workflow completes:

- Confirm the GitHub Release appears at `https://github.com/alhussein-jamil/TensorTorrent/releases`.
- Confirm the package appears on PyPI: `pip index versions tensortorrent`.
- Pull and smoke-test the published wheel:

  ```bash
  pip install tensortorrent==X.Y.Z
  tensortorrent doctor
  ```

- Update the `Unreleased` section in `CHANGELOG.md` to a blank slate for the
  next development cycle.
