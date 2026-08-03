# Contributing

Keep changes small enough to review and test. Correctness comes first;
performance changes need measurements.

## Ground rules

- Keep vendor logic in `python/tensortorrent/backends`, including collective
	communication behavior.
- Never bake the development host's topology or capabilities into the planner.
- Add regression coverage for planner, discovery, validation, and runtime fixes.
- Keep hardware tests explicitly marked: they may allocate most available VRAM
	or spill space and do not run in the architecture-neutral CI suite.
- Do not weaken validation to make an unsupported target appear healthy.

## Local setup

```bash
make sync
make pre-commit-install          # once per clone
make pre-commit
make check
make native-gate
```

Pre-commit covers whitespace, YAML/TOML/JSON, private keys, Ruff, codespell,
mypy, project-version consistency, `cargo fmt`, `cargo check`, and (on push)
`cargo clippy`.

`make check` is the complete architecture-neutral gate. Run `make hardware-test`
separately on every deployment target affected by a hardware or backend change.

## Pull requests

Explain the behavior change, the reason for it, and the checks you ran. Include
before/after measurements for performance work. Avoid unrelated cleanup in the
same change.

## Releases

Versions follow SemVer and release tags use `vMAJOR.MINOR.PATCH`. See
[docs/RELEASING.md](docs/RELEASING.md) for the checklist and tag validation.
