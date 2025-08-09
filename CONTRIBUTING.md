# Contributing to Spectra AI

Thank you for your interest in improving Spectra AI. This project emphasizes:

- Emotional intelligence & empathetic interaction
- Dynamic, real-time architecture (no static model corpora)
- Clean, observable, test-backed backend code

## Ground Rules

1. No inclusion of static training datasets or embedded knowledge dumps.
2. All timestamps must be UTC & timezone-aware.
3. Avoid introducing persistent state unless explicitly approved (must be ephemeral or opt-in).
4. Maintain structured logging (`structlog`). New logs should include context fields.
5. Keep external dependencies minimal and purposeful.

## Pull Request Checklist

- [ ] Tests added or updated (pytest) for new behavior
- [ ] No breaking changes to public API without documenting in CHANGELOG
- [ ] Updated `CHANGELOG.md` (Unreleased section)
- [ ] Added/confirmed logging for significant branches (success + failure)
- [ ] Ensured `model` and `model_used` both set in chat responses
- [ ] All timestamps UTC (`datetime.now(timezone.utc)`) where applicable

## Coding Style

- Python: PEP 8; meaningful names; docstrings for public functions/classes
- Frontend: TypeScript + React, functional components, clear props typing
- Avoid premature abstraction; prefer clarity over cleverness.

## Tests

Run the suite:

```bash
pytest -q
```

Add focused tests beside existing ones in `tests/`.

## Commit Messages

Use concise, present-tense verbs:

```text
feat: add streaming response scaffold
fix: normalize UTC timestamp in metrics endpoint
refactor: extract model selection heuristic
docs: update environment variable descriptions
```

## Security

Do not commit secrets. Use environment variables. Report concerns privately.

## Release Process

1. Merge PRs into `main` after review & green tests
2. Update `VERSION` file and tag (`git tag vX.Y.Z && git push --tags`)
3. Update `CHANGELOG.md` moving Unreleased items under the new version

---
Thank you for helping evolve Spectra AI responsibly.
