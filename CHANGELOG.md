# Changelog

All notable changes to Spectra AI will be documented in this file.

The format loosely follows Keep a Changelog and adheres to Semantic Versioning where practical.

## [Unreleased]

### Added

- (none yet)

### Changed

- (none yet)

### Fixed

- (none yet)

## [3.0.0-dev] - 2025-08-09

### Planned / In Progress

- Multi-provider abstraction (Ollama + OpenAI + Anthropic) with dynamic routing.
- Provider capability discovery endpoint.
- Enhanced streaming chat endpoint.
- Expanded test suite: provider fallbacks & error taxonomy.
- Observability: basic Prometheus counters for provider usage.

### Breaking Changes (to be documented)

- Potential restructuring of /api/chat request schema to allow provider override.

## [2.1.0] - 2025-08-09

### Fixed (v2.1.0)

- Indentation and optional history length guard in chat endpoint.

## [2.0.1] - 2025-08-09

### Added (v2.0.1)

- `model_used` field for backward-compatible chat responses.
- UTC timezone-aware timestamps across all API responses.
- README refactor with professional sections & schema documentation.
- CHANGELOG, CONTRIBUTING, CODE_OF_CONDUCT for project professionalism.

### Changed (v2.0.1)

- Structured logging standardized (JSON / console selectable).
- Metrics endpoint now returns UTC timestamps.

### Fixed (v2.0.1)

- Pytest import path stabilization via `conftest.py`.

## [2.0.0] - 2025-08-02

### Added (v2.0.0)

- FastAPI primary backend initialization.
- Dynamic model selection (context aware: creative / technical / concise).
- Personality hot-reload with hashing.
- Metrics: request counts, avg processing time, failed models tracking.

---

Links:

- 2.0.1: internal (not yet tagged)
- 2.0.0: initial professional baseline snapshot
