# AGENTS.md – Laser Slicer Development Guide

> Guidance for autonomous coding agents (OpenAI Codex CLI, Copilot Agent Mode, Cursor, etc.)
> Read this before writing, editing, or executing anything in this repo.

---

## 0 Important files to read

- [AGENTS.md](AGENTS.md) - This file
- [README.md](README.md) - Project README
- doc/AGENT_ONBOARDING.md - Agent onboarding guide

## 1 Preferred operating mode

| Tool | Default approval mode | Notes |
|------|-----------------------|-------|
| OpenAI Codex CLI | `suggest` | Ask before **any** shell command other than `pytest`, `uv`, `ruff`, or `npm run <script>` |
| GitHub Copilot Agent | "assistant" (ask-first) | Use **pull-request flow**—do not push directly to `main` |
| Cursor / IDE bots | Ask before writing files outside `core/`, `frontend/`, or `tests/` |

*Override by adding `# agent: auto-edit` or `# agent: full-auto` in a task description.*

---

## 2 Repository map & access rules

| Path/folder | Agent action |
|-------------|--------------|
| `core/` | ✅ Django backend code |
| `frontend/` | ✅ React + Vite TypeScript code |
| `tests/` | ✅ Keep tests green; add tests for new code |
| `cache/`, `data/`, `tmp/`, `media/`, `job_results/` | 🚫 Generated artefacts – do not commit |
| `srtm_cache/` | 🚫 Large external data – read only |
| `logs/`, `db.sqlite3` | 🚫 Ignore completely |
| `docker-compose*.yml`, `Dockerfile*` | ✅ May edit, but ask first—production setups depend on them |

---

## 3 Environment & setup commands

```bash
uv venv                # create .venv
uv sync                # install Python deps
npm ci --prefix frontend
```

Databases & services:
The SQLite file in the repo is test-only.
For local Postgres + Redis run `docker compose up db redis -d` or follow `dev_startup.md`.

---

## 4 Formatting & linting
- Python: `ruff format .` then `ruff check .`
- Type hints are strongly encouraged
- Front-end: `npm run lint`
- No Prettier – use ESLint's auto-fix

Agents should run linters before proposing commits or PRs.

## 5 Testing protocol

```bash
```bash
pytest -q           # run tests
```
Use `pytest-snapshots` for SVG fixtures.
Failing tests block merges.
Add tests alongside new code.

## 6 Mobile Testing
To test on mobile, use `npm run dev -- --host` or edit `vite.config.ts` to enable network access.
See [doc/MOBILE_TESTING_GUIDE.md](doc/MOBILE_TESTING_GUIDE.md) for details.

## 7 Commit & PR etiquette
- Conventional Commits (`feat:`, `fix:`, `chore:` …)
- PR description must list:
  1. Purpose / linked issue
  2. Key files changed
  3. `pytest`, `ruff`, and `npm run lint` results
  4. Screenshots for UI tweaks

## 7 House rules & coding style
- Prefer Django class-based views
- Domain logic lives in `core/services/`
- React components: functional with hooks; state via Zustand
- Tailwind CSS utility-first, no styled-components
- Write docstrings (Google style) and inline comments for non-obvious code
- Prefer `pathlib`, `typing.Annotated`, and f-strings
