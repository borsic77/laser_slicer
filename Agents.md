# AGENTS.md – laserslicer - Development Guidelines



> Guidance for autonomous coding agents (OpenAI Codex CLI, Copilot Agent Mode, Cursor, etc.).
> Read this before writing, editing, or executing anything in this repo.

---

## 1 Preferred operating mode

| Tool               | Default approval mode | Notes                                               |
|--------------------|-----------------------|-----------------------------------------------------|
| OpenAI Codex CLI   | `suggest`             | Ask before **any** shell command other than `pytest`, `uv`, `ruff`, or `npm run <script>` |
| GitHub Copilot Agent | “assistant” (ask-first) | Use **pull-request flow**—do not push directly to `main` |
| Cursor / IDE bots  | Ask before writing files outside `core/`, `frontend/`, or `tests/` |

*Override by adding `# agent: auto-edit` or `# agent: full-auto` in a task description.*

---

## 2 Repository map & access rules

| Path/folder              | Agent action                               |
|--------------------------|-------------------------------------------|
| `core/`                  | ✅ May read/write Python (Django) code.   |
| `frontend/`              | ✅ May read/write React + Vite TS code.   |
| `tests/`                 | ✅ Must keep tests green; add new tests.  |
| `data/`, `tmp/`, `media/`, `job_results/` | 🚫 Read-only. Never commit generated or binary files. |
| `srtm_cache/`            | 🚫 Read-only, large external data.        |
| `logs/`, `db.sqlite3`    | 🚫 Ignore completely.                     |
| `docker-compose*.yml`, `Dockerfile*` | ✅ May edit, but ask first—production environments depend on these. |

---

## 3 Environment & setup commands

```bash
uv venv                         # creates .venv
uv sync
npm ci --prefix frontend
```

Databases & services
The SQLite file in the repo is test-only.
For local Postgres + Redis, call docker compose up db redis –d.

⸻

## 4 Formatting, linting & typing
	-	Python: run ruff format . then ruff check .
	-	Type hints are mandatory; fail CI if mypy warnings > 0
	-	Front-end: npm run lint (ESLint + TypeScript)
	-	No prettier for now—use ESLint’s auto-fix

Agents should run all linters before proposing commits or PRs.

## 5 Testing protocol

### Fast unit tests
pytest -q

### Full suite incl. slow CLI & integration tests
pytest -m "not slow" -q

Failing tests bar merges.
If adding code, auto-generate companion tests.
Use pytest-snapshots for fixture-heavy SVG output comparisons.

## 6 Commit & PR etiquette
	-	Conventional Commits (feat:, fix:, chore:…).
	-	PR description must list:
        1.	Purpose / linked issue
        2.	Key files changed
        3.	pytest, ruff, npm run lint results
        4.	Screenshots for UI tweaks
## 7 House rules & coding style
	-	Backend prefer class-based views.
	-	Service layer lives in core/services/; new domain logic goes there.
	-	React components:
        -	Functional + hooks
        -	State via Zustand, never Redux
        -	Tailwind CSS utility-first, no styled-components
	-	Write docstrings (Google style) and inline comments for non-obvious bits.
	-	Prefer pathlib, typing.Annotated, and f-strings.