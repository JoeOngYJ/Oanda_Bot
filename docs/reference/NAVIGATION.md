# Navigation Guide

## Primary Working Areas

- `agents/` - live agent pipeline components (`market_data`, `strategy`, `risk`, `execution`, `monitoring`)
- `backtesting/` - simulation engine, strategy research tools, analytics
- `shared/` - shared contracts and infrastructure (`models`, `message_bus`, `config`)
- `scripts/` - runnable entrypoints and operations utilities
- `config/` - runtime and research configuration files
- `tests/` - test suites grouped by subsystem and test type

## Supporting Areas

- `docs/` - architecture, runbooks, plans, and references
- `docs/notes/` - ad-hoc status/fix notes
- `data/` - datasets, research outputs, and reports
- `models/` - promoted model artifacts (`active/`, `archive/`)
- `deploy/` - deployment assets (systemd templates)
- `notebooks/` - exploratory notebooks

## Script Conventions

- `scripts/` - production/research runners
- `scripts/dev/` - one-off local smoke/debug scripts

## Root File Conventions

Keep only core project control files at repo root:

- build/test config: `pyproject.toml`, `setup.cfg`, `pytest.ini`, `Makefile`
- environment/bootstrap: `.env.example`, `requirements.txt`, `docker-compose.yml`
- project overview: `README.md`, `AGENTS.md`, `SKILLS.md`

Avoid adding ad-hoc test files or temporary notes at root.
