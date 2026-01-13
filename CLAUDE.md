# Claude Code Instructions for Outrigger Project

## Python Environment

This project uses `uv` for Python package management. Always use `uv run` to execute Python scripts:

```bash
# Run scripts
uv run python script.py

# Run examples
cd outrigger_opt && uv run python example.py 6

# Run tests
uv run pytest
```

Do NOT use bare `python` or `python3` commands - they will fail due to system Python restrictions.

## Project Structure

- `outrigger_opt/` - Python package for rotation optimization (MIP solver)
  - `outrigger_opt/model.py` - Core MIP solvers (`solve_rotation_full`, `solve_rotation_cycle`)
  - `outrigger_opt/fatigue.py` - Fatigue curve functions
  - `outrigger_opt/meta.py` - Meta-optimization over stint durations
  - `example.py` - Usage examples (run with `uv run python example.py [1-6]`)

## Key Documentation

- `OPTIMIZATION_PLAN.md` - Detailed optimization model specification and implementation status
