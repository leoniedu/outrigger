# Claude Code Instructions for Outrigger Project

## Python Environment

This project uses `uv` for Python package management. Always use `uv run` to execute Python scripts:

```bash
# Run scripts
uv run python script.py

# Run examples
cd python && uv run python example.py 6

# Run tests
uv run pytest
```

Do NOT use bare `python` or `python3` commands - they will fail due to system Python restrictions.

## Project Structure

- `python/` - Python project directory
  - `outrigger_opt/` - Main package (MIP solver for rotation optimization)
    - `model.py` - Core MIP solver (`solve_rotation_cycle`)
    - `fatigue.py` - Fatigue curve functions
    - `meta.py` - Meta-optimization over stint durations
  - `example.py` - Usage examples (run with `uv run python example.py [1-6]`)

## Key Documentation

- `OPTIMIZATION_PLAN.md` - Detailed optimization model specification and implementation status
