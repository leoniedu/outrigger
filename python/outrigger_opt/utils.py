import pandas as pd

def demo_paddlers():
    """Return a sample paddler DataFrame for testing."""
    return pd.DataFrame({
        "name": ["Ana", "Ben", "Carlos", "Diana", "Eve", "Gina", "Hiro", "Frank", "Ivan"]
    })
