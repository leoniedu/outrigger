import pandas as pd
from .model import solve_rotation_full

def optimize_stint_range(paddlers,
                         stint_range=(30,35,40,45,50,55,60),
                         max_consecutive=6):
    results = []
    for stint_min in stint_range:
        res = solve_rotation_full(paddlers, stint_min=stint_min, max_consecutive=max_consecutive)
        results.append({
            "stint_min": stint_min,
            "n_stints": res["parameters"]["n_stints"],
            "avg_output": res["avg_output"],
            "race_time": res["race_time"],
            "schedule": res["schedule"]
        })
    summary = pd.DataFrame([{k:r[k] for k in ["stint_min","n_stints","avg_output","race_time"]} for r in results])
    best = min(results, key=lambda r: r["race_time"])
    return {"summary": summary, "best": best, "results": results}
