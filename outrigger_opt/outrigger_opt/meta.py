import pandas as pd
from .model import solve_rotation_cycle


def optimize_stint_range(paddlers,
                         stint_range=(30, 35, 40, 45, 50, 55, 60),
                         max_consecutive=6,
                         distance_km=60,
                         speed_kmh=10,
                         switch_time_min=1.5,
                         seat_eligibility=None,
                         seat_weights=None,
                         seat_entry_weights=None,
                         paddler_ability=None,
                         n_seats=6,
                         n_resting=3,
                         solver_time_secs=60,
                         gap_tolerance=0.01):
    """Grid search over stint durations to find optimal stint length.

    Args:
        paddlers: DataFrame with 'name' column
        stint_range: Tuple of stint durations to test (in minutes)
        max_consecutive: Maximum consecutive stints (variable bound)
        distance_km: Race distance in kilometers
        speed_kmh: Base speed at output 1.0 in km/h
        switch_time_min: Time penalty per crew switch in minutes
        seat_eligibility: Optional (n_paddlers, n_seats) eligibility matrix
        seat_weights: Optional list of seat importance weights
        seat_entry_weights: Optional list of entry ease weights per seat
        paddler_ability: Optional list of ability multipliers per paddler
        n_seats: Number of seats in canoe (default 6)
        n_resting: Number of paddlers resting each stint (default 3)
        solver_time_secs: Maximum solver computation time in seconds per stint duration
        gap_tolerance: Acceptable gap from optimal (default 0.01 = 1%)

    Returns:
        dict with 'summary' (DataFrame), 'best' (dict with lowest race_time),
        and 'results' (list of all results)
    """
    results = []
    for stint_min in stint_range:
        res = solve_rotation_cycle(
            paddlers,
            stint_min=stint_min,
            max_consecutive=max_consecutive,
            distance_km=distance_km,
            speed_kmh=speed_kmh,
            switch_time_min=switch_time_min,
            seat_eligibility=seat_eligibility,
            seat_weights=seat_weights,
            seat_entry_weights=seat_entry_weights,
            paddler_ability=paddler_ability,
            n_seats=n_seats,
            n_resting=n_resting,
            solver_time_secs=solver_time_secs,
            gap_tolerance=gap_tolerance,
        )
        results.append({
            "stint_min": stint_min,
            "n_stints": res["parameters"]["n_stints"],
            "avg_output": res["avg_output"],
            "race_time": res["race_time"],
            "schedule": res["schedule"],
            "cycle_schedule": res["cycle_schedule"],
            "cycle_rules": res["cycle_rules"],
        })
    summary = pd.DataFrame([
        {k: r[k] for k in ["stint_min", "n_stints", "avg_output", "race_time"]}
        for r in results
    ])
    best = min(results, key=lambda r: r["race_time"])
    return {"summary": summary, "best": best, "results": results}
