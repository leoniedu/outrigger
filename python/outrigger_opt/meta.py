import pandas as pd
from .model import solve_rotation_cycle


def optimize_stint_range(paddlers,
                         stint_km_range=(1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0),
                         max_consecutive=6,
                         distance_km=60,
                         speed_kmh=10,
                         switch_time_secs=90,
                         seat_eligibility=None,
                         seat_weights=None,
                         seat_entry_weights=None,
                         paddler_ability=None,
                         paddler_weight=None,
                         trim_penalty_weight=0.0,
                         moi_penalty_weight=0.0,
                         steerer_paddle_fraction=0.6,
                         n_seats=6,
                         n_resting=3,
                         solver_time_secs=60,
                         gap_tolerance=0.01):
    """Grid search over stint distances to find optimal stint length.

    Uses distance-based stints where n_stints = ceil(distance_km / stint_km)
    is deterministic, creating smooth optimization curves.

    Args:
        paddlers: DataFrame with 'name' column
        stint_km_range: Tuple of stint distances to test (in kilometers)
        max_consecutive: Maximum consecutive stints (variable bound)
        distance_km: Race distance in kilometers
        speed_kmh: Base speed at output 1.0 in km/h
        switch_time_secs: Time penalty per crew switch in seconds
        seat_eligibility: Optional (n_paddlers, n_seats) eligibility matrix
        seat_weights: Optional list of seat importance weights
        seat_entry_weights: Optional list of entry ease weights per seat
        paddler_ability: Optional list of ability multipliers per paddler
        paddler_weight: Optional list of weights per paddler (kg or relative)
        trim_penalty_weight: Penalty weight for max abs trim (default 0.0 = disabled)
        moi_penalty_weight: Penalty weight for weight concentration at ends (default 0.0 = disabled)
        steerer_paddle_fraction: Fraction of time steerer paddles vs steers (default 0.6)
        n_seats: Number of seats in canoe (default 6)
        n_resting: Number of paddlers resting each stint (default 3)
        solver_time_secs: Maximum solver computation time in seconds per stint distance
        gap_tolerance: Acceptable gap from optimal (default 0.01 = 1%)

    Returns:
        dict with 'summary' (DataFrame), 'best' (dict with lowest race_time),
        and 'results' (list of all results)
    """
    results = []
    for stint_km in stint_km_range:
        res = solve_rotation_cycle(
            paddlers,
            stint_km=stint_km,
            max_consecutive=max_consecutive,
            distance_km=distance_km,
            speed_kmh=speed_kmh,
            switch_time_secs=switch_time_secs,
            seat_eligibility=seat_eligibility,
            seat_weights=seat_weights,
            seat_entry_weights=seat_entry_weights,
            paddler_ability=paddler_ability,
            paddler_weight=paddler_weight,
            trim_penalty_weight=trim_penalty_weight,
            moi_penalty_weight=moi_penalty_weight,
            steerer_paddle_fraction=steerer_paddle_fraction,
            n_seats=n_seats,
            n_resting=n_resting,
            solver_time_secs=solver_time_secs,
            gap_tolerance=gap_tolerance,
        )
        results.append({
            "stint_km": stint_km,
            "n_stints": res["parameters"]["n_stints"],
            "avg_stint_time_min": res["parameters"]["avg_stint_time_min"],
            "avg_output": res["avg_output"],
            "race_time": res["race_time"],
            "schedule": res["schedule"],
            "cycle_schedule": res["cycle_schedule"],
            "cycle_rules": res["cycle_rules"],
            "parameters": res["parameters"],
        })
    summary = pd.DataFrame([
        {k: r[k] for k in ["stint_km", "n_stints", "avg_stint_time_min", "avg_output", "race_time"]}
        for r in results
    ])
    best = min(results, key=lambda r: r["race_time"])
    return {"summary": summary, "best": best, "results": results}
