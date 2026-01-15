#!/usr/bin/env python3
"""Desafio 60K 2025 - Outrigger rotation optimization."""

import pandas as pd
import numpy as np
from outrigger_opt import optimize_stint_range

def main():
    print("=" * 60)
    print("DESAFIO 60K 2025 - Rotation Optimization")
    print("=" * 60)

    paddlers = pd.DataFrame({
        'name': ['Eduardo', 'Vitor', 'Guilherme', 'Ricardo', 'Airton',
                 'Everson', 'Sergio', 'Marcelo', 'Zé']
    })

    # Define who can sit where (rows=paddlers, cols=seats 1-6)
    eligibility = np.array([
        [1, 1, 0, 0, 0, 0],  #  front only
        [1, 1, 0, 0, 0, 0],  #  front only
        [1, 1, 1, 0, 0, 0],  #  front + seat 3
        [0, 0, 1, 1, 0, 0],  #  middle front
        [0, 0, 1, 1, 0, 0],  #  middle front
        [0, 0, 0, 1, 1, 0],  #  middle back
        [0, 0, 0, 1, 1, 0],  #  middle back
        [0, 0, 0, 0, 1, 1],  #  back + steering
        [0, 0, 0, 0, 1, 1],  #  back + steering
    ])

    print("\nPaddlers with seat eligibility:")
    print("            Seat1 Seat2 Seat3 Seat4 Seat5 Seat6")
    for i, name in enumerate(paddlers.name):
        print(f"{name:11} {' '.join(str(x).center(5) for x in eligibility[i])}")

    print("\nOptimizing stint durations from 10 to 20 minutes...")
    print("(This may take a few minutes)\n")

    results = optimize_stint_range(
        paddlers,
        stint_range=range(10, 15),  # 10, 11, 12, ..., 20 minutes
        seat_eligibility=eligibility,
        seat_weights=[1.1, 1, 1, 1, 1, 1.1],
        seat_entry_weights=[1, 4, 1.2, 1.2, 1.2, 1],
        paddler_ability=[1.02, 1.01, 1, 1, 1, 1.01, .99, 1, 1.02],
        distance_km=60,
        speed_kmh=10,
        switch_time_secs=15,
        max_consecutive=3,
        solver_time_secs=60,
        gap_tolerance=0.001,
    )

    print("=" * 60)
    print("RESULTS COMPARISON")
    print("=" * 60)
    print(results['summary'].to_string(index=False))

    print("\n" + "=" * 60)
    print(f"BEST STINT DURATION: {results['best']['stint_min']} min")
    print(f"BEST RACE TIME: {results['best']['race_time']:.1f} min")
    print("=" * 60)

    print("\nCycle Rules (* = starting position):")
    for name, rule in results['best']['cycle_rules'].items():
        print(f"  {name}: {rule}")

    print("\nFull Schedule:")
    print(results['best']['schedule'])

    return results


if __name__ == "__main__":
    main()
