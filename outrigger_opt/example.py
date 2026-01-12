#!/usr/bin/env python3
"""Example usage of the outrigger rotation optimizer."""

import pandas as pd
import numpy as np
from outrigger_opt import solve_rotation_full, optimize_stint_range, make_eligibility_from_roles


def example_role_based():
    """Option 1: Role-based eligibility (simplest)."""
    print("=" * 60)
    print("OPTION 1: Role-based eligibility")
    print("=" * 60)

    paddlers = pd.DataFrame({
        'name': ['Eduardo', 'Vitor', 'Guilherme', 'Airton', 'Ricardo', 'Sergio', 'Everson', 'Marcelo',  'Zé'],
        'role': ['pacer', 'pacer', 'pacer', 'regular', 'regular', 'regular', 'regular', 'steerer', 'steerer']
    })

    print("\nPaddlers:")
    print(paddlers)

    # Show generated eligibility
    eligibility = make_eligibility_from_roles(paddlers)
    print("\nGenerated eligibility matrix (1=can sit in seat):")
    print("         Seat1 Seat2 Seat3 Seat4 Seat5 Seat6")
    for i, name in enumerate(paddlers.name):
        print(f"{name:8} {' '.join(str(x).center(5) for x in eligibility[i])}")

    print("\nSolving...")
    result = solve_rotation_full(
        paddlers,
        stint_min=30,
        distance_km=60,
        speed_kmh=9.5,
        switch_time_min=2,
        time_limit=30
    )

    print(f"\nStatus: {result['status']}")
    print(f"Race time: {result['race_time']:.1f} min")
    print(f"Avg output: {result['avg_output']:.3f}")
    print(f"\nSchedule:")
    print(result['schedule'])

    return result


def example_custom_eligibility():
    """Option 2: Custom eligibility matrix."""
    print("\n" + "=" * 60)
    print("OPTION 2: Custom eligibility matrix")
    print("=" * 60)

    paddlers = pd.DataFrame({
        'name': ['V1', 'V2', 'V3', 'H1', 'H2', 'H3', 'H4', 'L1', 'L2']
    })

    # Define who can sit where (rows=paddlers, cols=seats 1-6)
    eligibility = np.array([
        [1, 1, 0, 0, 0, 0],  # 
        [1, 1, 0, 0, 0, 0],  # 
        [1, 1, 1, 0, 0, 0],  # 
        [0, 0, 1, 1, 0, 0],  # 
        [0, 0, 1, 1, 0, 0],  # 
        [0, 0, 0, 1, 1, 0],  # 
        [0, 0, 0, 1, 1, 0],  # 
        [0, 0, 0, 0, 1, 1],  # 
        [0, 0, 0, 0, 1, 1],  # 
    ])

    print("\nPaddlers with custom eligibility:")
    print("         Seat1 Seat2 Seat3 Seat4 Seat5 Seat6")
    for i, name in enumerate(paddlers.name):
        print(f"{name:8} {' '.join(str(x).center(5) for x in eligibility[i])}")

    print("\nSolving...")
    result = solve_rotation_full(
        paddlers,
        seat_eligibility=eligibility,
        seat_weights=[1.2, 1.15, 1.1, 1, 1,1.2],
        stint_min=30,
        time_limit=40,
        switch_time_min=1,

    )

    print(f"\nStatus: {result['status']}")
    print(f"Race time: {result['race_time']:.1f} min")
    print(f"\nSchedule:")
    print(result['schedule'])

    return result


def example_different_crew_size():
    """Option 3: Different crew size (8 paddlers, 5 seats)."""
    print("\n" + "=" * 60)
    print("OPTION 3: Different crew size (8 paddlers, 5 seats)")
    print("=" * 60)

    paddlers = pd.DataFrame({
        'name': ['P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8']
    })

    # Everyone can sit anywhere
    eligibility = np.ones((8, 5), dtype=int)

    print(f"\nPaddlers: {list(paddlers.name)}")
    print("All paddlers eligible for all 5 seats")

    print("\nSolving...")
    result = solve_rotation_full(
        paddlers,
        seat_eligibility=eligibility,
        n_seats=5,
        n_resting=3,
        seat_weights=[1.2, 1.1, 1.0, 0.9, 1.1],
        stint_min=30,
        time_limit=30
    )

    print(f"\nStatus: {result['status']}")
    print(f"Race time: {result['race_time']:.1f} min")
    print(f"\nSchedule:")
    print(result['schedule'])

    return result


def example_meta_optimization():
    """Option 4: Find optimal stint duration."""
    print("\n" + "=" * 60)
    print("OPTION 4: Meta-optimization (find best stint duration)")
    print("=" * 60)

    paddlers = pd.DataFrame({
        'name': ['Ana', 'Ben', 'Carlos', 'Diana', 'Eve', 'Gina', 'Hiro', 'Frank', 'Ivan'],
        'role': ['pacer', 'pacer', 'pacer', 'regular', 'regular', 'regular', 'regular', 'steerer', 'steerer']
    })

    print("\nTesting stint durations: 30, 40, 50, 60 minutes")
    print("(Using shorter list for faster demo)\n")

    results = optimize_stint_range(
        paddlers,
        stint_range=(30, 40, 50, 60),
        max_consecutive=6
    )

    print("Comparison:")
    print(results['summary'].to_string(index=False))

    print(f"\nBest stint duration: {results['best']['stint_min']} min")
    print(f"Best race time: {results['best']['race_time']:.1f} min")
    print(f"\nBest schedule:")
    print(results['best']['schedule'])

    return results


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        option = sys.argv[1]
        if option == "1":
            example_role_based()
        elif option == "2":
            example_custom_eligibility()
        elif option == "3":
            example_different_crew_size()
        elif option == "4":
            example_meta_optimization()
        else:
            print(f"Unknown option: {option}")
            print("Usage: python example.py [1|2|3|4]")
    else:
        # Run all examples
        example_role_based()
        example_custom_eligibility()
        example_different_crew_size()
        example_meta_optimization()
