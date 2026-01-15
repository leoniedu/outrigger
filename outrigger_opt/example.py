#!/usr/bin/env python3
"""Example usage of the outrigger rotation optimizer."""

import pandas as pd
import numpy as np
from outrigger_opt import solve_rotation_cycle, optimize_stint_range


def example_basic():
    """Example 1: Basic usage with all paddlers eligible for all seats."""
    print("=" * 60)
    print("EXAMPLE 1: Basic usage (all paddlers eligible)")
    print("=" * 60)

    paddlers = pd.DataFrame({
        'name': ['Eduardo', 'Vitor', 'Guilherme', 'Airton', 'Ricardo',
                 'Sergio', 'Everson', 'Marcelo', 'Ze']
    })

    print("\nPaddlers:")
    print(paddlers)
    print("\nAll 9 paddlers can sit in any of the 6 seats.")

    print("\nSolving...")
    result = solve_rotation_cycle(
        paddlers,
        stint_min=30,
        distance_km=60,
        speed_kmh=9.5,
        switch_time_min=1,
        solver_time_secs=30
    )

    print(f"\nStatus: {result['status']}")
    print(f"Race time: {result['race_time']:.1f} min")
    print(f"Avg output: {result['avg_output']:.3f}")
    print(f"\nCycle Rules (* = starting position):")
    for name, rule in result['cycle_rules'].items():
        print(f"  {name}: {rule}")
    print(f"\nFull Schedule:")
    print(result['schedule'])

    return result


def example_custom_eligibility():
    """Example 2: Custom eligibility matrix."""
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Custom eligibility matrix")
    print("=" * 60)

    paddlers = pd.DataFrame({
        'name': ['Eduardo', 'Vitor', 'Guilherme', 'Ricardo', 'Airton', 'Everson', 'Sergio', 'Marcelo', 'Zé']
    })

    # Define who can sit where (rows=paddlers, cols=seats 1-6)
    eligibility = np.array([
        [1, 1, 0, 0, 0, 0],  # V1: front only
        [1, 1, 0, 0, 0, 0],  # V2: front only
        [1, 1, 1, 0, 0, 0],  # V3: front + seat 3
        [0, 0, 1, 1, 0, 0],  # F1: middle front
        [0, 0, 1, 1, 0, 0],  # F2: middle front
        [0, 0, 0, 1, 1, 0],  # F3: middle back
        [0, 0, 0, 1, 1, 0],  # F4: middle back
        [0, 0, 0, 0, 1, 1],  # L1: back + steering
        [0, 0, 0, 0, 1, 1],  # L2: back + steering
    ])

    print("\nPaddlers with custom eligibility:")
    print("         Seat1 Seat2 Seat3 Seat4 Seat5 Seat6")
    for i, name in enumerate(paddlers.name):
        print(f"{name:8} {' '.join(str(x).center(5) for x in eligibility[i])}")

    print("\nSolving...")
    result = solve_rotation_cycle(
        paddlers,
        seat_eligibility=eligibility,
        seat_weights=[1.1, 1, 1, 1, 1, 1.1],
        paddler_ability=[1.02, 1.01, 1, 1, 1, 1.02, .98, 1,1.02],
        seat_entry_weights=[1, 2, 1.2, 1.2, 1.2, 1],
        stint_min=15,
        distance_km=60,
        speed_kmh=10,
        solver_time_secs=60,
        switch_time_min=0.5,
        max_consecutive=3,
        gap_tolerance=0.001,
    )

    print(f"\nStatus: {result['status']}")
    print(f"Race time: {result['race_time']:.1f} min")
    print(f"\nCycle Rules (* = starting position):")
    for name, rule in result['cycle_rules'].items():
        print(f"  {name}: {rule}")
    print(f"\nFull Schedule:")
    print(result['schedule'])

    print(f"\nPaddler Summary:")
    print(result['paddler_summary'].to_string(index=False))

    print(f"\nAggregate Stats:")
    for key, val in result['summary_stats'].items():
        if isinstance(val, float):
            print(f"  {key}: {val:.1f}")
        else:
            print(f"  {key}: {val}")

    return result


def example_different_crew_size():
    """Example 3: Different crew size (8 paddlers, 4 seats)."""
    print("\n" + "=" * 60)
    print("EXAMPLE 3: Different crew size (8 paddlers, 4 seats)")
    print("=" * 60)

    paddlers = pd.DataFrame({
        'name': ['P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8']
    })

    print(f"\nPaddlers: {list(paddlers.name)}")
    print("All paddlers eligible for all 4 seats, 4 resting per stint")

    print("\nSolving...")
    result = solve_rotation_cycle(
        paddlers,
        n_seats=4,
        n_resting=4,
        seat_weights=[1.2, 1.1, 1.0, 0.9],
        stint_min=30,
        solver_time_secs=30
    )

    print(f"\nStatus: {result['status']}")
    print(f"Race time: {result['race_time']:.1f} min")
    print(f"Cycle length: {result['parameters']['cycle_length']} stints")
    print(f"\nCycle Rules (* = starting position):")
    for name, rule in result['cycle_rules'].items():
        print(f"  {name}: {rule}")
    print(f"\nFull Schedule:")
    print(result['schedule'])

    return result


def example_meta_optimization():
    """Example 4: Find optimal stint duration."""
    print("\n" + "=" * 60)
    print("EXAMPLE 4: Meta-optimization (find best stint duration)")
    print("=" * 60)

    paddlers = pd.DataFrame({
        'name': ['Ana', 'Ben', 'Carlos', 'Diana', 'Eve',
                 'Gina', 'Hiro', 'Frank', 'Ivan']
    })

    print("\nTesting stint durations: 30, 40, 50, 60 minutes")
    print("(Using shorter list for faster demo)\n")

    results = optimize_stint_range(
        paddlers,
        stint_range=(30, 40, 50, 60),
        solver_time_secs=30
    )

    print("Comparison:")
    print(results['summary'].to_string(index=False))

    print(f"\nBest stint duration: {results['best']['stint_min']} min")
    print(f"Best race time: {results['best']['race_time']:.1f} min")
    print(f"\nBest cycle rules (* = starting position):")
    for name, rule in results['best']['cycle_rules'].items():
        print(f"  {name}: {rule}")

    return results


def example_paddler_ability():
    """Example 5: Paddler ability (some paddlers are stronger)."""
    print("\n" + "=" * 60)
    print("EXAMPLE 5: Paddler ability (varying strength)")
    print("=" * 60)

    paddlers = pd.DataFrame({
        'name': ['Ana', 'Ben', 'Carlos', 'Diana', 'Eve',
                 'Gina', 'Hiro', 'Frank', 'Ivan']
    })

    # Ana is 50% stronger, Ben is 20% weaker
    ability = [1.5, 0.8, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

    print("\nPaddler abilities:")
    for name, ab in zip(paddlers.name, ability):
        print(f"  {name}: {ab:.1f}x")

    print("\nSolving with equal ability...")
    result_equal = solve_rotation_cycle(
        paddlers,
        stint_min=40,
        distance_km=60,
        speed_kmh=10,
        solver_time_secs=30
    )

    print("\nSolving with varying ability...")
    result_varying = solve_rotation_cycle(
        paddlers,
        stint_min=40,
        distance_km=60,
        speed_kmh=10,
        paddler_ability=ability,
        solver_time_secs=30
    )

    print(f"\nComparison:")
    print(f"  Equal ability race time: {result_equal['race_time']:.1f} min")
    print(f"  Varying ability race time: {result_varying['race_time']:.1f} min")
    print(f"  Time saved: {result_equal['race_time'] - result_varying['race_time']:.1f} min")

    print(f"\nCycle Rules (with varying ability, * = starting position):")
    for name, rule in result_varying['cycle_rules'].items():
        print(f"  {name}: {rule}")
    print(f"\nFull Schedule:")
    print(result_varying['schedule'])
    print("\nNote: Ana (1.5x) gets high-value Seat 1 for consecutive stints")

    return result_equal, result_varying


if __name__ == "__main__":
    import sys

    examples = {
        "1": example_basic,
        "2": example_custom_eligibility,
        "3": example_different_crew_size,
        "4": example_meta_optimization,
        "5": example_paddler_ability,
    }

    if len(sys.argv) > 1:
        option = sys.argv[1]
        if option in examples:
            examples[option]()
        else:
            print(f"Unknown option: {option}")
            print("Usage: python example.py [1|2|3|4|5]")
    else:
        # Run all examples
        for fn in examples.values():
            fn()
