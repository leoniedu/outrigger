#!/usr/bin/env python3
"""Example usage of the outrigger rotation optimizer."""

import pandas as pd
import numpy as np
import time
from outrigger_opt import solve_rotation_full, solve_rotation_cycle, optimize_stint_range


def example_all_eligible():
    """Option 1: All paddlers eligible for all seats (simplest)."""
    print("=" * 60)
    print("OPTION 1: All paddlers eligible for all seats")
    print("=" * 60)

    paddlers = pd.DataFrame({
        'name': ['Eduardo', 'Vitor', 'Guilherme', 'Airton', 'Ricardo',
                 'Sergio', 'Everson', 'Marcelo', 'Ze']
    })

    print("\nPaddlers:")
    print(paddlers)
    print("\nAll 9 paddlers can sit in any of the 6 seats.")

    print("\nSolving...")
    result = solve_rotation_full(
        paddlers,
        stint_min=30,
        distance_km=60,
        speed_kmh=9.5,
        switch_time_min=1,
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
        'name': ['V1', 'V2', 'V3', 'F1', 'F2', 'F3', 'F4', 'L1', 'L2']
    })

    # Define who can sit where (rows=paddlers, cols=seats 1-6)
    eligibility = np.array([
        [1, 1, 0, 0, 0, 0],  # V1: front only
        [1, 1, 0, 0, 0, 0],  # V2: front only
        [1, 1, 1, 0, 0, 0],  # V3: front + seat 3
        [0, 0, 1, 1, 0, 0],  # H1: middle front
        [0, 0, 1, 1, 0, 0],  # H2: middle front
        [0, 0, 0, 1, 1, 0],  # H3: middle back
        [0, 0, 0, 1, 1, 0],  # H4: middle back
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
        seat_entry_weight=[1, 1.2, 1.2, 1.2, 1.2, 1],
        stint_min=20,
        time_limit=120,
        switch_time_min=.5,
        max_consecutive=3,
        gap_tolerance=.001,
        
    )

    print(f"\nStatus: {result['status']}")
    print(f"Race time: {result['race_time']:.1f} min")
    print(f"\nSchedule:")
    print(result['schedule'])

    print(f"\nPaddler Summary:")
    print(result['paddler_summary'].to_string(index=False))

    print(f"\nAggregate Stats:")
    for key, val in result['summary_stats'].items():
        print(f"  {key}: {val:.1f}")

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
        'name': ['Ana', 'Ben', 'Carlos', 'Diana', 'Eve',
                 'Gina', 'Hiro', 'Frank', 'Ivan']
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


def example_pattern_penalties():
    """Option 5: Pattern consistency penalties."""
    print("\n" + "=" * 60)
    print("OPTION 5: Pattern consistency penalties")
    print("=" * 60)

    paddlers = pd.DataFrame({
        'name': ['Ana', 'Ben', 'Carlos', 'Diana', 'Eve',
                 'Gina', 'Hiro', 'Frank', 'Ivan']
    })

    print("\nComparing schedules with and without pattern penalties...")
    print("Pattern penalties encourage simpler rotation rules.\n")

    # Without penalties (pure output optimization)
    print("-" * 40)
    print("WITHOUT pattern penalties:")
    print("-" * 40)
    result_no_penalty = solve_rotation_full(
        paddlers,
        stint_min=40,
        distance_km=60,
        speed_kmh=10,
        time_limit=30,
        entry_rule_penalty=0.0,
        switch_rule_penalty=0.0
    )
    print(f"Status: {result_no_penalty['status']}")
    print(f"Race time: {result_no_penalty['race_time']:.1f} min")
    print(f"Avg output: {result_no_penalty['avg_output']:.3f}")
    print(f"\nSchedule:")
    print(result_no_penalty['schedule'])

    # With penalties (encourage simpler patterns)
    print("\n" + "-" * 40)
    print("WITH pattern penalties (entry=0.05, switch=0.05):")
    print("-" * 40)
    result_with_penalty = solve_rotation_full(
        paddlers,
        stint_min=40,
        distance_km=60,
        speed_kmh=10,
        time_limit=300,
        entry_rule_penalty=0.05,
        switch_rule_penalty=0.05
    )
    print(f"Status: {result_with_penalty['status']}")
    print(f"Race time: {result_with_penalty['race_time']:.1f} min")
    print(f"Avg output: {result_with_penalty['avg_output']:.3f}")
    print(f"\nSchedule:")
    print(result_with_penalty['schedule'])

    # Compare pattern complexity
    print("\n" + "-" * 40)
    print("Pattern Complexity Comparison:")
    print("-" * 40)
    ps_no = result_no_penalty['pattern_stats']
    ps_with = result_with_penalty['pattern_stats']

    print(f"                          No Penalty    With Penalty")
    print(f"Total entry rules:        {ps_no['total_entry_rules']:>10}    {ps_with['total_entry_rules']:>12}")
    print(f"Total switch rules:       {ps_no['total_switch_rules']:>10}    {ps_with['total_switch_rules']:>12}")
    print(f"Avg entry rules/paddler:  {ps_no['avg_entry_rules_per_paddler']:>10.1f}    {ps_with['avg_entry_rules_per_paddler']:>12.1f}")
    print(f"Avg switch rules/paddler: {ps_no['avg_switch_rules_per_paddler']:>10.1f}    {ps_with['avg_switch_rules_per_paddler']:>12.1f}")

    print("\n" + "-" * 40)
    print("Per-paddler breakdown (with penalties):")
    print("-" * 40)
    cols = ['name', 'stints_paddled', 'entry_rules', 'switch_rules']
    print(result_with_penalty['paddler_summary'][cols].to_string(index=False))

    return result_no_penalty, result_with_penalty


def example_cycle_vs_full():
    """Option 6: Compare cycle-based solver with full-race solver."""
    print("\n" + "=" * 60)
    print("OPTION 6: Cycle-based solver vs Full-race solver")
    print("=" * 60)

    paddlers = pd.DataFrame({
        'name': ['Ana', 'Ben', 'Carlos', 'Diana', 'Eve',
                 'Gina', 'Hiro', 'Frank', 'Ivan']
    })

    print("\nComparing two approaches:")
    print("- Full-race: Models all stints (more variables)")
    print("- Cycle-based: Models one repeating cycle (fewer variables)")
    print("  For 9 paddlers, 6 seats, 3 resting: cycle = 3 stints\n")

    # Full-race solver
    print("-" * 40)
    print("FULL-RACE SOLVER:")
    print("-" * 40)
    start_full = time.time()
    result_full = solve_rotation_full(
        paddlers,
        stint_min=40,
        distance_km=60,
        speed_kmh=10,
        time_limit=60
    )
    elapsed_full = time.time() - start_full
    print(f"Status: {result_full['status']}")
    print(f"Solve time: {elapsed_full:.2f}s")
    print(f"n_stints: {result_full['parameters']['n_stints']}")
    print(f"Race time: {result_full['race_time']:.1f} min")
    print(f"Avg output: {result_full['avg_output']:.3f}")
    print(f"\nSchedule:")
    print(result_full['schedule'])

    # Cycle-based solver
    print("\n" + "-" * 40)
    print("CYCLE-BASED SOLVER:")
    print("-" * 40)
    start_cycle = time.time()
    result_cycle = solve_rotation_cycle(
        paddlers,
        stint_min=40,
        distance_km=60,
        speed_kmh=10,
        time_limit=60
    )
    elapsed_cycle = time.time() - start_cycle
    print(f"Status: {result_cycle['status']}")
    print(f"Solve time: {elapsed_cycle:.2f}s")
    print(f"Cycle length: {result_cycle['parameters']['cycle_length']} stints")
    print(f"n_stints (full race): {result_cycle['parameters']['n_stints']}")
    print(f"Race time: {result_cycle['race_time']:.1f} min")
    print(f"Avg output: {result_cycle['avg_output']:.3f}")
    print(f"\nCycle Schedule (repeating pattern):")
    print(result_cycle['cycle_schedule'])
    print(f"\nFull Schedule (cycle expanded):")
    print(result_cycle['schedule'])

    # Comparison
    print("\n" + "-" * 40)
    print("COMPARISON:")
    print("-" * 40)
    print(f"                    Full-Race    Cycle-Based")
    print(f"Solve time:         {elapsed_full:>8.2f}s    {elapsed_cycle:>11.2f}s")
    print(f"Race time:          {result_full['race_time']:>8.1f} min  {result_cycle['race_time']:>9.1f} min")
    print(f"Avg output:         {result_full['avg_output']:>8.3f}      {result_cycle['avg_output']:>11.3f}")

    speedup = elapsed_full / elapsed_cycle if elapsed_cycle > 0 else float('inf')
    print(f"\nSpeedup: {speedup:.1f}x faster with cycle-based approach")

    diff = abs(result_full['race_time'] - result_cycle['race_time'])
    print(f"Race time difference: {diff:.1f} min")

    return result_full, result_cycle


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        option = sys.argv[1]
        if option == "1":
            example_all_eligible()
        elif option == "2":
            example_custom_eligibility()
        elif option == "3":
            example_different_crew_size()
        elif option == "4":
            example_meta_optimization()
        elif option == "5":
            example_pattern_penalties()
        elif option == "6":
            example_cycle_vs_full()
        else:
            print(f"Unknown option: {option}")
            print("Usage: python example.py [1|2|3|4|5|6]")
    else:
        # Run all examples
        example_all_eligible()
        example_custom_eligibility()
        example_different_crew_size()
        example_meta_optimization()
        example_pattern_penalties()
        example_cycle_vs_full()
