# Full MILP with S + Y linearization and race time
import itertools, math
import pandas as pd
import numpy as np
from pulp import LpProblem, LpMaximize, LpVariable, lpSum, LpStatus, value, PULP_CBC_CMD
from .fatigue import compute_output_table


def validate_paddlers(paddlers, n_seats=6, n_paddlers=9):
    """Validate paddler DataFrame has required structure.

    Args:
        paddlers: DataFrame with 'name' column
        n_seats: Number of seats in canoe (default 6)
        n_paddlers: Expected crew size (default 9)

    Returns:
        DataFrame with reset index

    Raises:
        ValueError: If validation fails
    """
    # Check required columns
    if 'name' not in paddlers.columns:
        raise ValueError("Missing required column: 'name'")

    # Check paddler count
    if len(paddlers) != n_paddlers:
        raise ValueError(
            f"Expected {n_paddlers} paddlers, got {len(paddlers)}. "
            f"Need {n_paddlers} paddlers to fill {n_seats} seats with "
            f"{n_paddlers - n_seats} resting."
        )

    # Check unique names
    if paddlers.name.duplicated().any():
        dupes = paddlers.name[paddlers.name.duplicated()].tolist()
        raise ValueError(f"Duplicate paddler names: {dupes}")

    # Return with reset index for consistent indexing
    return paddlers.reset_index(drop=True)


def validate_eligibility(eligibility, n_paddlers, n_seats):
    """Validate seat eligibility matrix.

    Args:
        eligibility: numpy array of shape (n_paddlers, n_seats)
        n_paddlers: Expected number of paddlers
        n_seats: Expected number of seats

    Raises:
        ValueError: If eligibility matrix is invalid or infeasible
    """
    eligibility = np.asarray(eligibility)

    if eligibility.shape != (n_paddlers, n_seats):
        raise ValueError(
            f"Eligibility matrix must be ({n_paddlers}, {n_seats}), "
            f"got {eligibility.shape}"
        )

    # Check each seat has at least one eligible paddler
    for s in range(n_seats):
        n_eligible = eligibility[:, s].sum()
        if n_eligible == 0:
            raise ValueError(f"Seat {s+1} has no eligible paddlers")

    # Check each paddler is eligible for at least one seat
    for p in range(n_paddlers):
        n_seats_eligible = eligibility[p, :].sum()
        if n_seats_eligible == 0:
            raise ValueError(f"Paddler {p} is not eligible for any seat")


def validate_params(stint_min, max_consecutive, distance_km, speed_kmh, switch_time_min):
    """Validate optimization parameters.

    Raises:
        ValueError: If any parameter is invalid
    """
    if stint_min <= 0:
        raise ValueError(f"stint_min must be positive, got {stint_min}")
    if max_consecutive < 1:
        raise ValueError(f"max_consecutive must be >= 1, got {max_consecutive}")
    if distance_km <= 0:
        raise ValueError(f"distance_km must be positive, got {distance_km}")
    if speed_kmh <= 0:
        raise ValueError(f"speed_kmh must be positive, got {speed_kmh}")
    if switch_time_min < 0:
        raise ValueError(f"switch_time_min must be non-negative, got {switch_time_min}")


def expand_cycle_to_race(cycle_schedule, cycle_length, n_stints):
    """Expand cycle schedule to full race by repeating pattern.

    Args:
        cycle_schedule: DataFrame with schedule for one cycle
        cycle_length: Number of stints in one cycle
        n_stints: Total number of stints in the race

    Returns:
        DataFrame with full race schedule
    """
    full = pd.DataFrame(index=range(n_stints), columns=cycle_schedule.columns)
    for t in range(n_stints):
        full.iloc[t] = cycle_schedule.iloc[t % cycle_length]
    return full


def format_cycle_rules(cycle_schedule, paddlers):
    """Format cycle schedule as per-paddler rotation rules.

    Args:
        cycle_schedule: DataFrame with stint rows (index), seat columns
        paddlers: DataFrame with 'name' column

    Returns:
        dict mapping paddler name to rule string
        e.g. {'Eduardo': 'Seat 3* -> Rest -> Seat 6'}
        where * marks the starting position (stint 0)
    """
    rules = {}
    cycle_length = len(cycle_schedule)

    for name in paddlers.name:
        positions = []
        for t in range(cycle_length):
            # Find where this paddler is at stint t
            found = False
            for col in cycle_schedule.columns:
                if cycle_schedule.iloc[t][col] == name:
                    seat_num = col.replace('seat', '')
                    pos = f"Seat {seat_num}"
                    if t == 0:
                        pos += "*"
                    positions.append(pos)
                    found = True
                    break
            if not found:
                pos = "Rest"
                if t == 0:
                    pos += "*"
                positions.append(pos)

        rules[name] = " -> ".join(positions)

    return rules


def solve_rotation_cycle(paddlers,
                         stint_min=40,
                         max_consecutive=6,
                         distance_km=60,
                         speed_kmh=10,
                         switch_time_min=1.5,
                         seat_eligibility=None,
                         seat_weights=None,
                         seat_entry_weight=None,
                         paddler_ability=None,
                         n_seats=6,
                         n_resting=3,
                         time_limit=60,
                         gap_tolerance=0.01):
    """Solve the crew rotation optimization using a cycle-based model.

    Models a single rotation cycle instead of the entire race, with wrap-around
    fatigue tracking. The cycle repeats for the full race duration.

    For 9 paddlers, 6 seats, 3 resting: cycle = 3 stints (paddle 2, rest 1).
    This reduces model size by ~66% compared to full-race model.

    Note: Pattern penalties (entry_rule_penalty, switch_rule_penalty) are not
    supported in the cycle model since the repeating cycle already produces
    a simple, consistent pattern.

    Args:
        paddlers: DataFrame with 'name' column
        stint_min: Duration of each stint in minutes
        max_consecutive: Maximum consecutive stints (variable bound)
        distance_km: Race distance in kilometers
        speed_kmh: Base speed at output 1.0 in km/h
        switch_time_min: Time penalty per crew switch in minutes
        seat_eligibility: Optional (n_paddlers, n_seats) matrix where 1 = paddler can sit in seat.
                         If None, all paddlers are eligible for all seats.
        seat_weights: Optional list of seat importance weights. Default [1.2, 1.1, 0.9, 0.9, 0.9, 1.1]
        seat_entry_weight: Optional list of entry ease weights per seat. Default all 1.0.
                          >1 = easier to enter, <1 = harder to enter.
        paddler_ability: Optional list of ability multipliers per paddler. Default all 1.0.
                        >1 = stronger paddler, <1 = weaker paddler.
        n_seats: Number of seats in canoe (default 6)
        n_resting: Number of paddlers resting each stint (default 3)
        time_limit: Maximum solver time in seconds (default 60)
        gap_tolerance: Acceptable gap from optimal (default 0.01 = 1%)

    Returns:
        dict with 'status', 'schedule', 'cycle_schedule', 'avg_output', 'race_time',
        'parameters', 'paddler_summary', 'summary_stats'
    """
    # Validate inputs
    n_paddlers = n_seats + n_resting
    paddlers = validate_paddlers(paddlers, n_seats=n_seats, n_paddlers=n_paddlers)
    validate_params(stint_min, max_consecutive, distance_km, speed_kmh, switch_time_min)

    P = len(paddlers)
    S = n_seats

    # Compute cycle length
    if n_paddlers % n_resting != 0:
        raise ValueError(f"n_paddlers ({n_paddlers}) must be divisible by n_resting ({n_resting})")
    cycle_length = n_paddlers // n_resting  # e.g., 9 // 3 = 3

    # Build or validate eligibility matrix
    if seat_eligibility is None:
        eligibility = np.ones((P, S), dtype=int)
    else:
        eligibility = np.asarray(seat_eligibility)
    validate_eligibility(eligibility, P, S)

    # Default seat weights
    if seat_weights is None:
        if S == 6:
            seat_weights = [1.2, 1.1, 0.9, 0.9, 0.9, 1.1]
        else:
            seat_weights = [1.0] * S

    if len(seat_weights) != S:
        raise ValueError(f"seat_weights must have {S} elements, got {len(seat_weights)}")

    # Default seat entry weights (all 1.0 = normal difficulty)
    if seat_entry_weight is None:
        seat_entry_weight = [1.0] * S
    if len(seat_entry_weight) != S:
        raise ValueError(f"seat_entry_weight must have {S} elements, got {len(seat_entry_weight)}")
    if any(w <= 0 for w in seat_entry_weight):
        raise ValueError("seat_entry_weight values must be positive")

    # Check if we need entry weight optimization (any weight != 1.0)
    use_entry_weights = any(w != 1.0 for w in seat_entry_weight)

    # Default paddler ability (all 1.0 = equal ability)
    if paddler_ability is None:
        paddler_ability = [1.0] * P
    if len(paddler_ability) != P:
        raise ValueError(f"paddler_ability must have {P} elements, got {len(paddler_ability)}")
    if any(a <= 0 for a in paddler_ability):
        raise ValueError("paddler_ability values must be positive")

    # Compute number of stints for race time calculation
    n_stints = math.ceil((distance_km/speed_kmh*60)/stint_min)

    output_table = compute_output_table(stint_min, max_consecutive)

    prob = LpProblem("OC6_Cycle", LpMaximize)

    # Decision variables - only for cycle_length stints
    eligible_pairs = [(p, s) for p in range(P) for s in range(S) if eligibility[p, s]]
    X = {(p,s,t): LpVariable(f"X_{p}_{s}_{t}", cat="Binary")
         for (p,s) in eligible_pairs for t in range(cycle_length)}
    R = {(p,t): LpVariable(f"R_{p}_{t}", cat="Binary")
         for p,t in itertools.product(range(P), range(cycle_length))}
    Scon = {(p,t): LpVariable(f"S_{p}_{t}",
                              lowBound=0, upBound=max_consecutive, cat="Integer")
            for p,t in itertools.product(range(P), range(cycle_length))}
    Y = {(p,t,k): LpVariable(f"Y_{p}_{t}_{k}", cat="Binary")
         for p,t,k in itertools.product(range(P), range(cycle_length), range(1,max_consecutive+1))}

    # Auxiliary variables Q for linearization
    max_weight = max(seat_weights)
    Q = {(p,t,k): LpVariable(f"Q_{p}_{t}_{k}", lowBound=0, upBound=max_weight, cat="Continuous")
         for p,t,k in itertools.product(range(P), range(cycle_length), range(1,max_consecutive+1))}

    # NEW: Link variable for wrap-around consecutive tracking
    Link = {p: LpVariable(f"Link_{p}", lowBound=0, upBound=max_consecutive, cat="Integer")
            for p in range(P)}

    # Entry variables for seat entry weight optimization
    if use_entry_weights:
        # Entry[p,s,t] = 1 if paddler p enters seat s at stint t from rest
        # Include t=0 for wrap-around (entering from rest at t=cycle_length-1)
        Entry = {(p, s, t): LpVariable(f"Entry_{p}_{s}_{t}", cat="Binary")
                 for (p, s) in eligible_pairs for t in range(cycle_length)}

    # Seat assignment constraints: exactly one paddler per seat
    for s,t in itertools.product(range(S), range(cycle_length)):
        eligible_for_seat = [p for p in range(P) if eligibility[p, s]]
        prob += lpSum(X[p,s,t] for p in eligible_for_seat) == 1

    # Paddler in one place: either in one eligible seat or resting
    for p,t in itertools.product(range(P), range(cycle_length)):
        eligible_seats = [s for s in range(S) if eligibility[p, s]]
        prob += lpSum(X[p,s,t] for s in eligible_seats) + R[p,t] == 1

    # Exactly n_resting paddlers resting each stint
    for t in range(cycle_length):
        prob += lpSum(R[p,t] for p in range(P)) == n_resting

    # Wrap-around consecutive tracking
    # Link[p] = Scon[p, cycle_length-1] if paddling at end of cycle, else 0
    for p in range(P):
        prob += Link[p] <= Scon[p, cycle_length-1]
        prob += Link[p] <= max_consecutive * (1 - R[p, cycle_length-1])
        prob += Link[p] >= Scon[p, cycle_length-1] - max_consecutive * R[p, cycle_length-1]

    # Scon[p,0] with wrap-around: continues from Link if paddling
    for p in range(P):
        prob += Scon[p, 0] <= max_consecutive * (1 - R[p, 0])
        prob += Scon[p, 0] >= Link[p] + 1 - max_consecutive * R[p, 0]
        prob += Scon[p, 0] <= Link[p] + 1

    # Standard consecutive tracking for t > 0
    for p in range(P):
        for t in range(1, cycle_length):
            prob += Scon[p, t] <= Scon[p, t-1] + 1
            prob += Scon[p, t] >= Scon[p, t-1] + 1 - max_consecutive * R[p, t]
            prob += Scon[p, t] <= max_consecutive * (1 - R[p, t])

    # Y-variable linearization
    for p,t in itertools.product(range(P), range(cycle_length)):
        prob += lpSum(Y[p,t,k] for k in range(1,max_consecutive+1)) == 1 - R[p,t]
        prob += Scon[p,t] == lpSum(k*Y[p,t,k] for k in range(1,max_consecutive+1))

    # Q-variable linearization
    for p,t,k in itertools.product(range(P), range(cycle_length), range(1,max_consecutive+1)):
        eligible_seats = [s for s in range(S) if eligibility[p, s]]
        V_pt = lpSum(X[p,s,t] * seat_weights[s] for s in eligible_seats)
        prob += Q[p,t,k] <= max_weight * Y[p,t,k]
        prob += Q[p,t,k] <= V_pt
        prob += Q[p,t,k] >= V_pt - max_weight * (1 - Y[p,t,k])

    # Entry weight constraints
    if use_entry_weights:
        for (p, s) in eligible_pairs:
            # t=0: wrap-around entry (from rest at t=cycle_length-1)
            prob += Entry[p, s, 0] >= X[p, s, 0] + R[p, cycle_length-1] - 1
            prob += Entry[p, s, 0] <= X[p, s, 0]
            prob += Entry[p, s, 0] <= R[p, cycle_length-1]
            # t > 0: standard entry from previous stint
            for t in range(1, cycle_length):
                prob += Entry[p, s, t] >= X[p, s, t] + R[p, t-1] - 1
                prob += Entry[p, s, t] <= X[p, s, t]
                prob += Entry[p, s, t] <= R[p, t-1]

    # Objective: Maximize weighted output minus entry weight penalties
    weighted_output = lpSum(
        Q[p,t,k] * output_table[k] * paddler_ability[p]
        for p,t,k in itertools.product(range(P), range(cycle_length), range(1,max_consecutive+1))
    )

    objective = weighted_output

    if use_entry_weights:
        nominal_paddle_time = distance_km / speed_kmh * 60
        entry_weight_scale = switch_time_min / nominal_paddle_time * sum(seat_weights)
        # t=0 entries only happen (n_cycles - 1) times (first stint of race is free)
        # t>0 entries happen n_cycles times
        n_cycles = n_stints / cycle_length
        t0_weight = (n_cycles - 1) / n_cycles if n_cycles > 1 else 0.0
        entry_weight_penalty = entry_weight_scale * (
            lpSum(
                Entry[p, s, 0] * (1.0 / seat_entry_weight[s] - 1.0) * t0_weight
                for (p, s) in eligible_pairs
            ) +
            lpSum(
                Entry[p, s, t] * (1.0 / seat_entry_weight[s] - 1.0)
                for (p, s) in eligible_pairs for t in range(1, cycle_length)
            )
        )
        objective = objective - entry_weight_penalty

    prob += objective

    # Solve
    solver = PULP_CBC_CMD(msg=0, timeLimit=time_limit, gapRel=gap_tolerance)
    prob.solve(solver)

    # Extract cycle schedule
    cycle_sched = pd.DataFrame("-", index=range(cycle_length), columns=[f"seat{s+1}" for s in range(S)])
    for (p,s,t), var in X.items():
        if var.value() and var.value() > 0.5:
            cycle_sched.iloc[t,s] = paddlers.name.iloc[p]

    # Compute stint outputs from cycle (steady-state with wrap-around)
    # The solver computed Scon values assuming wrap-around is in effect
    steady_stint_outputs = []
    for t in range(cycle_length):
        num = 0
        den = sum(seat_weights)
        for p in range(P):
            eligible_seats = [s for s in range(S) if eligibility[p, s]]
            for s in eligible_seats:
                if (p,s,t) in X and X[p,s,t].value() and X[p,s,t].value() > 0.5:
                    for k in output_table.keys():
                        if Y[p,t,k].value() and Y[p,t,k].value() > 0.5:
                            num += seat_weights[s] * output_table[k] * paddler_ability[p]
        steady_stint_outputs.append(num / den)

    # Exact race simulation stint-by-stint
    # This handles fresh start, wrap-around, and partial cycles correctly
    stint_outputs = []
    consecutive = {p: 0 for p in range(P)}  # Track consecutive from fresh start
    # Track per-paddler stats for exact calculation
    total_stints_paddled = {p: 0 for p in range(P)}
    max_consecutive_streak = {p: 0 for p in range(P)}

    for t in range(n_stints):
        cycle_t = t % cycle_length

        # Determine who is paddling this stint (from the cycle pattern)
        paddling_this_stint = {}
        for p in range(P):
            is_paddling = R[p, cycle_t].value() is not None and R[p, cycle_t].value() < 0.5
            paddling_this_stint[p] = is_paddling

        # Update consecutive counts and per-paddler stats
        for p in range(P):
            if paddling_this_stint[p]:
                consecutive[p] += 1
                total_stints_paddled[p] += 1
                max_consecutive_streak[p] = max(max_consecutive_streak[p], consecutive[p])
            else:
                consecutive[p] = 0

        # Calculate stint output
        num = 0
        den = sum(seat_weights)
        for p in range(P):
            if paddling_this_stint[p]:
                eligible_seats = [s for s in range(S) if eligibility[p, s]]
                for s in eligible_seats:
                    if (p, s, cycle_t) in X and X[p, s, cycle_t].value() and X[p, s, cycle_t].value() > 0.5:
                        k = min(consecutive[p], max_consecutive)
                        if k > 0:
                            num += seat_weights[s] * output_table[k] * paddler_ability[p]
        stint_outputs.append(num / den)

    # Calculate race time from exact stint outputs
    total_output = sum(stint_outputs)
    avg_output = total_output / n_stints

    steady_cycle_output = sum(steady_stint_outputs)  # Keep for stats

    nominal = distance_km / speed_kmh * 60
    switches = n_stints - 1
    race_time = nominal / avg_output + switches * switch_time_min

    # Expand cycle to full race schedule
    full_sched = expand_cycle_to_race(cycle_sched, cycle_length, n_stints)

    # Compute paddler summary statistics (exact, from simulation)
    paddler_stats = []
    for p in range(P):
        # Per-cycle stats (from cycle pattern)
        stints_per_cycle = sum(1 for t in range(cycle_length)
                               if R[p,t].value() is not None and R[p,t].value() < 0.5)

        paddler_stats.append({
            'name': paddlers.name.iloc[p],
            'stints_paddled': total_stints_paddled[p],
            'stints_rested': n_stints - total_stints_paddled[p],
            'total_time_min': float(total_stints_paddled[p] * stint_min),
            'longest_stretch_stints': max_consecutive_streak[p],
            'longest_stretch_min': float(max_consecutive_streak[p] * stint_min),
            'stints_paddled_per_cycle': stints_per_cycle,
        })

    paddler_summary = pd.DataFrame(paddler_stats)

    # Compute aggregate stats (exact)
    summary_stats = {
        'cycle_length': cycle_length,
        'n_stints': n_stints,
        'n_cycles': n_stints / cycle_length,
        'avg_time_per_paddler_min': float(paddler_summary['total_time_min'].mean()),
        'max_time_any_paddler_min': float(paddler_summary['total_time_min'].max()),
        'min_time_any_paddler_min': float(paddler_summary['total_time_min'].min()),
        'max_consecutive_stretch_min': float(paddler_summary['longest_stretch_min'].max()),
    }

    return {
        "status": LpStatus[prob.status],
        "schedule": full_sched,
        "cycle_schedule": cycle_sched,
        "cycle_rules": format_cycle_rules(cycle_sched, paddlers),
        "avg_output": avg_output,
        "race_time": race_time,
        "parameters": {
            "stint_min": stint_min,
            "n_stints": n_stints,
            "cycle_length": cycle_length,
            "seat_entry_weight": seat_entry_weight,
            "paddler_ability": paddler_ability,
        },
        "paddler_summary": paddler_summary,
        "summary_stats": summary_stats,
    }
