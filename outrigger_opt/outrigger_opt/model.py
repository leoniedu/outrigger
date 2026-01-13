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


def solve_rotation_full(paddlers,
                        stint_min=40,
                        max_consecutive=6,
                        distance_km=60,
                        speed_kmh=10,
                        switch_time_min=1.5,
                        seat_eligibility=None,
                        seat_weights=None,
                        seat_entry_weight=None,
                        n_seats=6,
                        n_resting=3,
                        time_limit=60,
                        gap_tolerance=0.01,
                        entry_rule_penalty=0.0,
                        switch_rule_penalty=0.0):
    """Solve the crew rotation optimization problem.

    Args:
        paddlers: DataFrame with 'name' column
        stint_min: Duration of each stint in minutes
        max_consecutive: Maximum consecutive stints before required rest
        distance_km: Race distance in kilometers
        speed_kmh: Base speed at output 1.0 in km/h
        switch_time_min: Time penalty per crew switch in minutes
        seat_eligibility: Optional (n_paddlers, n_seats) matrix where 1 = paddler can sit in seat.
                         If None, all paddlers are eligible for all seats.
        seat_weights: Optional list of seat importance weights. Default [1.2, 1.1, 0.9, 0.9, 0.9, 1.1]
        seat_entry_weight: Optional list of entry ease weights per seat. Default all 1.0.
                          >1 = easier to enter (reduces effective switch time),
                          <1 = harder to enter (increases effective switch time).
                          Affects both optimization and race_time calculation.
        n_seats: Number of seats in canoe (default 6)
        n_resting: Number of paddlers resting each stint (default 3)
        time_limit: Maximum solver time in seconds (default 60)
        gap_tolerance: Acceptable gap from optimal (default 0.01 = 1%)
        entry_rule_penalty: Penalty per entry rule per paddler (default 0.0 = disabled)
        switch_rule_penalty: Penalty per switch rule per paddler (default 0.0 = disabled)

    Returns:
        dict with 'status', 'schedule', 'avg_output', 'race_time', 'parameters',
        'paddler_summary', 'summary_stats', 'pattern_stats'
    """
    # Validate inputs
    n_paddlers = n_seats + n_resting
    paddlers = validate_paddlers(paddlers, n_seats=n_seats, n_paddlers=n_paddlers)
    validate_params(stint_min, max_consecutive, distance_km, speed_kmh, switch_time_min)

    P = len(paddlers)
    S = n_seats

    # Build or validate eligibility matrix
    if seat_eligibility is None:
        # Default: all paddlers eligible for all seats
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

    n_stints = math.ceil((distance_km/speed_kmh*60)/stint_min)
    output_table = compute_output_table(stint_min, max_consecutive)

    prob = LpProblem("OC6", LpMaximize)

    # Decision variables - only create X for eligible (paddler, seat) pairs
    eligible_pairs = [(p, s) for p in range(P) for s in range(S) if eligibility[p, s]]
    X = {(p,s,t): LpVariable(f"X_{p}_{s}_{t}", cat="Binary")
         for (p,s) in eligible_pairs for t in range(n_stints)}
    R = {(p,t): LpVariable(f"R_{p}_{t}", cat="Binary")
         for p,t in itertools.product(range(P), range(n_stints))}
    Scon = {(p,t): LpVariable(f"S_{p}_{t}",
                              lowBound=0, upBound=max_consecutive, cat="Integer")
            for p,t in itertools.product(range(P), range(n_stints))}
    Y = {(p,t,k): LpVariable(f"Y_{p}_{t}_{k}", cat="Binary")
         for p,t,k in itertools.product(range(P), range(n_stints), range(1,max_consecutive+1))}

    # Auxiliary variables Q[p,t,k] to linearize (Σ_s X[p,s,t]*seat_weight[s]) * Y[p,t,k]
    max_weight = max(seat_weights)
    Q = {(p,t,k): LpVariable(f"Q_{p}_{t}_{k}", lowBound=0, upBound=max_weight, cat="Continuous")
         for p,t,k in itertools.product(range(P), range(n_stints), range(1,max_consecutive+1))}

    # Pattern consistency variables (only if penalties are non-zero)
    use_pattern_penalties = entry_rule_penalty > 0 or switch_rule_penalty > 0

    if use_pattern_penalties:
        # EntryUsed[p,s] = 1 if paddler p ever enters seat s from rest
        EntryUsed = {(p,s): LpVariable(f"EntryUsed_{p}_{s}", cat="Binary")
                     for p,s in eligible_pairs}

        # TransitionUsed[p,s,s'] = 1 if paddler p ever transitions from seat s to s' while paddling
        # Include s=s' for "stay in same seat" transitions
        TransitionUsed = {}
        for p in range(P):
            eligible_seats_p = [s for s in range(S) if eligibility[p, s]]
            for s in eligible_seats_p:
                for s_prime in eligible_seats_p:
                    TransitionUsed[(p, s, s_prime)] = LpVariable(
                        f"TransitionUsed_{p}_{s}_{s_prime}", cat="Binary")

    # Entry variables for seat entry weight optimization (tracks each entry from rest)
    if use_entry_weights:
        # Entry[p,s,t] = 1 if paddler p enters seat s at stint t from rest (t > 0 only)
        Entry = {(p, s, t): LpVariable(f"Entry_{p}_{s}_{t}", cat="Binary")
                 for (p, s) in eligible_pairs for t in range(1, n_stints)}

    # Seat assignment constraints: exactly one paddler per seat (from eligible paddlers)
    for s,t in itertools.product(range(S), range(n_stints)):
        eligible_for_seat = [p for p in range(P) if eligibility[p, s]]
        prob += lpSum(X[p,s,t] for p in eligible_for_seat) == 1

    # Paddler in one place: either in one eligible seat or resting
    for p,t in itertools.product(range(P), range(n_stints)):
        eligible_seats = [s for s in range(S) if eligibility[p, s]]
        prob += lpSum(X[p,s,t] for s in eligible_seats) + R[p,t] == 1

    # Exactly n_resting paddlers resting each stint
    for t in range(n_stints):
        prob += lpSum(R[p,t] for p in range(P)) == n_resting

    # Consecutive stint tracking
    for p in range(P):
        prob += Scon[p,0] == 1 - R[p,0]
        for t in range(1, n_stints):
            prob += Scon[p,t] <= Scon[p,t-1] + 1
            prob += Scon[p,t] >= Scon[p,t-1] + 1 - max_consecutive * R[p,t]
            prob += Scon[p,t] <= max_consecutive * (1 - R[p,t])

    # Y-variable linearization (link Y to consecutive stint count S)
    for p,t in itertools.product(range(P), range(n_stints)):
        prob += lpSum(Y[p,t,k] for k in range(1,max_consecutive+1)) == 1 - R[p,t]
        prob += Scon[p,t] == lpSum(k*Y[p,t,k] for k in range(1,max_consecutive+1))

    # Q-variable linearization: Q[p,t,k] = (Σ_s X[p,s,t]*seat_weight[s]) * Y[p,t,k]
    for p,t,k in itertools.product(range(P), range(n_stints), range(1,max_consecutive+1)):
        eligible_seats = [s for s in range(S) if eligibility[p, s]]
        V_pt = lpSum(X[p,s,t] * seat_weights[s] for s in eligible_seats)
        prob += Q[p,t,k] <= max_weight * Y[p,t,k]
        prob += Q[p,t,k] <= V_pt
        prob += Q[p,t,k] >= V_pt - max_weight * (1 - Y[p,t,k])

    # Pattern consistency constraints
    if use_pattern_penalties:
        # EntryUsed[p,s] >= X[p,s,t] when entering from rest (R[p,t-1] = 1)
        # Note: t=0 is the starting position - doesn't count as "entry from rest"
        for p,s in eligible_pairs:
            # For t > 0, entry from rest means R[p,t-1]=1 and now paddling
            for t in range(1, n_stints):
                # EntryUsed[p,s] >= X[p,s,t] + R[p,t-1] - 1
                # This activates when X[p,s,t]=1 AND R[p,t-1]=1
                prob += EntryUsed[p,s] >= X[p,s,t] + R[p,t-1] - 1

        # TransitionUsed[p,s,s'] >= X[p,s,t-1] + X[p,s',t] - 1 when both paddling
        # This activates when paddler is in seat s at t-1, seat s' at t, and not resting either time
        for p in range(P):
            eligible_seats_p = [s for s in range(S) if eligibility[p, s]]
            for t in range(1, n_stints):
                for s in eligible_seats_p:
                    for s_prime in eligible_seats_p:
                        # Need: X[p,s,t-1]=1, X[p,s',t]=1, R[p,t-1]=0, R[p,t]=0
                        # TransitionUsed >= X[p,s,t-1] + X[p,s',t] - 1 - R[p,t-1] - R[p,t]
                        # Simplify: if R[p,t-1]=0 and R[p,t]=0, then this becomes X + X' - 1
                        # If either R=1, constraint is satisfied (RHS <= 0)
                        prob += TransitionUsed[p,s,s_prime] >= (
                            X[p,s,t-1] + X[p,s_prime,t] - 1 - R[p,t-1] - R[p,t]
                        )

    # Entry weight constraints: track when paddlers enter each seat from rest
    if use_entry_weights:
        for (p, s) in eligible_pairs:
            for t in range(1, n_stints):
                # Entry[p,s,t] = 1 when X[p,s,t] = 1 AND R[p,t-1] = 1
                # Lower bound: Entry >= X[p,s,t] + R[p,t-1] - 1
                prob += Entry[p, s, t] >= X[p, s, t] + R[p, t-1] - 1
                # Upper bounds to make Entry tight (only 1 when both conditions met)
                prob += Entry[p, s, t] <= X[p, s, t]
                prob += Entry[p, s, t] <= R[p, t-1]

    # Objective: Maximize weighted output minus pattern penalties minus entry difficulty penalties
    weighted_output = lpSum(
        Q[p,t,k] * output_table[k]
        for p,t,k in itertools.product(range(P), range(n_stints), range(1,max_consecutive+1))
    )

    objective = weighted_output

    if use_pattern_penalties:
        entry_penalty_term = entry_rule_penalty * lpSum(
            EntryUsed[p,s] for p,s in eligible_pairs
        )
        switch_penalty_term = switch_rule_penalty * lpSum(
            TransitionUsed[p,s,s_prime]
            for p in range(P)
            for s in range(S) if eligibility[p,s]
            for s_prime in range(S) if eligibility[p,s_prime]
        )
        objective = objective - entry_penalty_term - switch_penalty_term

    if use_entry_weights:
        # Penalty for entering hard-to-enter seats, bonus for easy seats
        # Scale factor converts switch time to output-equivalent units
        nominal_paddle_time = distance_km / speed_kmh * 60
        entry_weight_scale = switch_time_min / nominal_paddle_time * sum(seat_weights)
        # Penalty term: (1/entry_weight - 1) is positive for hard seats, negative for easy
        entry_weight_penalty = entry_weight_scale * lpSum(
            Entry[p, s, t] * (1.0 / seat_entry_weight[s] - 1.0)
            for (p, s) in eligible_pairs for t in range(1, n_stints)
        )
        objective = objective - entry_weight_penalty

    prob += objective

    # Solve with time limit and gap tolerance
    solver = PULP_CBC_CMD(msg=0, timeLimit=time_limit, gapRel=gap_tolerance)
    prob.solve(solver)

    # Extract schedule from solution
    sched = pd.DataFrame("-", index=range(n_stints), columns=[f"seat{s+1}" for s in range(S)])
    for (p,s,t), var in X.items():
        if var.value() and var.value() > 0.5:
            sched.iloc[t,s] = paddlers.name.iloc[p]

    # Compute stint outputs from solution
    stint_outputs = []
    for t in range(n_stints):
        num = 0
        den = sum(seat_weights)
        for p in range(P):
            eligible_seats = [s for s in range(S) if eligibility[p, s]]
            for s in eligible_seats:
                if (p,s,t) in X and X[p,s,t].value() and X[p,s,t].value() > 0.5:
                    for k in output_table.keys():
                        if Y[p,t,k].value() and Y[p,t,k].value() > 0.5:
                            num += seat_weights[s] * output_table[k]
        stint_outputs.append(num / den)

    avg_output = sum(stint_outputs) / n_stints
    nominal = distance_km/speed_kmh * 60
    switches = n_stints - 1
    race_time = nominal / avg_output + switches * switch_time_min

    # Compute paddler summary statistics
    paddler_stats = []
    for p in range(P):
        # Count stints paddled and find longest consecutive stretch
        stints_paddled = 0
        current_streak = 0
        max_streak = 0

        for t in range(n_stints):
            is_paddling = R[p,t].value() is not None and R[p,t].value() < 0.5
            if is_paddling:
                stints_paddled += 1
                current_streak += 1
                max_streak = max(max_streak, current_streak)
            else:
                current_streak = 0

        # Count pattern rules for this paddler
        entry_rules = 0
        switch_rules = 0
        if use_pattern_penalties:
            eligible_seats_p = [s for s in range(S) if eligibility[p, s]]
            for s in eligible_seats_p:
                if (p,s) in EntryUsed and EntryUsed[p,s].value() and EntryUsed[p,s].value() > 0.5:
                    entry_rules += 1
            for s in eligible_seats_p:
                for s_prime in eligible_seats_p:
                    if ((p,s,s_prime) in TransitionUsed and
                        TransitionUsed[p,s,s_prime].value() and
                        TransitionUsed[p,s,s_prime].value() > 0.5):
                        switch_rules += 1

        paddler_stats.append({
            'name': paddlers.name.iloc[p],
            'stints_paddled': stints_paddled,
            'stints_rested': n_stints - stints_paddled,
            'total_time_min': stints_paddled * stint_min,
            'longest_stretch_stints': max_streak,
            'longest_stretch_min': max_streak * stint_min,
            'entry_rules': entry_rules,
            'switch_rules': switch_rules,
        })

    paddler_summary = pd.DataFrame(paddler_stats)

    # Compute aggregate stats
    summary_stats = {
        'avg_time_per_paddler_min': paddler_summary['total_time_min'].mean(),
        'max_time_any_paddler_min': paddler_summary['total_time_min'].max(),
        'min_time_any_paddler_min': paddler_summary['total_time_min'].min(),
        'max_consecutive_stretch_min': paddler_summary['longest_stretch_min'].max(),
        'avg_consecutive_stretch_min': paddler_summary['longest_stretch_min'].mean(),
    }

    # Compute pattern stats
    pattern_stats = {
        'total_entry_rules': int(paddler_summary['entry_rules'].sum()),
        'total_switch_rules': int(paddler_summary['switch_rules'].sum()),
        'avg_entry_rules_per_paddler': paddler_summary['entry_rules'].mean(),
        'avg_switch_rules_per_paddler': paddler_summary['switch_rules'].mean(),
        'entry_rule_penalty': entry_rule_penalty,
        'switch_rule_penalty': switch_rule_penalty,
    }

    return {
        "status": LpStatus[prob.status],
        "schedule": sched,
        "avg_output": avg_output,
        "race_time": race_time,
        "parameters": {"stint_min": stint_min, "n_stints": n_stints, "seat_entry_weight": seat_entry_weight},
        "paddler_summary": paddler_summary,
        "summary_stats": summary_stats,
        "pattern_stats": pattern_stats,
    }


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


def solve_rotation_cycle(paddlers,
                         stint_min=40,
                         max_consecutive=6,
                         distance_km=60,
                         speed_kmh=10,
                         switch_time_min=1.5,
                         seat_eligibility=None,
                         seat_weights=None,
                         seat_entry_weight=None,
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
        Q[p,t,k] * output_table[k]
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
                            num += seat_weights[s] * output_table[k]
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
                            num += seat_weights[s] * output_table[k]
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
            'total_time_min': total_stints_paddled[p] * stint_min,
            'longest_stretch_stints': max_consecutive_streak[p],
            'longest_stretch_min': max_consecutive_streak[p] * stint_min,
            'stints_paddled_per_cycle': stints_per_cycle,
        })

    paddler_summary = pd.DataFrame(paddler_stats)

    # Compute aggregate stats (exact)
    summary_stats = {
        'cycle_length': cycle_length,
        'n_stints': n_stints,
        'n_cycles': n_stints / cycle_length,
        'avg_time_per_paddler_min': paddler_summary['total_time_min'].mean(),
        'max_time_any_paddler_min': paddler_summary['total_time_min'].max(),
        'min_time_any_paddler_min': paddler_summary['total_time_min'].min(),
        'max_consecutive_stretch_min': paddler_summary['longest_stretch_min'].max(),
    }

    return {
        "status": LpStatus[prob.status],
        "schedule": full_sched,
        "cycle_schedule": cycle_sched,
        "avg_output": avg_output,
        "race_time": race_time,
        "parameters": {
            "stint_min": stint_min,
            "n_stints": n_stints,
            "cycle_length": cycle_length,
            "seat_entry_weight": seat_entry_weight,
        },
        "paddler_summary": paddler_summary,
        "summary_stats": summary_stats,
    }
