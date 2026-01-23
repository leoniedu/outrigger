# Full MILP with S + Y linearization and race time
import itertools, math
import pandas as pd
import numpy as np
from pulp import LpProblem, LpMaximize, LpVariable, lpSum, LpStatus, value, PULP_CBC_CMD
from .fatigue import (compute_output_table, compute_cycle_output_table,
                      compute_cycle_stint_table, compute_stint_time,
                      power_to_speed, update_fatigue)


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


def validate_params(stint_km, max_consecutive, distance_km, speed_kmh, switch_time_secs):
    """Validate optimization parameters.

    Raises:
        ValueError: If any parameter is invalid
    """
    if stint_km <= 0:
        raise ValueError(f"stint_km must be positive, got {stint_km}")
    if max_consecutive < 1:
        raise ValueError(f"max_consecutive must be >= 1, got {max_consecutive}")
    if distance_km <= 0:
        raise ValueError(f"distance_km must be positive, got {distance_km}")
    if speed_kmh <= 0:
        raise ValueError(f"speed_kmh must be positive, got {speed_kmh}")
    if switch_time_secs < 0:
        raise ValueError(f"switch_time_secs must be non-negative, got {switch_time_secs}")


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
        e.g. {'Eduardo': 'Banco 3 → Banco 6 → Descanso*'}
        where * marks the starting position (stint 0)
        Rules are rotated so Descanso is always last.
    """
    rules = {}
    cycle_length = len(cycle_schedule)

    for name in paddlers.name:
        # Build list of (position_string, is_stint_zero) tuples
        positions = []
        for t in range(cycle_length):
            # Find where this paddler is at stint t
            found = False
            for col in cycle_schedule.columns:
                if cycle_schedule.iloc[t][col] == name:
                    seat_num = col.replace('Banco ', '')
                    positions.append((f"Banco {seat_num}", t == 0))
                    found = True
                    break
            if not found:
                positions.append(("Descanso", t == 0))

        # Rotate so Descanso is last
        rest_idx = next((i for i, (pos, _) in enumerate(positions) if pos == "Descanso"), None)
        if rest_idx is not None and rest_idx < cycle_length - 1:
            # Rotate: move everything before rest to after rest
            positions = positions[rest_idx + 1:] + positions[:rest_idx + 1]

        # Format with * marking the starting position
        formatted = []
        for pos, is_start in positions:
            if is_start:
                formatted.append(f"{pos}*")
            else:
                formatted.append(pos)

        rules[name] = " → ".join(formatted)

    return rules


def solve_rotation_cycle(paddlers,
                         stint_km=3.0,
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
                         gap_tolerance=0.01,
                         fatigue_work_rate=0.015,
                         fatigue_tau_recovery=7,
                         use_stateful_fatigue=True,
                         power_speed_exponent=0.4):
    """Solve the crew rotation optimization using a cycle-based model.

    Models a single rotation cycle instead of the entire race, with wrap-around
    fatigue tracking. The cycle repeats for the full race duration.

    For 9 paddlers, 6 seats, 3 resting: cycle = 3 stints (paddle 2, rest 1).
    This reduces model size by ~66% compared to full-race model.

    Uses distance-based stints: n_stints = ceil(distance_km / stint_km), which is
    deterministic and independent of fatigue/output. Stint times vary based on
    actual paddler output.

    Note: Pattern penalties (entry_rule_penalty, switch_rule_penalty) are not
    supported in the cycle model since the repeating cycle already produces
    a simple, consistent pattern.

    Args:
        paddlers: DataFrame with 'name' column
        stint_km: Distance per stint in kilometers (default 3.0)
        max_consecutive: Maximum consecutive stints (variable bound)
        distance_km: Race distance in kilometers
        speed_kmh: Base speed at output 1.0 in km/h
        switch_time_secs: Time penalty per crew switch in seconds
        seat_eligibility: Optional (n_paddlers, n_seats) matrix where 1 = paddler can sit in seat.
                         If None, all paddlers are eligible for all seats.
        seat_weights: Optional list of seat importance weights. Default [1.2, 1.1, 0.9, 0.9, 0.9, 1.1]
        seat_entry_weights: Optional list of entry ease weights per seat. Default all 1.0.
                          >1 = easier to enter, <1 = harder to enter.
        paddler_ability: Optional list of ability multipliers per paddler. Default all 1.0.
                        >1 = stronger paddler, <1 = weaker paddler.
        paddler_weight: Optional list of weights per paddler (kg or relative). Default all 75.0.
                       Used for trim (fore-aft balance) calculation.
        trim_penalty_weight: Penalty weight for max abs trim (default 0.0 = disabled).
                            Uses minimax: minimizes the worst-case trim imbalance across stints.
        moi_penalty_weight: Penalty weight for moment of inertia (default 0.0 = disabled).
                           Positive = prefer weight in middle, negative = prefer weight at ends.
        steerer_paddle_fraction: Fraction of time steerer (seat 6) paddles vs steers (default 0.6).
                                Affects seat 6 output contribution and dead weight penalty.
                                0.7-0.8 for flat/sprint, 0.5-0.6 moderate, 0.3-0.4 rough water.
        n_seats: Number of seats in canoe (default 6)
        n_resting: Number of paddlers resting each stint (default 3)
        solver_time_secs: Maximum solver computation time in seconds (default 60)
        gap_tolerance: Acceptable gap from optimal (default 0.01 = 1%)
        fatigue_work_rate: W' depletion per minute of work (default 0.015 = 15% per 10 min)
        fatigue_tau_recovery: Recovery time constant in minutes (default 7, half-life ~5 min)
        use_stateful_fatigue: Use stateful fatigue model for race simulation (default True)
        power_speed_exponent: Exponent for power-to-speed conversion (default 0.4).
                             speed = power^exponent. Use 1.0 for linear (legacy behavior).

    Returns:
        dict with 'status', 'schedule', 'cycle_schedule', 'avg_output', 'race_time',
        'parameters', 'paddler_summary', 'summary_stats', 'stint_times'
    """
    # Validate inputs
    if n_seats % n_resting != 0:
        raise ValueError(
            f"n_resting ({n_resting}) must be a divisor of n_seats ({n_seats}). "
            f"Valid values for {n_seats} seats: {[i for i in range(1, n_seats+1) if n_seats % i == 0]}"
        )
    n_paddlers = n_seats + n_resting
    paddlers = validate_paddlers(paddlers, n_seats=n_seats, n_paddlers=n_paddlers)
    validate_params(stint_km, max_consecutive, distance_km, speed_kmh, switch_time_secs)

    P = len(paddlers)
    S = n_seats

    # Compute cycle length for fair rotation
    # Since n_resting divides n_seats, it also divides n_paddlers
    # Each paddler paddles (cycle_length - 1) stints and rests 1 stint per cycle
    cycle_length = n_paddlers // n_resting
    # e.g., 9 paddlers, 3 resting: cycle=3 (paddle 2, rest 1)
    # e.g., 8 paddlers, 2 resting: cycle=4 (paddle 3, rest 1)

    # Build or validate eligibility matrix
    if seat_eligibility is None:
        eligibility = np.ones((P, S), dtype=int)
    else:
        eligibility = np.asarray(seat_eligibility)
    validate_eligibility(eligibility, P, S)

    # Validate steerer_paddle_fraction
    if not 0.0 <= steerer_paddle_fraction <= 1.0:
        raise ValueError(f"steerer_paddle_fraction must be between 0 and 1, got {steerer_paddle_fraction}")

    # Default seat weights (before steerer adjustment)
    if seat_weights is None:
        if S == 6:
            seat_weights = [1.2, 1.1, 0.9, 0.9, 0.9, 1.1]
        else:
            seat_weights = [1.0] * S

    if len(seat_weights) != S:
        raise ValueError(f"seat_weights must have {S} elements, got {len(seat_weights)}")

    # Adjust steerer (seat 6 / last seat) weight by paddle fraction
    # Steerer only contributes to output when paddling, not when steering
    seat_weights = list(seat_weights)  # Make mutable copy
    steerer_seat = S - 1  # Last seat (index S-1)
    seat_weights[steerer_seat] *= steerer_paddle_fraction

    # Default seat entry weights (all 1.0 = normal difficulty)
    if seat_entry_weights is None:
        seat_entry_weights = [1.0] * S
    if len(seat_entry_weights) != S:
        raise ValueError(f"seat_entry_weights must have {S} elements, got {len(seat_entry_weights)}")
    if any(w <= 0 for w in seat_entry_weights):
        raise ValueError("seat_entry_weights values must be positive")

    # Check if we need entry weight optimization (any weight != 1.0)
    use_entry_weights = any(w != 1.0 for w in seat_entry_weights)

    # Default paddler ability (all 1.0 = equal ability)
    if paddler_ability is None:
        paddler_ability = [1.0] * P
    if len(paddler_ability) != P:
        raise ValueError(f"paddler_ability must have {P} elements, got {len(paddler_ability)}")
    if any(a <= 0 for a in paddler_ability):
        raise ValueError("paddler_ability values must be positive")

    # Default paddler weight (all 75 kg)
    if paddler_weight is None:
        paddler_weight = [75.0] * P
    if len(paddler_weight) != P:
        raise ValueError(f"paddler_weight must have {P} elements, got {len(paddler_weight)}")
    if any(w <= 0 for w in paddler_weight):
        raise ValueError("paddler_weight values must be positive")

    # Validate penalty weights
    use_trim_penalty = trim_penalty_weight != 0
    use_moi_penalty = moi_penalty_weight != 0

    # Compute average paddler weight for normalization (makes penalties comparable across crews)
    avg_paddler_weight = sum(paddler_weight) / len(paddler_weight)

    # Seat positions for trim calculation (meters from center, negative = bow)
    seat_positions = [-(S-1)/2 + i for i in range(S)]  # e.g., [-2.5, -1.5, -0.5, 0.5, 1.5, 2.5] for S=6

    # Compute number of stints - deterministic based on distance
    n_stints = math.ceil(distance_km / stint_km)

    # Estimate stint time for optimization output table (assume ~85% avg output)
    estimated_stint_min = (stint_km / speed_kmh) * 60 / 0.85
    output_table = compute_output_table(estimated_stint_min, max_consecutive)

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

    # Trim penalty variables (for absolute value linearization and minimax)
    if use_trim_penalty:
        # Calculate max possible trim moment for variable bounds
        max_paddler_weight = max(paddler_weight)
        max_seat_pos = max(abs(p) for p in seat_positions)
        max_trim_moment = S * max_paddler_weight * max_seat_pos

        # TrimPos - TrimNeg = trim_moment, TrimPos + TrimNeg = |trim_moment|
        TrimPos = {t: LpVariable(f"TrimPos_{t}", lowBound=0, upBound=max_trim_moment, cat="Continuous")
                   for t in range(cycle_length)}
        TrimNeg = {t: LpVariable(f"TrimNeg_{t}", lowBound=0, upBound=max_trim_moment, cat="Continuous")
                   for t in range(cycle_length)}

        # MaxAbsTrim for minimax formulation: minimize the worst-case trim
        MaxAbsTrim = LpVariable("MaxAbsTrim", lowBound=0, upBound=max_trim_moment, cat="Continuous")

    # MOI (moment of inertia) variables for weight concentration penalty
    if use_moi_penalty:
        max_paddler_weight = max(paddler_weight)
        max_seat_pos_sq = max(p**2 for p in seat_positions)
        max_moi = S * max_paddler_weight * max_seat_pos_sq
        MOI = {t: LpVariable(f"MOI_{t}", lowBound=0, upBound=max_moi, cat="Continuous")
               for t in range(cycle_length)}

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
    # Precompute V_pt (weighted seat sum) - only depends on (p, t), not k
    V_pt = {}
    for p, t in itertools.product(range(P), range(cycle_length)):
        eligible_seats = [s for s in range(S) if eligibility[p, s]]
        V_pt[(p, t)] = lpSum(X[p,s,t] * seat_weights[s] for s in eligible_seats)

    for p,t,k in itertools.product(range(P), range(cycle_length), range(1,max_consecutive+1)):
        prob += Q[p,t,k] <= max_weight * Y[p,t,k]
        prob += Q[p,t,k] <= V_pt[(p, t)]
        prob += Q[p,t,k] >= V_pt[(p, t)] - max_weight * (1 - Y[p,t,k])

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

    # Trim balance constraints (absolute value linearization + minimax)
    if use_trim_penalty:
        for t in range(cycle_length):
            # Calculate trim moment as linear expression
            # Positive moment = stern-heavy, negative moment = bow-heavy
            trim_moment = lpSum(
                X[p, s, t] * paddler_weight[p] * seat_positions[s]
                for (p, s) in eligible_pairs
            )
            # TrimPos - TrimNeg = trim_moment (for absolute value)
            prob += TrimPos[t] - TrimNeg[t] == trim_moment, f"Trim_balance_{t}"

            # Minimax constraint: MaxAbsTrim >= |trim_moment| for all stints
            prob += MaxAbsTrim >= TrimPos[t] + TrimNeg[t], f"MaxAbsTrim_{t}"

    # MOI constraints (weight concentration - lower = better)
    if use_moi_penalty:
        for t in range(cycle_length):
            # MOI = sum of weight × position² (always non-negative)
            moi_expr = lpSum(
                X[p, s, t] * paddler_weight[p] * (seat_positions[s] ** 2)
                for (p, s) in eligible_pairs
            )
            prob += MOI[t] == moi_expr, f"MOI_{t}"

    # Seat transition constraint: if paddling consecutively, can only move ±1 seat
    # If resting in previous stint, paddler can enter any eligible seat
    for p in range(P):
        for t in range(cycle_length):
            t_prev = (t - 1) % cycle_length  # wrap-around for t=0
            for s in range(S):
                if not eligibility[p, s]:
                    continue
                for s_next in range(S):
                    if not eligibility[p, s_next]:
                        continue
                    if abs(s_next - s) > 1:  # non-adjacent seats
                        prob += X[p, s, t_prev] + X[p, s_next, t] <= 1, \
                            f"SeatTransition_p{p}_t{t}_s{s}_to_{s_next}"

    # Objective: Maximize weighted output minus entry weight penalties
    weighted_output = lpSum(
        Q[p,t,k] * output_table[k] * paddler_ability[p]
        for p,t,k in itertools.product(range(P), range(cycle_length), range(1,max_consecutive+1))
    )

    objective = weighted_output

    if use_entry_weights:
        # Scale entry weight penalty relative to per-cycle output, not total race time
        # Each cycle produces ~cycle_length stints worth of weighted output
        cycle_paddle_time = cycle_length * estimated_stint_min
        entry_weight_scale = (switch_time_secs / 60) / cycle_paddle_time * sum(seat_weights)
        # t=0 entries only happen (n_cycles - 1) times (first stint of race is free)
        # t>0 entries happen n_cycles times
        n_cycles = n_stints / cycle_length
        t0_weight = (n_cycles - 1) / n_cycles if n_cycles > 1 else 0.0
        entry_weight_penalty = entry_weight_scale * (
            lpSum(
                Entry[p, s, 0] * (1.0 / seat_entry_weights[s] - 1.0) * t0_weight
                for (p, s) in eligible_pairs
            ) +
            lpSum(
                Entry[p, s, t] * (1.0 / seat_entry_weights[s] - 1.0)
                for (p, s) in eligible_pairs for t in range(1, cycle_length)
            )
        )
        objective = objective - entry_weight_penalty

    # Normalization factors scaled by avg paddler weight (makes penalties comparable across crews)
    # Position factors based on typical OC6 geometry
    trim_normalizer = avg_paddler_weight * 2.5  # ~max seat position (m)
    sum_pos_sq = sum(p**2 for p in seat_positions)  # e.g., 17.5 for 6-seat
    moi_normalizer = avg_paddler_weight * sum_pos_sq  # total MOI if all avg weight

    if use_trim_penalty:
        # Using minimax: penalize the worst (maximum) absolute trim across all stints
        # Normalize by avg_weight × max_seat_position
        trim_scale = trim_penalty_weight / (trim_normalizer * sum(seat_weights) / cycle_length)
        trim_penalty = trim_scale * MaxAbsTrim * cycle_length
        objective = objective - trim_penalty

    if use_moi_penalty:
        # Normalize by avg_weight × sum(position²) / n_seats
        moi_scale = moi_penalty_weight / (moi_normalizer * sum(seat_weights) / cycle_length)
        # Total MOI across all stints in cycle (lower = weight concentrated in middle)
        moi_penalty = moi_scale * lpSum(MOI[t] for t in range(cycle_length))
        objective = objective - moi_penalty

    # Dead weight penalty for steerer (seat 6)
    # When steering (not paddling), the steerer's weight is pure drag without power contribution
    # Penalize heavier steerers proportional to: weight × (1 - paddle_fraction)
    if steerer_paddle_fraction < 1.0:
        # Eligible paddlers for steerer seat
        steerer_eligible = [p for p in range(P) if eligibility[p, steerer_seat]]
        # Dead weight contribution for each paddler if they're the steerer
        # Normalized by avg_paddler_weight so penalty is in comparable units
        dead_weight_penalty = lpSum(
            X[p, steerer_seat, t] * (paddler_weight[p] / avg_paddler_weight - 1.0) * (1.0 - steerer_paddle_fraction)
            for p in steerer_eligible
            for t in range(cycle_length)
        )
        # Scale relative to output (light steerer saves ~5-10% of output value)
        dead_weight_scale = 0.1 * sum(seat_weights) / cycle_length
        objective = objective - dead_weight_scale * dead_weight_penalty

    prob += objective

    # Solve
    solver = PULP_CBC_CMD(msg=0, timeLimit=solver_time_secs, gapRel=gap_tolerance)
    prob.solve(solver)

    # Extract cycle schedule
    cycle_sched = pd.DataFrame("-", index=range(cycle_length), columns=[f"Banco {s+1}" for s in range(S)])
    for (p,s,t), var in X.items():
        if var.value() and var.value() > 0.5:
            cycle_sched.iloc[t,s] = paddlers.name.iloc[p]

    # Compute weighted stint outputs (for optimization objective, uses seat_weights)
    # Note: This is NOT used for race_time - race_time uses uniform weights
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

    # Extract cycle pattern from solution (True = paddling)
    cycle_pattern = {}
    for p in range(P):
        cycle_pattern[p] = [
            R[p, t].value() is not None and R[p, t].value() < 0.5
            for t in range(cycle_length)
        ]

    # Compute cycle output and stint time table using stateful fatigue model
    if use_stateful_fatigue:
        cycle_stint_table = compute_cycle_stint_table(
            cycle_pattern, stint_km, speed_kmh,
            work_rate=fatigue_work_rate,
            tau_recovery=fatigue_tau_recovery,
            power_speed_exponent=power_speed_exponent
        )
        cycle_output_table = cycle_stint_table['outputs']
        cycle_stint_times = cycle_stint_table['stint_times']
    else:
        # Fallback: use estimated stint time
        cycle_output_table = None
        cycle_stint_times = None

    # Race simulation with distance-based stints and variable times
    # n_stints is deterministic: ceil(distance_km / stint_km)
    stint_outputs = []
    stint_times = []
    consecutive = {p: 0 for p in range(P)}
    total_stints_paddled = {p: 0 for p in range(P)}
    total_time_paddled = {p: 0.0 for p in range(P)}
    max_consecutive_streak = {p: 0 for p in range(P)}
    current_stretch_time = {p: 0.0 for p in range(P)}  # Track actual time in current streak
    longest_stretch_time = {p: 0.0 for p in range(P)}  # Track longest stretch time
    fatigue_state = {p: 0.0 for p in range(P)}

    for stint_idx in range(n_stints):
        cycle_t = stint_idx % cycle_length

        # Handle partial final stint
        current_stint_km = stint_km
        if stint_idx == n_stints - 1:
            remaining = distance_km - stint_idx * stint_km
            current_stint_km = min(stint_km, max(remaining, 0.001))

        # Determine who is paddling this stint
        paddling_this_stint = {}
        for p in range(P):
            is_paddling = R[p, cycle_t].value() is not None and R[p, cycle_t].value() < 0.5
            paddling_this_stint[p] = is_paddling

        # Compute stint time and output for paddling crew
        if use_stateful_fatigue and cycle_stint_times is not None:
            # Use precomputed cycle stint times (scaled for partial stint)
            paddling_times = []
            paddling_outputs = []
            for p in range(P):
                if paddling_this_stint[p]:
                    # Scale precomputed time by distance ratio for partial stints
                    base_time = cycle_stint_times[p][cycle_t]
                    stint_time_p = base_time * (current_stint_km / stint_km)
                    paddling_times.append(stint_time_p)
                    paddling_outputs.append(cycle_output_table[p][cycle_t] * paddler_ability[p])

            if paddling_times:
                # Average stint time for crew (they move together)
                avg_stint_time = np.mean(paddling_times)
                avg_output = np.mean(paddling_outputs) / S * len(paddling_outputs)
            else:
                avg_stint_time = estimated_stint_min * (current_stint_km / stint_km)
                avg_output = 0.0
        else:
            # Fallback: use simple simulation for each paddler
            paddling_results = []
            for p in range(P):
                if paddling_this_stint[p]:
                    result = compute_stint_time(
                        current_stint_km, speed_kmh, fatigue_state[p],
                        work_rate=fatigue_work_rate,
                        power_speed_exponent=power_speed_exponent
                    )
                    paddling_results.append(result)

            if paddling_results:
                avg_stint_time = np.mean([r['stint_time_min'] for r in paddling_results])
                avg_output = np.mean([r['avg_output'] for r in paddling_results])
            else:
                avg_stint_time = estimated_stint_min * (current_stint_km / stint_km)
                avg_output = 0.0

        stint_times.append(avg_stint_time)
        stint_outputs.append(avg_output)

        # Update consecutive counts, fatigue, and per-paddler stats
        for p in range(P):
            if paddling_this_stint[p]:
                consecutive[p] += 1
                total_stints_paddled[p] += 1
                total_time_paddled[p] += avg_stint_time
                current_stretch_time[p] += avg_stint_time  # Track actual stretch time
                max_consecutive_streak[p] = max(max_consecutive_streak[p], consecutive[p])
                longest_stretch_time[p] = max(longest_stretch_time[p], current_stretch_time[p])
                # Update fatigue for paddling
                fatigue_state[p] = update_fatigue(
                    fatigue_state[p], avg_stint_time, 0,
                    work_rate=fatigue_work_rate, tau_recovery=fatigue_tau_recovery
                )
            else:
                # Finalize current stretch before resetting
                longest_stretch_time[p] = max(longest_stretch_time[p], current_stretch_time[p])
                current_stretch_time[p] = 0.0  # Reset streak time
                consecutive[p] = 0
                # Update fatigue for resting
                fatigue_state[p] = update_fatigue(
                    fatigue_state[p], 0, avg_stint_time,
                    work_rate=fatigue_work_rate, tau_recovery=fatigue_tau_recovery
                )

    # Calculate race time from actual stint times
    switches = n_stints - 1
    race_time = sum(stint_times) + switches * (switch_time_secs / 60)

    # Calculate average output from actual stints
    avg_output = sum(stint_outputs) / n_stints if stint_outputs else 0.0
    avg_stint_time_min = sum(stint_times) / n_stints if stint_times else estimated_stint_min

    steady_cycle_output = sum(steady_stint_outputs)  # Keep for stats

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
            'total_time_min': float(total_time_paddled[p]),
            'longest_stretch_stints': max_consecutive_streak[p],
            'longest_stretch_min': float(longest_stretch_time[p]),  # Use tracked time
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

    # Compute cycle-level average outputs (crew average per cycle position)
    cycle_avg_outputs = []
    for t in range(cycle_length):
        if cycle_output_table is not None:
            # Use stateful fatigue outputs - average of paddling crew
            paddling_outputs = [cycle_output_table[p][t] * paddler_ability[p]
                               for p in range(P) if cycle_pattern[p][t]]
            cycle_avg_outputs.append(np.mean(paddling_outputs) if paddling_outputs else 0.0)
        else:
            # Fall back to steady state outputs
            cycle_avg_outputs.append(steady_stint_outputs[t] if t < len(steady_stint_outputs) else 0.0)

    # Compute trim and MOI statistics with per-seat breakdown (always, for transparency)
    trim_moments = []
    moi_values = []
    seat_breakdown = []  # Per-stint breakdown of contributions by seat

    for t in range(cycle_length):
        # Calculate actual trim moment and MOI from solution with per-seat details
        moment = 0
        moi = 0
        stint_seats = []
        for s in range(S):
            # Find which paddler is in this seat
            for p in range(P):
                if eligibility[p, s] and X[p, s, t].value() and X[p, s, t].value() > 0.5:
                    weight = paddler_weight[p]
                    position = seat_positions[s]
                    trim_contrib = weight * position
                    moi_contrib = weight * (position ** 2)
                    moment += trim_contrib
                    moi += moi_contrib
                    stint_seats.append({
                        'seat': s + 1,
                        'name': paddlers.name.iloc[p],
                        'weight': weight,
                        'position': position,
                        'trim_contrib': trim_contrib,
                        'moi_contrib': moi_contrib,
                    })
                    break
        trim_moments.append(moment)
        moi_values.append(moi)
        seat_breakdown.append(stint_seats)

    # Trim/MOI statistics
    trim_stats = {
        'trim_moments': trim_moments,
        'max_abs_trim_moment': max(abs(m) for m in trim_moments),
        'moi_values': moi_values,
        'avg_moi': sum(moi_values) / cycle_length,
        'seat_positions': seat_positions,
        'cycle_avg_outputs': cycle_avg_outputs,
        'seat_breakdown': seat_breakdown,  # Per-seat contributions for each stint
        # Normalized values (comparable across crews)
        'avg_paddler_weight': avg_paddler_weight,
        'normalized_max_abs_trim': max(abs(m) for m in trim_moments) / (avg_paddler_weight * 2.5),
        'normalized_avg_moi': sum(moi_values) / cycle_length / (avg_paddler_weight * sum_pos_sq),
    }

    return {
        "status": LpStatus[prob.status],
        "schedule": full_sched,
        "cycle_schedule": cycle_sched,
        "cycle_rules": format_cycle_rules(cycle_sched, paddlers),
        "avg_output": avg_output,
        "race_time": race_time,
        "stint_times": stint_times,
        "parameters": {
            "stint_km": stint_km,
            "n_stints": n_stints,
            "avg_stint_time_min": avg_stint_time_min,
            "cycle_length": cycle_length,
            "seat_entry_weights": seat_entry_weights,
            "paddler_ability": paddler_ability,
            "paddler_weight": paddler_weight,
            "trim_penalty_weight": trim_penalty_weight,
            "moi_penalty_weight": moi_penalty_weight,
            "steerer_paddle_fraction": steerer_paddle_fraction,
            "trim_stats": trim_stats,
        },
        "paddler_summary": paddler_summary,
        "summary_stats": summary_stats,
    }
