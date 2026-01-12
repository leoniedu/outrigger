# Full MILP with S + Y linearization and race time
import itertools, math
import pandas as pd
import numpy as np
from pulp import LpProblem, LpMaximize, LpVariable, lpSum, LpStatus, value, PULP_CBC_CMD
from .fatigue import compute_output_table


def validate_paddlers(paddlers, n_seats=6, n_paddlers=9):
    """Validate paddler DataFrame has required structure.

    Args:
        paddlers: DataFrame with 'name' column (and optionally 'role')
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


def make_eligibility_from_roles(paddlers, n_seats=6):
    """Create seat eligibility matrix from paddler roles.

    Default outrigger canoe rules:
    - First 2 seats (indices 0-1): pacers only (front/stroke seats)
    - Middle seats (indices 2 to n_seats-2): anyone
    - Last seat (index n_seats-1): steerers only (steering seat)

    For OC6 (6 seats):
    - Seats 1-2: pacers only
    - Seats 3-5: anyone
    - Seat 6: steerers only

    Args:
        paddlers: DataFrame with 'role' column ('pacer', 'regular', 'steerer')
        n_seats: Number of seats (default 6, minimum 3)

    Returns:
        numpy array of shape (n_paddlers, n_seats) where 1 = eligible
    """
    if 'role' not in paddlers.columns:
        raise ValueError("paddlers must have 'role' column to generate eligibility from roles")

    if n_seats < 3:
        raise ValueError(f"n_seats must be at least 3 for role-based eligibility, got {n_seats}")

    P = len(paddlers)
    eligibility = np.zeros((P, n_seats), dtype=int)

    # Define seat ranges based on n_seats
    front_end = 2                    # First 2 seats (indices 0, 1) are front/pacer seats
    middle_start = 2                 # Middle starts at index 2
    middle_end = n_seats - 1         # Middle ends before last seat
    last_seat = n_seats - 1          # Last seat is steering

    for p in range(P):
        role = paddlers.role.iloc[p]
        if role == 'pacer':
            # Pacers can sit in front two seats
            eligibility[p, 0:2] = 1
        elif role == 'steerer':
            # Steerers can sit in  last and next to last seats
            eligibility[p, (n_seats-2):n_seats] = 1
        elif role == 'regular':
            # Regular paddlers can sit in middle seats only
            eligibility[p, middle_start:middle_end] = 1
        else:
            raise ValueError(f"Unknown role '{role}' for paddler {paddlers.name.iloc[p]}")

    return eligibility


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
                        n_seats=6,
                        n_resting=3,
                        time_limit=60,
                        gap_tolerance=0.01):
    """Solve the crew rotation optimization problem.

    Args:
        paddlers: DataFrame with 'name' column (and 'role' if using default eligibility)
        stint_min: Duration of each stint in minutes
        max_consecutive: Maximum consecutive stints before required rest
        distance_km: Race distance in kilometers
        speed_kmh: Base speed at output 1.0 in km/h
        switch_time_min: Time penalty per crew switch in minutes
        seat_eligibility: Optional (n_paddlers, n_seats) matrix where 1 = paddler can sit in seat.
                         If None, generated from paddler roles using make_eligibility_from_roles().
        seat_weights: Optional list of seat importance weights. Default [1.2, 1.0, 0.9, 0.9, 0.9, 1.1]
        n_seats: Number of seats in canoe (default 6)
        n_resting: Number of paddlers resting each stint (default 3)
        time_limit: Maximum solver time in seconds (default 60)
        gap_tolerance: Acceptable gap from optimal (default 0.02 = 2%)

    Returns:
        dict with 'status', 'schedule', 'avg_output', 'race_time', 'parameters'
    """
    # Validate inputs
    n_paddlers = n_seats + n_resting
    paddlers = validate_paddlers(paddlers, n_seats=n_seats, n_paddlers=n_paddlers)
    validate_params(stint_min, max_consecutive, distance_km, speed_kmh, switch_time_min)

    P = len(paddlers)
    S = n_seats

    # Build or validate eligibility matrix
    if seat_eligibility is None:
        eligibility = make_eligibility_from_roles(paddlers, n_seats=S)
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
    # For continuous × binary: Q = V * Y where V = Σ_s X[p,s,t]*seat_weight[s]
    #   Q <= max_weight * Y  (Q=0 if not paddling with k consecutive)
    #   Q <= V               (Q bounded by weighted seat)
    #   Q >= V - max_weight * (1 - Y)  (Q=V if Y=1)
    for p,t,k in itertools.product(range(P), range(n_stints), range(1,max_consecutive+1)):
        eligible_seats = [s for s in range(S) if eligibility[p, s]]
        V_pt = lpSum(X[p,s,t] * seat_weights[s] for s in eligible_seats)
        prob += Q[p,t,k] <= max_weight * Y[p,t,k]
        prob += Q[p,t,k] <= V_pt
        prob += Q[p,t,k] >= V_pt - max_weight * (1 - Y[p,t,k])

    # Objective: Maximize weighted output (linear in Q)
    prob += lpSum(
        Q[p,t,k] * output_table[k]
        for p,t,k in itertools.product(range(P), range(n_stints), range(1,max_consecutive+1))
    )

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
    race_time = nominal / avg_output + switches*switch_time_min

    return {
        "status": LpStatus[prob.status],
        "schedule": sched,
        "avg_output": avg_output,
        "race_time": race_time,
        "parameters": {"stint_min": stint_min, "n_stints": n_stints}
    }
