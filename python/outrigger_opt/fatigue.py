import numpy as np

def output_curve(tau, start=0.8, peak=5, plateau=5, decay=0.01):
    if tau <= peak:
        return start * (1/start)**(tau/peak)
    if tau <= peak + plateau:
        return 1.0
    return (1-decay)**(tau - peak - plateau)

def average_output(stint_min, k, step=0.25):
    vals = []
    for minute in np.arange(0, stint_min, step):
        vals.append(output_curve(minute + stint_min*(k-1)))
    return float(np.mean(vals))

def compute_output_table(stint_min, max_consecutive=6):
    return {k: average_output(stint_min, k)
            for k in range(1, max_consecutive+1)}


# --- Stateful fatigue model ---

def power_to_speed(power, exponent=0.4):
    """
    Convert power fraction to speed fraction using drag model.

    For water resistance: drag ∝ v², power = drag × v ∝ v³
    So v ∝ P^(1/3). Using exponent=0.4 accounts for hull dynamics.

    Args:
        power: power fraction (0-1)
        exponent: power-to-speed exponent (default 0.4)

    Returns:
        speed fraction (0-1)
    """
    if power <= 0:
        return 0.0
    return power ** exponent


def output_power(t, fatigue, tau_warmup=2):
    """
    Power output based on exercise physiology principles.

    - Warmup: exponential rise from ~80% to 100% (neural activation, blood flow)
    - Fatigue: reduces available capacity (W' depletion model)

    Based on critical power model from exercise physiology research.

    Args:
        t: minutes into stint
        fatigue: accumulated fatigue (0 = fresh, represents W' depletion fraction)
        tau_warmup: warmup time constant in minutes (default 2, ~6 min to 95%)

    Returns:
        power fraction 0-1
    """
    warmup = 1 - 0.2 * np.exp(-t / tau_warmup)
    return warmup * max(0, 1 - fatigue)


def update_fatigue(fatigue, mins_worked, mins_rest, work_rate=0.015, tau_recovery=7):
    """
    Update fatigue based on exercise physiology W' balance model.

    - Work: depletes W' (anaerobic work capacity) at work_rate per minute
    - Rest: W' recovers exponentially with half-life ~5 min (tau ~7 min)

    Based on critical power research showing W' recovery half-time of ~5 minutes.

    Args:
        fatigue: current fatigue (0-1, represents fraction of W' depleted)
        mins_worked: minutes paddled in this stint
        mins_rest: minutes rested
        work_rate: W' depletion per minute of work (default 0.015 = 15% per 10 min)
        tau_recovery: recovery time constant in minutes (default 7, half-life ~5 min)

    Returns:
        updated fatigue value
    """
    # W' depletion during work
    fatigue += work_rate * mins_worked
    # W' recovery during rest (exponential with tau ~7 min)
    fatigue *= np.exp(-mins_rest / tau_recovery)
    return min(fatigue, 1.0)  # Cap at 100% depleted


def average_output_stateful(stint_min, fatigue_at_start, step=0.25,
                            tau_warmup=2, work_rate=0.015):
    """
    Compute average output over a stint with fatigue accumulating during work.

    Args:
        stint_min: duration of stint in minutes
        fatigue_at_start: fatigue level at the start of the stint
        step: integration step size
        tau_warmup: warmup time constant
        work_rate: fatigue accumulation rate per minute

    Returns:
        average power output over the stint
    """
    vals = []
    fatigue = fatigue_at_start
    for minute in np.arange(0, stint_min, step):
        vals.append(output_power(minute, fatigue, tau_warmup))
        fatigue += work_rate * step  # Accumulate fatigue during stint
    return float(np.mean(vals)) if vals else 1.0


def compute_stint_time(stint_km, speed_kmh, fatigue_at_start,
                       step_seconds=15, tau_warmup=2, work_rate=0.015,
                       power_speed_exponent=0.4):
    """
    Compute time to cover stint_km given fatigue state.

    Iteratively simulates: power -> speed -> distance until target reached.

    Args:
        stint_km: distance to cover in this stint (km)
        speed_kmh: base speed at power=1.0 (km/h)
        fatigue_at_start: fatigue level at start (0=fresh, 1=exhausted)
        step_seconds: simulation time step in seconds (default 15)
        tau_warmup: warmup time constant in minutes (default 2)
        work_rate: W' depletion per minute of work (default 0.015)
        power_speed_exponent: exponent for power-to-speed (default 0.4)

    Returns:
        dict with 'stint_time_min', 'avg_output', 'fatigue_at_end'
    """
    distance_covered = 0.0
    time_elapsed_min = 0.0
    fatigue = fatigue_at_start
    outputs = []

    step_min = step_seconds / 60.0

    while distance_covered < stint_km:
        power = output_power(time_elapsed_min, fatigue, tau_warmup)
        outputs.append(power)
        speed_kmh_actual = power_to_speed(power, power_speed_exponent) * speed_kmh

        # Advance by time step
        distance_covered += speed_kmh_actual * (step_min / 60.0)  # km
        time_elapsed_min += step_min
        fatigue += work_rate * step_min

    return {
        'stint_time_min': time_elapsed_min,
        'avg_output': float(np.mean(outputs)) if outputs else 1.0,
        'fatigue_at_end': min(fatigue, 1.0)
    }


def compute_cycle_output_table(cycle_pattern, stint_min, n_cycles_converge=10,
                               work_rate=0.015, tau_recovery=7):
    """
    Precompute fatigue-adjusted output for each paddler at each cycle position.

    Simulates fatigue through multiple cycles until steady-state is reached.

    Args:
        cycle_pattern: dict mapping paddler_id -> list of bools (True=paddling)
        stint_min: duration of each stint in minutes
        n_cycles_converge: number of cycles to simulate for steady-state
        work_rate: W' depletion per minute of work
        tau_recovery: recovery time constant in minutes

    Returns:
        dict mapping paddler_id -> list of average output values per cycle position
    """
    cycle_length = len(next(iter(cycle_pattern.values())))

    # Initialize fatigue for each paddler
    fatigue = {p: 0.0 for p in cycle_pattern}

    # Simulate multiple cycles to reach steady state
    cycle_fatigue_start = {p: [] for p in cycle_pattern}

    for cycle_idx in range(n_cycles_converge):
        if cycle_idx == n_cycles_converge - 1:
            cycle_fatigue_start = {p: [] for p in cycle_pattern}

        for t in range(cycle_length):
            for p, pattern in cycle_pattern.items():
                if cycle_idx == n_cycles_converge - 1:
                    cycle_fatigue_start[p].append(fatigue[p])

                if pattern[t]:  # paddling
                    fatigue[p] = update_fatigue(fatigue[p], stint_min, 0, work_rate, tau_recovery)
                else:  # resting
                    fatigue[p] = update_fatigue(fatigue[p], 0, stint_min, work_rate, tau_recovery)

    # Compute outputs using fatigue at start of each position
    result = {}
    for p, pattern in cycle_pattern.items():
        outputs = []
        for t in range(cycle_length):
            if pattern[t]:
                avg_out = average_output_stateful(stint_min, cycle_fatigue_start[p][t],
                                                   work_rate=work_rate)
                outputs.append(avg_out)
            else:
                outputs.append(0.0)
        result[p] = outputs

    return result


def compute_cycle_stint_table(cycle_pattern, stint_km, speed_kmh,
                               n_cycles_converge=10, work_rate=0.015,
                               tau_recovery=7, power_speed_exponent=0.4):
    """
    Precompute fatigue-adjusted output and stint times for distance-based stints.

    Simulates fatigue through multiple cycles until steady-state is reached.
    Unlike compute_cycle_output_table, this computes actual stint times based
    on the distance to cover and the paddler's varying power output.

    Args:
        cycle_pattern: dict mapping paddler_id -> list of bools (True=paddling)
        stint_km: distance per stint in kilometers
        speed_kmh: base speed at power=1.0 (km/h)
        n_cycles_converge: number of cycles to simulate for steady-state
        work_rate: W' depletion per minute of work
        tau_recovery: recovery time constant in minutes
        power_speed_exponent: exponent for power-to-speed conversion

    Returns:
        dict with:
            'outputs': dict mapping paddler_id -> list of avg output per cycle position
            'stint_times': dict mapping paddler_id -> list of stint times (min) per position
            'fatigue_at_start': dict mapping paddler_id -> list of fatigue at start of each position
    """
    cycle_length = len(next(iter(cycle_pattern.values())))

    # Initialize fatigue for each paddler
    fatigue = {p: 0.0 for p in cycle_pattern}

    # First, estimate typical stint time for convergence simulation
    # Use compute_stint_time with fresh paddler to get ballpark
    est_result = compute_stint_time(stint_km, speed_kmh, 0.0,
                                     work_rate=work_rate,
                                     power_speed_exponent=power_speed_exponent)
    estimated_stint_min = est_result['stint_time_min']

    # Simulate multiple cycles to reach steady state using estimated stint time
    for cycle_idx in range(n_cycles_converge - 1):
        for t in range(cycle_length):
            for p, pattern in cycle_pattern.items():
                if pattern[t]:  # paddling
                    fatigue[p] = update_fatigue(fatigue[p], estimated_stint_min, 0,
                                                 work_rate, tau_recovery)
                else:  # resting
                    fatigue[p] = update_fatigue(fatigue[p], 0, estimated_stint_min,
                                                 work_rate, tau_recovery)

    # Store fatigue at start of final cycle for accurate computation
    cycle_fatigue_start = {p: [] for p in cycle_pattern}
    for t in range(cycle_length):
        for p, pattern in cycle_pattern.items():
            cycle_fatigue_start[p].append(fatigue[p])

            if pattern[t]:  # paddling
                fatigue[p] = update_fatigue(fatigue[p], estimated_stint_min, 0,
                                             work_rate, tau_recovery)
            else:  # resting
                fatigue[p] = update_fatigue(fatigue[p], 0, estimated_stint_min,
                                             work_rate, tau_recovery)

    # Compute outputs and stint times using actual simulation
    outputs = {}
    stint_times = {}

    for p, pattern in cycle_pattern.items():
        p_outputs = []
        p_stint_times = []
        for t in range(cycle_length):
            if pattern[t]:
                result = compute_stint_time(stint_km, speed_kmh,
                                             cycle_fatigue_start[p][t],
                                             work_rate=work_rate,
                                             power_speed_exponent=power_speed_exponent)
                p_outputs.append(result['avg_output'])
                p_stint_times.append(result['stint_time_min'])
            else:
                p_outputs.append(0.0)
                p_stint_times.append(0.0)  # Resting, no time contribution
        outputs[p] = p_outputs
        stint_times[p] = p_stint_times

    return {
        'outputs': outputs,
        'stint_times': stint_times,
        'fatigue_at_start': cycle_fatigue_start
    }
