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
