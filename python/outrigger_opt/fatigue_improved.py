def output_power(mins_paddling, fatigue):
    """
    Base power production at a given minute of a stint,
    modified by accumulated fatigue (0 = fresh).
    
    mins_paddling : minutes into current stint
    fatigue       : fatigue scalar (larger = more tired)
    Returns       : power fraction 0–1
    """
    # Start slightly below peak, ramp to 1.0 by minute 8
    if mins_paddling <= 8:
        base = 0.85 + 0.15 * (mins_paddling / 8)
    else:
        # after minute 8, linear fade at ~0.7% per minute
        base = max(0, 1.0 - 0.007*(mins_paddling - 8))
    
    # Apply fatigue reduction (each fatigue unit subtracts power)
    power = base * max(0, 1 - fatigue)
    return power


def update_fatigue(fatigue, mins_worked, mins_rest, work_rate=0.05, recovery_rate=0.03):
    """
    Update fatigue based on time paddling and time resting.
    
    fatigue      : current fatigue state (>=0 recommended)
    mins_worked  : minutes paddled in this stint
    mins_rest    : minutes rested since last stint
    
    work_rate     : fatigue gained per active minute
    recovery_rate : fatigue shed per rest minute
    """
    fatigue += work_rate * mins_worked
    fatigue = max(0, fatigue - recovery_rate * mins_rest)
    return fatigue


def power_to_speed(power, exponent=0.45):
    """
    Convert power fraction (0–1) to a speed fraction using drag model.
    """
    return power ** exponent
