import numpy as np

def output_curve(tau, start=0.8, peak=12, plateau=10, decay=0.01):
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
