# Outrigger Rotation Optimization Plan

## Overview

The `outrigger_opt` package provides a mixed-integer programming (MIP) optimizer that finds the optimal crew rotation schedule, minimizing estimated race time while accounting for fatigue, recovery, and switching overhead.

**Design principle:** The optimization module is fully self-contained. All necessary calculations (e.g., `n_stints` from distance/speed) are computed independently within the optimization code.

## Implementation Status

| Stage | Status | Notes |
|-------|--------|-------|
| Stage 1: Fatigue Curve | ✅ Complete | `fatigue.py` |
| Stage 2: Race Time Calculation | ✅ Complete | Exact stint-by-stint simulation |
| Stage 3: Parameters & Validation | ✅ Complete | `model.py` |
| Stage 4: MIP Model (Cycle-Based) | ✅ Complete | `solve_rotation_cycle()` with wrap-around fatigue |
| Stage 5: Output Formatting | ✅ Complete | Returns schedule + paddler stats |
| Stage 6: Visualization | 🔲 Not Started | `plot.py` planned |
| Stage 7: Meta-Optimization | ✅ Complete | `meta.py` |
| Stage 8: Integration | ✅ Complete | `example.py` with 5 examples |

**Package location:** `outrigger_opt/`

**Key files:**
- `model.py` - Core MIP solver (`solve_rotation_cycle()`, `expand_cycle_to_race()`)
- `fatigue.py` - Fatigue curve functions
- `meta.py` - Stint duration grid search (`optimize_stint_range()`)
- `example.py` - Working usage examples (5 examples)

## Problem Formulation

### Objective

**Minimize estimated race time:**

```
race_time = nominal_paddle_time / avg_output + n_switches × switch_time_min
```

Where:
- `nominal_paddle_time = distance_km / speed_kmh × 60` (minutes at output 1.0)
- `avg_output` = weighted average output across all stints
- `n_switches × switch_time_min` = total switching overhead

**Interpretation:**
- At output 1.0, race takes `nominal_paddle_time` minutes of paddling
- Lower output = slower speed = proportionally longer race
- Switching adds fixed overhead regardless of output

**Average output calculation:**

```
avg_output = (1 / n_stints) × Σ_t crew_output_t

crew_output_t = (Σ_s Σ_p X[p,s,t] × seat_weight[s] × paddler_ability[p] × output_multiplier[p,t]) / Σ_s seat_weight[s]
```

**Why this metric:**
- Directly comparable across different stint durations
- Captures the tradeoff:
  - Shorter stints → more switches (overhead), but higher avg_output (less fatigue)
  - Longer stints → fewer switches, but lower avg_output (more fatigue)
- Enables meta-optimization over stint_min parameter

**Note on linearity:**
To minimize `nominal_paddle_time / avg_output`, we equivalently maximize `avg_output` (since nominal_paddle_time is fixed for a given stint duration). The switching overhead `n_switches × switch_time_min` is a constant for fixed stint duration.

So for a fixed stint_min, the MIP objective simplifies to:

```
Maximize: Σ_t Σ_s Σ_p X[p,s,t] × seat_weight[s] × paddler_ability[p] × output_multiplier[p,t]
```

This is linear and tractable. Race time is computed post-hoc from the solution.

### Cycle-Based Model

The solver uses a **cycle-based approach** that models a single rotation cycle instead of the entire race.

**Key insight:** For 9 paddlers, 6 seats, 3 resting:
- Cycle length = n_paddlers / n_resting = 3 stints
- Each paddler paddles 2 stints, rests 1 stint per cycle
- The pattern repeats for the entire race

**Benefits:**
- ~66% reduction in variables (549 vs 1620 for typical race)
- Solve time ~3x faster
- Naturally produces repeating patterns (easier for crew to remember)
- Wrap-around fatigue ensures pattern is sustainable when repeated

### Decision Variables

| Variable | Type | Description |
|----------|------|-------------|
| `X[p,s,t]` | Binary | Paddler p in seat s at stint t (only created for eligible p,s pairs) |
| `R[p,t]` | Binary | Paddler p is resting at stint t |
| `S[p,t]` | Integer [0, max_consecutive] | Consecutive stints paddling for p at t |
| `Y[p,t,k]` | Binary | Paddler p at stint t has exactly k consecutive stints |
| `Q[p,t,k]` | Continuous [0, max_weight] | Linearization variable for weighted seat × fatigue output |
| `Link[p]` | Integer [0, max_consecutive] | Wrap-around consecutive count from end of cycle |

**Seat entry weight variables (optional, when any weight != 1.0):**

| Variable | Type | Description |
|----------|------|-------------|
| `Entry[p,s,t]` | Binary | Paddler p enters seat s at stint t from rest |

### Seat Eligibility

Seat assignments are controlled by an **eligibility matrix** `E[p,s]` where 1 means paddler p can sit in seat s.

**Default eligibility:** If no eligibility matrix is provided, all paddlers are eligible for all seats (matrix of all 1s).

**Custom eligibility:** Users can provide any (n_paddlers × n_seats) binary matrix to specify seat restrictions.

**Example eligibility matrix:**

```
           Seat1 Seat2 Seat3 Seat4 Seat5 Seat6
Alice        1     1     0     0     0     0    # front only
Bob          1     1     0     0     0     0    # front only
Carol        1     1     1     1     1     0    # anywhere except steering
Dave         0     0     1     1     1     0    # middle only
Eve          0     0     1     1     1     0    # middle only
Frank        0     0     1     1     1     1    # back + steering
Grace        0     0     1     1     1     1    # back + steering
Hank         0     0     1     1     1     0    # middle only
Ivy          0     0     1     1     1     0    # middle only
```

### Constraints

**Seat assignment:**

| Constraint | Formula |
|------------|---------|
| One paddler per seat | `Σ_{p: E[p,s]=1} X[p,s,t] = 1` for each s, t |
| Paddler in one place | `Σ_{s: E[p,s]=1} X[p,s,t] + R[p,t] = 1` for each p, t |
| Exactly n_resting resting | `Σ_p R[p,t] = n_resting` for each t |

Note: X variables are only created for eligible (p,s) pairs where E[p,s]=1, reducing model size.

**Wrap-around consecutive tracking:**

| Constraint | Formula |
|------------|---------|
| Link captures end-of-cycle | `Link[p] = Scon[p,C-1]` if paddling, else 0 |
| Scon[p,0] continues from Link | `Scon[p,0] = Link[p] + 1` if paddling at t=0 |
| Standard tracking for t > 0 | Same as before |

**Linearizing output curve lookup (Y variables):**

| Constraint | Formula |
|------------|---------|
| Exactly one k active | `Σ_k Y[p,t,k] = 1 - R[p,t]` |
| Link Y to S | `S[p,t] = Σ_k k × Y[p,t,k]` |

**Linearizing objective (Q variables):**

The objective requires computing `X[p,s,t] × seat_weight[s] × paddler_ability[p] × output[k] × Y[p,t,k]`, which contains a product of binary variables X and Y. This is linearized using auxiliary continuous variables Q.

Define: `V[p,t] = Σ_{s: E[p,s]=1} X[p,s,t] × seat_weight[s]` (weighted seat contribution, linear in X)

Then: `Q[p,t,k] = V[p,t] × Y[p,t,k]` (product of continuous × binary)

**McCormick linearization for Q:**

| Constraint | Formula |
|------------|---------|
| Upper bound (Y=0) | `Q[p,t,k] ≤ max_weight × Y[p,t,k]` |
| Upper bound (V) | `Q[p,t,k] ≤ V[p,t]` |
| Lower bound | `Q[p,t,k] ≥ V[p,t] - max_weight × (1 - Y[p,t,k])` |

**Linearized objective:**

```
Maximize: Σ_{p,t,k} Q[p,t,k] × output_table[k] × paddler_ability[p]
```

This is fully linear in the decision variables.

---

## Meta-Optimization: Stint Duration

The MIP optimizes seat assignments for a *fixed* stint duration. To find the optimal stint duration itself, we use a grid search wrapper.

### Why Grid Search?

`stint_min` determines `n_stints`, which determines model dimensions. Since model structure depends on stint duration, we can't include it as a decision variable in a single MIP.

### Approach

```python
optimize_stint_range(
    paddlers,
    stint_range=(30, 35, 40, 45, 50, 55, 60),
    max_consecutive=6
)
```

**Algorithm:**

```
for each stint_min in stint_range:
    1. Compute n_stints = ceiling(race_duration_min / stint_min)
    2. Solve MIP for optimal seat assignments
    3. Record: stint_min, estimated_race_time, schedule

return stint_min with lowest estimated_race_time
```

### Output

| stint_min | n_stints | n_switches | est_race_time | status |
|-----------|----------|------------|---------------|--------|
| 30 | 12 | 11 | 387.2 | optimal |
| 35 | 11 | 10 | 379.5 | optimal |
| 40 | 9 | 8 | 374.1 | optimal |
| 45 | 8 | 7 | 372.8 | **best** |
| 50 | 8 | 7 | 375.3 | optimal |
| 55 | 7 | 6 | 381.2 | optimal |
| 60 | 6 | 5 | 390.4 | optimal |

Returns:
- Comparison table across all stint durations
- Best stint_min and its optimal schedule

---

## Fatigue Model

### Output as Function of Cumulative Minutes

Fatigue is based on cumulative minutes paddling, not stint count.

**Parameters:**

| Parameter | Description | Default |
|-----------|-------------|---------|
| `start_output` | Output at τ=0 | 0.80 |
| `peak_time` | Minutes to reach peak | 12 |
| `plateau_duration` | Minutes at peak | 10 |
| `decay_rate` | Proportional loss per minute | 0.01 |

**Formula:**

```
plateau_end = peak_time + plateau_duration

output(τ) =
  if τ ≤ peak_time:     start_output × (1/start_output)^(τ/peak_time)
  if τ ≤ plateau_end:   1.00
  if τ > plateau_end:   (1 - decay_rate)^(τ - plateau_end)
```

**Curve shape:**

```
Output
1.00 |        ___________
     |      /             \
0.90 |    /                 \
     |   /                    \
0.80 |__/                       \____
     |________________________________ Minutes
     0    12    22    32    42    62
         peak  end
               plateau
```

### Averaging Over Stints

For a stint of length `L` minutes, paddler starting at cumulative minute `τ_start`:

```
stint_output = (1/L) × ∫[τ_start to τ_start + L] output(τ) dτ
```

Pre-compute average output for each consecutive stint given stint length.

### Recovery Model

**Simplified: Full reset after any rest stint**

- After any rest stint → cumulative minutes resets to 0
- Next paddling stint starts fresh on the warmup curve
- No partial recovery or residual fatigue

---

## Parameters

### Race Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `distance_km` | Race distance in kilometers | 60 |
| `speed_kmh` | Base speed in km/h (at output = 1.0) | 10 |
| `stint_min` | Stint length in minutes | 40 |
| `switch_time_min` | Time per crew switch in minutes | 1.5 |
| `n_stints` | Number of stints in race | computed: ceiling(distance/speed*60/stint_min) |

### Crew Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `n_seats` | Seats in canoe | 6 |
| `n_resting` | Number of paddlers resting each stint | 3 |
| `n_paddlers` | Total crew size | computed: n_seats + n_resting |
| `seat_weights` | Seat importance for optimization (does NOT affect race_time) | [1.2, 1.1, 0.9, 0.9, 0.9, 1.1] |
| `seat_entry_weights` | Entry ease for optimization (does NOT affect race_time) | [1.0] * n_seats |
| `seat_eligibility` | (n_paddlers × n_seats) matrix of eligible assignments | all 1s (all eligible) |
| `paddler_ability` | Ability multiplier per paddler (>1 stronger, <1 weaker) | [1.0] * n_paddlers |

### Fatigue Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `max_consecutive` | Max consecutive stints (variable bound) | 6 |
| `start_output` | Output at start of paddling (τ=0) | 0.80 |
| `peak_time` | Minutes to reach peak output | 12 |
| `plateau_duration` | Minutes at peak before decay | 10 |
| `decay_rate` | Proportional output loss per minute | 0.01 |

### Solver Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `solver_time_secs` | Maximum solver computation time in seconds | 60 |
| `gap_tolerance` | Acceptable optimality gap (e.g., 0.01 = 1%) | 0.01 |

### Seat Weight Rationale

| Seat | Weight | Reason |
|------|--------|--------|
| 1 | 1.2 | Pacer/voga - sets rhythm, highest priority for rest |
| 2 | 1.1 | Pacer - important rhythm seat |
| 3-5 | 0.9 | Regular seats - flexible |
| 6 | 1.1 | Steerer - tired steerer creates drag |

### Seat Entry Weight

Controls how easy it is to enter each seat from the escort boat during a crew switch.

| Value | Meaning |
|-------|---------|
| 1.0 | Normal difficulty (default) |
| > 1.0 | Easier to enter (optimizer prefers these seats for entries) |
| < 1.0 | Harder to enter (optimizer avoids these for entries) |

**Example:** Middle seats might be harder to reach from the escort boat:

```python
seat_entry_weights = [1.2, 1.0, 0.8, 0.8, 1.0, 1.2]  # ends easier, middle harder
```

**Effect on optimization:**
- The optimizer penalizes entries (from rest) into hard-to-enter seats
- First stint of the race is "free" (no entry penalty)
- Wrap-around entries in subsequent cycles are penalized

### Paddler Ability

Controls the relative strength/output of each paddler.

| Value | Meaning |
|-------|---------|
| 1.0 | Normal ability (default) |
| > 1.0 | Stronger paddler (contributes more output) |
| < 1.0 | Weaker paddler (contributes less output) |

**Example:** Ana is 50% stronger, Ben is 20% weaker:

```python
paddler_ability = [1.5, 0.8, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
```

**Effect on optimization:**
- Stronger paddlers are scheduled to paddle more often
- Weaker paddlers rest more
- Ability multiplier directly affects output contribution

---

## Input/Output Specification

### Input: `paddlers` Data Frame

Required column: `name`

**Example:**

| name |
|------|
| Alice |
| Bob |
| Carol |
| Dave |
| Eve |
| Frank |
| Grace |
| Hank |
| Ivy |

### Input: `seat_eligibility` (Optional)

Binary matrix of shape (n_paddlers × n_seats) where 1 = paddler can sit in seat.

If not provided, defaults to all 1s (all paddlers eligible for all seats).

### Output: `solve_rotation_cycle()` Return Dict

The function returns a dictionary with the following keys:

**`status`** (string): Solver status
- "Optimal" - found optimal solution
- "Not Solved" - solver failed or timed out
- "Infeasible" - no feasible solution exists

**`cycle_schedule`** (DataFrame): The repeating cycle pattern

| | seat1 | seat2 | seat3 | seat4 | seat5 | seat6 |
|---|-------|-------|-------|-------|-------|-------|
| 0 | Ana | Ben | Carlos | Diana | Eve | Frank |
| 1 | Ana | Ben | Carlos | Gina | Hiro | Frank |
| 2 | Ben | Ana | Hiro | Diana | Gina | Ivan |

**`schedule`** (DataFrame): Full race schedule (cycle expanded)

**`avg_output`** (float): Weighted average output across all stints (e.g., 0.85)

**`race_time`** (float): Estimated race time in minutes

**`parameters`** (dict): Computed race parameters
```python
{
    "stint_min": 40,
    "n_stints": 9,
    "cycle_length": 3,
    "seat_entry_weights": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    "paddler_ability": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
}
```

**`paddler_summary`** (DataFrame): Per-paddler statistics

| name | stints_paddled | stints_rested | total_time_min | longest_stretch_stints | longest_stretch_min | stints_paddled_per_cycle |
|------|----------------|---------------|----------------|------------------------|---------------------|--------------------------|
| Ana | 6 | 3 | 240.0 | 2 | 80.0 | 2 |
| Frank | 6 | 3 | 240.0 | 2 | 80.0 | 2 |

**`summary_stats`** (dict): Aggregate crew statistics
```python
{
    'cycle_length': 3,
    'n_stints': 9,
    'n_cycles': 3.0,
    'avg_time_per_paddler_min': 240.0,
    'max_time_any_paddler_min': 240.0,
    'min_time_any_paddler_min': 240.0,
    'max_consecutive_stretch_min': 80.0
}
```

### Output: `optimize_stint_range()` Return Dict

**`summary`** (DataFrame): Comparison across stint durations

| stint_min | n_stints | avg_output | race_time |
|-----------|----------|------------|-----------|
| 30 | 12 | 0.90 | 416.5 |
| 40 | 9 | 0.87 | 411.2 |
| 50 | 8 | 0.82 | 427.3 |

**`best`** (dict): Result with lowest race_time (includes full schedule)

**`results`** (list): All individual solve results

---

## Implementation Details

### Dependencies

**Python packages (in pyproject.toml):**

```toml
dependencies = [
    "pulp>=2.8",
    "pandas>=2.0",
    "numpy>=1.23"
]
```

**Solver:** PuLP's bundled CBC (COIN-OR Branch and Cut) is used by default. No external solver installation required.

### Implemented Functions

**Core optimization (model.py):**

| Function | Purpose |
|----------|---------|
| `solve_rotation_cycle()` | Solve cycle-based MIP, returns schedule + metrics |
| `expand_cycle_to_race()` | Expand cycle schedule to full race |
| `validate_paddlers()` | Validates paddler DataFrame structure |
| `validate_params()` | Validates optimization parameters |
| `validate_eligibility()` | Validates eligibility matrix dimensions and feasibility |

**Fatigue model (fatigue.py):**

| Function | Purpose |
|----------|---------|
| `output_curve(tau, start, peak, plateau, decay)` | Returns output multiplier for given cumulative minutes |
| `average_output(stint_min, k, step)` | Averages curve over a consecutive stint |
| `compute_output_table(stint_min, max_consecutive)` | Pre-computes output for each consecutive stint count |

**Meta-optimization (meta.py):**

| Function | Purpose |
|----------|---------|
| `optimize_stint_range()` | Grid search over stint durations |

### Race Time Calculation (Exact Simulation)

The race time is computed by simulating the actual race stint-by-stint using the cycle pattern:

```python
cycle_length = n_paddlers // n_resting  # e.g., 3
n_stints = ceil((distance_km/speed_kmh*60)/stint_min)

# Simulate actual race stint by stint
stint_outputs = []
consecutive = {p: 0 for p in range(P)}  # Track from fresh start

for t in range(n_stints):
    cycle_t = t % cycle_length
    # Update consecutive counts (resets on rest)
    # Calculate stint output using actual consecutive count
    stint_outputs.append(stint_output)

avg_output = sum(stint_outputs) / n_stints
race_time = nominal / avg_output + (n_stints - 1) * switch_time_min
```

This exact simulation handles:
- First cycle fresh start (no wrap-around effect)
- Subsequent cycles with wrap-around (consecutive carries across cycle boundary)
- Partial cycles at the end (just simulates fewer stints)
- Short races with n_stints < cycle_length (works correctly)

### Model Size

For cycle_length=3, 9 paddlers, 6 seats, max_consecutive = 6:

| Variable | Count | Notes |
|----------|-------|-------|
| X[p,s,t] | ≤ 9 × 6 × 3 = 162 | Only for eligible (p,s) pairs |
| R[p,t] | 9 × 3 = 27 | |
| S[p,t] | 9 × 3 = 27 | |
| Y[p,t,k] | 9 × 3 × 6 = 162 | |
| Q[p,t,k] | 9 × 3 × 6 = 162 | Continuous (linearization) |
| Link[p] | 9 | Wrap-around consecutive tracking |
| **Total** | ~549 variables | |

This is very tractable for CBC (typically solves in 1-10 seconds).

---

## Examples

The `example.py` file contains 5 working examples:

1. **Basic usage** - All paddlers eligible for all seats
2. **Custom eligibility** - Seat restrictions with eligibility matrix
3. **Different crew size** - 8 paddlers, 4 seats, 4 resting
4. **Meta-optimization** - Find optimal stint duration
5. **Paddler ability** - Varying paddler strength

Run examples:
```bash
python example.py      # Run all examples
python example.py 1    # Run specific example
```

---

## Testing Strategy

| Test Category | What to Test |
|---------------|--------------|
| Fatigue curve | Correct values at key points (0, 12, 22, 60 min) |
| Stint averaging | Integral matches numerical average |
| Constraint satisfaction | All seats filled, eligibility respected, one place per paddler |
| Consecutive tracking | S resets on rest, increments when paddling |
| Wrap-around fatigue | Paddler paddling at stints C-1 and 0 has Scon[p,0] = 2 |
| Output curve lookup | Correct multiplier selected via Y variables |
| Race time calculation | Estimated time computed correctly from simulation |
| Cycle expansion | Expanded schedule correctly repeats cycle pattern |
| Meta-optimization | Grid search returns best stint_min, all solves complete |
| Default eligibility | No eligibility = all 1s (all paddlers can sit anywhere) |
| Custom eligibility | Solver respects arbitrary eligibility matrix |
| Eligibility validation | Invalid eligibility (empty seats/paddlers) raises error |
| Input validation | Missing columns, wrong counts, invalid params raise errors |
| Paddler ability | Stronger paddlers scheduled more, affects output |

---

## Future Enhancements (Out of Scope for Now)

- Visualization functions (plot.py)
- Partial recovery based on rest duration
- Weather/conditions affecting fatigue rates
- Mid-race strategy adjustments
- Multi-objective optimization (time vs. fairness)
- Stochastic optimization for uncertain conditions
- Formal test suite (pytest)
