# Outrigger Rotation Optimization Plan

## Overview

Extend the `outrigger` package with a mixed-integer programming (MIP) optimizer that finds the optimal crew rotation schedule, minimizing estimated race time while accounting for fatigue, recovery, and switching overhead.

**Design principle:** The optimization module is fully self-contained. It does not depend on existing package functions (`analyze_scenario()`, `run_scenarios()`, etc.), which may be refactored or removed later. All necessary calculations (e.g., `n_stints` from distance/speed) are computed independently within the optimization code.

## Implementation Status

| Stage | Status | Notes |
|-------|--------|-------|
| Stage 1: Fatigue Curve | ✅ Complete | `fatigue.py` |
| Stage 2: Race Time Calculation | ✅ Complete | Inline in `solve_rotation_full()` |
| Stage 3: Parameters & Validation | ✅ Complete | `model.py` |
| Stage 4: MIP Model (Full Race) | ✅ Complete | `model.py` with PuLP/CBC |
| Stage 5: Output Formatting | ✅ Complete | Returns schedule + paddler stats |
| Stage 6: Visualization | 🔲 Not Started | `plot.py` planned |
| Stage 7: Meta-Optimization | ✅ Complete | `meta.py` |
| Stage 8: Integration | ✅ Mostly Complete | `example.py` working, tests pending |
| Stage 9: Pattern Consistency | ✅ Complete | Entry rule + switch rule penalties |
| Stage 10: Cycle-Based Model | ✅ Complete | `solve_rotation_cycle()` with wrap-around fatigue |

**Package location:** `outrigger_opt/`

**Key files:**
- `model.py` - Core MIP solvers (`solve_rotation_full()`, `solve_rotation_cycle()`)
- `fatigue.py` - Fatigue curve functions
- `meta.py` - Stint duration grid search (`optimize_stint_range()`)
- `utils.py` - Helper functions (`demo_paddlers()`)
- `example.py` - Working usage examples (6 examples)

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

crew_output_t = (Σ_s Σ_p X[p,s,t] × seat_weight[s] × output_multiplier[p,t]) / Σ_s seat_weight[s]
```

**Why this metric:**
- Directly comparable across different stint durations
- Captures the tradeoff:
  - Shorter stints → more switches (overhead), but higher avg_output (less fatigue)
  - Longer stints → fewer switches, but lower avg_output (more fatigue)
- Enables meta-optimization over stint_min parameter

**Example comparison (60 km, 10 km/h base, 1.5 min switch):**

| Stint duration | n_stints | n_switches | avg_output | race_time |
|----------------|----------|------------|------------|-----------|
| 30 min | 12 | 11 | 0.90 | 416.5 min |
| 50 min | 8 | 7 | 0.82 | 449.5 min |
| 60 min | 6 | 5 | 0.75 | 487.5 min |

**Note on linearity:**
To minimize `nominal_paddle_time / avg_output`, we equivalently maximize `avg_output` (since nominal_paddle_time is fixed for a given stint duration). The switching overhead `n_switches × switch_time_min` is a constant for fixed stint duration.

So for a fixed stint_min, the MIP objective simplifies to:

```
Maximize: Σ_t Σ_s Σ_p X[p,s,t] × seat_weight[s] × output_multiplier[p,t]
```

This is linear and tractable. Race time is computed post-hoc from the solution.

### Decision Variables

| Variable | Type | Description |
|----------|------|-------------|
| `X[p,s,t]` | Binary | Paddler p in seat s at stint t (only created for eligible p,s pairs) |
| `R[p,t]` | Binary | Paddler p is resting at stint t |
| `S[p,t]` | Integer [0, max_consecutive] | Consecutive stints paddling for p at t |
| `Y[p,t,k]` | Binary | Paddler p at stint t has exactly k consecutive stints |
| `Q[p,t,k]` | Continuous [0, max_weight] | Linearization variable for weighted seat × fatigue output |

**Pattern consistency variables (optional, for reducing confusion):**

| Variable | Type | Description |
|----------|------|-------------|
| `EntryUsed[p,s]` | Binary | Paddler p ever enters seat s from rest |
| `TransitionUsed[p,s,s']` | Binary | Paddler p ever transitions from seat s to s' while paddling |

Note: These variables count rules to remember. Fewer rules = simpler pattern. Set penalties to 0 to disable.

**Seat entry weight variables (optional, for entry difficulty):**

| Variable | Type | Description |
|----------|------|-------------|
| `Entry[p,s,t]` | Binary | Paddler p enters seat s at stint t from rest (t > 0 only) |

Note: Only created when any seat_entry_weight != 1.0. Penalizes entries into hard-to-enter seats.

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

**Consecutive stint tracking:**

| Constraint | Formula |
|------------|---------|
| Reset on rest | `S[p,t] ≤ max_consecutive × (1 - R[p,t])` |
| Increment when paddling | `S[p,t] ≥ S[p,t-1] + 1 - max_consecutive × R[p,t]` |
| Cap increment | `S[p,t] ≤ S[p,t-1] + 1` |
| Initial condition | `S[p,1] = 1 - R[p,1]` |

**Linearizing output curve lookup (Y variables):**

| Constraint | Formula |
|------------|---------|
| Exactly one k active | `Σ_k Y[p,t,k] = 1 - R[p,t]` |
| Link Y to S | `S[p,t] = Σ_k k × Y[p,t,k]` |

**Linearizing objective (Q variables):**

The objective requires computing `X[p,s,t] × seat_weight[s] × output[k] × Y[p,t,k]`, which contains a product of binary variables X and Y. This is linearized using auxiliary continuous variables Q.

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
Maximize: Σ_{p,t,k} Q[p,t,k] × output_table[k]
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
- Visualization of race time vs stint duration

### Parallelization

Each stint duration is independent - can solve in parallel:

```python
from concurrent.futures import ProcessPoolExecutor

with ProcessPoolExecutor(max_workers=4) as executor:
    results = list(executor.map(
        lambda s: solve_rotation_full(paddlers, stint_min=s),
        stint_range
    ))
```

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

**With defaults (start=0.80, peak=12, plateau=10, decay=1%):**

```
output(τ) =
  if τ ≤ 12:   0.80 × 1.25^(τ/12)    # exponential rise
  if τ ≤ 22:   1.00                   # plateau
  if τ > 22:   0.99^(τ - 22)          # exponential decay
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

**Key values:**

| Minutes | Output | Phase |
|---------|--------|-------|
| 0 | 0.80 | start |
| 6 | 0.89 | rising |
| 12 | 1.00 | peak |
| 22 | 1.00 | end plateau |
| 32 | 0.90 | decaying |
| 42 | 0.82 | decaying |
| 62 | 0.67 | fatigued |
| 82 | 0.55 | tired |
| 102 | 0.45 | exhausted |

### Averaging Over Stints

For a stint of length `L` minutes, paddler starting at cumulative minute `τ_start`:

```
stint_output = (1/L) × ∫[τ_start to τ_start + L] output(τ) dτ
```

Pre-compute average output for each consecutive stint given stint length.

**Example: 50-minute stints**

| Consecutive Stint | Minutes Paddling | Avg Output |
|-------------------|------------------|------------|
| 1 | 0–50 | 0.91 |
| 2 | 50–100 | 0.59 |
| 3 | 100–150 | 0.36 |
| 4 | 150–200 | 0.22 |

Note: With 50-min stints, paddlers should rarely go beyond 2 consecutive stints.

**Example: 30-minute stints**

| Consecutive Stint | Minutes Paddling | Avg Output |
|-------------------|------------------|------------|
| 1 | 0–30 | 0.95 |
| 2 | 30–60 | 0.80 |
| 3 | 60–90 | 0.59 |
| 4 | 90–120 | 0.44 |

Note: 30-min stints allow more flexibility—stint 2 still has good output.

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
| `stint_min` | Stint length in minutes | 50 |
| `switch_time_min` | Time per crew switch in minutes | 1.5 |
| `n_stints` | Number of stints in race | computed: ceiling(distance/speed*60/stint_min) |

### Crew Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `n_seats` | Seats in canoe | 6 |
| `n_resting` | Number of paddlers resting each stint | 3 |
| `n_paddlers` | Total crew size | computed: n_seats + n_resting |
| `seat_weights` | Importance weight by seat | [1.2, 1.1, 0.9, 0.9, 0.9, 1.1] |
| `seat_entry_weight` | Entry ease weight by seat (>1 easier, <1 harder) | [1.0] * n_seats |
| `seat_eligibility` | (n_paddlers × n_seats) matrix of eligible assignments | generated from roles |

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
| `time_limit` | Maximum solver time in seconds | 60 |
| `gap_tolerance` | Acceptable optimality gap (e.g., 0.01 = 1%) | 0.01 |

### Switching Model

**`switch_time_min`:** Time (in minutes) the boat stops/slows during crew changes. This is real time added to the race.

**Crew switch** (rest ↔ paddle): Happens at stint boundaries. Adds `switch_time_min` per switch.

**Seat switch** (seat A → seat B while staying in boat): No additional time cost. Happens during the same crew change. Matters only because:
- Different seats have different weights
- Allows placing highest-output paddlers in highest-weight seats

**Key insight:** A seat switch does NOT reset fatigue. Only resting resets the consecutive stint counter.

### Pattern Consistency (Reducing Confusion)

Paddlers need to remember rules for:
1. **Which seat to enter** when coming back from rest
2. **What to do next** at each seat (stay or move to which seat)

**Fewer rules = less confusion = fewer mistakes.**

#### Counting Rules

**Entry rules** = number of different seats a paddler enters from rest

| Paddler A entries | Entry rules |
|-------------------|-------------|
| Always enters seat 2 | 1 |
| Sometimes seat 2, sometimes seat 3 | 2 |
| Enters seats 2, 3, and 4 at different times | 3 |

**Switch rules** = number of different seat transitions while paddling (including staying in same seat)

| Paddler A transitions | Switch rules |
|-----------------------|--------------|
| Always stays in seat 2 (2→2) | 1 |
| Stays in seat 2 (2→2) OR moves 2→3 | 2 |
| 2→2, 2→3, and 3→3 | 3 |

#### Variables

**`EntryUsed[p,s]`** = 1 if paddler p ever enters seat s from rest

```
EntryUsed[p,s] ≥ X[p,s,t]  when R[p,t-1] = 1 (entering from rest)
```

**`TransitionUsed[p,s,s']`** = 1 if paddler p ever transitions from seat s to seat s' while paddling

```
TransitionUsed[p,s,s'] ≥ X[p,s,t-1] + X[p,s',t] - 1  when R[p,t-1] = 0 and R[p,t] = 0
```

#### Rule Counts

```
entry_rules[p] = Σ_s EntryUsed[p,s]
switch_rules[p] = Σ_{s,s'} TransitionUsed[p,s,s']
```

#### Objective

```
Maximize: weighted_output
          - λ_entry × Σ_p entry_rules[p]
          - λ_switch × Σ_p switch_rules[p]
```

**Ideal scenario:** Each paddler has 1 entry rule and minimal switch rules.

**Verify** The first assignment (starting position) shouldnt count for complexity. 

#### Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `entry_rule_penalty` | Penalty per entry rule per paddler | 0.02 |
| `switch_rule_penalty` | Penalty per switch rule per paddler | 0.02 |

**Interpretation:**
- With 9 paddlers, if everyone has 1 entry rule: penalty = 9 × 0.02 = 0.18
- If average is 2 entry rules: penalty = 18 × 0.02 = 0.36
- The optimizer will trade off output vs. pattern simplicity
- Set to 0 to disable (pure output optimization)

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
seat_entry_weight = [1.2, 1.0, 0.8, 0.8, 1.0, 1.2]  # ends easier, middle harder
```

**Effect on optimization:**
- The optimizer penalizes entries (from rest) into hard-to-enter seats
- Does NOT affect race_time calculation (only optimization preference)
- Scale factor converts entry difficulty to output-equivalent units
- Set all to 1.0 to disable (default behavior)

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

### Output: `solve_rotation_full()` Return Dict

The function returns a dictionary with the following keys:

**`status`** (string): Solver status
- "Optimal" - found optimal solution
- "Not Solved" - solver failed or timed out
- "Infeasible" - no feasible solution exists

**`schedule`** (DataFrame): Wide format schedule

| | seat1 | seat2 | seat3 | seat4 | seat5 | seat6 |
|---|-------|-------|-------|-------|-------|-------|
| 0 | Ana | Ben | Carlos | Diana | Eve | Frank |
| 1 | Ana | Ben | Carlos | Diana | Gina | Frank |
| 2 | Ben | Ana | Hiro | Diana | Gina | Frank |

**`avg_output`** (float): Weighted average output across all stints (e.g., 0.85)

**`race_time`** (float): Estimated race time in minutes

**`parameters`** (dict): Computed race parameters
```python
{"stint_min": 40, "n_stints": 9, "seat_entry_weight": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]}
```

**`paddler_summary`** (DataFrame): Per-paddler statistics

| name | stints_paddled | stints_rested | total_time_min | longest_stretch_stints | longest_stretch_min |
|------|----------------|---------------|----------------|------------------------|---------------------|
| Ana | 6 | 2 | 240 | 3 | 120 |
| Frank | 5 | 3 | 200 | 2 | 80 |

**`summary_stats`** (dict): Aggregate crew statistics
```python
{
    'avg_time_per_paddler_min': 213.3,
    'max_time_any_paddler_min': 280.0,
    'min_time_any_paddler_min': 160.0,
    'max_consecutive_stretch_min': 120.0,
    'avg_consecutive_stretch_min': 80.0
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

## Implementation Plan

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

| Function | Purpose | Status |
|----------|---------|--------|
| `solve_rotation_full()` | Solve MIP for fixed stint duration, returns schedule + metrics | ✅ Implemented |
| `validate_paddlers()` | Validates paddler DataFrame structure | ✅ Implemented |
| `validate_params()` | Validates optimization parameters | ✅ Implemented |
| `validate_eligibility()` | Validates eligibility matrix dimensions and feasibility | ✅ Implemented |

**Fatigue model (fatigue.py):**

| Function | Purpose | Status |
|----------|---------|--------|
| `output_curve(tau, start, peak, plateau, decay)` | Returns output multiplier for given cumulative minutes | ✅ Implemented |
| `average_output(stint_min, k, step)` | Averages curve over a consecutive stint | ✅ Implemented |
| `compute_output_table(stint_min, max_consecutive)` | Pre-computes output for each consecutive stint count | ✅ Implemented |

**Meta-optimization (meta.py):**

| Function | Purpose | Status |
|----------|---------|--------|
| `optimize_stint_range()` | Grid search over stint durations | ✅ Implemented |

**Utilities (utils.py):**

| Function | Purpose | Status |
|----------|---------|--------|
| `demo_paddlers()` | Returns sample paddler DataFrame for testing | ✅ Implemented |

**Visualization (not yet implemented):**

| Function | File | Purpose |
|----------|------|---------|
| `plot_schedule()` | plot.py | Heatmap of paddler × stint assignments |
| `plot_output_curve()` | plot.py | Visualize the fatigue curve |
| `plot_stint_output()` | plot.py | Bar chart of total output per stint |
| `plot_stint_comparison()` | plot.py | Race time vs stint duration (meta-optimization result) |

### Implementation Stages

Each stage must be completed, tested, and verified before proceeding to the next.

---

#### Stage 1: Fatigue Curve ✅ COMPLETE

**Goal:** Implement the output curve that models paddler fatigue over time.

**Implemented in `fatigue.py`:**
- `output_curve(tau, start=0.8, peak=12, plateau=10, decay=0.01)` → returns output multiplier for cumulative minutes τ
- `average_output(stint_min, k, step=0.25)` → averages curve over consecutive stint k using numerical integration
- `compute_output_table(stint_min, max_consecutive)` → returns dict {k: avg_output} for k=1..max_consecutive

---

#### Stage 2: Race Time Calculation ✅ COMPLETE

**Goal:** Given a schedule (paddler assignments), compute estimated race time.

**Implemented inline in `solve_rotation_full()`:**

Race time is computed from the MIP solution using:
```python
nominal = distance_km/speed_kmh * 60
switches = n_stints - 1
race_time = nominal / avg_output + switches * switch_time_min
```

---

#### Stage 3: Parameters and Paddler Roster ✅ COMPLETE

**Goal:** Clean interface for setting up optimization inputs.

**Implemented in `model.py`:**
- `validate_paddlers(paddlers, n_seats, n_paddlers)` → validates and returns DataFrame with reset index
- `validate_params(stint_min, max_consecutive, distance_km, speed_kmh, switch_time_min)` → raises ValueError on invalid params

**Note:** Parameters are passed directly to `solve_rotation_full()` rather than via a separate params object.

---

#### Stage 4: MIP Model (Single Stint Duration) ✅ COMPLETE

**Goal:** Build and solve the optimization model for fixed stint duration.

**Implemented in `model.py`:**
- `solve_rotation_full()` → complete MIP with PuLP/CBC solver

**Model components implemented:**
- Decision variables: X[p,s,t] (only for eligible pairs), R[p,t], S[p,t], Y[p,t,k], Q[p,t,k]
- All constraints: seat assignment, paddler placement, resting count, consecutive tracking, Y/Q linearization
- Objective: maximize weighted output (equivalent to minimizing race time)

**Solver configuration:**
- Time limit (default 60s)
- Gap tolerance (default 1%)

---

#### Stage 5: Output Formatting ✅ COMPLETE

**Goal:** Convert solution into user-friendly formats.

**Implemented in `solve_rotation_full()` return dict:**

| Output | Format | Description |
|--------|--------|-------------|
| `status` | string | Solver status ("Optimal", "Infeasible", etc.) |
| `schedule` | DataFrame | Wide format: stint × seats with paddler names |
| `avg_output` | float | Weighted average output across all stints |
| `race_time` | float | Estimated race time in minutes |
| `parameters` | dict | Contains `stint_min`, `n_stints` |
| `paddler_summary` | DataFrame | Per-paddler stats (stints paddled/rested, total time, longest stretch) |
| `summary_stats` | dict | Aggregate metrics (avg/max/min time per paddler, max consecutive stretch) |

**Note:** `schedule_long` and `transitions` tables from original plan not implemented. Current output focuses on practical metrics.

---

#### Stage 6: Visualization 🔲 NOT STARTED

**Goal:** Visual outputs for schedule and analysis.

**Planned functions (not yet implemented):**
- `plot_schedule(result)` → heatmap of paddler × stint assignments
- `plot_output_curve(params)` → fatigue curve visualization
- `plot_stint_output(result)` → bar chart of crew output per stint

---

#### Stage 7: Meta-Optimization (Stint Duration) ✅ COMPLETE

**Goal:** Find optimal stint duration via grid search.

**Implemented in `meta.py`:**
- `optimize_stint_range(paddlers, stint_range, max_consecutive)` → returns dict with summary table and best result

**Returns:**
- `summary`: DataFrame comparing all stint durations (stint_min, n_stints, avg_output, race_time)
- `best`: dict with lowest race_time result including full schedule
- `results`: list of all individual results

**Note:** Parallelization mentioned in original plan not implemented (runs sequentially).

---

#### Stage 8: Integration and Documentation ✅ MOSTLY COMPLETE

**Goal:** Full workflow tested end-to-end.

**Completed:**
- Working optimization module with pip-installable package structure
- `example.py` with five usage examples:
  1. All paddlers eligible (default, simplest)
  2. Custom eligibility matrix
  3. Different crew sizes (configurable n_seats, n_resting)
  4. Meta-optimization across stint durations
  5. Pattern consistency penalties

**Not completed:**
- Formal test suite (pytest tests)
- Baseline comparison (naive round-robin)
- Comprehensive documentation beyond example.py

---

#### Stage 9: Pattern Consistency ✅ COMPLETE

**Goal:** Penalize complex rotation patterns by counting rules each paddler must remember.

**Implemented in `model.py`:**

New parameters added to `solve_rotation_full()`:
- `entry_rule_penalty` - Penalty per entry rule per paddler (default: 0.0 = disabled)
- `switch_rule_penalty` - Penalty per switch rule per paddler (default: 0.0 = disabled)

**Variables (only created when penalties > 0):**
- `EntryUsed[p,s]` - Binary, 1 if paddler p ever enters seat s from rest
- `TransitionUsed[p,s,s']` - Binary, 1 if paddler p ever transitions s→s' while paddling

**Constraints:**
```python
# Track entry seats used (t=0 counts as entering from rest)
EntryUsed[p,s] >= X[p,s,0]
EntryUsed[p,s] >= X[p,s,t] + R[p,t-1] - 1  # for t > 0

# Track transitions used (consecutive paddling stints)
TransitionUsed[p,s,s'] >= X[p,s,t-1] + X[p,s',t] - 1 - R[p,t-1] - R[p,t]
```

**Modified objective:**
```
Maximize: weighted_output
          - entry_rule_penalty × Σ EntryUsed[p,s]
          - switch_rule_penalty × Σ TransitionUsed[p,s,s']
```

**New output:** `pattern_stats` dict with:
- `total_entry_rules`, `total_switch_rules`
- `avg_entry_rules_per_paddler`, `avg_switch_rules_per_paddler`
- Per-paddler `entry_rules` and `switch_rules` in `paddler_summary`

**Example:** `python example.py 5` demonstrates the effect of pattern penalties

---

#### Stage 10: Cycle-Based Model ✅ COMPLETE

**Goal:** Simplify the MIP by modeling a single rotation cycle instead of the entire race.

**Key insight:** For 9 paddlers, 6 seats, 3 resting:
- Minimum cycle length = n_paddlers / n_resting = 3 stints
- Each paddler paddles 2 stints, rests 1 stint per cycle
- The pattern repeats for the entire race

**Benefits:**
- ~66% reduction in variables (549 vs 1620 for typical race)
- Solve time ~3x faster
- Naturally produces repeating patterns (easier for crew to remember)
- Wrap-around fatigue ensures pattern is sustainable when repeated

**Implemented in `model.py`:**

New function `solve_rotation_cycle()` with same interface as `solve_rotation_full()`.

**New variable for wrap-around fatigue:**
- `Link[p]` - Integer, captures consecutive count at end of cycle for wrap-around

**Wrap-around constraints:**
```python
# Link[p] = Scon[p,C-1] if paddling at end of cycle, else 0
Link[p] <= Scon[p, cycle_length-1]
Link[p] <= max_consecutive * (1 - R[p, cycle_length-1])
Link[p] >= Scon[p, cycle_length-1] - max_consecutive * R[p, cycle_length-1]

# Scon[p,0] with wrap-around: continues from Link if paddling
Scon[p, 0] <= max_consecutive * (1 - R[p, 0])
Scon[p, 0] >= Link[p] + 1 - max_consecutive * R[p, 0]
Scon[p, 0] <= Link[p] + 1
```

**Race time calculation (exact simulation):**

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

**Most parameters from `solve_rotation_full()` are supported:**
- `seat_eligibility` - Custom seat restrictions
- `seat_weights` - Seat importance weights
- `seat_entry_weight` - Entry difficulty weights

**Not supported in cycle model:**
- `entry_rule_penalty` and `switch_rule_penalty` - Pattern penalties are not applicable
  because the repeating cycle already produces a simple, consistent pattern

**Output includes:**
- `cycle_schedule` - The 3-stint repeating pattern
- `schedule` - Full race schedule (cycle expanded)
- `parameters.cycle_length` - Cycle length used
- `parameters.seat_entry_weight` - Entry weights used

**Example:** `python example.py 6` compares cycle-based vs full-race solver

---

### Model Size Estimate

#### Full-Race Model (`solve_rotation_full`)

For 9 stints, 9 paddlers, 6 seats, max_consecutive = 6:

**Core variables:**

| Variable | Count | Notes |
|----------|-------|-------|
| X[p,s,t] | ≤ 9 × 6 × 9 = 486 | Only for eligible (p,s) pairs |
| R[p,t] | 9 × 9 = 81 | |
| S[p,t] | 9 × 9 = 81 | |
| Y[p,t,k] | 9 × 9 × 6 = 486 | |
| Q[p,t,k] | 9 × 9 × 6 = 486 | Continuous (linearization) |
| **Subtotal** | ~1,620 variables | |

#### Cycle-Based Model (`solve_rotation_cycle`)

For cycle_length=3, 9 paddlers, 6 seats, max_consecutive = 6:

**Core variables:**

| Variable | Count | Notes |
|----------|-------|-------|
| X[p,s,t] | ≤ 9 × 6 × 3 = 162 | Only for eligible (p,s) pairs |
| R[p,t] | 9 × 3 = 27 | |
| S[p,t] | 9 × 3 = 27 | |
| Y[p,t,k] | 9 × 3 × 6 = 162 | |
| Q[p,t,k] | 9 × 3 × 6 = 162 | Continuous (linearization) |
| Link[p] | 9 | Wrap-around consecutive tracking |
| **Subtotal** | ~549 variables | |

**Comparison:** Cycle-based model has **66% fewer variables** than full-race model.

#### Optional Variables (Full-Race Model)

**Pattern consistency variables (when penalties > 0):**

| Variable | Count | Notes |
|----------|-------|-------|
| EntryUsed[p,s] | 9 × 6 = 54 | Which seats each paddler enters from rest |
| TransitionUsed[p,s,s'] | 9 × 6 × 6 = 324 | Which transitions each paddler uses |
| **Subtotal** | ~378 variables | |

**Seat entry weight variables (when any weight != 1.0):**

| Variable | Count | Notes |
|----------|-------|-------|
| Entry[p,s,t] | ≤ 9 × 6 × 7 = 378 | Per-stint entries (t > 0 only, eligible pairs) |
| **Subtotal** | ~378 variables | |

| **Total with all optional features** | ~2,196 variables | |

**Constraint counts:**

| Constraint Type | Count |
|-----------------|-------|
| Seat assignment | 6 × 8 = 48 |
| Paddler in one place | 9 × 8 = 72 |
| Resting count | 8 |
| Consecutive tracking | 9 × (1 + 3×7) = 198 |
| Y linearization | 9 × 8 × 2 = 144 |
| Q linearization | 9 × 8 × 6 × 3 = 1,296 |
| **Total** | ~1,766 constraints |

This is tractable for CBC (typically solves in 10-60 seconds). The time limit parameter prevents excessive solve times.

---

## Testing Strategy

| Test Category | What to Test |
|---------------|--------------|
| Fatigue curve | Correct values at key points (0, 12, 22, 60 min) |
| Stint averaging | Integral matches numerical average |
| Constraint satisfaction | All seats filled, eligibility respected, one place per paddler |
| Consecutive tracking | S resets on rest, increments when paddling |
| Output curve lookup | Correct multiplier selected via Y variables |
| Race time calculation | Estimated time computed correctly from solution |
| Known optimal | 3-stint toy problem verified by hand |
| Baseline comparison | Optimizer beats naive round-robin |
| Output format | Schedule has correct structure and dimensions |
| Meta-optimization | Grid search returns best stint_min, all solves complete |
| Stint comparison | Shorter stints have more switches, longer stints have more fatigue |
| Default eligibility | No eligibility = all 1s (all paddlers can sit anywhere) |
| Custom eligibility | Solver respects arbitrary eligibility matrix |
| Eligibility validation | Invalid eligibility (empty seats/paddlers) raises error |
| Input validation | Missing columns, wrong counts, invalid params raise errors |
| Entry rules count | EntryUsed correctly tracks which seats each paddler enters from rest |
| Switch rules count | TransitionUsed correctly tracks seat transitions while paddling |
| Rule penalties | Higher penalties → fewer total rules (simpler patterns) |
| Penalty=0 equivalence | With penalties=0, solution matches original (pure output optimization) |
| Cycle length | Correct cycle_length = n_paddlers / n_resting |
| Wrap-around fatigue | Paddler paddling at stints C-1 and 0 has Scon[p,0] = 2 |
| Cycle expansion | Expanded schedule correctly repeats cycle pattern |
| Cycle vs full comparison | Both solvers produce similar race times |

---

## Future Enhancements (Out of Scope for Now)

- Partial recovery based on rest duration
- Individual paddler strength/endurance parameters
- Weather/conditions affecting fatigue rates
- Mid-race strategy adjustments
- Multi-objective optimization (time vs. fairness)
- Stochastic optimization for uncertain conditions

