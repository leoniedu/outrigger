# Outrigger Rotation Optimization Plan

## Overview

Extend the `outrigger` package with a mixed-integer programming (MIP) optimizer that finds the optimal crew rotation schedule, minimizing estimated race time while accounting for fatigue, recovery, and switching overhead.

**Design principle:** The optimization module is fully self-contained. It does not depend on existing package functions (`analyze_scenario()`, `run_scenarios()`, etc.), which may be refactored or removed later. All necessary calculations (e.g., `n_stints` from distance/speed) are computed independently within the optimization code.

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

Note: Transition variables (`crew_switch`, `seat_switch`) removed from model. Transitions are computed post-hoc from the solution for reporting.

### Seat Eligibility

Seat assignments are controlled by an **eligibility matrix** `E[p,s]` where 1 means paddler p can sit in seat s.

**Default eligibility (from roles):**

| Role | Eligible Seats | Description |
|------|----------------|-------------|
| pacer | 1-5 | Front seats and middle |
| regular | 3-5 | Middle seats only |
| steerer | 3-6 | Middle seats and steering |

**Custom eligibility:** Users can provide any (n_paddlers × n_seats) binary matrix to specify arbitrary seat restrictions.

**Example custom eligibility:**

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
| `seat_weights` | Importance weight by seat | [1.2, 1.0, 0.9, 0.9, 0.9, 1.1] |
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
| `gap_tolerance` | Acceptable optimality gap (e.g., 0.02 = 2%) | 0.02 |

### Switching Model

**`switch_time_min`:** Time (in minutes) the boat stops/slows during crew changes. This is real time added to the race.

**Crew switch** (rest ↔ paddle): Happens at stint boundaries. Adds `switch_time_min` per switch.

**Seat switch** (seat A → seat B while staying in boat): No additional time cost. Happens during the same crew change. Matters only because:
- Different seats have different weights
- Allows placing highest-output paddlers in highest-weight seats

**Key insight:** A seat switch does NOT reset fatigue. Only resting resets the consecutive stint counter.

Note: `crew_switch_cost` and `seat_switch_cost` penalty terms removed from objective. Switching overhead is captured directly via `switch_time_min` in the race time formula.

### Seat Weight Rationale

| Seat | Weight | Reason |
|------|--------|--------|
| 1 | 1.2 | Pacer/voga - sets rhythm, highest priority for rest |
| 2 | 1.0 | Pacer - important but less critical than seat 1 |
| 3-5 | 0.9 | Regular seats - flexible |
| 6 | 1.1 | Steerer - tired steerer creates drag |

---

## Input/Output Specification

### Input: `paddlers` Data Frame

Required column: `name`

Optional column: `role` (used if `seat_eligibility` not provided)

**Example with roles (generates eligibility automatically):**

| name | role |
|------|------|
| Ana | pacer |
| Ben | pacer |
| Carlos | pacer |
| Diana | regular |
| Eve | regular |
| Gina | regular |
| Hiro | regular |
| Frank | steerer |
| Ivan | steerer |

**Example without roles (requires custom eligibility):**

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

If not provided, generated from `role` column using `make_eligibility_from_roles()`.

### Output: `schedule` (Wide Format)

| stint | seat_1 | seat_2 | seat_3 | seat_4 | seat_5 | seat_6 | resting |
|-------|--------|--------|--------|--------|--------|--------|---------|
| 1 | Ana | Ben | Carlos | Diana | Eve | Frank | Gina, Hiro, Ivan |
| 2 | Ana | Ben | Carlos | Diana | Gina | Frank | Eve, Hiro, Ivan |
| 3 | Ben | Ana | Hiro | Diana | Gina | Frank | Carlos, Eve, Ivan |

### Output: `schedule_long` (Long Format)

| stint | seat | paddler | role | consecutive_stint | output_multiplier |
|-------|------|---------|------|-------------------|-------------------|
| 1 | 1 | Ana | pacer | 1 | 0.93 |
| 1 | 2 | Ben | pacer | 1 | 0.93 |
| 1 | 6 | Frank | steerer | 1 | 0.93 |
| 1 | NA | Gina | regular | 0 | NA |

### Output: `transitions`

| stint | type | paddler | from | to |
|-------|------|---------|------|-----|
| 2 | seat_switch | Ana | 1 | 2 |
| 2 | crew_switch_in | Gina | rest | 5 |
| 2 | crew_switch_out | Eve | 5 | rest |

### Output: `paddler_summary`

| paddler | role | stints_paddled | stints_rested | total_output | avg_output |
|---------|------|----------------|---------------|--------------|------------|
| Ana | pacer | 6 | 2 | 5.43 | 0.91 |
| Frank | steerer | 5 | 3 | 4.51 | 0.90 |

---

## Implementation Plan

### Dependencies

**Python packages (in pyproject.toml):**

```toml
dependencies = [
    "pandas",
    "numpy",
    "pulp",  # MIP solver interface (includes CBC)
]
```

**Solver:** PuLP's bundled CBC (COIN-OR Branch and Cut) is used by default. No external solver installation required.

### New Functions

**Core optimization:**

| Function | File | Purpose |
|----------|------|---------|
| `solve_rotation_full()` | model.py | Solve MIP for fixed stint duration |
| `optimize_stint_range()` | meta.py | Grid search over stint durations (meta-optimization) |

**Fatigue model:**

| Function | File | Purpose |
|----------|------|---------|
| `output_curve()` | fatigue.py | Returns output multiplier for given cumulative minutes |
| `average_output()` | fatigue.py | Averages curve over a consecutive stint |
| `compute_output_table()` | fatigue.py | Pre-computes output for each consecutive stint count |

**Validation and eligibility:**

| Function | File | Purpose |
|----------|------|---------|
| `validate_paddlers()` | model.py | Validates paddler DataFrame structure |
| `validate_params()` | model.py | Validates optimization parameters |
| `make_eligibility_from_roles()` | model.py | Generates eligibility matrix from role column |
| `validate_eligibility()` | model.py | Validates eligibility matrix dimensions and feasibility |

**Visualization (planned):**

| Function | File | Purpose |
|----------|------|---------|
| `plot_schedule()` | plot.py | Heatmap of paddler × stint assignments |
| `plot_output_curve()` | plot.py | Visualize the fatigue curve |
| `plot_stint_output()` | plot.py | Bar chart of total output per stint |
| `plot_stint_comparison()` | plot.py | Race time vs stint duration (meta-optimization result) |

### Implementation Stages

Each stage must be completed, tested, and verified before proceeding to the next.

---

#### Stage 1: Fatigue Curve

**Goal:** Implement the output curve that models paddler fatigue over time.

**Functions:**
- `output_curve(start_output, peak_time, plateau_duration, decay_rate)` → returns function τ → output
- `compute_stint_outputs(stint_min, curve_fn, max_consecutive)` → returns vector of averaged outputs

**Tests:**
- Curve returns correct values at τ = 0, 12, 22, 32, 62
- Stint averages computed correctly (compare to numerical integration)

**Deliverables:**
- Can call `output_curve()` and get multipliers
- Can see averaged output for consecutive stints 1, 2, 3, ...
- Plot of the fatigue curve

**Verification:** Present curve values and stint averages for review.

---

#### Stage 2: Race Time Calculation

**Goal:** Given a schedule (paddler assignments), compute estimated race time.

**Functions:**
- `compute_race_time(schedule, params, stint_outputs, seat_weights)` → race time in minutes

**Inputs:**
- `schedule`: matrix or data frame of paddler assignments (who's in which seat each stint)
- `stint_outputs`: output multiplier for each paddler in each stint (based on consecutive stints)
- `params`: includes distance_km, speed_kmh, stint_min, switch_time_min

**Formula:**
```
nominal_paddle_time = distance_km / speed_kmh × 60
avg_output = mean of weighted crew outputs across all stints
race_time = nominal_paddle_time / avg_output + n_switches × switch_time_min
```

**Interpretation:**
- `nominal_paddle_time` = time to finish at output 1.0 (no fatigue)
- Lower avg_output = slower average speed = proportionally longer race
- Switching adds fixed overhead

**Tests:**
- All paddlers at output 1.0 → race_time = nominal_paddle_time + switching
- avg_output = 0.8 → paddle time is 25% longer
- Hand-calculated example with known schedule

**Deliverables:**
- Can compute race time for any hypothetical schedule
- Understand the tradeoff: fatigue vs switching overhead
- Compare schedules for same stint duration

**Verification:** Present race time calculation for sample schedules.

---

#### Stage 3: Parameters and Paddler Roster

**Goal:** Clean interface for setting up optimization inputs.

**Functions:**
- `rotation_params(distance_km, speed_kmh, stint_min, switch_time_min, ...)` → parameter list
- `validate_paddlers(paddlers)` → validated paddler data frame

**Computed values:**
- `n_stints = ceiling((distance_km / speed_kmh * 60) / stint_min)`
- `n_switches = n_stints - 1`
- Pre-computed stint outputs for this stint_min

**Tests:**
- Parameter validation (positive values, etc.)
- Paddler roster has required columns (id, name, role)
- Role counts: enough pacers for seats 1-2, steerers for seat 6

**Deliverables:**
- `rotation_params()` returns complete parameter list
- Clear error messages for invalid inputs

**Verification:** Present parameter list for a sample race.

---

#### Stage 4: MIP Model (Single Stint Duration)

**Goal:** Build and solve the optimization model for fixed stint duration.

**Functions:**
- `create_rotation_model(paddlers, params)` → ompr model object
- `optimize_rotation(paddlers, params)` → solved model + extracted schedule

**Model components:**
- Decision variables: X[p,s,t], R[p,t], S[p,t], Y[p,t,k]
- Constraints: seat assignment, role, consecutive tracking, linearization
- Objective: minimize estimated race time

**Tests:**
- All constraints satisfied (seats filled, roles respected, one place per paddler)
- S[p,t] resets on rest, increments when paddling
- Small example (3 stints) verified by hand

**Deliverables:**
- MIP solves successfully
- Returns optimal schedule with race time

**Verification:** Present model summary and solution for small example.

---

#### Stage 5: Output Formatting

**Goal:** Convert solution into user-friendly formats.

**Functions:**
- `extract_schedule(solution, paddlers, params)` → list of formatted outputs

**Outputs:**
- `schedule_wide`: stint × seats table with paddler names
- `schedule_long`: tidy format with stint, seat, paddler, output
- `transitions`: crew switches and seat switches between stints
- `paddler_summary`: per-paddler statistics

**Tests:**
- Output dimensions correct
- All paddlers appear correct number of times
- Transitions computed correctly from schedule

**Deliverables:**
- Clean, readable schedule output
- Summary statistics

**Verification:** Present formatted outputs for review.

---

#### Stage 6: Visualization

**Goal:** Visual outputs for schedule and analysis.

**Functions:**
- `plot_schedule(result)` → heatmap of paddler × stint assignments
- `plot_output_curve(params)` → fatigue curve visualization
- `plot_stint_output(result)` → bar chart of crew output per stint

**Tests:**
- Plots render without error
- Correct data shown

**Deliverables:**
- Publication-ready visualizations

**Verification:** Present plots for review.

---

#### Stage 7: Meta-Optimization (Stint Duration)

**Goal:** Find optimal stint duration via grid search.

**Functions:**
- `optimize_stint_duration(paddlers, params, stint_range)` → best stint_min + all results
- `plot_stint_comparison(results)` → race time vs stint duration

**Algorithm:**
- For each stint_min in range, solve MIP
- Compare race times, return best

**Tests:**
- All stint durations solve successfully
- Best stint_min has lowest race time
- Results table complete

**Deliverables:**
- Comparison table across stint durations
- Optimal stint_min identified
- Visualization of tradeoff

**Verification:** Present comparison table and recommendation.

---

#### Stage 8: Integration and Documentation

**Goal:** Full workflow tested end-to-end.

**Tests:**
- Complete workflow: params → optimize → format → plot
- Compare to naive round-robin baseline
- Performance on realistic race (8 stints, 9 paddlers)

**Deliverables:**
- Working optimization module
- Example usage in documentation
- All tests passing

**Verification:** Final review of complete system.

### Model Size Estimate

For 8 stints, 9 paddlers, 6 seats, max_consecutive = 6:

| Variable | Count | Notes |
|----------|-------|-------|
| X[p,s,t] | ≤ 9 × 6 × 8 = 432 | Only for eligible (p,s) pairs |
| R[p,t] | 9 × 8 = 72 | |
| S[p,t] | 9 × 8 = 72 | |
| Y[p,t,k] | 9 × 8 × 6 = 432 | |
| Q[p,t,k] | 9 × 8 × 6 = 432 | Continuous (linearization) |
| **Total** | ~1,440 variables | |

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
| Eligibility from roles | Default eligibility matrix matches role definitions |
| Custom eligibility | Solver respects arbitrary eligibility matrix |
| Eligibility validation | Invalid eligibility (empty seats/paddlers) raises error |
| Input validation | Missing columns, wrong counts, invalid params raise errors |

---

## Future Enhancements (Out of Scope for Now)

- Partial recovery based on rest duration
- Individual paddler strength/endurance parameters
- Weather/conditions affecting fatigue rates
- Mid-race strategy adjustments
- Multi-objective optimization (time vs. fairness)
- Stochastic optimization for uncertain conditions
