# Outrigger Canoe Race Rotation Analyzer

An R-based tool for optimizing crew rotation strategies in long-distance outrigger canoe (OC6) races. This analyzer helps race coordinators balance workload across crew members while accounting for role-specific constraints (pacers, steerers, regular paddlers).

## Problem Statement

In a 60km outrigger race with:
- **9 paddlers** competing for **6 seats** in the canoe
- **Role constraints**: Only 3 can pace (seats 1-2), only 2 can steer (seat 6)
- **Crew considerations**: Average age ~53, some slightly overweight
- **Trade-offs**: Longer stints = more fatigue but less switching overhead

**Key Question**: How many rotation stints optimize the balance between crew fatigue and switching efficiency?

## Features

- **Scenario comparison**: Analyze multiple stint length and pace combinations
- **Role-based workload tracking**: Separate analysis for pacers, steerers, and regular paddlers
- **Constraint validation**: Ensures feasible rotations given crew composition
- **Balance metrics**: Identifies most equitable workload distribution
- **Time analysis**: Calculates switching overhead and total race duration
- **Visualizations**: Comparative plots of workload and switching costs

## Installation

### Requirements
- R (≥ 4.0.0)
- tidyverse package

```r
install.packages("tidyverse")
```

## Usage

### Quick Start

```bash
Rscript outrigger_rotation_analyzer.R
```

### Customizing Parameters

Edit the configuration section in `outrigger_rotation_analyzer.R`:

```r
# CONFIGURATION ----
DISTANCE <- 60          # Race distance in km
SWITCH_TIME <- 1.5      # Time per crew switch in minutes
N_PACERS <- 3           # Number of paddlers who can pace (seats 1-2)
N_STEERERS <- 2         # Number of paddlers who can steer (seat 6)
N_REGULAR <- 4          # Number of regular paddlers (seats 3-5 only)
```

### Defining Scenarios

Modify the scenarios table to test different strategies:

```r
scenarios <- tribble(
  ~scenario_name,           ~stint_min,  ~speed_kmh,
  "Conservative (60min)",    60,          10,
  "Moderate (50min)",        50,          10.5,
  "Active (40min)",          40,          11,
  # Add your own scenarios here
)
```

## Output

The script generates:

1. **Console output**: 
   - Scenario overview table
   - Workload breakdown by role
   - Feasibility and balance metrics
   - Recommendations for optimal strategy

2. **CSV file**: `outrigger_scenarios.csv` with detailed results

3. **Visualizations**:
   - Workload distribution across scenarios
   - Switching overhead vs stint length

### Example Output

```
SCENARIO OVERVIEW:
# A tibble: 6 × 8
  scenario          race_duration_hours speed_kmh stint_length_min n_stints n_switches
  <chr>                           <dbl>     <dbl>            <dbl>    <dbl>      <dbl>
1 Conservative (6…                  6        10                  60        6          5
2 Moderate (50min)                  5.7      10.5                50        7          6
3 Active (40min)                    5.5      11                  40        9          8

WORKLOAD BY ROLE:
# A tibble: 6 × 7
  scenario          pacer_stints pacer_rest_min steerer_stints steerer_rest_min
  <chr>                    <dbl>          <dbl>          <dbl>            <dbl>
1 Conservative (6…             4            120              3              180
2 Moderate (50min)             5            100              4              150
3 Active (40min)               6             80              5              120
```

## Understanding the Results

### Key Metrics

- **Stints**: Number of rotation periods during the race
- **Workload variance**: Lower values indicate more balanced distribution (< 0.5 is good)
- **Switching %**: Percentage of total race time spent switching crews
- **Rest time**: Minutes each role spends resting between stints

### Recommendations

The script identifies:
- **Best balance**: Scenario with most equitable workload distribution
- **Most rest**: Scenario maximizing recovery time (may have more switches)
- **Least switching**: Scenario minimizing time overhead (may have longer, tiring stints)

## Methodology

### Seat Constraints

- **Seats 1-2**: Require pacer skills (voga)
- **Seat 6**: Requires steering skills
- **Seats 3-5**: Can be filled by any paddler
- **Seat 5**: Can be filled by steerers as an alternative to steering

### Workload Calculation

1. **Pacers**: Must cover minimum required stints for seats 1-2
2. **Steerers**: Must cover minimum required stints for seat 6
3. **Regular**: Fill remaining slots in seats 3-5
4. **Balance**: Algorithm attempts to equalize total stints across all crew

### Feasibility Check

A scenario is feasible if:
```
Available capacity (seats 3-5) ≥ Required slots (seats 3-5)
```

Where capacity includes:
- Pacer time when not in seats 1-2
- Regular paddler time
- Steerer time when paddling seat 5 (if allowed)

## Contributing

Contributions welcome! Areas for enhancement:

- [ ] Add Shiny web interface for interactive scenario building
- [ ] Implement rotation pattern optimization algorithm
- [ ] Add export to printable rotation schedule
- [ ] Include fatigue modeling (e.g., exponential decline in performance)
- [ ] Support for irregular stint lengths
- [ ] Weather/conditions impact on switching time

## License

MIT License - feel free to use and modify for your crew's needs.

## Author

Created for a 60km outrigger race with a masters crew (average age 53).

## Acknowledgments

Thanks to the outrigger racing community for insights on crew rotation strategies.

---

**Note**: This tool provides analytical guidance but cannot replace experienced race strategy and crew coordination. Always practice your switching procedures before race day!
