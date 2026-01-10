# Quick Start Guide

## 1. Install R and Required Packages

```r
# Install tidyverse if you haven't already
install.packages("tidyverse")
```

## 2. Run the Default Analysis

```bash
Rscript outrigger_rotation_analyzer.R
```

This will analyze 6 pre-configured scenarios for a 60km race.

## 3. Customize for Your Race

### Option A: Edit the main script directly

Open `outrigger_rotation_analyzer.R` and modify the CONFIGURATION section:

```r
DISTANCE <- 60          # Your race distance
SWITCH_TIME <- 1.5      # Your crew's average switch time
N_PACERS <- 3           # How many can pace
N_STEERERS <- 2         # How many can steer
N_REGULAR <- 4          # Regular paddlers
```

### Option B: Use the configuration template

1. Copy `config_example.R` to `config.R`
2. Edit `config.R` with your parameters
3. Modify the main script to source your config

## 4. Interpret Results

### Look for:

**Best Balance Scenario**
- Lowest workload variance
- Everyone does similar number of stints

**Most Rest Scenario**
- Maximizes recovery between stints
- Good for older/less fit crews
- But may have more switching overhead

**Least Switching Scenario**
- Minimizes time lost to crew changes
- But longer stints = more fatigue

### Red Flags:

⚠️ High workload variance (>1.0) = unbalanced rotation
⚠️ Switching % >5% = too many switches
⚠️ Stint length >70 min = too tiring for most crews
⚠️ Rest time <60 min between stints = insufficient recovery

## 5. Next Steps

1. **Test your chosen strategy** in training
2. **Practice switches** to reduce switch_time
3. **Adjust** based on actual crew performance
4. **Re-run analysis** with updated parameters

## Example Workflow

```r
# 1. Install packages
install.packages("tidyverse")

# 2. Run default analysis
Rscript outrigger_rotation_analyzer.R

# 3. Review output
# Check outrigger_scenarios.csv for detailed results

# 4. Adjust parameters based on results
# Edit DISTANCE, SWITCH_TIME, etc. in the script

# 5. Re-run with new parameters
Rscript outrigger_rotation_analyzer.R

# 6. Compare results and choose optimal strategy
```

## Common Customizations

### Add a new scenario

```r
scenarios <- tribble(
  ~scenario_name,           ~stint_min,  ~speed_kmh,
  "Conservative (60min)",    60,          10,
  "Moderate (50min)",        50,          10.5,
  "My Custom Strategy",      45,          10.8,  # <-- Add here
)
```

### Change crew composition

```r
# Example: Smaller crew (7 people instead of 9)
N_PACERS <- 2
N_STEERERS <- 2
N_REGULAR <- 3
```

### Different race distance

```r
# Example: 40km marathon
DISTANCE <- 40
```

## Troubleshooting

**Error: "could not find function"**
→ Install tidyverse: `install.packages("tidyverse")`

**Warning: "Infeasible scenario"**
→ Not enough paddlers for the rotation
→ Reduce stint length or adjust crew composition

**Plots not showing**
→ Make sure you're in RStudio or have X11 forwarding
→ Or save plots to file instead

## Support

- Check the main [README.md](README.md) for detailed documentation
- File issues on GitHub
- Share your rotation strategies with the community!
