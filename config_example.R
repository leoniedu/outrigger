# Example Configuration
# Copy this to create your own custom scenarios

# Race Parameters
DISTANCE <- 60          # Race distance in km
SWITCH_TIME <- 1.5      # Average time to switch crews (minutes)

# Crew Composition
N_PACERS <- 3           # Paddlers who can pace (seats 1-2, voga)
N_STEERERS <- 2         # Paddlers who can steer (seat 6)
N_REGULAR <- 4          # Regular paddlers (seats 3-5 only)

# Scenarios to Compare
# Each row defines: scenario_name, stint_length (min), average_speed (km/h)

scenarios <- tribble(
  ~scenario_name,           ~stint_min,  ~speed_kmh,
  
  # Conservative approach - longer rest
  "Conservative (60min)",    60,          10,
  
  # Moderate - balanced
  "Moderate (50min)",        50,          10.5,
  
  # Active - more switches
  "Active (40min)",          40,          11,
  
  # Aggressive - very frequent switches
  "Aggressive (30min)",      30,          11.5,
  
  # Speed variations
  "Fast pace (50min)",       50,          11.5,
  "Slow pace (50min)",       50,          10,
  
  # Add your custom scenarios below:
  # "My Custom",            45,          10.8,
)

# Notes on choosing parameters:
# - stint_min: 30-60 typical range for masters crews
# - speed_kmh: 
#     * 9-10: Recreational/training pace
#     * 10-11: Competitive recreational
#     * 11-13: Competitive masters
#     * 13+: Elite
# - switch_time_min: 
#     * 1.0: Well-practiced crew, calm conditions
#     * 1.5-2.0: Average
#     * 2.5+: Rough conditions or inexperienced
