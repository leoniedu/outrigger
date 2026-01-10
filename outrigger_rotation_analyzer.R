#!/usr/bin/env Rscript
# Outrigger Canoe Race Rotation Analyzer
# Analyzes crew rotation strategies for long-distance outrigger races
# considering role constraints (pacers, steerers, regular paddlers)

library(tidyverse)

# FUNCTION TO ANALYZE ONE SCENARIO ----
analyze_scenario <- function(distance_km, speed_kmh, stint_min, switch_time_min,
                            n_pacers = 3, n_steerers = 2, n_regular = 4,
                            steerers_can_paddle_5 = TRUE,
                            scenario_name = NULL) {
  
  # Basic race metrics
  race_duration_hours <- distance_km / speed_kmh
  race_duration_min <- race_duration_hours * 60
  n_stints <- ceiling(race_duration_min / stint_min)
  n_switches <- n_stints - 1
  total_crew <- n_pacers + n_steerers + n_regular
  
  # Slot requirements
  slots_seats_1_2 <- n_stints * 2  # Must be pacers
  slots_seat_6 <- n_stints * 1     # Must be steerers
  slots_seats_3_5 <- n_stints * 3  # Flexible
  total_slots <- n_stints * 6
  
  # Workload calculations
  # Pacers must cover seats 1-2 first
  min_pacer_stints <- ceiling(slots_seats_1_2 / n_pacers)
  
  # Steerers must cover seat 6
  min_steerer_stints <- ceiling(slots_seat_6 / n_steerers)
  
  # Ideal stints per person (if perfectly balanced)
  ideal_stints_per_person <- n_stints * 6 / total_crew
  
  # Regular paddlers
  # Total capacity for seats 3-5
  capacity_3_5 <- (n_pacers * n_stints - slots_seats_1_2) + # Pacers when not in 1-2
                  (n_regular * n_stints) + # All regular paddler time
                  (if(steerers_can_paddle_5) n_steerers * n_stints - slots_seat_6 else 0) # Steerers in seat 5
  
  feasible <- capacity_3_5 >= slots_seats_3_5
  
  # If feasible, calculate how regular paddlers share the load
  if(feasible) {
    # After pacers and steerers take their minimums, how many seat 3-5 slots remain?
    remaining_3_5_slots <- slots_seats_3_5
    
    # Distribute remaining work
    # This is simplified - assumes even distribution where possible
    regular_stints <- round(ideal_stints_per_person)
  } else {
    regular_stints <- NA
  }
  
  # Time analysis
  actual_paddling_time <- n_stints * stint_min
  switching_time <- n_switches * switch_time_min
  total_time <- actual_paddling_time + switching_time
  switching_pct <- 100 * switching_time / total_time
  
  # Rest time calculations (minutes)
  rest_time_pacer <- (n_stints - min_pacer_stints) * stint_min
  rest_time_steerer <- (n_stints - min_steerer_stints) * stint_min
  rest_time_regular <- (n_stints - regular_stints) * stint_min
  
  # Return summary
  tibble(
    scenario = scenario_name %||% paste0(stint_min, "min stints"),
    distance_km = distance_km,
    speed_kmh = speed_kmh,
    stint_length_min = stint_min,
    switch_time_min = switch_time_min,
    
    race_duration_hours = round(race_duration_hours, 1),
    n_stints = n_stints,
    n_switches = n_switches,
    
    # Workload per role
    pacer_stints = min_pacer_stints,
    pacer_rest_stints = n_stints - min_pacer_stints,
    pacer_rest_min = rest_time_pacer,
    
    steerer_stints = min_steerer_stints,
    steerer_rest_stints = n_stints - min_steerer_stints,
    steerer_rest_min = rest_time_steerer,
    
    regular_stints = regular_stints,
    regular_rest_stints = n_stints - regular_stints,
    regular_rest_min = rest_time_regular,
    
    # Time breakdown
    paddling_time_min = actual_paddling_time,
    switching_time_min = switching_time,
    total_time_min = total_time,
    switching_pct = round(switching_pct, 1),
    
    # Feasibility
    feasible = feasible,
    
    # Balance metric (lower is more balanced)
    workload_variance = var(c(min_pacer_stints, min_steerer_stints, regular_stints), na.rm = TRUE)
  )
}

# DEFINE SCENARIOS ----
scenarios <- tribble(
  ~scenario_name,           ~stint_min,  ~speed_kmh,
  "Conservative (60min)",    60,          10,
  "Moderate (50min)",        50,          10.5,
  "Active (40min)",          40,          11,
  "Aggressive (30min)",      30,          11.5,
  "Fast pace (50min)",       50,          11.5,
  "Slow pace (50min)",       50,          10
)

# CONFIGURATION ----
# Edit these parameters for your race
DISTANCE <- 60          # Race distance in km
SWITCH_TIME <- 1.5      # Time per crew switch in minutes
N_PACERS <- 3           # Number of paddlers who can pace (seats 1-2)
N_STEERERS <- 2         # Number of paddlers who can steer (seat 6)
N_REGULAR <- 4          # Number of regular paddlers (seats 3-5 only)

# RUN ALL SCENARIOS ----
results <- scenarios %>%
  pmap_dfr(function(scenario_name, stint_min, speed_kmh) {
    analyze_scenario(
      distance_km = DISTANCE,
      speed_kmh = speed_kmh,
      stint_min = stint_min,
      switch_time_min = SWITCH_TIME,
      n_pacers = N_PACERS,
      n_steerers = N_STEERERS,
      n_regular = N_REGULAR,
      scenario_name = scenario_name
    )
  })

# DISPLAY RESULTS ----
cat("\n╔══════════════════════════════════════════════════════════╗\n")
cat("║  OUTRIGGER RACE ROTATION - SCENARIO COMPARISON          ║\n")
cat("╚══════════════════════════════════════════════════════════╝\n\n")

cat("RACE SETUP:\n")
cat(sprintf("  Distance: %d km\n", DISTANCE))
cat(sprintf("  Crew: %d pacers + %d steerers + %d regular = %d total\n", 
            N_PACERS, N_STEERERS, N_REGULAR, N_PACERS + N_STEERERS + N_REGULAR))
cat(sprintf("  Switch time: %.1f minutes\n\n", SWITCH_TIME))

# Summary table
cat("SCENARIO OVERVIEW:\n")
results %>%
  select(scenario, race_duration_hours, speed_kmh, stint_length_min, 
         n_stints, n_switches, switching_time_min, switching_pct) %>%
  print(n = Inf)

cat("\n\nWORKLOAD BY ROLE:\n")
results %>%
  select(scenario, 
         pacer_stints, pacer_rest_min,
         steerer_stints, steerer_rest_min,
         regular_stints, regular_rest_min) %>%
  print(n = Inf)

cat("\n\nFEASIBILITY & BALANCE:\n")
results %>%
  select(scenario, feasible, workload_variance, switching_pct) %>%
  mutate(
    feasible = ifelse(feasible, "✓", "✗"),
    balance_rating = case_when(
      workload_variance < 0.3 ~ "Excellent",
      workload_variance < 0.5 ~ "Good",
      workload_variance < 1.0 ~ "Fair",
      TRUE ~ "Unbalanced"
    )
  ) %>%
  print(n = Inf)

# RECOMMENDATIONS ----
cat("\n\n╔══════════════════════════════════════════════════════════╗\n")
cat("║  RECOMMENDATIONS                                         ║\n")
cat("╚══════════════════════════════════════════════════════════╝\n\n")

best_balance <- results %>%
  filter(feasible) %>%
  arrange(workload_variance) %>%
  slice(1)

best_rest <- results %>%
  filter(feasible) %>%
  arrange(desc(regular_rest_min)) %>%
  slice(1)

lowest_switching <- results %>%
  filter(feasible) %>%
  arrange(switching_pct) %>%
  slice(1)

cat("Best workload balance:\n")
cat(sprintf("  → %s (variance: %.2f)\n", best_balance$scenario, best_balance$workload_variance))
cat(sprintf("     Pacers: %d stints, Steerers: %d stints, Regular: %d stints\n\n",
            best_balance$pacer_stints, best_balance$steerer_stints, best_balance$regular_stints))

cat("Most rest for crew:\n")
cat(sprintf("  → %s (regular paddlers get %.0f min rest)\n", 
            best_rest$scenario, best_rest$regular_rest_min))
cat(sprintf("     But requires %d switches (%.1f%% of race time)\n\n",
            best_rest$n_switches, best_rest$switching_pct))

cat("Least time switching:\n")
cat(sprintf("  → %s (only %.1f%% switching)\n", 
            lowest_switching$scenario, lowest_switching$switching_pct))
cat(sprintf("     But stints are %.0f min (tiring for older crews)\n\n",
            lowest_switching$stint_length_min))

# VISUALIZATION ----
cat("\n")
cat("Generating comparison plots...\n\n")

# Reshape for plotting
plot_data <- results %>%
  select(scenario, pacer_stints, steerer_stints, regular_stints, switching_pct) %>%
  pivot_longer(cols = c(pacer_stints, steerer_stints, regular_stints),
               names_to = "role", 
               values_to = "stints") %>%
  mutate(role = str_remove(role, "_stints"),
         role = str_to_title(role))

p1 <- ggplot(plot_data, aes(x = scenario, y = stints, fill = role)) +
  geom_col(position = "dodge") +
  geom_hline(yintercept = mean(plot_data$stints, na.rm = TRUE), 
             linetype = "dashed", alpha = 0.5) +
  labs(title = "Workload Distribution Across Scenarios",
       subtitle = "Stints per person by role",
       x = NULL, y = "Number of stints",
       fill = "Role") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1),
        legend.position = "bottom")

p2 <- ggplot(results, aes(x = stint_length_min, y = switching_pct, 
                          color = as.factor(speed_kmh), size = n_stints)) +
  geom_point() +
  geom_line(aes(group = speed_kmh)) +
  labs(title = "Switching Overhead vs Stint Length",
       subtitle = "By pace (km/h)",
       x = "Stint length (minutes)", 
       y = "% of race time switching",
       color = "Speed (km/h)",
       size = "# Stints") +
  theme_minimal()

print(p1)
cat("\n")
print(p2)

# SAVE RESULTS ----
output_file <- "outrigger_scenarios.csv"
write_csv(results, output_file)
cat(sprintf("\nResults saved to: %s\n", output_file))
