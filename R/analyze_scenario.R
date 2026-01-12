#' Analyze a crew rotation scenario
#'
#' Calculates workload distribution, feasibility, and timing metrics for a
#' specific crew rotation configuration in an outrigger canoe race.
#'
#' @param distance_km Race distance in kilometers.
#' @param speed_kmh Expected average speed in kilometers per hour.
#' @param stint_min Length of each paddling stint in minutes.
#' @param switch_time_min Time required for each crew switch in minutes.
#' @param n_pacers Number of paddlers who can pace (seats 1-2). Default is 3.
#' @param n_steerers Number of paddlers who can steer (seat 6). Default is 2.
#' @param n_regular Number of regular paddlers (seats 3-5 only). Default is 4.
#' @param steerers_can_paddle_5 Logical. Can steerers also paddle in seat 5?
#'   Default is TRUE.

#' @param scenario_name Optional name for the scenario. If NULL, a name is
#'   generated from the stint length.
#'
#' @return A tibble with one row containing:
#' \describe{
#'   \item{scenario}{Name of the scenario}
#'   \item{distance_km}{Race distance}
#'   \item{speed_kmh}{Average speed}
#'   \item{stint_length_min}{Stint duration}
#'   \item{switch_time_min}{Time per switch}
#'   \item{race_duration_hours}{Total race duration}
#'   \item{n_stints}{Number of stints in the race}
#'   \item{n_switches}{Number of crew switches}
#'   \item{pacer_stints}{Stints per pacer}
#'   \item{pacer_rest_stints}{Rest stints per pacer}
#'   \item{pacer_rest_min}{Rest time per pacer in minutes}
#'   \item{steerer_stints}{Stints per steerer}
#'   \item{steerer_rest_stints}{Rest stints per steerer}
#'   \item{steerer_rest_min}{Rest time per steerer in minutes}
#'   \item{regular_stints}{Stints per regular paddler}
#'   \item{regular_rest_stints}{Rest stints per regular paddler}
#'   \item{regular_rest_min}{Rest time per regular paddler in minutes}
#'   \item{paddling_time_min}{Total paddling time}
#'   \item{switching_time_min}{Total time spent switching}
#'   \item{total_time_min}{Total race time including switches}
#'   \item{switching_pct}{Percentage of race time spent switching}
#'   \item{feasible}{Logical. Is this configuration feasible?}
#'   \item{workload_variance}{Variance in stints across roles (lower = more balanced)}
#' }
#'
#' @export
#'
#' @examples
#' # Analyze a 60km race with 50-minute stints
#' analyze_scenario(
#'   distance_km = 60,
#'   speed_kmh = 10,
#'   stint_min = 50,
#'   switch_time_min = 1.5
#' )
#'
#' # Analyze with a custom scenario name
#' analyze_scenario(
#'   distance_km = 60,
#'   speed_kmh = 11,
#'   stint_min = 40,
#'   switch_time_min = 1.5,
#'   scenario_name = "Fast Active"
#' )
analyze_scenario <- function(distance_km,
                             speed_kmh,
                             stint_min,
                             switch_time_min,
                             n_pacers = 3,
                             n_steerers = 2,
                             n_regular = 4,
                             steerers_can_paddle_5 = TRUE,
                             scenario_name = NULL) {
  # Input validation

  if (!is.numeric(distance_km) || distance_km <= 0) {
    stop("`distance_km` must be a positive number", call. = FALSE)
  }
  if (!is.numeric(speed_kmh) || speed_kmh <= 0) {
    stop("`speed_kmh` must be a positive number", call. = FALSE)
  }
  if (!is.numeric(stint_min) || stint_min <= 0) {
    stop("`stint_min` must be a positive number", call. = FALSE)
  }
  if (!is.numeric(switch_time_min) || switch_time_min < 0) {
    stop("`switch_time_min` must be a non-negative number", call. = FALSE)
  }
  if (!is.numeric(n_pacers) || n_pacers < 1) {
    stop("`n_pacers` must be at least 1", call. = FALSE)
  }
  if (!is.numeric(n_steerers) || n_steerers < 1) {
    stop("`n_steerers` must be at least 1", call. = FALSE)
  }
  if (!is.numeric(n_regular) || n_regular < 0) {
    stop("`n_regular` must be non-negative", call. = FALSE)
  }

  # Basic race metrics
  race_duration_hours <- distance_km / speed_kmh
  race_duration_min <- race_duration_hours * 60
  n_stints <- ceiling(race_duration_min / stint_min)
  n_switches <- n_stints - 1

  total_crew <- n_pacers + n_steerers + n_regular

  # Slot requirements

  slots_seats_1_2 <- n_stints * 2 # Must be pacers
  slots_seat_6 <- n_stints * 1 # Must be steerers
  slots_seats_3_5 <- n_stints * 3 # Flexible

  # Workload calculations
  # Pacers must cover seats 1-2 first
  min_pacer_stints <- ceiling(slots_seats_1_2 / n_pacers)

  # Steerers must cover seat 6
  min_steerer_stints <- ceiling(slots_seat_6 / n_steerers)

  # Ideal stints per person (if perfectly balanced)
  ideal_stints_per_person <- n_stints * 6 / total_crew

  # Regular paddlers

  # Total capacity for seats 3-5
  steerer_capacity <- if (steerers_can_paddle_5) {
    n_steerers * n_stints - slots_seat_6
  } else {
    0
  }
  capacity_3_5 <- (n_pacers * n_stints - slots_seats_1_2) +
    (n_regular * n_stints) +
    steerer_capacity

  feasible <- capacity_3_5 >= slots_seats_3_5

  # If feasible, calculate how regular paddlers share the load
  if (feasible) {
    regular_stints <- round(ideal_stints_per_person)
  } else {
    regular_stints <- NA_real_
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
    workload_variance = var(
      c(min_pacer_stints, min_steerer_stints, regular_stints),
      na.rm = TRUE
    )
  )
}
