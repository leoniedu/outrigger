#' Run multiple rotation scenarios
#'
#' Analyzes multiple crew rotation scenarios and returns combined results.
#' This is useful for comparing different stint lengths and speeds.
#'
#' @param scenarios A data frame with columns:
#'   \describe{
#'     \item{scenario_name}{Character. Name for the scenario.}
#'     \item{stint_min}{Numeric. Stint length in minutes.}
#'     \item{speed_kmh}{Numeric. Expected speed in km/h.}
#'   }
#' @param distance_km Race distance in kilometers.
#' @param switch_time_min Time required for each crew switch in minutes.
#' @param n_pacers Number of paddlers who can pace (seats 1-2). Default is 3.
#' @param n_steerers Number of paddlers who can steer (seat 6). Default is 2.
#' @param n_regular Number of regular paddlers (seats 3-5 only). Default is 4.
#' @param steerers_can_paddle_5 Logical. Can steerers also paddle in seat 5?
#'   Default is TRUE.
#'
#' @return A tibble with one row per scenario, containing all metrics from
#'   [analyze_scenario()]. Results have class `outrigger_results` for
#'   custom printing.
#'
#' @export
#'
#' @examples
#' # Define scenarios
#' scenarios <- data.frame(
#'   scenario_name = c("Conservative", "Moderate", "Aggressive"),
#'   stint_min = c(60, 50, 30),
#'   speed_kmh = c(10, 10.5, 11.5)
#' )
#'
#' # Run all scenarios
#' results <- run_scenarios(
#'   scenarios = scenarios,
#'   distance_km = 60,
#'   switch_time_min = 1.5
#' )
#'
#' results
run_scenarios <- function(scenarios,
                          distance_km,
                          switch_time_min,
                          n_pacers = 3,
                          n_steerers = 2,
                          n_regular = 4,
                          steerers_can_paddle_5 = TRUE) {
  # Input validation
  required_cols <- c("scenario_name", "stint_min", "speed_kmh")
  missing_cols <- setdiff(required_cols, names(scenarios))
  if (length(missing_cols) > 0) {
    stop(
      "Missing required columns in `scenarios`: ",
      paste(missing_cols, collapse = ", "),
      call. = FALSE
    )
  }

  if (nrow(scenarios) == 0) {
    stop("`scenarios` must have at least one row", call. = FALSE)
  }

  results <- pmap_dfr(
    scenarios,
    function(scenario_name, stint_min, speed_kmh, ...) {
      analyze_scenario(
        distance_km = distance_km,
        speed_kmh = speed_kmh,
        stint_min = stint_min,
        switch_time_min = switch_time_min,
        n_pacers = n_pacers,
        n_steerers = n_steerers,
        n_regular = n_regular,
        steerers_can_paddle_5 = steerers_can_paddle_5,
        scenario_name = scenario_name
      )
    }
  )

  class(results) <- c("outrigger_results", class(results))
  results
}


#' Default scenarios for outrigger race analysis
#'
#' Returns a data frame with commonly used scenarios for comparison.
#'
#' @return A tibble with columns `scenario_name`, `stint_min`, and `speed_kmh`.
#'
#' @export
#'
#' @examples
#' default_scenarios()
default_scenarios <- function() {
  tibble(
    scenario_name = c(
      "Conservative (60min)",
      "Moderate (50min)",
      "Active (40min)",
      "Aggressive (30min)",
      "Fast pace (50min)",
      "Slow pace (50min)"
    ),
    stint_min = c(60, 50, 40, 30, 50, 50),
    speed_kmh = c(10, 10.5, 11, 11.5, 11.5, 10)
  )
}
