#' Print outrigger results
#'
#' @param x An object of class `outrigger_results`.
#' @param ... Additional arguments passed to print methods.
#'
#' @return Invisibly returns the input object.
#'
#' @export
print.outrigger_results <- function(x, ...) {
  cat("\nOUTRIGGER RACE ROTATION - SCENARIO COMPARISON\n")
  cat(strrep("=", 50), "\n\n")

  if (nrow(x) == 0) {
    cat("No scenarios to display.\n")
    return(invisible(x))
  }

  # Race setup info
  cat("RACE SETUP:\n")
  cat(sprintf("  Distance: %.0f km\n", x$distance_km[1]))
  cat(sprintf("  Switch time: %.1f minutes\n\n", x$switch_time_min[1]))

  # Summary table - strip class to avoid infinite recursion
  cat("SCENARIO OVERVIEW:\n")
  overview <- tibble::as_tibble(x[, c(
    "scenario", "race_duration_hours", "speed_kmh",
    "stint_length_min", "n_stints", "n_switches",
    "switching_time_min", "switching_pct"
  )])
  print(overview, n = Inf)

  cat("\n\nWORKLOAD BY ROLE:\n")
  workload <- tibble::as_tibble(x[, c(
    "scenario",
    "pacer_stints", "pacer_rest_min",
    "steerer_stints", "steerer_rest_min",
    "regular_stints", "regular_rest_min"
  )])
  print(workload, n = Inf)

  cat("\n\nFEASIBILITY & BALANCE:\n")
  feasibility <- tibble::as_tibble(x) |>
    select("scenario", "feasible", "workload_variance", "switching_pct") |>
    mutate(
      feasible = ifelse(.data$feasible, "Yes", "No"),
      balance_rating = dplyr::case_when(
        .data$workload_variance < 0.3 ~ "Excellent",
        .data$workload_variance < 0.5 ~ "Good",
        .data$workload_variance < 1.0 ~ "Fair",
        TRUE ~ "Unbalanced"
      )
    )
  print(feasibility, n = Inf)

  invisible(x)
}


#' Summarize outrigger results
#'
#' @param object An object of class `outrigger_results`.
#' @param ... Additional arguments (unused).
#'
#' @return A list with summary statistics.
#'
#' @export
summary.outrigger_results <- function(object, ...) {
  n_feasible <- sum(object$feasible, na.rm = TRUE)
  n_total <- nrow(object)

  list(
    n_scenarios = n_total,
    n_feasible = n_feasible,
    n_infeasible = n_total - n_feasible,
    avg_switching_pct = mean(object$switching_pct, na.rm = TRUE),
    min_workload_variance = min(object$workload_variance, na.rm = TRUE),
    max_workload_variance = max(object$workload_variance, na.rm = TRUE)
  )
}
