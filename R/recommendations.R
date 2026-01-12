#' Get scenario recommendations
#'
#' Analyzes scenario results and returns the best scenarios for different
#' optimization criteria.
#'
#' @param results A tibble of scenario results from [run_scenarios()] or
#'   [analyze_scenario()].
#'
#' @return A list with class `outrigger_recommendations` containing:
#' \describe{
#'   \item{best_balance}{Scenario with the most balanced workload distribution}
#'   \item{best_rest}{Scenario providing the most rest time for regular paddlers}
#'   \item{lowest_switching}{Scenario with the least time spent switching crews}
#' }
#'
#' @export
#'
#' @examples
#' scenarios <- default_scenarios()
#' results <- run_scenarios(
#'   scenarios = scenarios,
#'   distance_km = 60,
#'   switch_time_min = 1.5
#' )
#' recommendations <- get_recommendations(results)
#' recommendations
get_recommendations <- function(results) {
  if (!is.data.frame(results)) {
    stop("`results` must be a data frame", call. = FALSE)
  }

  required_cols <- c(
    "scenario", "feasible", "workload_variance",
    "regular_rest_min", "switching_pct"
  )
  missing_cols <- setdiff(required_cols, names(results))
  if (length(missing_cols) > 0) {
    stop(
      "Missing required columns: ",
      paste(missing_cols, collapse = ", "),
      call. = FALSE
    )
  }

  feasible_results <- filter(results, .data$feasible)

  if (nrow(feasible_results) == 0) {
    warning("No feasible scenarios found", call. = FALSE)
    recommendations <- list(
      best_balance = NULL,
      best_rest = NULL,
      lowest_switching = NULL
    )
    class(recommendations) <- "outrigger_recommendations"
    return(recommendations)
  }

  best_balance <- feasible_results |>
    arrange(.data$workload_variance) |>
    slice(1)

  best_rest <- feasible_results |>
    arrange(dplyr::desc(.data$regular_rest_min)) |>
    slice(1)

  lowest_switching <- feasible_results |>
    arrange(.data$switching_pct) |>
    slice(1)

  recommendations <- list(
    best_balance = best_balance,
    best_rest = best_rest,
    lowest_switching = lowest_switching
  )
  class(recommendations) <- "outrigger_recommendations"
  recommendations
}


#' @export
print.outrigger_recommendations <- function(x, ...) {
  cat("\nOUTRIGGER RACE RECOMMENDATIONS\n")
  cat(strrep("=", 50), "\n\n")

  if (is.null(x$best_balance)) {
    cat("No feasible scenarios found.\n")
    return(invisible(x))
  }

  cat("Best workload balance:\n")
  cat(sprintf(
    "  -> %s (variance: %.2f)\n",
    x$best_balance$scenario,
    x$best_balance$workload_variance
  ))
  cat(sprintf(
    "     Pacers: %d stints, Steerers: %d stints, Regular: %d stints\n\n",
    x$best_balance$pacer_stints,
    x$best_balance$steerer_stints,
    x$best_balance$regular_stints
  ))

  cat("Most rest for crew:\n")
  cat(sprintf(
    "  -> %s (regular paddlers get %.0f min rest)\n",
    x$best_rest$scenario,
    x$best_rest$regular_rest_min
  ))
  cat(sprintf(
    "     But requires %d switches (%.1f%% of race time)\n\n",
    x$best_rest$n_switches,
    x$best_rest$switching_pct
  ))

  cat("Least time switching:\n")
  cat(sprintf(
    "  -> %s (only %.1f%% switching)\n",
    x$lowest_switching$scenario,
    x$lowest_switching$switching_pct
  ))
  cat(sprintf(
    "     But stints are %.0f min (may be tiring for older crews)\n\n",
    x$lowest_switching$stint_length_min
  ))

  invisible(x)
}
