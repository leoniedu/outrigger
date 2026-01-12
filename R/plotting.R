#' Plot workload distribution across scenarios
#'
#' Creates a bar chart showing the number of stints per role across different
#' scenarios.
#'
#' @param results A tibble of scenario results from [run_scenarios()] or
#'   [analyze_scenario()].
#'
#' @return A ggplot2 object.
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
#' plot_workload(results)
plot_workload <- function(results) {
  if (!is.data.frame(results)) {
    stop("`results` must be a data frame", call. = FALSE)
  }

  required_cols <- c("scenario", "pacer_stints", "steerer_stints", "regular_stints")
  missing_cols <- setdiff(required_cols, names(results))
  if (length(missing_cols) > 0) {
    stop(
      "Missing required columns: ",
      paste(missing_cols, collapse = ", "),
      call. = FALSE
    )
  }

  plot_data <- results |>
    select(
      "scenario",
      "pacer_stints",
      "steerer_stints",
      "regular_stints",
      "switching_pct"
    ) |>
    pivot_longer(
      cols = c("pacer_stints", "steerer_stints", "regular_stints"),
      names_to = "role",
      values_to = "stints"
    ) |>
    mutate(
      role = str_remove(.data$role, "_stints"),
      role = str_to_title(.data$role)
    )

  mean_stints <- mean(plot_data$stints, na.rm = TRUE)

  ggplot2::ggplot(plot_data, ggplot2::aes(x = .data$scenario, y = .data$stints, fill = .data$role)) +
    ggplot2::geom_col(position = "dodge") +
    ggplot2::geom_hline(yintercept = mean_stints, linetype = "dashed", alpha = 0.5) +
    ggplot2::labs(
      title = "Workload Distribution Across Scenarios",
      subtitle = "Stints per person by role",
      x = NULL,
      y = "Number of stints",
      fill = "Role"
    ) +
    ggplot2::theme_minimal() +
    ggplot2::theme(
      axis.text.x = ggplot2::element_text(angle = 45, hjust = 1),
      legend.position = "bottom"
    )
}


#' Plot switching overhead vs stint length
#'
#' Creates a scatter plot showing the relationship between stint length and
#' the percentage of race time spent switching crews.
#'
#' @param results A tibble of scenario results from [run_scenarios()] or
#'   [analyze_scenario()].
#'
#' @return A ggplot2 object.
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
#' plot_switching(results)
plot_switching <- function(results) {
  if (!is.data.frame(results)) {
    stop("`results` must be a data frame", call. = FALSE)
  }

  required_cols <- c("stint_length_min", "switching_pct", "speed_kmh", "n_stints")
  missing_cols <- setdiff(required_cols, names(results))
  if (length(missing_cols) > 0) {
    stop(
      "Missing required columns: ",
      paste(missing_cols, collapse = ", "),
      call. = FALSE
    )
  }

  ggplot2::ggplot(
    results,
    ggplot2::aes(
      x = .data$stint_length_min,
      y = .data$switching_pct,
      color = as.factor(.data$speed_kmh),
      size = .data$n_stints
    )
  ) +
    ggplot2::geom_point() +
    ggplot2::geom_line(ggplot2::aes(group = .data$speed_kmh)) +
    ggplot2::labs(
      title = "Switching Overhead vs Stint Length",
      subtitle = "By pace (km/h)",
      x = "Stint length (minutes)",
      y = "% of race time switching",
      color = "Speed (km/h)",
      size = "# Stints"
    ) +
    ggplot2::theme_minimal()
}
