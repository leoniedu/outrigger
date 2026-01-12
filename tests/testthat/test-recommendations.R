test_that("get_recommendations returns outrigger_recommendations class", {
  scenarios <- default_scenarios()
  results <- run_scenarios(
    scenarios = scenarios,
    distance_km = 60,
    switch_time_min = 1.5
  )

  recommendations <- get_recommendations(results)

  expect_s3_class(recommendations, "outrigger_recommendations")
})

test_that("get_recommendations returns list with expected elements", {
  scenarios <- default_scenarios()
  results <- run_scenarios(
    scenarios = scenarios,
    distance_km = 60,
    switch_time_min = 1.5
  )

  recommendations <- get_recommendations(results)

  expect_true("best_balance" %in% names(recommendations))
  expect_true("best_rest" %in% names(recommendations))
  expect_true("lowest_switching" %in% names(recommendations))
})

test_that("get_recommendations finds best balance correctly", {
  scenarios <- data.frame(
    scenario_name = c("High Variance", "Low Variance"),
    stint_min = c(30, 60),
    speed_kmh = c(11.5, 10)
  )

  results <- run_scenarios(
    scenarios = scenarios,
    distance_km = 60,
    switch_time_min = 1.5
  )

  recommendations <- get_recommendations(results)

  # The scenario with lower workload_variance should be best_balance
  min_variance_scenario <- results$scenario[which.min(results$workload_variance)]
  expect_equal(recommendations$best_balance$scenario, min_variance_scenario)
})

test_that("get_recommendations finds best rest correctly", {
  scenarios <- data.frame(
    scenario_name = c("Short stints", "Long stints"),
    stint_min = c(30, 60),
    speed_kmh = c(10, 10)
  )

  results <- run_scenarios(
    scenarios = scenarios,
    distance_km = 60,
    switch_time_min = 1.5
  )

  recommendations <- get_recommendations(results)

  # The scenario with higher regular_rest_min should be best_rest
  max_rest_scenario <- results$scenario[which.max(results$regular_rest_min)]
  expect_equal(recommendations$best_rest$scenario, max_rest_scenario)
})

test_that("get_recommendations finds lowest switching correctly", {
  scenarios <- data.frame(
    scenario_name = c("Short stints", "Long stints"),
    stint_min = c(30, 60),
    speed_kmh = c(10, 10)
  )

  results <- run_scenarios(
    scenarios = scenarios,
    distance_km = 60,
    switch_time_min = 1.5
  )

  recommendations <- get_recommendations(results)

  # The scenario with lower switching_pct should be lowest_switching
  min_switching_scenario <- results$scenario[which.min(results$switching_pct)]
  expect_equal(recommendations$lowest_switching$scenario, min_switching_scenario)
})

test_that("get_recommendations only considers feasible scenarios", {
  # Create a results data frame with one infeasible scenario
  scenarios <- data.frame(
    scenario_name = c("Feasible", "Infeasible"),
    stint_min = c(50, 50),
    speed_kmh = c(10, 10)
  )

  results <- run_scenarios(
    scenarios = scenarios,
    distance_km = 60,
    switch_time_min = 1.5
  )

  # Manually mark one as infeasible for testing
  results$feasible[2] <- FALSE

  recommendations <- get_recommendations(results)

  expect_equal(recommendations$best_balance$scenario, "Feasible")
})

test_that("get_recommendations handles no feasible scenarios", {
  scenarios <- data.frame(
    scenario_name = "Infeasible",
    stint_min = 50,
    speed_kmh = 10
  )

  results <- run_scenarios(
    scenarios = scenarios,
    distance_km = 60,
    switch_time_min = 1.5
  )

  # Force infeasible
  results$feasible <- FALSE

  expect_warning(
    recommendations <- get_recommendations(results),
    "No feasible scenarios found"
  )

  expect_null(recommendations$best_balance)
  expect_null(recommendations$best_rest)
  expect_null(recommendations$lowest_switching)
})

test_that("get_recommendations validates input is data frame", {
  expect_error(
    get_recommendations("not a data frame"),
    "`results` must be a data frame"
  )
})

test_that("get_recommendations validates required columns", {
  bad_df <- data.frame(x = 1, y = 2)

  expect_error(
    get_recommendations(bad_df),
    "Missing required columns"
  )
})

test_that("print.outrigger_recommendations works", {
  scenarios <- default_scenarios()
  results <- run_scenarios(
    scenarios = scenarios,
    distance_km = 60,
    switch_time_min = 1.5
  )

  recommendations <- get_recommendations(results)

  expect_output(print(recommendations), "OUTRIGGER RACE RECOMMENDATIONS")
  expect_output(print(recommendations), "Best workload balance")
  expect_output(print(recommendations), "Most rest for crew")
  expect_output(print(recommendations), "Least time switching")
})

test_that("print.outrigger_recommendations handles no feasible", {
  scenarios <- data.frame(
    scenario_name = "Infeasible",
    stint_min = 50,
    speed_kmh = 10
  )

  results <- run_scenarios(
    scenarios = scenarios,
    distance_km = 60,
    switch_time_min = 1.5
  )
  results$feasible <- FALSE

  recommendations <- suppressWarnings(get_recommendations(results))

  expect_output(print(recommendations), "No feasible scenarios found")
})

test_that("get_recommendations works with single row results", {
  result <- analyze_scenario(
    distance_km = 60,
    speed_kmh = 10,
    stint_min = 50,
    switch_time_min = 1.5
  )

  recommendations <- get_recommendations(result)

  expect_s3_class(recommendations, "outrigger_recommendations")
  expect_equal(recommendations$best_balance$scenario, result$scenario)
})
