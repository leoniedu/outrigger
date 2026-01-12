test_that("print.outrigger_results produces output", {
  scenarios <- default_scenarios()
  results <- run_scenarios(
    scenarios = scenarios,
    distance_km = 60,
    switch_time_min = 1.5
  )

  expect_output(print(results), "OUTRIGGER RACE ROTATION")
  expect_output(print(results), "RACE SETUP")
  expect_output(print(results), "SCENARIO OVERVIEW")
  expect_output(print(results), "WORKLOAD BY ROLE")
  expect_output(print(results), "FEASIBILITY & BALANCE")
})

test_that("print.outrigger_results handles empty results", {
  empty_results <- tibble::tibble(
    scenario = character(),
    distance_km = numeric(),
    speed_kmh = numeric(),
    stint_length_min = numeric(),
    switch_time_min = numeric(),
    race_duration_hours = numeric(),
    n_stints = numeric(),
    n_switches = numeric(),
    pacer_stints = numeric(),
    pacer_rest_stints = numeric(),
    pacer_rest_min = numeric(),
    steerer_stints = numeric(),
    steerer_rest_stints = numeric(),
    steerer_rest_min = numeric(),
    regular_stints = numeric(),
    regular_rest_stints = numeric(),
    regular_rest_min = numeric(),
    paddling_time_min = numeric(),
    switching_time_min = numeric(),
    total_time_min = numeric(),
    switching_pct = numeric(),
    feasible = logical(),
    workload_variance = numeric()
  )
  class(empty_results) <- c("outrigger_results", class(empty_results))

  expect_output(print(empty_results), "No scenarios to display")
})

test_that("print.outrigger_results returns invisible", {
  scenarios <- default_scenarios()
  results <- run_scenarios(
    scenarios = scenarios,
    distance_km = 60,
    switch_time_min = 1.5
  )

  result <- capture.output(invisible_return <- print(results))

  expect_identical(invisible_return, results)
})

test_that("summary.outrigger_results returns expected statistics", {
  scenarios <- default_scenarios()
  results <- run_scenarios(
    scenarios = scenarios,
    distance_km = 60,
    switch_time_min = 1.5
  )

  s <- summary(results)

  expect_type(s, "list")
  expect_equal(s$n_scenarios, 6)
  expect_equal(s$n_feasible, 6)
  expect_equal(s$n_infeasible, 0)
  expect_true(is.numeric(s$avg_switching_pct))
  expect_true(is.numeric(s$min_workload_variance))
  expect_true(is.numeric(s$max_workload_variance))
})

test_that("summary.outrigger_results handles mixed feasibility", {
  scenarios <- default_scenarios()
  results <- run_scenarios(
    scenarios = scenarios,
    distance_km = 60,
    switch_time_min = 1.5
  )

  # Make some scenarios infeasible
  results$feasible[1:2] <- FALSE

  s <- summary(results)

  expect_equal(s$n_feasible, 4)
  expect_equal(s$n_infeasible, 2)
})
