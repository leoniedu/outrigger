test_that("run_scenarios returns outrigger_results class", {
  scenarios <- data.frame(
    scenario_name = c("Test1", "Test2"),
    stint_min = c(50, 40),
    speed_kmh = c(10, 11)
  )

  result <- run_scenarios(
    scenarios = scenarios,
    distance_km = 60,
    switch_time_min = 1.5
  )

  expect_s3_class(result, "outrigger_results")
  expect_s3_class(result, "tbl_df")
})

test_that("run_scenarios returns correct number of rows", {
  scenarios <- data.frame(
    scenario_name = c("Test1", "Test2", "Test3"),
    stint_min = c(50, 40, 30),
    speed_kmh = c(10, 11, 12)
  )

  result <- run_scenarios(
    scenarios = scenarios,
    distance_km = 60,
    switch_time_min = 1.5
  )

  expect_equal(nrow(result), 3)
})

test_that("run_scenarios preserves scenario names", {
  scenarios <- data.frame(
    scenario_name = c("Alpha", "Beta", "Gamma"),
    stint_min = c(50, 40, 30),
    speed_kmh = c(10, 10, 10)
  )

  result <- run_scenarios(
    scenarios = scenarios,
    distance_km = 60,
    switch_time_min = 1.5
  )

  expect_equal(result$scenario, c("Alpha", "Beta", "Gamma"))
})

test_that("run_scenarios passes crew parameters correctly", {
  scenarios <- data.frame(
    scenario_name = "Test",
    stint_min = 50,
    speed_kmh = 10
  )

  result <- run_scenarios(
    scenarios = scenarios,
    distance_km = 60,
    switch_time_min = 1.5,
    n_pacers = 4,
    n_steerers = 3,
    n_regular = 5
  )

  # With 4 pacers covering 2 seats for 8 stints = 16 slots
  # ceiling(16/4) = 4 stints per pacer
  expect_equal(result$pacer_stints, 4)
})

test_that("run_scenarios validates required columns", {
  # Missing scenario_name
  scenarios <- data.frame(
    stint_min = c(50, 40),
    speed_kmh = c(10, 11)
  )

  expect_error(
    run_scenarios(scenarios = scenarios, distance_km = 60, switch_time_min = 1.5),
    "Missing required columns"
  )
})

test_that("run_scenarios validates non-empty scenarios", {
  scenarios <- data.frame(
    scenario_name = character(),
    stint_min = numeric(),
    speed_kmh = numeric()
  )

  expect_error(
    run_scenarios(scenarios = scenarios, distance_km = 60, switch_time_min = 1.5),
    "`scenarios` must have at least one row"
  )
})

test_that("run_scenarios handles tibble input", {
  scenarios <- tibble::tibble(
    scenario_name = c("Test1", "Test2"),
    stint_min = c(50, 40),
    speed_kmh = c(10, 11)
  )

  result <- run_scenarios(
    scenarios = scenarios,
    distance_km = 60,
    switch_time_min = 1.5
  )

  expect_equal(nrow(result), 2)
})

test_that("run_scenarios ignores extra columns in scenarios", {
  scenarios <- data.frame(
    scenario_name = "Test",
    stint_min = 50,
    speed_kmh = 10,
    extra_column = "ignored"
  )

  expect_no_error(
    run_scenarios(scenarios = scenarios, distance_km = 60, switch_time_min = 1.5)
  )
})

test_that("default_scenarios returns expected structure", {
  scenarios <- default_scenarios()

  expect_s3_class(scenarios, "tbl_df")
  expect_true("scenario_name" %in% names(scenarios))
  expect_true("stint_min" %in% names(scenarios))
  expect_true("speed_kmh" %in% names(scenarios))
})

test_that("default_scenarios returns 6 scenarios", {
  scenarios <- default_scenarios()

  expect_equal(nrow(scenarios), 6)
})

test_that("default_scenarios can be used with run_scenarios", {
  scenarios <- default_scenarios()

  result <- run_scenarios(
    scenarios = scenarios,
    distance_km = 60,
    switch_time_min = 1.5
  )

  expect_equal(nrow(result), 6)
  expect_s3_class(result, "outrigger_results")
})

test_that("run_scenarios with default_scenarios produces all feasible results", {
  scenarios <- default_scenarios()

  result <- run_scenarios(
    scenarios = scenarios,
    distance_km = 60,
    switch_time_min = 1.5
  )

  expect_true(all(result$feasible))
})
