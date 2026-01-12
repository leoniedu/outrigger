test_that("plot_workload returns ggplot object", {
  scenarios <- default_scenarios()
  results <- run_scenarios(
    scenarios = scenarios,
    distance_km = 60,
    switch_time_min = 1.5
  )

  p <- plot_workload(results)

  expect_s3_class(p, "gg")
  expect_s3_class(p, "ggplot")
})

test_that("plot_workload validates input is data frame", {
  expect_error(
    plot_workload("not a data frame"),
    "`results` must be a data frame"
  )
})

test_that("plot_workload validates required columns", {
  bad_df <- data.frame(x = 1, y = 2)

  expect_error(
    plot_workload(bad_df),
    "Missing required columns"
  )
})

test_that("plot_workload has correct labels", {
  scenarios <- default_scenarios()
  results <- run_scenarios(
    scenarios = scenarios,
    distance_km = 60,
    switch_time_min = 1.5
  )

  p <- plot_workload(results)

  expect_equal(p$labels$title, "Workload Distribution Across Scenarios")
  expect_equal(p$labels$y, "Number of stints")
  expect_equal(p$labels$fill, "Role")
})

test_that("plot_workload works with single scenario", {
  result <- analyze_scenario(
    distance_km = 60,
    speed_kmh = 10,
    stint_min = 50,
    switch_time_min = 1.5
  )

  p <- plot_workload(result)

  expect_s3_class(p, "ggplot")
})

test_that("plot_switching returns ggplot object", {
  scenarios <- default_scenarios()
  results <- run_scenarios(
    scenarios = scenarios,
    distance_km = 60,
    switch_time_min = 1.5
  )

  p <- plot_switching(results)

  expect_s3_class(p, "gg")
  expect_s3_class(p, "ggplot")
})

test_that("plot_switching validates input is data frame", {
  expect_error(
    plot_switching("not a data frame"),
    "`results` must be a data frame"
  )
})

test_that("plot_switching validates required columns", {
  bad_df <- data.frame(x = 1, y = 2)

  expect_error(
    plot_switching(bad_df),
    "Missing required columns"
  )
})

test_that("plot_switching has correct labels", {
  scenarios <- default_scenarios()
  results <- run_scenarios(
    scenarios = scenarios,
    distance_km = 60,
    switch_time_min = 1.5
  )

  p <- plot_switching(results)

  expect_equal(p$labels$title, "Switching Overhead vs Stint Length")
  expect_equal(p$labels$x, "Stint length (minutes)")
  expect_equal(p$labels$y, "% of race time switching")
})

test_that("plot_switching works with single scenario", {
  result <- analyze_scenario(
    distance_km = 60,
    speed_kmh = 10,
    stint_min = 50,
    switch_time_min = 1.5
  )

  p <- plot_switching(result)

  expect_s3_class(p, "ggplot")
})

test_that("plot_workload handles NA values in stints", {
  result <- analyze_scenario(
    distance_km = 60,
    speed_kmh = 10,
    stint_min = 50,
    switch_time_min = 1.5,
    n_pacers = 1,
    n_steerers = 1,
    n_regular = 0,
    steerers_can_paddle_5 = FALSE
  )

  # This should create an infeasible scenario with NA regular_stints
  # The plot should still work (with warning about removed NA values)
  expect_s3_class(plot_workload(result), "ggplot")
})
