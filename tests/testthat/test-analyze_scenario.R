test_that("analyze_scenario returns tibble with expected columns", {
  result <- analyze_scenario(
    distance_km = 60,
    speed_kmh = 10,
    stint_min = 50,
    switch_time_min = 1.5
  )

  expect_s3_class(result, "tbl_df")
  expect_equal(nrow(result), 1)

  expected_cols <- c(
    "scenario", "distance_km", "speed_kmh", "stint_length_min",
    "switch_time_min", "race_duration_hours", "n_stints", "n_switches",
    "pacer_stints", "pacer_rest_stints", "pacer_rest_min",
    "steerer_stints", "steerer_rest_stints", "steerer_rest_min",
    "regular_stints", "regular_rest_stints", "regular_rest_min",
    "paddling_time_min", "switching_time_min", "total_time_min",
    "switching_pct", "feasible", "workload_variance"
  )
  expect_true(all(expected_cols %in% names(result)))
})

test_that("analyze_scenario calculates race duration correctly", {
  result <- analyze_scenario(
    distance_km = 60,
    speed_kmh = 10,
    stint_min = 50,
    switch_time_min = 1.5
  )

  expect_equal(result$race_duration_hours, 6.0)
})

test_that("analyze_scenario calculates number of stints correctly", {
  # 60km at 10km/h = 6 hours = 360 minutes

# 360 minutes / 50 min stints = 7.2, ceiling = 8 stints
  result <- analyze_scenario(
    distance_km = 60,
    speed_kmh = 10,
    stint_min = 50,
    switch_time_min = 1.5
  )

  expect_equal(result$n_stints, 8)
  expect_equal(result$n_switches, 7)
})

test_that("analyze_scenario calculates switching time correctly", {
  result <- analyze_scenario(
    distance_km = 60,
    speed_kmh = 10,
    stint_min = 50,
    switch_time_min = 1.5
  )

  # 7 switches * 1.5 min = 10.5 min
  expect_equal(result$switching_time_min, 10.5)
})

test_that("analyze_scenario uses custom scenario name", {
  result <- analyze_scenario(
    distance_km = 60,
    speed_kmh = 10,
    stint_min = 50,
    switch_time_min = 1.5,
    scenario_name = "Custom Name"
  )

  expect_equal(result$scenario, "Custom Name")
})

test_that("analyze_scenario generates default scenario name", {
  result <- analyze_scenario(
    distance_km = 60,
    speed_kmh = 10,
    stint_min = 50,
    switch_time_min = 1.5
  )

  expect_equal(result$scenario, "50min stints")
})

test_that("analyze_scenario calculates pacer stints correctly", {
  # With 8 stints and seats 1-2 needing pacers: 8 * 2 = 16 slots
  # With 3 pacers: ceiling(16/3) = 6 stints per pacer
  result <- analyze_scenario(
    distance_km = 60,
    speed_kmh = 10,
    stint_min = 50,
    switch_time_min = 1.5,
    n_pacers = 3
  )

  expect_equal(result$pacer_stints, 6)
  expect_equal(result$pacer_rest_stints, 2)
})

test_that("analyze_scenario calculates steerer stints correctly", {
  # With 8 stints and seat 6 needing steerers: 8 * 1 = 8 slots
  # With 2 steerers: ceiling(8/2) = 4 stints per steerer
  result <- analyze_scenario(
    distance_km = 60,
    speed_kmh = 10,
    stint_min = 50,
    switch_time_min = 1.5,
    n_steerers = 2
  )

  expect_equal(result$steerer_stints, 4)
  expect_equal(result$steerer_rest_stints, 4)
})

test_that("analyze_scenario determines feasibility correctly", {
  # Default crew should be feasible
  result <- analyze_scenario(
    distance_km = 60,
    speed_kmh = 10,
    stint_min = 50,
    switch_time_min = 1.5
  )
  expect_true(result$feasible)
})

test_that("analyze_scenario handles infeasible scenarios", {
  # With only 1 pacer for seats 1-2, this might be infeasible
  # But let's test with an extreme case
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

  # With n_regular = 0 and steerers can't paddle seat 5, seats 3-5 can't be filled
  expect_false(result$feasible)
  expect_true(is.na(result$regular_stints))
})

test_that("analyze_scenario handles steerers_can_paddle_5 = FALSE", {
  result_can <- analyze_scenario(
    distance_km = 60,
    speed_kmh = 10,
    stint_min = 50,
    switch_time_min = 1.5,
    steerers_can_paddle_5 = TRUE
  )

  result_cannot <- analyze_scenario(
    distance_km = 60,
    speed_kmh = 10,
    stint_min = 50,
    switch_time_min = 1.5,
    steerers_can_paddle_5 = FALSE
  )

  # Both should be feasible with default crew
  expect_true(result_can$feasible)
  expect_true(result_cannot$feasible)
})

test_that("analyze_scenario validates distance_km", {
  expect_error(
    analyze_scenario(distance_km = -10, speed_kmh = 10, stint_min = 50, switch_time_min = 1.5),
    "`distance_km` must be a positive number"
  )
  expect_error(
    analyze_scenario(distance_km = 0, speed_kmh = 10, stint_min = 50, switch_time_min = 1.5),
    "`distance_km` must be a positive number"
  )
  expect_error(
    analyze_scenario(distance_km = "abc", speed_kmh = 10, stint_min = 50, switch_time_min = 1.5),
    "`distance_km` must be a positive number"
  )
})

test_that("analyze_scenario validates speed_kmh", {
  expect_error(
    analyze_scenario(distance_km = 60, speed_kmh = -5, stint_min = 50, switch_time_min = 1.5),
    "`speed_kmh` must be a positive number"
  )
  expect_error(
    analyze_scenario(distance_km = 60, speed_kmh = 0, stint_min = 50, switch_time_min = 1.5),
    "`speed_kmh` must be a positive number"
  )
})

test_that("analyze_scenario validates stint_min", {
  expect_error(
    analyze_scenario(distance_km = 60, speed_kmh = 10, stint_min = -30, switch_time_min = 1.5),
    "`stint_min` must be a positive number"
  )
  expect_error(
    analyze_scenario(distance_km = 60, speed_kmh = 10, stint_min = 0, switch_time_min = 1.5),
    "`stint_min` must be a positive number"
  )
})

test_that("analyze_scenario validates switch_time_min", {
  expect_error(
    analyze_scenario(distance_km = 60, speed_kmh = 10, stint_min = 50, switch_time_min = -1),
    "`switch_time_min` must be a non-negative number"
  )
  # Zero switch time should be allowed
  expect_no_error(
    analyze_scenario(distance_km = 60, speed_kmh = 10, stint_min = 50, switch_time_min = 0)
  )
})

test_that("analyze_scenario validates crew counts", {
  expect_error(
    analyze_scenario(distance_km = 60, speed_kmh = 10, stint_min = 50, switch_time_min = 1.5, n_pacers = 0),
    "`n_pacers` must be at least 1"
  )
  expect_error(
    analyze_scenario(distance_km = 60, speed_kmh = 10, stint_min = 50, switch_time_min = 1.5, n_steerers = 0),
    "`n_steerers` must be at least 1"
  )
  expect_error(
    analyze_scenario(distance_km = 60, speed_kmh = 10, stint_min = 50, switch_time_min = 1.5, n_regular = -1),
    "`n_regular` must be non-negative"
  )
})

test_that("analyze_scenario calculates rest time correctly", {
  result <- analyze_scenario(
    distance_km = 60,
    speed_kmh = 10,
    stint_min = 50,
    switch_time_min = 1.5
  )

  # pacer_rest_min = pacer_rest_stints * stint_min
  expect_equal(result$pacer_rest_min, result$pacer_rest_stints * result$stint_length_min)
  expect_equal(result$steerer_rest_min, result$steerer_rest_stints * result$stint_length_min)
})

test_that("analyze_scenario workload_variance is calculated", {
  result <- analyze_scenario(
    distance_km = 60,
    speed_kmh = 10,
    stint_min = 50,
    switch_time_min = 1.5
  )

  expect_true(is.numeric(result$workload_variance))
  expect_true(result$workload_variance >= 0)
})

test_that("analyze_scenario handles short races", {
  # 10km at 10km/h = 1 hour = 60 minutes
  # With 50 min stints = 2 stints
  result <- analyze_scenario(
    distance_km = 10,
    speed_kmh = 10,
    stint_min = 50,
    switch_time_min = 1.5
  )

  expect_equal(result$race_duration_hours, 1.0)
  expect_equal(result$n_stints, 2)
  expect_equal(result$n_switches, 1)
})

test_that("analyze_scenario handles long stints", {
  # 60km at 10km/h = 6 hours
  # With 120 min stints = 3 stints
  result <- analyze_scenario(
    distance_km = 60,
    speed_kmh = 10,
    stint_min = 120,
    switch_time_min = 1.5
  )

  expect_equal(result$n_stints, 3)
})
