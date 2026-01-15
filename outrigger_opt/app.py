#!/usr/bin/env python3
"""Streamlit web app for the outrigger rotation optimizer."""

import streamlit as st
import pandas as pd
import numpy as np
from outrigger_opt import solve_rotation_full

st.set_page_config(
    page_title="Outrigger Rotation Optimizer",
    page_icon="🚣",
    layout="wide"
)

st.title("Outrigger Rotation Optimizer")
st.markdown("Optimize crew rotation schedules to minimize race time while managing fatigue.")

# Sidebar for parameters
st.sidebar.header("Race Parameters")

distance_km = st.sidebar.number_input(
    "Distance (km)",
    min_value=1.0, max_value=200.0, value=60.0, step=1.0
)
speed_kmh = st.sidebar.number_input(
    "Base Speed (km/h)",
    min_value=1.0, max_value=20.0, value=10.0, step=0.5
)
stint_min = st.sidebar.number_input(
    "Stint Duration (min)",
    min_value=5, max_value=120, value=20, step=1
)
switch_time_min = st.sidebar.number_input(
    "Switch Time (min)",
    min_value=0.0, max_value=10.0, value=1.0, step=0.5
)
max_consecutive = st.sidebar.number_input(
    "Max Consecutive Stints",
    min_value=1, max_value=10, value=3, step=1
)

st.sidebar.header("Crew Configuration")
n_seats = st.sidebar.number_input(
    "Number of Seats",
    min_value=2, max_value=12, value=6, step=1
)
n_resting = st.sidebar.number_input(
    "Paddlers Resting per Stint",
    min_value=1, max_value=10, value=3, step=1
)

st.sidebar.header("Solver Settings")
time_limit = st.sidebar.number_input(
    "Time Limit (seconds)",
    min_value=5, max_value=600, value=120, step=5
)
gap_tolerance = st.sidebar.number_input(
    "Gap Tolerance",
    min_value=0.001, max_value=0.1, value=0.001, step=0.001, format="%.3f"
)

st.sidebar.header("Pattern Penalties")
entry_rule_penalty = st.sidebar.number_input(
    "Entry Rule Penalty",
    min_value=0.0, max_value=0.5, value=0.05, step=0.01,
    help="Penalty for each different seat a paddler enters from rest"
)
switch_rule_penalty = st.sidebar.number_input(
    "Switch Rule Penalty",
    min_value=0.0, max_value=0.5, value=0.05, step=0.01,
    help="Penalty for each different seat transition while paddling"
)

# Main content area
n_paddlers = n_seats + n_resting

# Paddler names
st.header("Paddlers")
st.markdown(f"Enter names for {n_paddlers} paddlers ({n_seats} seats + {n_resting} resting)")

# Default paddler names from example_custom_eligibility
default_names = ['V1', 'V2', 'V3', 'F1', 'F2', 'F3', 'F4', 'L1', 'L2']
if n_paddlers != 9:
    default_names = [f"P{i+1}" for i in range(n_paddlers)]

paddler_names = st.text_area(
    "Paddler Names (one per line or comma-separated)",
    value=", ".join(default_names[:n_paddlers]),
    height=100
)

# Parse paddler names
if "," in paddler_names:
    names = [n.strip() for n in paddler_names.split(",") if n.strip()]
else:
    names = [n.strip() for n in paddler_names.split("\n") if n.strip()]

if len(names) != n_paddlers:
    st.warning(f"Expected {n_paddlers} names, got {len(names)}. Please adjust.")

# Seat Weights
st.header("Seat Configuration")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Seat Weights")
    st.markdown("Importance weight for each seat (higher = more important)")

    # Default from example_custom_eligibility: [1, 1, 1, 1, 1, 1]
    default_weights = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    if n_seats != 6:
        default_weights = [1.0] * n_seats

    seat_weights = []
    cols = st.columns(min(n_seats, 6))
    for i in range(n_seats):
        with cols[i % 6]:
            w = st.number_input(
                f"Seat {i+1}",
                min_value=0.1, max_value=3.0,
                value=default_weights[i] if i < len(default_weights) else 1.0,
                step=0.1,
                key=f"weight_{i}"
            )
            seat_weights.append(w)

with col2:
    st.subheader("Seat Entry Weights")
    st.markdown("How easy to enter each seat (>1 easier, <1 harder)")

    # Default from example_custom_eligibility: [1, 1.5, 1.5, 1.5, 1.5, 1]
    default_entry_weights = [1.0, 1.5, 1.5, 1.5, 1.5, 1.0]
    if n_seats != 6:
        default_entry_weights = [1.0] * n_seats

    seat_entry_weight = []
    cols = st.columns(min(n_seats, 6))
    for i in range(n_seats):
        with cols[i % 6]:
            w = st.number_input(
                f"Seat {i+1}",
                min_value=0.1, max_value=3.0,
                value=default_entry_weights[i] if i < len(default_entry_weights) else 1.0,
                step=0.1,
                key=f"entry_weight_{i}"
            )
            seat_entry_weight.append(w)

# Eligibility Matrix
st.header("Seat Eligibility")
st.markdown("Check the seats each paddler can occupy")

# Default eligibility from example_custom_eligibility
default_eligibility = np.array([
    [1, 1, 0, 0, 0, 0],  # V1: front only
    [1, 1, 0, 0, 0, 0],  # V2: front only
    [1, 1, 1, 0, 0, 0],  # V3: front + seat 3
    [0, 0, 1, 1, 0, 0],  # F1: middle front
    [0, 0, 1, 1, 0, 0],  # F2: middle front
    [0, 0, 0, 1, 1, 0],  # F3: middle back
    [0, 0, 0, 1, 1, 0],  # F4: middle back
    [0, 0, 0, 0, 1, 1],  # L1: back + steering
    [0, 0, 0, 0, 1, 1],  # L2: back + steering
])

# Create eligibility DataFrame for editing
if n_paddlers == 9 and n_seats == 6:
    elig_data = {f"Seat {s+1}": default_eligibility[:, s].astype(bool) for s in range(n_seats)}
else:
    elig_data = {f"Seat {s+1}": [True] * n_paddlers for s in range(n_seats)}

elig_df = pd.DataFrame(elig_data, index=names[:n_paddlers] if len(names) == n_paddlers else [f"P{i+1}" for i in range(n_paddlers)])

edited_elig = st.data_editor(
    elig_df,
    use_container_width=True,
    hide_index=False
)

# Convert back to numpy array
eligibility = edited_elig.values.astype(int)

# Run optimization button
st.header("Optimize")

if st.button("Run Optimization", type="primary", use_container_width=True):
    if len(names) != n_paddlers:
        st.error(f"Please provide exactly {n_paddlers} paddler names.")
    else:
        paddlers = pd.DataFrame({"name": names})

        with st.spinner("Solving optimization problem..."):
            try:
                result = solve_rotation_full(
                    paddlers,
                    stint_min=stint_min,
                    max_consecutive=max_consecutive,
                    distance_km=distance_km,
                    speed_kmh=speed_kmh,
                    switch_time_min=switch_time_min,
                    seat_eligibility=eligibility,
                    seat_weights=seat_weights,
                    seat_entry_weight=seat_entry_weight,
                    n_seats=n_seats,
                    n_resting=n_resting,
                    time_limit=time_limit,
                    gap_tolerance=gap_tolerance,
                    entry_rule_penalty=entry_rule_penalty,
                    switch_rule_penalty=switch_rule_penalty
                )

                # Store result in session state
                st.session_state['result'] = result
                st.session_state['has_result'] = True

            except Exception as e:
                st.error(f"Optimization failed: {str(e)}")
                st.session_state['has_result'] = False

# Display results
if st.session_state.get('has_result', False):
    result = st.session_state['result']

    st.header("Results")

    # Status and key metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        status_color = "green" if result['status'] == 'Optimal' else "orange"
        st.metric("Status", result['status'])
    with col2:
        st.metric("Race Time", f"{result['race_time']:.1f} min")
    with col3:
        st.metric("Avg Output", f"{result['avg_output']:.3f}")
    with col4:
        st.metric("Stints", result['parameters']['n_stints'])

    # Schedule
    st.subheader("Rotation Schedule")
    schedule_display = result['schedule'].copy()
    schedule_display.index = [f"Stint {i+1}" for i in range(len(schedule_display))]
    schedule_display.index.name = ""
    st.dataframe(schedule_display, use_container_width=True)

    # Paddler summary
    st.subheader("Paddler Summary")
    paddler_summary = result['paddler_summary'].copy()
    st.dataframe(paddler_summary, use_container_width=True, hide_index=True)

    # Stats columns
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Aggregate Stats")
        stats = result['summary_stats']
        stats_df = pd.DataFrame({
            'Metric': [
                'Avg Time per Paddler',
                'Max Time (any paddler)',
                'Min Time (any paddler)',
                'Max Consecutive Stretch',
                'Avg Consecutive Stretch'
            ],
            'Value': [
                f"{stats['avg_time_per_paddler_min']:.1f} min",
                f"{stats['max_time_any_paddler_min']:.1f} min",
                f"{stats['min_time_any_paddler_min']:.1f} min",
                f"{stats['max_consecutive_stretch_min']:.1f} min",
                f"{stats['avg_consecutive_stretch_min']:.1f} min"
            ]
        })
        st.dataframe(stats_df, use_container_width=True, hide_index=True)

    with col2:
        st.subheader("Pattern Stats")
        pattern = result['pattern_stats']
        pattern_df = pd.DataFrame({
            'Metric': [
                'Total Entry Rules',
                'Total Switch Rules',
                'Avg Entry Rules/Paddler',
                'Avg Switch Rules/Paddler'
            ],
            'Value': [
                pattern['total_entry_rules'],
                pattern['total_switch_rules'],
                f"{pattern['avg_entry_rules_per_paddler']:.2f}",
                f"{pattern['avg_switch_rules_per_paddler']:.2f}"
            ]
        })
        st.dataframe(pattern_df, use_container_width=True, hide_index=True)

# Footer
st.markdown("---")
st.markdown("*Powered by PuLP/CBC mixed-integer programming solver*")
