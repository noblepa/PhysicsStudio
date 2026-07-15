"""
Banana Kick Simulator - Streamlit App
-------------------------------------
Interactive version of the Magnus-effect free kick simulation.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import streamlit as st

# This should be called only once by app.py
# Remove this line if app.py already contains st.set_page_config()
# st.set_page_config(page_title="Banana Kick Simulator", layout="wide")


def run():

    # ----------------------------
    # Sidebar controls
    # ----------------------------
    st.sidebar.title("⚽ Banana Kick Controls")

    st.sidebar.subheader("Kick")

    speed0 = st.sidebar.slider(
        "Initial speed (m/s)",
        15.0, 40.0, 30.0, 0.5,
        help="~108 km/h at 30 m/s"
    )

    angle_up = st.sidebar.slider(
        "Launch angle above ground (°)",
        0.0, 30.0, 16.0, 0.5
    )

    angle_side = st.sidebar.slider(
        "Aim angle, wide of goal center (°)",
        0.0, 30.0, 16.0, 0.5
    )

    st.sidebar.subheader("Spin (Magnus effect)")

    spin_dir = st.sidebar.radio(
        "Curve direction",
        ["Right → Left", "Left → Right"],
        index=0
    )

    spin_sign = -1.0 if spin_dir == "Right → Left" else 1.0

    Cl = st.sidebar.slider(
        "Spin strength (Magnus coefficient Cl)",
        0.0, 0.6, 0.30, 0.02,
        help="Higher = more spin = sharper curve"
    )

    st.sidebar.subheader("Ball & Air")

    Cd = st.sidebar.slider(
        "Drag coefficient Cd",
        0.10, 0.40, 0.22, 0.01
    )

    mass = st.sidebar.slider(
        "Ball mass (kg)",
        0.35, 0.50, 0.43, 0.01
    )

    st.sidebar.subheader("Pitch Geometry")

    GOAL_X = st.sidebar.slider(
        "Distance to goal (m)",
        18.0, 35.0, 30.0, 0.5
    )

    WALL_X = st.sidebar.slider(
        "Distance to wall (m)",
        5.0, 12.0, 9.15, 0.05
    )

    WALL_HALF_WIDTH = st.sidebar.slider(
        "Wall half-width (m)",
        0.5, 2.0, 1.0, 0.1
    )

    GOAL_HALF_WIDTH = 3.66
    GOAL_HEIGHT = 2.44
    WALL_HEIGHT = 2.0

    r = 0.11
    rho = 1.225
    A = np.pi * r**2
    g = 9.81

    @st.cache_data(show_spinner=False)
    def simulate(speed0, angle_up_deg, angle_side_deg,
                 Cl, Cd, mass, spin_sign, GOAL_X):

        angle_up = np.radians(angle_up_deg)
        angle_side = np.radians(angle_side_deg)

        def deriv(t, state):

            x, y, z, vx, vy, vz = state

            v = np.array([vx, vy, vz])
            speed = np.linalg.norm(v)

            if speed < 1e-6:
                speed = 1e-6

            drag = -0.5 * rho * Cd * A * speed * v

            omega_hat = np.array([0, 0, spin_sign])

            cross = np.cross(omega_hat, v)

            cn = np.linalg.norm(cross)

            n_hat = cross / cn if cn > 1e-9 else np.zeros(3)

            magnus = 0.5 * rho * Cl * A * speed**2 * n_hat

            F = drag + magnus + np.array([0, 0, -mass * g])

            a = F / mass

            return [vx, vy, vz, a[0], a[1], a[2]]