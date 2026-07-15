"""
Banana Kick Simulator - Streamlit App
-------------------------------------
Interactive version of the Magnus-effect free kick simulation.
"""
import numpy as np
import plotly.graph_objects as go
from scipy.integrate import solve_ivp
import streamlit as st
from pathlib import Path

ASSET_DIR = Path(__file__).parent.parent / "assets"

# This should be called only once by app.py
# Remove this line if app.py already contains st.set_page_config()
# st.set_page_config(page_title="Banana Kick Simulator", layout="wide")


def run():
    if st.button("🏠 Back to Home"):
        st.session_state.simulation = "home"
        st.rerun()

    #st.title("⚽ Banana Kick Simulator")
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

    # ----------------------------
    # Physics (cached so identical parameter combos don't re-integrate)
    # ----------------------------
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

        vx0 = speed0 * np.cos(angle_up) * np.cos(angle_side)
        vy0 = speed0 * np.cos(angle_up) * np.sin(angle_side)
        vz0 = speed0 * np.sin(angle_up)
        state0 = [0, 0, 0.05, vx0, vy0, vz0]

        def hit_ground(t, state):
            return state[2] + 0.01
        hit_ground.terminal = True
        hit_ground.direction = -1

        sol = solve_ivp(deriv, [0, 5], state0, max_step=0.004,
                         events=hit_ground, dense_output=True)

        mask = sol.y[0] <= GOAL_X
        if not mask.any():
            mask = np.ones_like(sol.y[0], dtype=bool)
        x, y, z = sol.y[0][mask], sol.y[1][mask], sol.y[2][mask]
        return x, y, z

    x, y, z = simulate(speed0, angle_up, angle_side, Cl, Cd, mass, spin_sign, GOAL_X)

    y_at_wall = np.interp(WALL_X, x, y) if x.max() >= WALL_X else None
    z_at_wall = np.interp(WALL_X, x, z) if x.max() >= WALL_X else None
    y_final, z_final = y[-1], z[-1]
    reached_goal_line = x.max() >= GOAL_X - 0.05

    is_goal = reached_goal_line and abs(y_final) < GOAL_HALF_WIDTH and z_final < GOAL_HEIGHT
    cleared_wall = (y_at_wall is not None) and (abs(y_at_wall) > WALL_HALF_WIDTH or z_at_wall > WALL_HEIGHT)

    # ----------------------------
    # Header + result banner
    # ----------------------------
    st.title("⚽ Banana Kick Simulator")

    banner = ASSET_DIR / "bananakick.png"

    st.image(
        str(banner),
        use_container_width=True
    )

    st.caption(
        "A free kick's curve comes from the **Magnus effect**: spin on the ball creates a sideways "
        "aerodynamic force, bending its path around the wall."
    )

    col_a, col_b, col_c = st.columns(3)
    col_a.metric(
        "At the wall",
        f"y = {y_at_wall:.2f} m" if y_at_wall is not None else "—",
        "clears it" if cleared_wall else "blocked!"
    )
    # col_b.metric("At the goal line", f"y = {y_final:.2f} m, z = {z_final:.2f} m")
    with col_b:
        st.metric("Lateral Position", f"{y_final:.2f} m")
        st.metric("Height", f"{z_final:.2f} m")
    col_c.metric("Result", "🥅 GOAL!" if is_goal else ("Blocked by wall" if not cleared_wall else "Missed target"))

    # ----------------------------
    # Plotly helpers
    # ----------------------------
    def wall_mesh_3d(x_pos, half_width, height, color="rgba(70,130,180,0.5)"):
        xs = [x_pos, x_pos, x_pos, x_pos]
        ys = [-half_width, half_width, half_width, -half_width]
        zs = [0, 0, height, height]
        return go.Mesh3d(x=xs, y=ys, z=zs, i=[0, 0], j=[1, 2], k=[2, 3],
                          color=color, opacity=0.5, showlegend=False, name="Wall")

    def goal_lines_3d(goal_x, half_width, height):
        xg = [goal_x, goal_x, goal_x, goal_x, goal_x]
        yg = [-half_width, -half_width, half_width, half_width, -half_width]
        zg = [0, height, height, 0, 0]
        return go.Scatter3d(x=xg, y=yg, z=zg, mode="lines",
                             line=dict(color="black", width=6), name="Goal", showlegend=False)

    # ----------------------------
    # 3D trajectory
    # ----------------------------
    fig3d = go.Figure()
    fig3d.add_trace(go.Scatter3d(x=x, y=y, z=z, mode="lines",
                                  line=dict(color="darkorange", width=6), name="Ball path"))
    fig3d.add_trace(go.Scatter3d(x=[0], y=[0], z=[0], mode="markers",
                                  marker=dict(size=4, color="black"), name="Kick spot"))
    fig3d.add_trace(wall_mesh_3d(WALL_X, WALL_HALF_WIDTH, WALL_HEIGHT))
    fig3d.add_trace(goal_lines_3d(GOAL_X, GOAL_HALF_WIDTH, GOAL_HEIGHT))
    fig3d.update_layout(
        title="3D Flight Path",
        scene=dict(
            xaxis_title="x (m) downfield",
            yaxis_title="y (m) lateral",
            zaxis_title="z (m) height",
            aspectmode="manual",
            aspectratio=dict(x=2, y=1, z=0.5),
        ),
        margin=dict(l=0, r=0, t=40, b=0),
        height=500,
    )

    # ----------------------------
    # Top-down view
    # ----------------------------
    fig_top = go.Figure()
    fig_top.add_trace(go.Scatter(x=x, y=y, mode="lines",
                                  line=dict(color="darkorange", width=3), name="Ball path"))
    fig_top.add_shape(type="rect", x0=WALL_X - 0.3, x1=WALL_X + 0.3,
                       y0=-WALL_HALF_WIDTH, y1=WALL_HALF_WIDTH,
                       fillcolor="steelblue", opacity=0.6, line=dict(width=0))
    fig_top.add_trace(go.Scatter(x=[GOAL_X, GOAL_X], y=[-GOAL_HALF_WIDTH, GOAL_HALF_WIDTH],
                                  mode="lines", line=dict(color="black", width=6), name="Goal"))
    fig_top.add_trace(go.Scatter(x=[0], y=[0], mode="markers",
                                  marker=dict(size=8, color="black"), name="Kick spot"))
    fig_top.update_layout(
        title="Top-Down View",
        xaxis_title="x (m) downfield",
        yaxis_title="y (m) lateral",
        yaxis=dict(scaleanchor="x", scaleratio=1),
        height=450,
        margin=dict(l=10, r=10, t=40, b=10),
    )

    # ----------------------------
    # Side view
    # ----------------------------
    fig_side = go.Figure()
    fig_side.add_trace(go.Scatter(x=x, y=z, mode="lines",
                                   line=dict(color="darkorange", width=3), name="Ball path"))
    fig_side.add_shape(type="rect", x0=WALL_X - 0.3, x1=WALL_X + 0.3, y0=0, y1=WALL_HEIGHT,
                        fillcolor="steelblue", opacity=0.6, line=dict(width=0))
    fig_side.add_trace(go.Scatter(x=[GOAL_X, GOAL_X], y=[0, GOAL_HEIGHT],
                                   mode="lines", line=dict(color="black", width=6), name="Goal"))
    fig_side.update_layout(
        title="Side View (height profile)",
        xaxis_title="x (m) downfield",
        yaxis_title="z (m) height",
        height=450,
        margin=dict(l=10, r=10, t=40, b=10),
    )

    # ----------------------------
    # Layout
    # ----------------------------
    st.plotly_chart(fig3d, width='stretch')
    col_left, col_right = st.columns(2)
    with col_left:
        st.plotly_chart(fig_top, width='stretch')
    with col_right:
        st.plotly_chart(fig_side, width='stretch')

    with st.expander("How this works"):
        st.markdown(
            r"""
The ball's motion is governed by three forces: gravity, drag, and the Magnus force from spin:

$$
m \frac{d\mathbf{v}}{dt} = -mg\hat{z} \;-\; \tfrac{1}{2}\rho C_d A |\mathbf{v}| \mathbf{v} \;+\; \tfrac{1}{2}\rho C_l A |\mathbf{v}|^2 \hat{n}
$$

where $\hat{n}$ points in the direction of $(\boldsymbol{\omega} \times \mathbf{v})$ — perpendicular to the
ball's velocity, in the plane set by its spin axis. This is integrated numerically with
`scipy.integrate.solve_ivp`. Try increasing **spin strength** for a sharper curve, or flipping
**curve direction** to bend the other way.
"""
        )


if __name__ == "__main__":
    st.set_page_config(page_title="Banana Kick Simulator", layout="wide")
    run()