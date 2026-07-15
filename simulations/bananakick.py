"""
Banana Kick Simulator - Streamlit App
----------------------------------------
Interactive version of the Magnus-effect free kick simulation. Adjust
ball speed, launch angle, spin, and pitch geometry in the sidebar and
watch the trajectory, top-down curve, and side view update live.

"""
def run(): 
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import streamlit as st

st.set_page_config(page_title="Banana Kick Simulator", layout="wide")

# ----------------------------
# Sidebar controls
# ----------------------------
st.sidebar.title("⚽ Banana Kick Controls")

st.sidebar.subheader("Kick")
speed0 = st.sidebar.slider("Initial speed (m/s)", 15.0, 40.0, 30.0, 0.5,
                            help="~108 km/h at 30 m/s")
angle_up = st.sidebar.slider("Launch angle above ground (°)", 0.0, 30.0, 16.0, 0.5)
angle_side = st.sidebar.slider("Aim angle, wide of goal center (°)", 0.0, 30.0, 16.0, 0.5)

st.sidebar.subheader("Spin (Magnus effect)")
spin_dir = st.sidebar.radio("Curve direction", ["Right → Left", "Left → Right"], index=0)
spin_sign = -1.0 if spin_dir == "Right → Left" else 1.0
Cl = st.sidebar.slider("Spin strength (Magnus coefficient Cl)", 0.0, 0.6, 0.30, 0.02,
                        help="Higher = more spin = sharper curve")

st.sidebar.subheader("Ball & Air")
Cd = st.sidebar.slider("Drag coefficient Cd", 0.10, 0.40, 0.22, 0.01)
mass = st.sidebar.slider("Ball mass (kg)", 0.35, 0.50, 0.43, 0.01)

st.sidebar.subheader("Pitch Geometry")
GOAL_X = st.sidebar.slider("Distance to goal (m)", 18.0, 35.0, 30.0, 0.5)
WALL_X = st.sidebar.slider("Distance to wall (m)", 5.0, 12.0, 9.15, 0.05)
WALL_HALF_WIDTH = st.sidebar.slider("Wall half-width (m)", 0.5, 2.0, 1.0, 0.1)

GOAL_HALF_WIDTH = 3.66
GOAL_HEIGHT = 2.44
WALL_HEIGHT = 2.0
r = 0.11
rho = 1.225
A = np.pi * r**2
g = 9.81

# ----------------------------
# Physics: cached so dragging one slider doesn't re-import/recompute more than needed
# ----------------------------
@st.cache_data(show_spinner=False)
def simulate(speed0, angle_up_deg, angle_side_deg, Cl, Cd, mass, spin_sign, GOAL_X):
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
st.caption("A free kick's curve comes from the **Magnus effect**: spin on the ball creates a sideways "
           "aerodynamic force, bending its path around the wall.")

col_a, col_b, col_c = st.columns(3)
col_a.metric("At the wall", f"y = {y_at_wall:.2f} m" if y_at_wall is not None else "—",
             "clears it" if cleared_wall else "blocked!")
col_b.metric("At the goal line", f"y = {y_final:.2f} m, z = {z_final:.2f} m")
col_c.metric("Result", "🥅 GOAL!" if is_goal else ("Blocked by wall" if not cleared_wall else "Missed target"))

# ----------------------------
# Plots
# ----------------------------
fig = plt.figure(figsize=(14, 5.5))

ax3d = fig.add_subplot(1, 3, 1, projection="3d")
ax3d.plot(x, y, z, color="darkorange", lw=2)
ax3d.scatter([0], [0], [0], color="black", s=25)
wx = np.array([WALL_X, WALL_X])
wy = np.array([-WALL_HALF_WIDTH, WALL_HALF_WIDTH])
wz = np.array([0, WALL_HEIGHT])
WY, WZ = np.meshgrid(wy, wz)
WX = np.full_like(WY, WALL_X)
ax3d.plot_surface(WX, WY, WZ, color="steelblue", alpha=0.4)
ax3d.plot([GOAL_X, GOAL_X], [-GOAL_HALF_WIDTH, -GOAL_HALF_WIDTH], [0, GOAL_HEIGHT], color="black", lw=3)
ax3d.plot([GOAL_X, GOAL_X], [GOAL_HALF_WIDTH, GOAL_HALF_WIDTH], [0, GOAL_HEIGHT], color="black", lw=3)
ax3d.plot([GOAL_X, GOAL_X], [-GOAL_HALF_WIDTH, GOAL_HALF_WIDTH], [GOAL_HEIGHT, GOAL_HEIGHT], color="black", lw=3)
ax3d.set_xlabel("x (m)")
ax3d.set_ylabel("y (m)")
ax3d.set_zlabel("z (m)")
ax3d.set_title("3D Flight Path")
ax3d.set_box_aspect([2, 1, 0.5])

ax_top = fig.add_subplot(1, 3, 2)
ax_top.plot(x, y, color="darkorange", lw=2, label="Ball path")
ax_top.fill_between([WALL_X - 0.3, WALL_X + 0.3], -WALL_HALF_WIDTH, WALL_HALF_WIDTH,
                     color="steelblue", alpha=0.6, label="Wall")
ax_top.plot([GOAL_X, GOAL_X], [-GOAL_HALF_WIDTH, GOAL_HALF_WIDTH], color="black", lw=4, label="Goal")
ax_top.scatter([0], [0], color="black", s=40)
ax_top.set_xlabel("x (m) downfield")
ax_top.set_ylabel("y (m) lateral")
ax_top.set_title("Top-Down View")
ax_top.legend(loc="upper left", fontsize=8)
ax_top.set_aspect("equal")
ax_top.grid(alpha=0.3)

ax_side = fig.add_subplot(1, 3, 3)
ax_side.plot(x, z, color="darkorange", lw=2)
ax_side.fill_between([WALL_X - 0.3, WALL_X + 0.3], 0, WALL_HEIGHT, color="steelblue", alpha=0.6)
ax_side.plot([GOAL_X, GOAL_X], [0, GOAL_HEIGHT], color="black", lw=4)
ax_side.set_xlabel("x (m) downfield")
ax_side.set_ylabel("z (m) height")
ax_side.set_title("Side View")
ax_side.grid(alpha=0.3)

fig.tight_layout()
st.pyplot(fig)

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
