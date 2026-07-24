"""
simulations.charged_particle
---------------------------------
Interactive charged-particle-in-E-and-B-field simulation (Lorentz force),
using Plotly for visuals. Exposes a single run() function, called from
app.py as:

    from simulations import charged_particle
    charged_particle.run()

Assumes st.set_page_config() has already been called by app.py.
"""
import numpy as np
import plotly.graph_objects as go
from scipy.integrate import solve_ivp
import streamlit as st
from pathlib import Path
ASSET_DIR = Path(__file__).parent.parent / "assets"

def run():
    if st.button("🏠 Back to Home"):
        st.session_state.simulation = "home"
        st.rerun()
    # ----------------------------
    # Sidebar controls
    # ----------------------------
    st.sidebar.title("⚛️ Charged Particle Controls")

    preset = st.sidebar.selectbox("Quick presets", [
        "Custom (use sliders below)",
        "Pure helix (E = 0)",
        "E × B drift",
    ])

    st.sidebar.subheader("Magnetic Field B (scaled units)")
    Bx = st.sidebar.slider("Bx", -2.0, 2.0, 0.0, 0.1)
    By = st.sidebar.slider("By", -2.0, 2.0, 0.0, 0.1)
    Bz = st.sidebar.slider("Bz", -2.0, 2.0, 1.0, 0.1)

    st.sidebar.subheader("Electric Field E (scaled units)")
    Ex = st.sidebar.slider("Ex", -2.0, 2.0, 0.0, 0.1)
    Ey = st.sidebar.slider("Ey", -2.0, 2.0, 0.0, 0.1)
    Ez = st.sidebar.slider("Ez", -2.0, 2.0, 0.0, 0.1)

    st.sidebar.subheader("Particle")
    q = st.sidebar.slider("Charge q", -2.0, 2.0, 1.0, 0.1)
    m = st.sidebar.slider("Mass m", 0.2, 3.0, 1.0, 0.1)

    st.sidebar.subheader("Initial Velocity")
    vx0 = st.sidebar.slider("vx₀", -2.0, 2.0, 1.2, 0.1)
    vy0 = st.sidebar.slider("vy₀", -2.0, 2.0, 0.0, 0.1)
    vz0 = st.sidebar.slider("vz₀", -2.0, 2.0, 0.5, 0.1)

    st.sidebar.subheader("Simulation")
    n_orbits = st.sidebar.slider("Number of cyclotron orbits to show", 1, 15, 4)

    B = np.array([Bx, By, Bz])
    E = np.array([Ex, Ey, Ez])
    v0 = np.array([vx0, vy0, vz0])

    if preset == "Pure helix (E = 0)":
        B = np.array([0.0, 0.0, 1.0])
        E = np.array([0.0, 0.0, 0.0])
        v0 = np.array([1.2, 0.0, 0.5])
    elif preset == "E × B drift":
        B = np.array([0.0, 0.0, 1.0])
        E = np.array([1.0, 0.0, 0.0])
        v0 = np.array([0.0, 0.0, 0.0])

    B_mag = np.linalg.norm(B)

    # ----------------------------
    # Physics (cached so identical parameter combos don't re-integrate)
    # ----------------------------
    @st.cache_data(show_spinner=False)
    def simulate(q, m, E, B, v0, n_orbits):
        B_mag = np.linalg.norm(B)
        if B_mag < 1e-6:
            omega_c = 0.0
            t_max = 10.0
        else:
            omega_c = abs(q) * B_mag / m
            T_c = 2 * np.pi / omega_c
            t_max = n_orbits * T_c

        def deriv(t, state):
            pos = state[:3]
            v = state[3:]
            F = q * (E + np.cross(v, B))
            a = F / m
            return [*v, *a]

        n_pts = max(400, 80 * n_orbits)
        t_eval = np.linspace(0, t_max, n_pts)
        state0 = [0, 0, 0, *v0]
        max_step = t_max / (n_pts / 2) if t_max > 0 else 0.01
        sol = solve_ivp(deriv, [0, t_max], state0, t_eval=t_eval, max_step=max_step)
        x, y, z = sol.y[0], sol.y[1], sol.y[2]
        vx, vy, vz = sol.y[3], sol.y[4], sol.y[5]
        return x, y, z, vx, vy, vz, omega_c

    x, y, z, vx, vy, vz, omega_c = simulate(q, m, E, B, v0, n_orbits)

    # Analytic quantities
    if B_mag > 1e-6:
        v_perp_vec = v0 - (np.dot(v0, B) / B_mag ** 2) * B
        v_perp = np.linalg.norm(v_perp_vec)
        larmor_radius = m * v_perp / (abs(q) * B_mag)
    else:
        v_perp = np.linalg.norm(v0)
        larmor_radius = float("nan")

    v_drift_analytic = np.cross(E, B) / np.dot(B, B) if B_mag > 1e-6 else np.array([np.nan] * 3)
    v_drift_sim = np.array([np.mean(vx), np.mean(vy), np.mean(vz)])

    # ----------------------------
    # Header + metrics
    # ----------------------------
    st.title("⚛️ Charged Particle in Electric & Magnetic Fields")
    st.caption(
        "Solves the Lorentz force equation m·dv/dt = q(E + v × B). Pure B gives helical motion around "
        "field lines; adding a perpendicular E gives a steady E×B drift."
    )
    banner = ASSET_DIR / "chargedparticle.png"
    st.image(
        str(banner),
        use_container_width=True
    )

    col1, col2, col3 = st.columns(3)
    col1.metric("Cyclotron frequency ω_c = |q|B/m",
                f"{omega_c:.3f} rad/s" if B_mag > 1e-6 else "n/a (no B)")
    col2.metric("Larmor radius", f"{larmor_radius:.3f}" if B_mag > 1e-6 else "n/a")
    col3.metric("Simulated avg drift velocity",
                f"({v_drift_sim[0]:.2f}, {v_drift_sim[1]:.2f}, {v_drift_sim[2]:.2f})")

    if B_mag > 1e-6 and np.linalg.norm(E) > 1e-6:
        st.info(
            f"Analytic E×B drift velocity = (E×B)/B² = "
            f"({v_drift_analytic[0]:.3f}, {v_drift_analytic[1]:.3f}, {v_drift_analytic[2]:.3f}) "
            f"— compare to the simulated average above."
        )

    # ----------------------------
    # 3D trajectory (Plotly)
    # ----------------------------
    fig3d = go.Figure()
    fig3d.add_trace(go.Scatter3d(
        x=x, y=y, z=z, mode="lines",
        line=dict(color="darkorange", width=4), name="Trajectory"
    ))
    fig3d.add_trace(go.Scatter3d(
        x=[0], y=[0], z=[0], mode="markers",
        marker=dict(size=4, color="black"), name="Start"
    ))
    fig3d.update_layout(
        title=dict(text="3D Trajectory", pad=dict(b=15)),
        scene=dict(xaxis_title="x", yaxis_title="y", zaxis_title="z"),
        margin=dict(l=0, r=0, t=50, b=0),
        height=520,
    )
    st.plotly_chart(fig3d, width='stretch')

    # ----------------------------
    # 2D projections (side by side)
    # ----------------------------
    col_left, col_right = st.columns(2)

    with col_left:
        fig_xy = go.Figure()
        fig_xy.add_trace(go.Scatter(x=x, y=y, mode="lines",
                                     line=dict(color="steelblue", width=1.5)))
        fig_xy.update_layout(
            title=dict(text="x–y Projection", pad=dict(b=15)),
            xaxis_title="x", yaxis_title="y",
            yaxis=dict(scaleanchor="x", scaleratio=1),
            height=420,
            margin=dict(l=10, r=10, t=50, b=10),
        )
        st.plotly_chart(fig_xy, width='stretch')

    with col_right:
        fig_xz = go.Figure()
        fig_xz.add_trace(go.Scatter(x=x, y=z, mode="lines",
                                     line=dict(color="firebrick", width=1.5)))
        fig_xz.update_layout(
            title=dict(text="x–z Projection", pad=dict(b=15)),
            xaxis_title="x", yaxis_title="z",
            height=420,
            margin=dict(l=10, r=10, t=50, b=10),
        )
        st.plotly_chart(fig_xz, width='stretch')

    with st.expander("How this works"):
        st.markdown(
            r"""
The particle obeys the **Lorentz force law**:

$$
m \frac{d\mathbf{v}}{dt} = q(\mathbf{E} + \mathbf{v} \times \mathbf{B})
$$

- With **only a magnetic field**, the velocity component perpendicular to $\mathbf{B}$ circles at the
  **cyclotron frequency** $\omega_c = |q|B/m$, with radius (the **Larmor radius**)
  $r_L = m v_\perp / (|q| B)$, while the parallel component carries the particle steadily along the
  field line — tracing a **helix**.
- Adding an **electric field perpendicular to B** superimposes a steady drift on top of the gyration,
  at velocity
  $$ \mathbf{v}_{drift} = \frac{\mathbf{E}\times\mathbf{B}}{B^2} $$
  independent of the particle's charge or mass. This is exactly how particles drift in a magnetron, a
  Hall thruster, or Earth's magnetosphere.
- Try the **presets** in the sidebar for the two classic cases, or build your own combination of
  **E**, **B**, and initial velocity.
"""
        )