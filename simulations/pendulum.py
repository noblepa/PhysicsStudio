"""
simulations.pendulum
------------------------
Interactive simple (nonlinear) pendulum simulation, using Plotly for
visuals - including an animated swing with a play/pause control.
Exposes a single run() function, called from app.py as:

    from simulations import pendulum
    pendulum.run()

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
    st.sidebar.title("🕰 Pendulum Controls")

    st.sidebar.subheader("Pendulum")
    L = st.sidebar.slider("Length L (m)", 0.2, 3.0, 1.0, 0.1)
    theta0_deg = st.sidebar.slider(
        "Initial angle θ₀ (°)", 1.0, 170.0, 12.0, 1.0,
        help="Large angles show clearly nonlinear (non-sinusoidal) motion"
    )
    omega0 = st.sidebar.slider("Initial angular velocity ω₀ (rad/s)", -5.0, 5.0, 0.0, 0.1)

    st.sidebar.subheader("Damping & Gravity")
    b = st.sidebar.slider(
        "Damping coefficient b", 0.0, 1.0, 0.15, 0.01,
        help="0 = ideal, undamped pendulum (closed loop in phase space)"
    )
    g = st.sidebar.slider("Gravity g (m/s²)", 1.0, 25.0, 9.81, 0.1)

    st.sidebar.subheader("Simulation")
    t_max = st.sidebar.slider("Simulation time (s)", 5, 40, 20, 1)
    fps = st.sidebar.slider("Animation frame rate (fps)", 10, 60, 30, 5)

    theta0 = np.radians(theta0_deg)

    # ----------------------------
    # Physics (cached so identical parameter combos don't re-integrate)
    # ----------------------------
    @st.cache_data(show_spinner=False)
    def simulate(L, theta0, omega0, b, g, t_max, fps):
        def pendulum_ode(t, y):
            theta, omega = y
            return [omega, -(g / L) * np.sin(theta) - b * omega]

        t_eval = np.linspace(0, t_max, int(t_max * fps))
        sol = solve_ivp(pendulum_ode, [0, t_max], [theta0, omega0],
                         t_eval=t_eval, method="RK45", rtol=1e-9, atol=1e-9)
        theta, omega = sol.y
        x = L * np.sin(theta)
        y_pos = -L * np.cos(theta)
        return sol.t, theta, omega, x, y_pos

    t, theta, omega, x, y_pos = simulate(L, theta0, omega0, b, g, t_max, fps)

    # Small-angle period, for comparison
    T_small_angle = 2 * np.pi * np.sqrt(L / g)
    # Estimate the actual (possibly amplitude-dependent) period from zero crossings of omega
    sign_changes = np.where(np.diff(np.sign(omega)))[0]
    if len(sign_changes) >= 2:
        T_simulated = 2 * np.mean(np.diff(t[sign_changes]))
    else:
        T_simulated = None

    # ----------------------------
    # Header + info banner
    # ----------------------------
    st.title("🕰 Simple Pendulum Simulator")
    st.caption(
        "Solves the exact nonlinear pendulum equation "
        "θ'' = -(g/L)sin(θ) - bθ' — not just the small-angle approximation."
    )

    banner = ASSET_DIR / "pendulum.png"
    st.image(
        str(banner),
        use_container_width=True
    )

    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Small-angle period (2π√(L/g))", f"{T_small_angle:.2f} s")
    col_b.metric("Simulated period", f"{T_simulated:.2f} s" if T_simulated else "n/a (damped out)")
    col_c.metric("Regime", "Ideal (undamped)" if b == 0 else "Damped")

    # ----------------------------
    # Animated pendulum (Plotly frames + play/pause)
    # ----------------------------
    max_frames = 300
    stride = max(1, len(t) // max_frames)
    idx = np.arange(0, len(t), stride)

    trail_len = 25  # points of fading trail

    frames = []
    for i in idx:
        start = max(0, i - trail_len)
        frames.append(go.Frame(
            data=[
                go.Scatter(x=[0, x[i]], y=[0, y_pos[i]], mode="lines+markers",
                           line=dict(color="steelblue", width=3),
                           marker=dict(size=[6, 16], color=["black", "firebrick"])),
                go.Scatter(x=x[start:i + 1], y=y_pos[start:i + 1], mode="lines",
                           line=dict(color="lightsteelblue", width=1)),
            ],
            name=str(i)
        ))

    fig_pend = go.Figure(
        data=[
            go.Scatter(x=[0, x[0]], y=[0, y_pos[0]], mode="lines+markers",
                       line=dict(color="steelblue", width=3),
                       marker=dict(size=[6, 16], color=["black", "firebrick"])),
            go.Scatter(x=[x[0]], y=[y_pos[0]], mode="lines",
                       line=dict(color="lightsteelblue", width=1)),
        ],
        frames=frames,
    )
    fig_pend.update_layout(
        title="Pendulum Motion",
        xaxis=dict(range=[-1.3 * L, 1.3 * L], title="x (m)"),
        yaxis=dict(range=[-1.3 * L, 0.3 * L], title="y (m)", scaleanchor="x", scaleratio=1),
        showlegend=False,
        height=500,
        margin=dict(l=10, r=10, t=40, b=10),
        updatemenus=[dict(
            type="buttons",
            showactive=False,
            y=0.95, x=0.09,
            buttons=[
                dict(label="▶ Play", method="animate",
                     args=[None, dict(frame=dict(duration=1000 / fps, redraw=True),
                                       fromcurrent=True, transition=dict(duration=0))]),
                dict(label="⏸ Pause", method="animate",
                     args=[[None], dict(frame=dict(duration=0, redraw=False),
                                         mode="immediate")]),
            ]
        )],
    )

    st.plotly_chart(fig_pend, width='stretch')

    # ----------------------------
    # Phase space + time series (side by side)
    # ----------------------------
    col_left, col_right = st.columns(2)

    with col_left:
        fig_phase = go.Figure()
        fig_phase.add_trace(go.Scatter(
            x=theta, y=omega, mode="lines",
            line=dict(color="darkorange", width=1.5), name="Phase path"
        ))
        fig_phase.add_trace(go.Scatter(
            x=[theta[0]], y=[omega[0]], mode="markers",
            marker=dict(size=9, color="darkorange", line=dict(color="black", width=1)),
            name="Start"
        ))
        fig_phase.update_layout(
            title=dict(text="Phase Space (θ vs ω)", pad=dict(b=15)),
            xaxis_title="θ (rad)",
            yaxis_title="ω = dθ/dt (rad/s)",
            height=420,
            margin=dict(l=10, r=10, t=40, b=10),
        )
        st.plotly_chart(fig_phase, width='stretch')

    with col_right:
        fig_time = go.Figure()
        fig_time.add_trace(go.Scatter(x=t, y=np.degrees(theta), mode="lines",
                                       line=dict(color="steelblue", width=1.5), name="θ (deg)"))
        fig_time.add_trace(go.Scatter(x=t, y=omega, mode="lines",
                                       line=dict(color="firebrick", width=1.5), name="ω (rad/s)"))
        fig_time.update_layout(
            title=dict(text="Angle & Angular Velocity vs Time", pad=dict(b=15)),
            xaxis_title="time (s)",
            yaxis_title="value",
            height=420,
            margin=dict(l=10, r=10, t=40, b=10),
            #legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        st.plotly_chart(fig_time, width='stretch')

    with st.expander("How this works"):
        st.markdown(
            r"""
The pendulum obeys the exact nonlinear equation of motion (not just the small-angle approximation):

$$
\frac{d^2\theta}{dt^2} = -\frac{g}{L}\sin\theta \;-\; b\,\frac{d\theta}{dt}
$$

- With **b = 0** (no damping), energy is conserved and the phase-space path is a closed loop — the
  pendulum swings forever at constant amplitude.
- With **b > 0**, energy bleeds away and the phase-space path spirals in toward the origin (theta=0,
  omega=0) — the pendulum settles to rest hanging straight down.
- At **large initial angles**, the period is noticeably longer than the small-angle formula
  $T = 2\pi\sqrt{L/g}$ predicts — that formula is only exact in the limit θ₀ → 0.

Try pushing **θ₀** close to 180° — the pendulum lingers near the unstable equilibrium at the top of its
swing before continuing, and the period grows dramatically.
"""
        )