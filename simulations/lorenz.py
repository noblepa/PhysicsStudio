"""
Lorenz Attractor Simulator - Streamlit App
----------------------------------------------
Interactive version of the Lorenz strange-attractor simulation, using
Plotly for interactive 3D/2D visuals.
"""
import numpy as np
import plotly.graph_objects as go
from scipy.integrate import solve_ivp
import streamlit as st
from pathlib import Path
ASSET_DIR = Path(__file__).parent.parent / "assets"

# This should be called only once by app.py
# Remove this line if app.py already contains st.set_page_config()
# st.set_page_config(page_title="Lorenz Attractor Simulator", layout="wide")


def run():
    if st.button("🏠 Back to Home"):
        st.session_state.simulation = "home"
        st.rerun()
    # ----------------------------
    # Sidebar controls
    # ----------------------------
    st.sidebar.title("🦋 Lorenz Attractor Controls")

    st.sidebar.subheader("System Parameters")
    sigma = st.sidebar.slider("σ (sigma, Prandtl-like number)", 0.5, 20.0, 10.0, 0.5)
    rho = st.sidebar.slider(
        "ρ (rho, Rayleigh-like number)", 0.0, 50.0, 28.0, 0.5,
        help="Chaos onset is around ρ ≈ 24.74 with the default σ, β"
    )
    beta = st.sidebar.slider("β (beta)", 0.5, 5.0, 8.0 / 3.0, 0.05)

    st.sidebar.subheader("Initial Condition")
    x0 = st.sidebar.slider("x₀", -20.0, 20.0, 1.0, 0.5)
    y0 = st.sidebar.slider("y₀", -20.0, 20.0, 1.0, 0.5)
    z0 = st.sidebar.slider("z₀", 0.0, 50.0, 1.0, 0.5)

    st.sidebar.subheader("Simulation")
    t_max = st.sidebar.slider("Integration time", 10, 80, 40, 5)
    n_pts = st.sidebar.slider("Resolution (points)", 1000, 10000, 6000, 500)

    st.sidebar.subheader("Butterfly Effect")
    show_butterfly = st.sidebar.checkbox("Show 2nd trajectory (perturbed start)", value=True)
    eps_exp = st.sidebar.slider(
        "Initial separation (10^x)", -8, -2, -5, 1,
        help="How far apart the two starting points are"
    )
    eps = 10.0 ** eps_exp

    # ----------------------------
    # Physics (cached so identical parameter combos don't re-integrate)
    # ----------------------------
    @st.cache_data(show_spinner=False)
    def simulate(sigma, rho, beta, x0, y0, z0, t_max, n_pts, eps):
        def lorenz(t, state):
            x, y, z = state
            return [sigma * (y - x), x * (rho - z) - y, x * y - beta * z]

        t_eval = np.linspace(0, t_max, n_pts)
        sol = solve_ivp(lorenz, [0, t_max], [x0, y0, z0], t_eval=t_eval,
                         method="RK45", rtol=1e-9, atol=1e-9)
        x, y, z = sol.y

        sol_b = solve_ivp(lorenz, [0, t_max], [x0 + eps, y0, z0], t_eval=t_eval,
                           method="RK45", rtol=1e-9, atol=1e-9)
        xb, yb, zb = sol_b.y

        separation = np.sqrt((x - xb) ** 2 + (y - yb) ** 2 + (z - zb) ** 2)
        return t_eval, x, y, z, xb, yb, zb, separation

    t_eval, x, y, z, xb, yb, zb, separation = simulate(
        sigma, rho, beta, x0, y0, z0, t_max, n_pts, eps
    )

    # Estimate Lyapunov exponent from a window before saturation, chosen adaptively
    lyap_exp = None
    fit_mask = None
    sep_range = separation.max() / max(separation.min(), 1e-300)
    if sep_range > 10:
        lower = eps * 10
        upper = 0.2 * max(np.ptp(x), np.ptp(y), np.ptp(z))
        fit_mask = (separation > lower) & (separation < upper)
        if fit_mask.sum() > 10:
            fit = np.polyfit(t_eval[fit_mask], np.log(separation[fit_mask]), 1)
            lyap_exp = fit[0]

    # ----------------------------
    # Header + info banner
    # ----------------------------
    st.title("🦋 Lorenz Attractor Simulator")
    st.caption(
        "A fully deterministic system with three simple equations — yet chaotic: trajectories never "
        "repeat, never settle down, and are exquisitely sensitive to their starting point."
    )

    banner = ASSET_DIR / "butterfly.png"
    st.image(
        str(banner),
        use_container_width=True
    )
  

    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Regime", "Chaotic" if rho > 24.74 and sigma > beta + 1 else "Likely non-chaotic (check ρ, σ, β)")
    col_b.metric(
        "Estimated Lyapunov exponent",
        f"{lyap_exp:.2f}" if lyap_exp is not None else "n/a",
        help="Positive = chaotic (nearby trajectories diverge exponentially). Classic value ≈0.90."
    )
    col_c.metric("Final separation", f"{separation[-1]:.3g}" if show_butterfly else "—")

    # ----------------------------
    # 3D attractor (Plotly)
    # ----------------------------
    fig3d = go.Figure()
    fig3d.add_trace(go.Scatter3d(
        x=x, y=y, z=z, mode="lines",
        line=dict(color="darkorange", width=2), name="Trajectory A"
    ))
    fig3d.update_layout(
        title="The Attractor",
        scene=dict(xaxis_title="x", yaxis_title="y", zaxis_title="z"),
        margin=dict(l=0, r=0, t=40, b=0),
        height=520,
    )

    # ----------------------------
    # x-z projection (Plotly)
    # ----------------------------
    fig_proj = go.Figure()
    fig_proj.add_trace(go.Scatter(
        x=x, y=z, mode="lines",
        line=dict(color="steelblue", width=1), name="Trajectory A"
    ))
    fig_proj.update_layout(
        title="x-z Projection (the classic 'butterfly wings')",
        xaxis_title="x", yaxis_title="z",
        height=450,
        margin=dict(l=10, r=10, t=40, b=10),
    )

    # ----------------------------
    # Divergence / time-series plot (Plotly)
    # ----------------------------
    fig_div = go.Figure()
    if show_butterfly:
        fig_div.add_trace(go.Scatter(
            x=t_eval, y=separation, mode="lines",
            line=dict(color="firebrick", width=2), name="|trajectory A − B|"
        ))
        if lyap_exp is not None and fit_mask is not None:
            anchor_t = t_eval[fit_mask][0]
            anchor_val = np.log(separation[fit_mask])[0]
            fit_line = np.exp(lyap_exp * (t_eval - anchor_t) + anchor_val)
            fig_div.add_trace(go.Scatter(
                x=t_eval, y=fit_line, mode="lines",
                line=dict(color="black", width=1.5, dash="dash"),
                name=f"fit (λ ≈ {lyap_exp:.2f})"
            ))
        fig_div.update_layout(
            title=f"Sensitive Dependence (initial separation = {eps:.0e})",
            xaxis_title="time",
            yaxis_title="separation (log scale)",
            yaxis_type="log",
            height=450,
            margin=dict(l=10, r=10, t=40, b=10),
        )
    else:
        fig_div.add_trace(go.Scatter(
            x=t_eval, y=x, mode="lines",
            line=dict(color="steelblue", width=1), name="x(t)"
        ))
        fig_div.update_layout(
            title="x(t) Time Series",
            xaxis_title="time",
            yaxis_title="x",
            height=450,
            margin=dict(l=10, r=10, t=40, b=10),
        )

    # ----------------------------
    # Layout
    # ----------------------------
    st.plotly_chart(fig3d, width='stretch')
    col_left, col_right = st.columns(2)
    with col_left:
        st.plotly_chart(fig_proj, width='stretch')
    with col_right:
        st.plotly_chart(fig_div, width='stretch')

    if show_butterfly:
        st.subheader("Two trajectories, side by side (x–z projection)")
        fig_compare = go.Figure()
        fig_compare.add_trace(go.Scatter(
            x=x, y=z, mode="lines",
            line=dict(color="darkorange", width=1), name="Trajectory A"
        ))
        fig_compare.add_trace(go.Scatter(
            x=xb, y=zb, mode="lines",
            line=dict(color="steelblue", width=1), name=f"Trajectory B (+{eps:.0e} start)"
        ))
        fig_compare.update_layout(
            xaxis_title="x", yaxis_title="z",
            height=500,
            margin=dict(l=10, r=10, t=30, b=10),
        )
        st.plotly_chart(fig_compare, width='stretch')

    with st.expander("How this works"):
        st.markdown(
            r"""
The Lorenz system was derived by Edward Lorenz in 1963 as a simplified model of atmospheric convection:

$$
\frac{dx}{dt} = \sigma(y - x), \qquad
\frac{dy}{dt} = x(\rho - z) - y, \qquad
\frac{dz}{dt} = xy - \beta z
$$

With the classic parameters (σ=10, ρ=28, β=8/3), the system is **chaotic**: trajectories are confined to
a bounded, infinitely-folded surface (the "strange attractor") but never repeat, and two trajectories
starting arbitrarily close together diverge exponentially — the **butterfly effect**. That divergence
rate is the **Lyapunov exponent**; a positive value is the signature of chaos.

Try dragging **ρ below ≈24.74** (with σ=10, β=8/3) — the system loses its chaotic behavior and settles
into simple fixed-point or periodic motion. This is the classic route to chaos in the Lorenz system.
"""
        )


if __name__ == "__main__":
    st.set_page_config(page_title="Lorenz Attractor Simulator", layout="wide")
    run()