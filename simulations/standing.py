"""
simulations.string_standing_wave
-------------------------------------
The classic standing wave on a string fixed at both ends. Exposes a
single run() function, called from app.py as:

    from simulations import string_standing_wave
    string_standing_wave.run()

Assumes st.set_page_config() has already been called by app.py.

PHYSICS NOTE
------------
A standing wave is what you get when a wave reflects back and forth
between two fixed ends and interferes with itself. Mathematically, the
sum of a right-moving and a left-moving traveling wave of equal
amplitude and frequency:

    y_R(x,t) = (A/2) sin(kx - wt)
    y_L(x,t) = (A/2) sin(kx + wt)

adds up (via a standard trig identity) to

    y(x,t) = y_R + y_L = A sin(kx) cos(wt)

which is a standing wave: its SHAPE (sin(kx)) never travels anywhere -
only its overall amplitude A*cos(wt) pulses in time. The ends being
fixed (y=0 at x=0 and x=L for all t) forces k = n*pi/L for a positive
integer n - only these discrete "harmonics" fit, which is why a
guitar string, organ pipe, or any bounded wave only rings at specific
frequencies.
"""
import numpy as np
import plotly.graph_objects as go
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
    st.sidebar.title("🎸 Standing Wave Controls")

    st.sidebar.subheader("String")
    n = st.sidebar.slider("Harmonic number n", 1, 8, 2, 1,
                           help="n=1 is the fundamental; n=2,3,... are the overtones/harmonics")
    A = st.sidebar.slider("Amplitude", 0.1, 1.0, 0.6, 0.05)
    speed = st.sidebar.slider("Wave speed", 0.2, 3.0, 1.0, 0.1,
                               help="Purely controls how fast it oscillates in this demo")

    st.sidebar.subheader("Display")
    show_components = st.sidebar.checkbox("Show the two traveling waves that add up to this", value=True)
    show_spacetime = st.sidebar.checkbox("Show space-time plot", value=True)

    st.sidebar.subheader("Playback")
    n_cycles = st.sidebar.slider("Cycles to animate", 1, 6, 3, 1)
    fps = st.sidebar.slider("Animation speed (fps)", 10, 40, 24, 2)

    L = 1.0
    k = n * np.pi / L
    omega = k * speed
    T = 2 * np.pi / omega  # period of one full oscillation

    # ----------------------------
    # Physics (cached so identical parameter combos don't recompute)
    # ----------------------------
    @st.cache_data(show_spinner=False)
    def simulate(n, A, k, omega, n_cycles, n_x=300, frames_per_cycle=40):
        x = np.linspace(0, L, n_x)
        t = np.linspace(0, n_cycles * 2 * np.pi / omega, n_cycles * frames_per_cycle)

        # standing wave: shape * time-envelope
        Y = A * np.sin(k * x)[None, :] * np.cos(omega * t)[:, None]
        # the two traveling-wave components (each half the amplitude)
        Y_right = (A / 2) * np.sin(k * x[None, :] - omega * t[:, None])
        Y_left = (A / 2) * np.sin(k * x[None, :] + omega * t[:, None])

        return x, t, Y, Y_right, Y_left

    x, t, Y, Y_right, Y_left = simulate(n, A, k, omega, n_cycles)

    node_x = np.array([j * L / n for j in range(n + 1)])
    antinode_x = np.array([(j + 0.5) * L / n for j in range(n)])

    # ----------------------------
    # Header + metrics
    # ----------------------------
    st.title("🎸 Standing Wave: String Fixed at Both Ends")
    st.caption(
        "A wave traveling right, reflecting off the fixed end, traveling back left, reflecting again — "
        "over and over. Where the two directions interfere destructively, the string never moves at "
        "all: those points are the **nodes**."
    )

    banner = ASSET_DIR / "standing.png"
    st.image(
        str(banner),
        use_container_width=True
    )    

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Wavelength λ = 2L/n", f"{2 * L / n:.3f}")
    col2.metric("Frequency f = ω/2π", f"{omega / (2 * np.pi):.3f}")
    col3.metric("Nodes (incl. ends)", f"{n + 1}")
    col4.metric("Antinodes", f"{n}")

    # ----------------------------
    # Main animated figure: the string itself
    # ----------------------------
    max_frames = 150
    n_t = len(t)
    stride = max(1, n_t // max_frames)
    idx = np.arange(0, n_t, stride)

    y_range = A * 0.75 if not show_components else A

    frames = []
    for i in idx:
        data = [
            go.Scatter(x=x, y=Y[i], mode="lines", line=dict(color="#2E86FF", width=4),
                       name="Standing wave (sum)"),
        ]
        if show_components:
            data.append(go.Scatter(x=x, y=Y_right[i], mode="lines",
                                    line=dict(color="firebrick", width=1.5, dash="dot"),
                                    name="Right-moving component"))
            data.append(go.Scatter(x=x, y=Y_left[i], mode="lines",
                                    line=dict(color="darkorange", width=1.5, dash="dot"),
                                    name="Left-moving component"))
        data.append(go.Scatter(x=node_x, y=np.zeros_like(node_x), mode="markers",
                                marker=dict(size=8, color="black"), name="Nodes (never move)"))
        frames.append(go.Frame(data=data, name=str(i)))

    fig = go.Figure(data=frames[0].data, frames=frames)
    fig.update_layout(
        title=dict(text=f"n = {n} harmonic", pad=dict(b=15)),
        xaxis=dict(title="Position along string (x/L)", range=[-0.05, 1.05]),
        yaxis=dict(title="Displacement y", range=[-1.15 * (A if show_components else A), 1.15 * (A if show_components else A)]),
        height=460,
        margin=dict(l=10, r=10, t=50, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        updatemenus=[dict(
            type="buttons",
            showactive=False,
            y=1.0, x=-0.1,
            buttons=[
                dict(label="▶ Play", method="animate",
                     args=[None, dict(frame=dict(duration=1000 / fps, redraw=True),
                                       fromcurrent=True, transition=dict(duration=0))]),
                dict(label="⏸ Pause", method="animate",
                     args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate")]),
            ],
        )],
    )
    st.plotly_chart(fig, width='stretch')

    if show_components:
        st.caption(
            "🔴 Dashed red and 🟠 dashed orange are the two traveling waves, moving in opposite "
            "directions. 🔵 The thick blue curve is their sum — notice it never travels, it only "
            "pulses in place. ⚫ Black dots are the nodes, fixed at both ends and evenly spaced between."
        )
    else:
        st.caption("⚫ Black dots mark the nodes — points that never move, no matter how large the amplitude gets.")

    # ----------------------------
    # Space-time plot: shows the nodal lines never move, for the WHOLE
    # duration, at a glance
    # ----------------------------
    if show_spacetime:
        st.subheader("Space-Time View")
        st.caption("Position along the x-axis, time running up the y-axis. The vertical white/pale stripes "
                   "are the nodes — you can see they stay in exactly the same place for all time.")
        fig_st = go.Figure(go.Heatmap(
            x=x, y=t, z=Y,
            colorscale="RdBu", zmid=0, showscale=False,
        ))
        fig_st.update_layout(
            xaxis_title="Position along string (x/L)",
            yaxis_title="Time",
            height=420,
            margin=dict(l=10, r=10, t=20, b=10),
        )
        st.plotly_chart(fig_st, width='stretch')

    with st.expander("How this works"):
        st.markdown(
            r"""
Two traveling waves of equal amplitude and frequency, moving in opposite directions:

$$
y_R(x,t) = \frac{A}{2}\sin(kx - \omega t), \qquad y_L(x,t) = \frac{A}{2}\sin(kx + \omega t)
$$

add together (using $\sin(\alpha \pm \beta) = \sin\alpha\cos\beta \pm \cos\alpha\sin\beta$) to give

$$
y(x,t) = y_R + y_L = A\sin(kx)\cos(\omega t)
$$

Because the *spatial part*, $\sin(kx)$, and the *time part*, $\cos(\omega t)$, are completely separate
factors, the wave's **shape never moves** — only its overall size pulses up and down. That's the
defining feature of a standing wave, as opposed to a traveling one.

Fixing both ends ($y=0$ at $x=0$ and $x=L$, always) forces $\sin(kL) = 0$, which only happens when

$$
k = \frac{n\pi}{L}, \qquad n = 1, 2, 3, \ldots
$$

Only these specific wavelengths ($\lambda = 2L/n$) "fit" between the fixed ends — which is exactly why
a guitar string, a struck bell, or a wind instrument's air column only rings at a discrete set of
harmonic frequencies, rather than any frequency at all.
"""
        )