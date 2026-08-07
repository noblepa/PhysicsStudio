"""
simulations.double_slit
----------------------------
An interactive recreation of the thought experiments in Feynman's
Lectures on Physics, Vol. III, Chapter 1: "Quantum Behavior"
(https://www.feynmanlectures.caltech.edu/III_01.html) - bullets, water
waves, and electrons, fired one at a time through two slits. Exposes a
single run() function, called from app.py as:

    from simulations import double_slit
    double_slit.run()

Assumes st.set_page_config() has already been called by app.py.

PHYSICS NOTE
------------
Bullets:  P12 = P1 + P2                                (no interference)
Waves:    I12 = |h1 + h2|^2 = I1 + I2 + 2*sqrt(I1 I2)*cos(delta)   (interference)
Electrons, unwatched:  P12 = |phi1 + phi2|^2, same math as waves
Electrons, watched (which-path known):  P12' = P1' + P2' = I1 + I2  (interference destroyed)

where delta(x) = 2*pi*d*x/(lambda*L) is the phase difference between
the two paths at screen position x (small-angle approximation), d is
the slit separation, L the screen distance, and lambda the wavelength.

The single-slit envelope E(x) = sinc(a*x/(lambda*L))^2 (a = slit
width) shapes all of I1, I2, I12 - it's the same envelope in every
case, since it only depends on diffraction through ONE slit at a time.

The four tabs mirror the chapter's four experiments in order (Sec.
1-2 through 1-6), plus a numbered-summary tab mirroring Sec. 1-7's
"First principles of quantum mechanics."
"""
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from pathlib import Path
ASSET_DIR = Path(__file__).parent.parent / "assets"


# ============================================================
# Shared helpers
# ============================================================
def envelope(x, a, lam, L):
    return np.sinc(a * x / (lam * L)) ** 2


def phase_diff(x, d, lam, L):
    return 2 * np.pi * d * x / (lam * L)


def sample_from_pdf(x, pdf, n_samples, seed=0):
    rng = np.random.default_rng(seed)
    pdf = np.clip(pdf, 0, None)
    pdf_max = pdf.max()
    if pdf_max <= 0:
        return np.zeros(n_samples)
    samples = []
    batch = max(n_samples * 3, 200)
    guard = 0
    while len(samples) < n_samples and guard < 200:
        xs = rng.uniform(x.min(), x.max(), batch)
        ys = rng.uniform(0, pdf_max, batch)
        pdf_interp = np.interp(xs, x, pdf)
        accepted = xs[ys < pdf_interp]
        samples.extend(accepted.tolist())
        guard += 1
    samples = samples[:n_samples] if len(samples) >= n_samples else samples + [0.0] * (n_samples - len(samples))
    return np.array(samples)


def setup_diagram(source_emoji, source_label):
    fig = go.Figure()
    fig.add_annotation(x=0.06, y=0.5, text=source_emoji, showarrow=False, font=dict(size=34), xref="paper", yref="paper")
    fig.add_annotation(x=0.06, y=0.22, text=source_label, showarrow=False, font=dict(size=11, color="#555"), xref="paper", yref="paper")
    # wall with two slits
    fig.add_shape(type="line", x0=0.42, x1=0.42, y0=0.0, y1=0.38, line=dict(color="#444", width=8))
    fig.add_shape(type="line", x0=0.42, x1=0.42, y0=0.46, y1=0.54, line=dict(color="#444", width=8))
    fig.add_shape(type="line", x0=0.42, x1=0.42, y0=0.62, y1=1.0, line=dict(color="#444", width=8))
    fig.add_annotation(x=0.42, y=1.08, text="wall", showarrow=False, font=dict(size=11, color="#555"), xref="paper", yref="paper")
    # screen
    fig.add_shape(type="line", x0=0.92, x1=0.92, y0=0.0, y1=1.0, line=dict(color="#888", width=6))
    fig.add_annotation(x=0.92, y=1.08, text="screen", showarrow=False, font=dict(size=11, color="#555"), xref="paper", yref="paper")
    fig.update_layout(
        xaxis=dict(visible=False, range=[0, 1]), yaxis=dict(visible=False, range=[-0.1, 1.15]),
        height=140, margin=dict(l=10, r=10, t=10, b=10), plot_bgcolor="white",
    )
    return fig


def buildup_figure(x, pdf, n_total, fps, color, title, max_frames=90, seed=0):
    samples = sample_from_pdf(x, pdf, n_total, seed=seed)
    n_bins = 46
    bin_edges = np.linspace(x.min(), x.max(), n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_idx = np.clip(np.digitize(samples, bin_edges) - 1, 0, n_bins - 1)

    stack_height = np.zeros(n_total, dtype=int)
    counts = np.zeros(n_bins, dtype=int)
    for i, b in enumerate(bin_idx):
        stack_height[i] = counts[b]
        counts[b] += 1
    dot_x = bin_centers[bin_idx]
    dot_y = stack_height.astype(float)

    stride = max(1, n_total // max_frames)
    reveal_points = list(range(stride, n_total + 1, stride))
    if not reveal_points or reveal_points[-1] != n_total:
        reveal_points.append(n_total)

    y_max = max(dot_y.max() + 2, 5)
    ref_curve = pdf / pdf.max() * y_max * 0.9

    frames = [
        go.Frame(data=[go.Scatter(x=dot_x[:n], y=dot_y[:n], mode="markers",
                                   marker=dict(size=4, color=color, opacity=0.75))],
                 name=str(n), traces=[1])
        for n in reveal_points
    ]

    fig = go.Figure(
        data=[
            go.Scatter(x=x, y=ref_curve, mode="lines", line=dict(color="rgba(120,120,120,0.35)", width=2, dash="dot"),
                       showlegend=False, hoverinfo="skip"),
            frames[0].data[0],
        ],
        frames=frames,
    )

    
    fig.update_layout(
        title=dict(text=title, pad=dict(b=15)),
        xaxis=dict(title="Position on screen (x)"),
        yaxis=dict(title="Count", range=[0, y_max]),
        height=400,
        margin=dict(l=10, r=10, t=100, b=35),
        showlegend=False,
        updatemenus=[dict(
            type="buttons", showactive=False, y=1.15, x=0.02,
            buttons=[
                dict(label="▶ Fire!", method="animate",
                     args=[None, dict(frame=dict(duration=1000 / fps, redraw=True), fromcurrent=True, transition=dict(duration=0))]),
                dict(label="⏸ Pause", method="animate",
                     args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate")]),
            ],
        )],
    )
    return fig


def curves_figure(x, P1, P2, P12, title, show_naive_sum, naive_color="gray"):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=P1, y=x, mode="lines", line=dict(color="#4D96FF", width=1.5, dash="dot"), name="P₁ (hole 1 alone)"))
    fig.add_trace(go.Scatter(x=P2, y=x, mode="lines", line=dict(color="#FF8C42", width=1.5, dash="dot"), name="P₂ (hole 2 alone)"))
    if show_naive_sum:
        fig.add_trace(go.Scatter(x=P1 + P2, y=x, mode="lines", line=dict(color=naive_color, width=1.5, dash="dash"),
                                  name="P₁+P₂ (naive sum)"))
    fig.add_trace(go.Scatter(x=P12, y=x, mode="lines", line=dict(color="#D2003C", width=3), name="Both holes open (observed)"))
    fig.update_layout(
        title=dict(text=title, pad=dict(b=15)),
        xaxis=dict(title="Probability / Intensity", range=[0,P12.max()*1.05]),
        yaxis=dict(title="Position on screen (x)"),
        height=420,
        margin=dict(l=10, r=10, t=100, b=10),
        legend=dict(orientation="h", yanchor="bottom",x=1.3, y=0.7),
    )
    return fig


# ============================================================
# Main entry point
# ============================================================
def run():
    if st.button("🏠 Back to Home"):
        st.session_state.simulation = "home"
        st.rerun()

    st.title("🔬 Young's Double Slit: Bullets, Waves, and Electrons")
    st.caption(
        "Recreating the thought experiments from Feynman's Lectures on Physics, Vol. III, Ch. 1, "
        "*\"Quantum Behavior\"* — the passage Feynman called the heart of quantum mechanics, containing "
        "\"the only mystery.\" [Read the original chapter here.]"
        "(https://www.feynmanlectures.caltech.edu/III_01.html)"
    )

    banner = ASSET_DIR / "doubleslit.png"
    st.image(
        str(banner),
        use_container_width=True
    )  

    # ----------------------------
    # Shared sidebar controls
    # ----------------------------
    st.sidebar.title("🔬 Double-Slit Controls")
    st.sidebar.subheader("Geometry")
    d = st.sidebar.slider("Slit separation d", 0.5, 4.0, 2.0, 0.1)
    a = st.sidebar.slider("Slit width a", 0.5, 2.5, 1.0, 0.05)
    L = st.sidebar.slider("Screen distance L", 5.0, 40.0, 20.0, 1.0)

    st.sidebar.subheader("Waves / Electrons")
    lam = st.sidebar.slider("Wavelength λ", 0.1, 2.0, 0.5, 0.05,
                             help="Shorter wavelength = finer interference fringes. Try dragging this "
                                  "toward 0 to see why bullets (effectively λ≈0) never show interference.")

    st.sidebar.subheader("Particle Buildup")
    n_particles = st.sidebar.slider("Number of particles fired", 200, 4000, 1500, 100)
    fps = st.sidebar.slider("Animation speed (fps)", 5, 40, 20, 5)

    x = np.linspace(-10, 10, 2000)
    E = envelope(x, a, lam, L)
    delta = phase_diff(x, d, lam, L)

    tab_bullets, tab_waves, tab_e_free, tab_e_watched, tab_summary = st.tabs(
        ["🔫 Bullets", "🌊 Water Waves", "⚛️ Electrons — Unobserved", "👁️ Electrons — Watched", "📜 Summary"]
    )

    # ============================================================
    # TAB 1: BULLETS
    # ============================================================
    with tab_bullets:
        st.subheader("An Experiment with Bullets")
        st.write(
            "A machine gun sprays indestructible bullets at a wall with two holes. Each bullet is a "
            "whole, indivisible lump — it goes through one hole or the other, bounces around a bit, "
            "and lands somewhere on the backstop."
        )
        st.plotly_chart(setup_diagram("🔫", "machine gun"), width='stretch')

        sigma = max(a * 1.5, 0.3)
        P1_b = np.exp(-((x - d / 2) ** 2) / (2 * sigma ** 2))
        P2_b = np.exp(-((x + d / 2) ** 2) / (2 * sigma ** 2))
        P12_b = P1_b + P2_b

        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(buildup_figure(x, P12_b, n_particles, fps, "#333333",
                                            "Bullets Landing, One at a Time", seed=1),
                             width='stretch')
        with col2:
            st.plotly_chart(curves_figure(x, P1_b, P2_b, P12_b, "Result: P₁₂ = P₁ + P₂",
                                           show_naive_sum=False),
                             width='stretch')

        max_diff = np.max(np.abs(P12_b - (P1_b + P2_b)))
        st.success(f"✅ P₁₂ = P₁ + P₂ exactly (max deviation: {max_diff:.2e}). **No interference.** "
                   "The probabilities just add — closing one hole never increases the count anywhere.")

    # ============================================================
    # TAB 2: WATER WAVES
    # ============================================================
    with tab_waves:
        st.subheader("An Experiment with Waves")
        st.write(
            "A wave source jiggles up and down, sending circular ripples through both slits. Unlike "
            "bullets, the intensity can take **any** value — waves don't arrive in lumps at all."
        )
        st.plotly_chart(setup_diagram("🌊", "wave source"), width='stretch')

        #st.markdown("**Ripple tank view** — watch the two circular wavefronts interfere:")

        n_grid = 130
        y_wall = 0.15
        y_screen = 1.0
        gx = np.linspace(-1, 1, n_grid)
        gy = np.linspace(0, y_screen, n_grid)
        GX, GY = np.meshgrid(gx, gy)
        scale = 10.0  # map the [-1,1] grid axis to the same physical x-units used elsewhere
        r1 = np.sqrt((GX * scale - d / 2) ** 2 + (np.maximum(GY - y_wall, 0) * L / (1 - y_wall)) ** 2)
        r2 = np.sqrt((GX * scale + d / 2) ** 2 + (np.maximum(GY - y_wall, 0) * L / (1 - y_wall)) ** 2)
        k = 2 * np.pi / lam
        cos_r1, sin_r1 = np.cos(k * r1), np.sin(k * r1)
        cos_r2, sin_r2 = np.cos(k * r2), np.sin(k * r2)
        mask_beyond_wall = GY >= y_wall



        I1_w, I2_w, I12_w = E, E, 2 * E * (1 + np.cos(delta))
        st.plotly_chart(curves_figure(x, I1_w, I2_w, I12_w, "Result: I₁₂ ≠ I₁ + I₂  (interference!)",
                                       show_naive_sum=True),
                         width='stretch')
        st.info(
            "🌊 The intensity with both holes open is **not** the sum of the two individual intensities "
            "(gray dashed line). At some spots the waves add constructively (bright fringes); at others "
            "they cancel destructively (dark fringes) — this is **interference**."
        )

    # ============================================================
    # TAB 3: ELECTRONS - UNOBSERVED
    # ============================================================
    with tab_e_free:
        st.subheader("An Experiment with Electrons")
        st.write(
            "An electron gun fires electrons toward the wall, one at a time. The detector clicks — "
            "sharp, discrete, all identical — exactly like the bullets. Surely they must go through "
            "one hole or the other... right?"
        )
        st.plotly_chart(setup_diagram("⚛️", "electron gun"), width='stretch')

        P1_e, P2_e = E, E
        P12_e = 2 * E * (1 + np.cos(delta))

        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(buildup_figure(x, P12_e, n_particles, fps, "#7B2FE0",
                                            "Electrons, One Click at a Time", seed=2),
                             width='stretch')
            st.caption(
                "Each electron lands as a single, random-looking click — yet the *pattern* that builds "
                "up is the wave interference pattern, not the simple sum of two bumps."
            )
        with col2:
            st.plotly_chart(curves_figure(x, P1_e, P2_e, P12_e, "Result: P₁₂ ≠ P₁ + P₂",
                                           show_naive_sum=True),
                             width='stretch')

        i0 = np.argmin(np.abs(x))
        ratio = P12_e[np.argmax(P12_e)] / (P1_e[np.argmax(P12_e)] + P2_e[np.argmax(P12_e)])
        st.warning(
            f"⚡ At the central peak, P₁₂ is about **{ratio:.2f}×** the naive sum P₁+P₂ — closing one "
            "hole can *increase* the count somewhere else, and *decrease* it here. Individual, "
            "indivisible particles are somehow sensitive to whether **both** holes are open, even "
            "though each one seems to land as a single, whole lump. This is the mystery."
        )
        st.caption(
            "Try dragging **wavelength λ** in the sidebar down toward 0 — the fringes get finer and "
            "finer until any real detector would only see their smoothed-out average: P₁+P₂. That's "
            "exactly why ordinary bullets, with their absurdly tiny effective wavelength, never show "
            "this effect in practice."
        )

    # ============================================================
    # TAB 4: ELECTRONS - WATCHED
    # ============================================================
    with tab_e_watched:
        st.subheader("Watching the Electrons")
        st.write(
            "Now we shine a light between the two slits so we can *see* which hole each electron "
            "actually goes through — and keep separate counts for 'seen at hole 1' and 'seen at hole 2.'"
        )
        st.plotly_chart(setup_diagram("👁️💡", "light source added"), width='stretch')

        lam_light = st.slider(
            "Wavelength of the watching light", 0.05, 8.0, 0.3, 0.05,
            help="Short wavelength = sharp, precise which-path information. Long wavelength = fuzzy, "
                 "uncertain which-path information (per Heisenberg's uncertainty principle)."
        )
        D = np.exp(-0.5 * (lam_light / d) ** 2)   # which-path distinguishability, 0..1
        V = np.sqrt(max(1 - D ** 2, 0))            # fringe visibility, 0..1

        P1_w2, P2_w2 = E, E
        P_mixed = (P1_w2 + P2_w2) + V * 2 * np.sqrt(P1_w2 * P2_w2) * np.cos(delta)

        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(buildup_figure(x, np.clip(P_mixed, 0, None), n_particles, fps, "#0AA36F",
                                            "Electrons, Watched This Time", seed=3),
                             width='stretch')
        with col2:
            st.plotly_chart(curves_figure(x, P1_w2, P2_w2, P_mixed, "Interference vs. Which-Path Knowledge",
                                           show_naive_sum=True),
                             width='stretch')

        colA, colB = st.columns(2)
        colA.metric("Which-path distinguishability", f"{D:.2f}", help="1 = we always know which hole; 0 = no idea")
        colB.metric("Interference fringe visibility", f"{V:.2f}", help="1 = full interference pattern; 0 = none")

        if D > 0.8:
            st.error(
                "🔦 With short-wavelength (precise) light, we can tell which hole every electron used — "
                "**and the interference pattern is destroyed**, exactly as when we watched with bright "
                "light. Look = no interference."
            )
        elif D < 0.2:
            st.success(
                "🌙 With very long-wavelength light, we can no longer resolve which hole was used — and "
                "the interference pattern is restored, even though the light is technically still "
                "'watching.' Can't tell = interference returns."
            )
        else:
            st.info(
                "🌗 With intermediate wavelength, we get **partial** which-path information and a "
                "**partially washed-out** interference pattern — this is the precise, continuous "
                "trade-off at the heart of Heisenberg's uncertainty principle."
            )

    # ============================================================
    # TAB 5: SUMMARY
    # ============================================================
    with tab_summary:
        st.subheader("First Principles of Quantum Mechanics")
        st.write(
            "Feynman distills all of this into three rules that apply to any \"ideal experiment\" — "
            "one with no uncontrolled outside disturbances:"
        )
        st.markdown(
            r"""
**1. Probability is the absolute square of a complex amplitude.**
$$
P = |\phi|^2
$$

**2. When an event can happen multiple indistinguishable ways, the amplitudes add — and there is interference.**
$$
\phi = \phi_1 + \phi_2, \qquad P = |\phi_1 + \phi_2|^2
$$

**3. If an experiment *could* tell you which way it happened, the interference is gone — probabilities add instead.**
$$
P = P_1 + P_2
$$
"""
        )
        st.write(
            "Bullets always give away 'which hole' just by their sheer clumsiness, so rule 3 applies and "
            "they never interfere. Waves are never particle-like lumps, so rule 2 applies throughout. "
            "Electrons are the strange case: rule 2 applies *until* something in the experiment — even in "
            "principle — could reveal which hole was used, at which point rule 3 takes over instead."
        )
        st.caption(
            "Explore all four tabs above with different wavelengths and slit geometries to see these "
            "three rules play out directly. Full chapter: "
            "https://www.feynmanlectures.caltech.edu/III_01.html"
        )