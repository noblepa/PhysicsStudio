"""
simulations.coriolis_simple
--------------------------------
A playful, jargon-light explanation of the Coriolis effect using the
classic "ball on a merry-go-round" story, instead of the full rotating-
Earth simulator. Aimed at people with no physics background. Exposes a
single run() function, called from app.py as:

    from simulations import coriolis_simple
    coriolis_simple.run()

Assumes st.set_page_config() has already been called by app.py.

The underlying physics is exactly the same free-particle /
coordinate-rotation trick used elsewhere in this project (a ball with
no real sideways force on it, viewed from a spinning platform) - just
told as a story, with a merry-go-round instead of a globe, and with the
sliders, labels, and pacing built for a first-time audience rather than
a physics student.
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
    # Friendly intro
    # ----------------------------
    st.title("🎠 The Coriolis Effect, Explained Simply")
    
    banner = ASSET_DIR / "coriolis1.png"
    st.image(
        str(banner),
        use_container_width=True
    )

    # ----------------------------
    # Sidebar controls - kept minimal and friendly
    # ----------------------------
    st.sidebar.title("🎠 Try It Yourself")

    spin_speed = st.sidebar.slider(
        "🐴💨 How fast is the ride spinning?", 0.0, 3.0, 1.2, 0.1,
        help="Drag this to 0 to stop the ride completely — watch what happens to the curve!"
    )
    throw_strength = st.sidebar.slider(
        "🎯 How hard do you throw the ball?", 0.05, 0.5, 0.25, 0.01
    )
    viewpoint = st.sidebar.radio(
        "👀 Whose eyes are you watching through?",
        ["🧍 Your friend, standing outside", "🎠 You, riding the merry-go-round"],
        index=1
    )

    with st.sidebar.expander("⚙️ Advanced (optional)"):
        throw_angle = st.slider("Throw direction (°)", 0, 360, 20, 5)
        t_max = st.slider("How long to watch (seconds)", 1.0, 10.0, 6.0, 0.5)
        fps = st.slider("Animation smoothness (fps)", 10, 40, 24, 2)

    riding = viewpoint.startswith("🎠")

    # ----------------------------
    # Physics: identical free-particle + coordinate-rotation model used
    # throughout this project, just relabeled for the story.
    # ----------------------------
    @st.cache_data(show_spinner=False)
    def simulate(spin_speed, throw_strength, throw_angle, t_max, n_pts=400):
        az = np.radians(throw_angle)
        v0 = throw_strength * np.array([np.cos(az), np.sin(az)])
        t = np.linspace(0, t_max, n_pts)
        # straight line, as seen by the friend standing outside
        pos_outside = np.outer(t, v0)
        # transform into the spinning ride's own point of view
        co, so = np.cos(spin_speed * t), np.sin(spin_speed * t)
        x_out, y_out = pos_outside[:, 0], pos_outside[:, 1]
        x_ride = x_out * co + y_out * so
        y_ride = -x_out * so + y_out * co
        pos_ride = np.stack([x_ride, y_ride], axis=1)
        deflection = np.linalg.norm(pos_ride - pos_outside, axis=1)
        return t, pos_outside, pos_ride, deflection

    t, pos_outside, pos_ride, deflection = simulate(spin_speed, throw_strength, throw_angle, t_max)
    pos_shown = pos_ride if riding else pos_outside

    R = 1.0  # merry-go-round radius, for drawing

    # ----------------------------
    # Friendly result callouts (no jargon, no metrics-speak)
    # ----------------------------
    col1, col2 = st.columns(2)
    with col1:
        if spin_speed == 0:
            st.success("🛑 The ride isn't spinning — so there's no curve at all. Both of you see the exact same "
                        "straight path.")
        elif riding:
            st.info(f"🎠 From **your** seat on the ride, the ball drifts off course by about "
                    f"**{deflection[-1]:.2f} ride-widths** by the time it reaches the edge.")
        else:
            st.info("🧍 From **outside**, the ball just rolls in a straight line the whole way — "
                    "no curve, no mystery.")
    with col2:
        st.write(
            "Try dragging **spin speed down to 0** — the curve disappears completely, because "
            "without spinning, there's no difference between the two points of view at all."
        )

    # ----------------------------
    # Playful platform: a colorful pinwheel merry-go-round
    # ----------------------------
    WEDGE_COLORS = ["#FF6B6B", "#FFD93D", "#6BCB77", "#4D96FF", "#C77DFF", "#FF9F45"]
    N_WEDGES = 8

    def wedge_xy(theta0, theta1, radius=R, n=12):
        angles = np.linspace(theta0, theta1, n)
        xs = np.concatenate(([0], radius * np.cos(angles), [0]))
        ys = np.concatenate(([0], radius * np.sin(angles), [0]))
        return xs, ys

    def platform_traces(angle_offset):
        traces = []
        dtheta = 2 * np.pi / N_WEDGES
        for i in range(N_WEDGES):
            th0, th1 = i * dtheta + angle_offset, (i + 1) * dtheta + angle_offset
            xs, ys = wedge_xy(th0, th1)
            traces.append(go.Scatter(
                x=xs, y=ys, fill="toself", fillcolor=WEDGE_COLORS[i % len(WEDGE_COLORS)],
                line=dict(width=0), mode="lines", opacity=0.85,
                showlegend=False, hoverinfo="skip",
            ))
        # a few little "horse" markers riding the rim for charm
        rim_angles = np.linspace(0, 2 * np.pi, N_WEDGES, endpoint=False) + angle_offset + dtheta / 2
        traces.append(go.Scatter(
            x=(R * 0.7) * np.cos(rim_angles), y=(R * 0.7) * np.sin(rim_angles),
            mode="text", text=["🐴"] * len(rim_angles), textfont=dict(size=18),
            showlegend=False, hoverinfo="skip",
        ))
        return traces

    def comet_trail(path, i, start, n_seg=8, rgb=(255, 255, 255)):
        seg_idx = np.unique(np.linspace(start, i, n_seg + 1).astype(int))
        traces = []
        n = max(len(seg_idx) - 1, 1)
        for k in range(n):
            a0, a1 = seg_idx[k], seg_idx[min(k + 1, len(seg_idx) - 1)]
            frac = (k + 1) / n
            alpha = 0.15 + 0.8 * frac ** 1.5
            width = 3 + 8 * frac ** 1.2
            traces.append(go.Scatter(
                x=path[a0:a1 + 1, 0], y=path[a0:a1 + 1, 1], mode="lines",
                line=dict(color=f"rgba({rgb[0]},{rgb[1]},{rgb[2]},{alpha:.2f})", width=width),
                showlegend=False, hoverinfo="skip",
            ))
        return traces

    def glowing_ball(point):
        return [
            go.Scatter(x=[point[0]], y=[point[1]], mode="markers",
                       marker=dict(size=26, color="rgba(255,255,255,0.35)"),
                       showlegend=False, hoverinfo="skip"),
            go.Scatter(x=[point[0]], y=[point[1]], mode="text", text=["⚾"],
                       textfont=dict(size=20), showlegend=False, hoverinfo="skip"),
        ]

    max_frames = 100
    stride = max(1, len(t) // max_frames)
    idx = np.arange(0, len(t), stride)
    trail_len = 30

    frames = []
    for i in idx:
        start = max(0, i - trail_len)
        angle_i = spin_speed * t[i] if not riding else 0.0
        data = platform_traces(angle_i)
        # dashed guide showing the whole path traced out so far, in both viewpoints
        data += [go.Scatter(
            x=pos_shown[:i + 1, 0], y=pos_shown[:i + 1, 1], mode="lines",
            line=dict(color="rgba(255,255,255,0.6)", width=2, dash="dot"),
            showlegend=False, hoverinfo="skip",
        )]
        data += comet_trail(pos_shown, i, start)
        data += glowing_ball(pos_shown[i])
        frames.append(go.Frame(data=data, name=str(i)))

    fig = go.Figure(data=frames[0].data, frames=frames)
    view_label = "🎠 Your view, riding the merry-go-round" if riding else "🧍 Your friend's view, standing outside"
    fig.update_layout(
        title=dict(text=view_label, pad=dict(b=15), font=dict(size=20)),
        xaxis=dict(range=[-1.6, 1.6], visible=False),
        yaxis=dict(range=[-1.6, 1.6], visible=False, scaleanchor="x", scaleratio=1),
        plot_bgcolor="#2e7d32" if not riding else "#1b5e20",
        height=560,
        margin=dict(l=10, r=10, t=60, b=10),
        updatemenus=[dict(
            type="buttons",
            showactive=False,
            y=1.0, x=0.02,
            buttons=[
                dict(label="▶ Throw the ball!", method="animate",
                     args=[None, dict(frame=dict(duration=1000 / fps, redraw=True),
                                       fromcurrent=True, transition=dict(duration=0))]),
                dict(label="⏸ Pause", method="animate",
                     args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate")]),
            ]
        )],
    )
    st.plotly_chart(fig, width='stretch')

    st.caption(
        "🧍 Standing outside, the ride spins underneath a ball that never stops going straight. "
        "🎠 Riding along, you're spinning too — so the same ball looks like it bends away from you."
    )

    # ----------------------------
    # Myth-buster callout
    # ----------------------------
    st.warning(
        "🚽 **Common myth, busted:** you may have heard that the Coriolis effect makes toilets and "
        "sinks drain in opposite directions in the Northern and Southern Hemispheres. That's not "
        "actually true! Your sink is far too small and drains far too fast for Earth's gentle spin "
        "to matter — the direction it swirls is decided by the shape of the basin and tiny leftover "
        "currents in the water, not the Coriolis effect. The real effect only shows up over huge "
        "distances and long times, like weather systems spanning hundreds of kilometers."
    )

    # ----------------------------
    # Bonus: hurricanes really do obey this
    # ----------------------------
    st.subheader("🌀 But hurricanes really do feel it")
    col_n, col_s = st.columns(2)
    with col_n:
        st.markdown("#### Northern Hemisphere")
        st.markdown("Hurricanes spin **counterclockwise** 🌀↺")
    with col_s:
        st.markdown("#### Southern Hemisphere")
        st.markdown("Cyclones spin **clockwise** 🌀↻")
    st.write(
        "A hurricane is just air rushing toward a low-pressure center — but because it travels over "
        "hundreds of kilometers and hours, the same 'merry-go-round' effect has plenty of room and "
        "time to act, curving the incoming wind into the recognizable spiral."
    )

    with st.expander("Want the actual math behind this?"):
        st.markdown(
            r"""
The ball never has any real sideways force on it — its path in the outside (non-spinning) view is
simply

$$
\mathbf{r}_{outside}(t) = \mathbf{v}_0 \, t
$$

To find what it looks like from the spinning ride, we just re-express that same straight line using
axes that rotate along with the ride, at angular speed Ω:

$$
\mathbf{r}_{ride}(t) = R(-\Omega t)\,\mathbf{r}_{outside}(t)
$$

where $R(-\Omega t)$ is a standard 2D rotation matrix. Differentiating this twice with respect to
time (holding the outside-view path fixed) reveals two *fictitious* accelerations that only appear
in the spinning frame:

$$
\mathbf{a}_{ride} = \underbrace{-2\,\Omega \hat{z} \times \mathbf{v}_{ride}}_{\text{Coriolis}} \;
\underbrace{- \,\Omega^2 \mathbf{r}_{ride}}_{\text{centrifugal}}
$$

The first term — the **Coriolis term** — is what curves the ball's path in this demo. Setting
Ω (spin speed) to 0 makes both terms vanish, and the two points of view become identical, exactly
as you saw above.
"""
        )