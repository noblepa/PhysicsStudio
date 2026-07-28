"""
simulations.coriolis
-------------------------
Interactive rotating-Earth Coriolis force simulator, using Plotly for
visuals (including an animated spinning-globe / curving-path view).
Exposes a single run() function, called from app.py as:

    from simulations import coriolis
    coriolis.run()

Assumes st.set_page_config() has already been called by app.py.

PHYSICS NOTE
------------
A projectile is launched from a chosen latitude with some speed and
compass direction. Two rotation rates are used, on purpose:

  - The FULL rotation rate Omega governs how the whole rigid Earth
    (drawn here as a globe with meridian/parallel gridlines) spins in
    real 3D space - this is shown directly in the "Space Frame" view.

  - The LOCAL horizontal deflection rate Omega_eff = Omega * sin(latitude)
    (the classic geophysical "Coriolis parameter" f = 2*Omega*sin(lat),
    here used as f/2) governs how a projectile's path curves within the
    local tangent plane at that latitude - this is the standard
    "f-plane" approximation used throughout meteorology and
    oceanography, valid for motion small compared to Earth's radius.

This is why the whole globe visibly spins at the same rate regardless
of latitude, while the LOCAL curving of a projectile's path is
strongest at the poles, vanishes at the equator, and flips sign between
hemispheres - exactly matching the textbook Coriolis effect.
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
    st.sidebar.title("🌍 Coriolis Force Controls")

    st.sidebar.subheader("Launch")
    latitude = st.sidebar.slider("Launch latitude (°)", -90.0, 90.0, 45.0, 1.0,
                                  help="0° = equator, +90° = North Pole, -90° = South Pole")
    azimuth = st.sidebar.slider("Launch direction (° from North, clockwise)", 0.0, 360.0, 90.0, 5.0,
                                 help="0°=North, 90°=East, 180°=South, 270°=West")
    speed = st.sidebar.slider("Launch speed", 0.02, 0.20, 0.08, 0.01,
                               help="In units of Earth radii per second")

    st.sidebar.subheader("Earth's Rotation")
    Omega = st.sidebar.slider("Rotation rate Ω (rad/s)", 0.0, 3.0, 1.2, 0.1,
                               help="Set to 0 to see the Coriolis effect disappear entirely. "
                                    "(Real Earth: Ω ≈ 7.29×10⁻⁵ rad/s — greatly exaggerated here for visibility.)")

    st.sidebar.subheader("Simulation")
    t_max = st.sidebar.slider("Flight duration (s)", 1.0, 10.0, 5.0, 0.5)
    fps = st.sidebar.slider("Animation frame rate (fps)", 10, 40, 24, 2)

    st.sidebar.subheader("View")
    frame_choice = st.sidebar.radio(
        "Reference frame to animate",
        ["Space (Inertial) Frame", "Earth (Rotating) Frame"],
        index=1
    )
    show_vector = st.sidebar.checkbox("Show Coriolis force vector", value=True)

    R = 1.0  # normalized Earth radius

    # ----------------------------
    # Physics (cached so identical parameter combos don't re-integrate)
    # ----------------------------
    @st.cache_data(show_spinner=False)
    def simulate(latitude, azimuth, speed, Omega, t_max, n_pts=300):
        phi = np.radians(latitude)
        az = np.radians(azimuth)
        Omega_eff = Omega * np.sin(phi)

        vE0 = speed * np.sin(az)
        vN0 = speed * np.cos(az)

        t = np.linspace(0, t_max, n_pts)
        xI, yI = vE0 * t, vN0 * t  # straight line in the local tangent plane

        co, so = np.cos(Omega_eff * t), np.sin(Omega_eff * t)
        xR = xI * co + yI * so
        yR = -xI * so + yI * co

        deflection = np.hypot(xR - xI, yR - yI)

        # Coriolis acceleration in the local rotating frame, via the relative
        # velocity in that frame (finite differences of the curved path)
        vE_rel = np.gradient(xR, t)
        vN_rel = np.gradient(yR, t)
        aE_cor = 2 * Omega_eff * vN_rel
        aN_cor = -2 * Omega_eff * vE_rel

        return t, xI, yI, xR, yR, deflection, Omega_eff, aE_cor, aN_cor

    t, xI, yI, xR, yR, deflection, Omega_eff, aE_cor, aN_cor = simulate(
        latitude, azimuth, speed, Omega, t_max
    )

    # Hover text for each point along each path, so mousing over the
    # trajectory actually tells you something (time, local displacement,
    # and how far the two views have diverged at that instant).
    hover_I = [f"t = {ti:.2f} s<br>East = {xi:+.3f} R<br>North = {yi:+.3f} R"
               for ti, xi, yi in zip(t, xI, yI)]
    hover_R = [f"t = {ti:.2f} s<br>East = {xi:+.3f} R<br>North = {yi:+.3f} R<br>Deflection = {d:.3f} R"
               for ti, xi, yi, d in zip(t, xR, yR, deflection)]

    # ----------------------------
    # Geometry: launch point + local East/North/Up basis on the globe
    # ----------------------------
    phi = np.radians(latitude)
    pos0 = R * np.array([np.cos(phi), 0.0, np.sin(phi)])   # launched at longitude = 0
    East_hat = np.array([0.0, 1.0, 0.0])
    North_hat = np.array([-np.sin(phi), 0.0, np.cos(phi)])

    def embed(x_local, y_local):
        """Embed local tangent-plane (East, North) coords into 3D world space."""
        return (pos0[None, :] + np.outer(x_local, East_hat) + np.outer(y_local, North_hat))

    world_I = embed(xI, yI)   # straight path, world coordinates
    world_R = embed(xR, yR)   # curved (apparent) path, world coordinates

    # ----------------------------
    # Header + metrics
    # ----------------------------
    st.title("🌍 Coriolis Force: Interactive Rotating-Earth Simulator")

    st.caption(
        "Launch a projectile from any latitude and watch it fly in a straight line — while the "
        "rotating Earth beneath it makes the path appear to curve. Toggle frames below to see both."
    )

    banner = ASSET_DIR / "coriolis.png"
    st.image(
        str(banner),
        use_container_width=True
    )

    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Local Coriolis parameter (Ω sin φ)", f"{Omega_eff:+.3f} rad/s")
    col_b.metric("Final deflection (apparent vs. straight)", f"{deflection[-1]:.4f} R")
    col_c.metric("Hemisphere behavior",
                 "No deflection (equator)" if abs(latitude) < 0.5 else
                 ("Deflects right of motion (N. Hemisphere)" if latitude > 0 else
                  "Deflects left of motion (S. Hemisphere)"))

    if Omega == 0:
        st.info("Ω = 0 — Earth isn't rotating, so there's no Coriolis effect: the two paths coincide exactly.")

    # ----------------------------
    # Helpers: globe surface + gridlines
    # ----------------------------
    def earth_surface(R=1.0, n=40):
        u = np.linspace(0, 2 * np.pi, n)
        v = np.linspace(0, np.pi, n // 2)
        x = R * np.outer(np.cos(u), np.sin(v))
        y = R * np.outer(np.sin(u), np.sin(v))
        z = R * np.outer(np.ones_like(u), np.cos(v))
        return go.Surface(x=x, y=y, z=z, opacity=0.95,
                           colorscale=[[0, "#bfe3f0"], [1, "#bfe3f0"]],
                           showscale=False, name="Earth", hoverinfo="skip")

    def meridian(lon_deg, R=1.0, n=60):
        lon = np.radians(lon_deg)
        v = np.linspace(0, np.pi, n)
        x = R * np.sin(v) * np.cos(lon)
        y = R * np.sin(v) * np.sin(lon)
        z = R * np.cos(v)
        return np.stack([x, y, z], axis=1)

    def parallel(lat_deg, R=1.0, n=60):
        lat = np.radians(lat_deg)
        u = np.linspace(0, 2 * np.pi, n)
        x = R * np.cos(lat) * np.cos(u)
        y = R * np.cos(lat) * np.sin(u)
        z = R * np.sin(lat) * np.ones_like(u)
        return np.stack([x, y, z], axis=1)

    def rot_z(points, angle):
        c, s = np.cos(angle), np.sin(angle)
        Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        return points @ Rz.T

    meridians = [meridian(lon) for lon in range(0, 360, 45)]
    parallels = [parallel(lat) for lat in [-45, 0, 45]]

    def star_field(n=180, radius=2.0, seed=7):
        """A fixed backdrop of 'stars', drawn only in the Space (Inertial)
        Frame view. Because these never move while the globe visibly spins
        beneath them, they give an unambiguous visual anchor for 'this is
        the frame that stays still.' Kept close enough that it doesn't
        shrink the (much smaller) trajectory to invisibility."""
        rng = np.random.default_rng(seed)
        phi_s = rng.uniform(0, 2 * np.pi, n)
        costheta = rng.uniform(-1, 1, n)
        theta_s = np.arccos(costheta)
        x = radius * np.sin(theta_s) * np.cos(phi_s)
        y = radius * np.sin(theta_s) * np.sin(phi_s)
        z = radius * np.cos(theta_s)
        return go.Scatter3d(x=x, y=y, z=z, mode="markers",
                             marker=dict(size=1.6, color="white", opacity=0.55),
                             showlegend=False, hoverinfo="skip", name="stars")

    # ----------------------------
    # Main animated 3D figure
    # ----------------------------
    max_frames = 120
    stride = max(1, len(t) // max_frames)
    idx = np.arange(0, len(t), stride)
    trail_len = 40

    grid_color = "rgba(70,130,180,0.45)"
    inertial_mode = frame_choice.startswith("Space")

    def gridline_traces(angle=0.0):
        traces = []
        for m in meridians:
            pts = rot_z(m, angle) if inertial_mode else m
            traces.append(go.Scatter3d(x=pts[:, 0], y=pts[:, 1], z=pts[:, 2], mode="lines",
                                        line=dict(color=grid_color, width=0.5),
                                        showlegend=False, hoverinfo="skip"))
        for p in parallels:
            pts = rot_z(p, angle) if inertial_mode else p
            traces.append(go.Scatter3d(x=pts[:, 0], y=pts[:, 1], z=pts[:, 2], mode="lines",
                                        line=dict(color=grid_color, width=0.5),
                                        showlegend=False, hoverinfo="skip"))
        return traces

    def launch_marker(angle=0.0):
        p = (rot_z(pos0[None, :], angle) if inertial_mode else pos0[None, :])[0]
        return go.Scatter3d(x=[p[0]], y=[p[1]], z=[p[2]], mode="markers",
                             marker=dict(size=5, color="black"), name="Launch point")

    def rgb_for(color_name):
        return {"brightblue": (2, 2, 2), "red": (230, 20, 20)}[color_name]

    def comet_trail(path, i, start, color_name, n_seg=8):
        """Render the recent trail as a series of segments that brighten and
        thicken toward the current point, giving a fading 'comet tail' look
        instead of one flat, uniform line."""
        rgb = rgb_for(color_name)
        seg_idx = np.unique(np.linspace(start, i, n_seg + 1).astype(int))
        traces = []
        n = len(seg_idx) - 1
        for k in range(max(n, 0)):
            a0, a1 = seg_idx[k], seg_idx[k + 1]
            frac = (k + 1) / max(n, 1)  # 0 -> 1, brightest at the head
            alpha = 0.12 + 0.85 * frac ** 1.5
            width = 6 + 7 * frac ** 1.2
            traces.append(go.Scatter3d(
                x=path[a0:a1 + 1, 0], y=path[a0:a1 + 1, 1], z=path[a0:a1 + 1, 2],
                mode="lines",
                line=dict(color=f"rgba({rgb[0]},{rgb[1]},{rgb[2]},{alpha:.2f})", width=width),
                showlegend=False, hoverinfo="skip",
            ))
        return traces

    def glowing_head(point, color_name, hover_text=None):
        rgb = rgb_for(color_name)
        return [
            go.Scatter3d(x=[point[0]], y=[point[1]], z=[point[2]], mode="markers",
                         marker=dict(size=16, color=f"rgba({rgb[0]},{rgb[1]},{rgb[2]},0.25)"),
                         showlegend=False, hoverinfo="skip"),
            go.Scatter3d(x=[point[0]], y=[point[1]], z=[point[2]], mode="markers",
                         marker=dict(size=7, color=f"rgb({rgb[0]},{rgb[1]},{rgb[2]})",
                                     line=dict(color="black", width=0)),
                         showlegend=False,
                         hoverinfo="text" if hover_text else "skip",
                         text=hover_text if hover_text else None),
        ]

    world_path = world_I if inertial_mode else world_R
    path_color = "red" if inertial_mode else "red"
    path_label = "Straight-line (inertial) path" if inertial_mode else "Apparent curved path"
    hover_path = hover_I if inertial_mode else hover_R

    faded_rgb = rgb_for(path_color)
    vec_scale = 0.6 * max(np.hypot(aE_cor, aN_cor).max(), 1e-9) ** -1 * 0.15 if show_vector else 0.0

    # Base (never-changing-by-index) traces: the Earth surface, plus a fixed
    # star field in the inertial view. These must be EXCLUDED from each
    # frame's trace list (via the `traces=` argument below) - otherwise
    # Plotly's default index-based frame updates silently overwrite them,
    # making the globe vanish the instant the animation starts.
    n_base_traces = 2 if inertial_mode else 1

    frames = []
    for i in idx:
        start = max(0, i - trail_len)
        angle_i = Omega * t[i] if inertial_mode else 0.0
        data = gridline_traces(angle_i) + [
            launch_marker(angle_i),
            # full path so far, with hover text so mousing over it actually
            # tells you the time and how far it had deflected by then
            go.Scatter3d(x=world_path[:i + 1, 0], y=world_path[:i + 1, 1], z=world_path[:i + 1, 2],
                         mode="lines",
                         line=dict(color=f"rgba({faded_rgb[0]},{faded_rgb[1]},{faded_rgb[2]},0.55)", width=3.5),
                         showlegend=False, hoverinfo="text", text=hover_path[:i + 1]),
        ]
        data += comet_trail(world_path, i, start, path_color)
        data += glowing_head(world_path[i], path_color, hover_text=hover_path[i])

        if show_vector and not inertial_mode:
            vx, vy = aE_cor[i] * vec_scale, aN_cor[i] * vec_scale
            tip = world_R[i] + vx * East_hat + vy * North_hat
            data.append(go.Scatter3d(
                x=[world_R[i, 0], tip[0]], y=[world_R[i, 1], tip[1]], z=[world_R[i, 2], tip[2]],
                mode="lines+markers", line=dict(color="gold", width=5),
                marker=dict(size=[0, 4], color="gold", symbol="diamond"),
                showlegend=False
            ))
        frames.append(go.Frame(
            data=data, name=str(i),
            traces=list(range(n_base_traces, n_base_traces + len(data))),
        ))

    base_data = [earth_surface()]
    if inertial_mode:
        base_data.append(star_field())
    fig = go.Figure(
        data=base_data + list(frames[0].data),
        frames=frames,
    )
    axis_extent = 2.2 if inertial_mode else 1.6
    fig.update_layout(
        title=dict(text=f"{path_label} — {frame_choice}", pad=dict(b=15)),
        uirevision="coriolis-camera",
        scene=dict(
            xaxis=dict(range=[-axis_extent, axis_extent], visible=False),
            yaxis=dict(range=[-axis_extent, axis_extent], visible=False),
            zaxis=dict(range=[-axis_extent, axis_extent], visible=False),
            aspectmode="cube",
            # An explicit starting camera + uirevision (set at BOTH the
            # scene level and the top-level layout above) is the combo that
            # actually stops Plotly from resetting the 3D camera every time
            # animate() redraws a frame - uirevision alone isn't sufficient
            # for 3D scenes once redraw:true is involved.
            camera=dict(eye=dict(x=1.6, y=1.6, z=1.1)),
            uirevision="coriolis-camera",
        ),
        height=560,
        margin=dict(l=0, r=0, t=50, b=0),
        updatemenus=[dict(
            type="buttons",
            showactive=False,
            y=1.0, x=0.02,
            buttons=[
                dict(label="▶ Play", method="animate",
                     args=[None, dict(frame=dict(duration=1000 / fps, redraw=True),
                                       fromcurrent=True, transition=dict(duration=0))]),
                dict(label="⏸ Pause", method="animate",
                     args=[[None], dict(frame=dict(duration=0, redraw=True), mode="immediate")]),
            ]
        )],
    )
    st.plotly_chart(fig, width='stretch')

    if inertial_mode:
        st.caption(
            "🌌 Notice the **white background stars never move** — they mark the fixed inertial frame. "
            "The **globe and its gridlines visibly spin** beneath the die-straight **red** path. "
            "Hover over the path or the moving dot for exact time/position. Scroll or drag to zoom and "
            "rotate the view — even while playing."
        )
    else:
        st.caption(
            "🌐 Now the globe and its gridlines **hold perfectly still** — you're watching from a point "
            "glued to the Earth's surface. The **same physical motion** now traces a visibly curved "
            "**red** path. Hover over the path or the moving dot for exact time/deflection. Scroll or "
            "drag to zoom and rotate the view — even while playing."
        )

    # ----------------------------
    # Static comparison: both paths, local tangent-plane view
    # ----------------------------
    st.subheader("Both Paths Compared (local tangent-plane view)")
    fig_cmp = go.Figure()
    fig_cmp.add_trace(go.Scatter(x=xI, y=yI, mode="lines",
                                  line=dict(color="#0066FF", width=2, dash="dash"),
                                  name="Straight-line (inertial) path"))
    fig_cmp.add_trace(go.Scatter(x=xR, y=yR, mode="lines",
                                  line=dict(color="red", width=2.5),
                                  name="Apparent curved (Earth-frame) path"))
    fig_cmp.add_trace(go.Scatter(x=[0], y=[0], mode="markers",
                                  marker=dict(size=8, color="black"), name="Launch point"))
    fig_cmp.update_layout(
        title=dict(text="East–North Displacement from Launch Point", pad=dict(b=15)),
        xaxis_title="East displacement (R)",
        yaxis_title="North displacement (R)",
        yaxis=dict(scaleanchor="x", scaleratio=1),
        height=460,
        margin=dict(l=10, r=10, t=50, b=10),
    )
    st.plotly_chart(fig_cmp, width='stretch')

    with st.expander("How this works"):
        st.markdown(
            r"""
This simulator uses two different rotation rates on purpose:

- The **full rotation rate Ω** spins the entire rigid Earth (the globe and its gridlines you see in the
  "Space Frame" view) — this is real and happens at the same rate everywhere on the planet.
- The **local horizontal deflection rate** Ω_eff = Ω·sin(latitude) — equivalent to half the classic
  **Coriolis parameter**, f = 2Ω sin(latitude) — governs how a *freely moving* object's path curves
  within the local tangent plane at that latitude. This is the standard **f-plane approximation** used
  throughout meteorology and oceanography, valid for motion that's small compared to Earth's radius.

That's why the whole globe spins at the same visible rate regardless of where you launch from, while the
**local curving of the projectile's path**:

- is **strongest at the poles** (sin 90° = 1),
- **vanishes exactly at the equator** (sin 0° = 0) — try dragging the latitude slider to 0°,
- and **flips direction** between hemispheres (deflects right of motion in the North, left in the South).

Setting **Ω = 0** removes the effect completely — the straight and curved paths become identical, since
without rotation there's no distinction between "inertial" and "rotating" frames at all.

The instantaneous Coriolis acceleration shown as a red arrow is
$$
\mathbf{a}_{Cor} = -2\,\Omega_{eff}\,\hat{z}_{local} \times \mathbf{v}_{rel}
$$
where $\mathbf{v}_{rel}$ is the object's velocity as measured in the rotating (Earth-fixed) frame.
"""
        )