"""
simulations.lens_imaging
-----------------------------
Interactive thin-lens ray-diagram simulator with a REALISTIC lens
cross-section, a choice of real-world objects (candle, tree,
photograph, ...), and an actual projection screen showing the real
image forming on it (or a faded "virtual image" rendering when no
screen could ever show it). Exposes a single run() function, called
from app.py as:

    from simulations import lens_imaging
    lenses.run()

Assumes st.set_page_config() has already been called by app.py.

PHYSICS NOTE
------------
The thin lens equation and magnification:

    1/di = 1/f - 1/do          m = -di/do          hi = m * ho

Every ray from a point on the object that passes through the lens
converges to (or, for a virtual image, appears to diverge from) the
single image point (di, hi) - this lets all three "principal rays" be
drawn with one unified rule (see the expander in the app for details).

For the object/image PICTURES: a real optical image is not just
flipped top-to-bottom, it's rotated a full 180 degrees (flipped both
vertically AND horizontally) - exactly like a photo held upside down.
This module pre-renders each object icon and a true 180-degree-rotated
version with PIL (verified pixel-by-pixel before shipping - the
flame of the candle icon sits at the top in the upright version and
at the bottom in the rotated one), and simply swaps in the correct
version instead of relying on any run-time flip trick.
"""
import base64
import io

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
import plotly.graph_objects as go
import streamlit as st
from pathlib import Path
ASSET_DIR = Path(__file__).parent.parent / "assets"

# ----------------------------
# Object icon generation (pure function, cached below)
# ----------------------------
def _render_icon(kind: str) -> Image.Image:
    fig, ax = plt.subplots(figsize=(1.4, 2.0), dpi=150)
    ax.axis("off")
    fig.patch.set_alpha(0)
    ax.patch.set_alpha(0)
    ax.set_xlim(0, 1)

    if kind == "candle":
        ax.set_ylim(0, 1.6)
        ax.add_patch(plt.Rectangle((0.35, 0.0), 0.3, 0.9, color="#F4E1C1", ec="#8B7355", lw=2))
        ax.plot([0.5, 0.5], [0.9, 1.05], color="black", lw=2)
        ax.add_patch(plt.Polygon([[0.5, 1.55], [0.4, 1.15], [0.5, 1.02], [0.62, 1.18]],
                                  closed=True, color="#FF8C00"))
        ax.add_patch(plt.Polygon([[0.5, 1.4], [0.45, 1.15], [0.5, 1.08], [0.56, 1.17]],
                                  closed=True, color="#FFD54A"))
    elif kind == "tree":
        ax.set_ylim(0, 1.3)
        ax.add_patch(plt.Rectangle((0.42, 0.0), 0.16, 0.25, color="#8B5A2B"))
        for w, y0, y1 in [(0.65, 0.18, 0.55), (0.5, 0.42, 0.85), (0.36, 0.68, 1.15)]:
            ax.add_patch(plt.Polygon([[0.5 - w / 2, y0], [0.5 + w / 2, y0], [0.5, y1]],
                                      closed=True, color="#2E7D32"))
    elif kind == "photograph":
        ax.set_ylim(0, 1.0)
        ax.add_patch(plt.Rectangle((0.0, 0.0), 1.0, 1.0, fill=False, ec="#8B5A2B", lw=9))
        ax.add_patch(plt.Rectangle((0.08, 0.08), 0.84, 0.84, color="#BFE3F5"))
        ax.add_patch(plt.Circle((0.72, 0.75), 0.1, color="#FFD54A"))
        ax.add_patch(plt.Polygon([[0.1, 0.15], [0.35, 0.55], [0.5, 0.3], [0.68, 0.5], [0.9, 0.15]],
                                  closed=True, color="#4B7A3E"))
    elif kind == "letter_f":
        ax.set_ylim(0, 1.2)
        ax.text(0.5, 0.6, "F", fontsize=110, fontweight="bold", ha="center", va="center",
                 family="DejaVu Sans")
    else:  # arrow
        ax.set_ylim(0, 1.2)
        ax.annotate("", xy=(0.5, 1.15), xytext=(0.5, 0.05),
                     arrowprops=dict(arrowstyle="-|>", lw=5, color="#1B8A3A", mutation_scale=30))

    buf = io.BytesIO()
    fig.savefig(buf, format="png", transparent=True, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGBA")


@st.cache_data(show_spinner=False)
def get_icon(kind: str):
    """Returns (upright_data_uri, rotated_180_data_uri, width/height aspect ratio)."""
    img = _render_icon(kind)
    img_rot = img.transpose(Image.ROTATE_180)

    def to_uri(im):
        b = io.BytesIO()
        im.save(b, format="PNG")
        return "data:image/png;base64," + base64.b64encode(b.getvalue()).decode()

    return to_uri(img), to_uri(img_rot), img.width / img.height


# ----------------------------
# Realistic lens cross-section (biconvex / biconcave polygon)
# ----------------------------
def lens_polygon(f, H, n=50):
    y = np.linspace(-H, H, n)
    if f > 0:
        thickness = 0.35 * (1 - (y / H) ** 2)
    else:
        thickness = 0.06 + (0.35 - 0.06) * (y / H) ** 2
    x_right = thickness / 2
    x_left = -thickness / 2
    xs = np.concatenate([x_right, x_left[::-1]])
    ys = np.concatenate([y, y[::-1]])
    return xs, ys


def run():
    if st.button("🏠 Back to Home"):
        st.session_state.simulation = "home"
        st.rerun()

    # ----------------------------
    # Sidebar controls
    # ----------------------------
    st.sidebar.title("🔍 Lens Imaging Controls")

    st.sidebar.subheader("Object")
    object_options = {
        "🕯️ Candle": "candle",
        "🌲 Tree": "tree",
        "🖼️ Photograph": "photograph",
        "🔤 Letter F (classic optics test object)": "letter_f",
        "➡️ Simple arrow": "arrow",
    }
    object_label = st.sidebar.selectbox("Choose an object", list(object_options.keys()), index=0)
    object_kind = object_options[object_label]

    st.sidebar.subheader("Lens")
    lens_type = st.sidebar.radio("Lens type", ["Converging (convex)", "Diverging (concave)"], index=0)
    f_mag = st.sidebar.slider("Focal length |f|", 0.5, 5.0, 2.0, 0.1)
    f = f_mag if lens_type.startswith("Converging") else -f_mag

    st.sidebar.subheader("Object Placement")
    do = st.sidebar.slider("Object distance dₒ", 0.3, 12.0, 5.0, 0.1)
    ho = st.sidebar.slider("Object height", 0.5, 3.0, 1.5, 0.1)

    st.sidebar.subheader("Sweep Animation")
    do_start = st.sidebar.slider("Sweep start distance", 0.3, 12.0, 10.0, 0.1)
    do_end = st.sidebar.slider("Sweep end distance", 0.3, 12.0, 0.5, 0.1)
    fps = st.sidebar.slider("Animation speed (fps)", 5, 30, 15, 1)
    max_axis_range = st.sidebar.slider(
        "Sweep view: max axis range", 5.0, 30.0, 15.0, 1.0,
        help="Caps how far the 'Distance along axis' extends in the sweep animation below — lower this "
             "if the view looks too zoomed out."
    )

    near_focus = abs(do - f_mag) < 0.03 if lens_type.startswith("Converging") else False
    icon_upright, icon_inverted, icon_aspect = get_icon(object_kind)

    # ----------------------------
    # Physics
    # ----------------------------
    def image_of(do, f, ho):
        if abs(do - f) < 1e-4:
            return None, None, None
        di = f * do / (do - f)
        m = -di / do
        hi = m * ho
        return di, m, hi

    def ray_geometry(do, f, ho, di, hi, x_max):
        segments = []
        for y_lens in (ho, hi):
            segments.append((np.array([-do, 0.0]), np.array([ho, y_lens]), False))
            slope = (hi - y_lens) / di if abs(di) > 1e-9 else (hi - y_lens) / 1e-9
            y_at_xmax = y_lens + slope * x_max
            segments.append((np.array([0.0, x_max]), np.array([y_lens, y_at_xmax]), False))
            if di < 0:
                segments.append((np.array([di, 0.0]), np.array([hi, y_lens]), True))
        slope_center = (0.0 - ho) / do
        y_at_xmax_center = ho + slope_center * (x_max + do)
        segments.append((np.array([-do, x_max]), np.array([ho, y_at_xmax_center]), False))
        return segments

    # ----------------------------
    # Header + metrics
    # ----------------------------
    st.title("🔍 Image Formation with a Thin Lens")
    st.caption(
        "Pick an object, drag the object distance and focal length in the sidebar, and watch a "
        "realistic image of it form — on a real projection screen if it's a real image, or as a "
        "faded 'phantom' if it's virtual."
    )

    banner = ASSET_DIR / "lenses.png"
    st.image(
        str(banner),
        use_container_width=True
    )    

    if near_focus:
        st.warning(
            "⚠️ The object is sitting almost exactly at the focal point — outgoing rays are nearly "
            "parallel and no image forms at any finite distance. Nudge the object distance slider "
            "slightly to see a well-defined image again."
        )
    else:
        di, m, hi = image_of(do, f, ho)

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Image distance dᵢ", f"{di:+.2f}")
        col2.metric("Magnification m", f"{m:+.2f}")
        col3.metric("Image type", "Real" if di > 0 else "Virtual")
        col4.metric("Orientation / size",
                     f"{'Inverted' if m < 0 else 'Upright'}, "
                     f"{'magnified' if abs(m) > 1 else 'reduced' if abs(m) < 1 else 'same size'}")

        # ----------------------------
        # Main ray-diagram figure
        # ----------------------------
        x_max = max(do, abs(f_mag), abs(di)) * 1.4 + 1.0
        x_min = -max(do, abs(f_mag), abs(di)) * 1.2 - 1.0
        y_extent = max(abs(ho), abs(hi), f_mag * 0.6) * 1.6 + 0.5

        fig = go.Figure()

        # optical axis
        fig.add_trace(go.Scatter(x=[x_min, x_max], y=[0, 0], mode="lines",
                                  line=dict(color="gray", width=1), showlegend=False, hoverinfo="skip"))

        # realistic lens cross-section
        lens_h = y_extent * 0.85
        lx, ly = lens_polygon(f, lens_h)
        fig.add_trace(go.Scatter(x=lx, y=ly, fill="toself", fillcolor="rgba(150,200,255,0.55)",
                                  line=dict(color="#2E86FF", width=2), showlegend=False, hoverinfo="skip"))

        # focal points
        fig.add_trace(go.Scatter(x=[-f, f], y=[0, 0], mode="markers+text",
                                  marker=dict(size=7, color="black"),
                                  text=["F", "F'"], textposition="top center",
                                  showlegend=False, hoverinfo="skip"))

        # screen (only meaningful for a real image)
        if di > 0:
            fig.add_shape(type="rect", x0=di - 0.04, x1=di + 0.04,
                          y0=-y_extent * 0.9, y1=y_extent * 0.9,
                          fillcolor="#DDDDDD", opacity=0.9, line=dict(color="#888888", width=1))
            fig.add_shape(type="line", x0=di - 0.35, x1=di + 0.35, y0=-y_extent * 0.9, y1=-y_extent * 0.9,
                          line=dict(color="#888888", width=3))

        # principal rays
        segments = ray_geometry(do, f, ho, di, hi, x_max)
        color_cycle = ["firebrick", "purple", "#555555"]
        seg_group, count_in_group = 0, 0
        for seg_x, seg_y, dashed in segments:
            fig.add_trace(go.Scatter(
                x=seg_x, y=seg_y, mode="lines",
                line=dict(color=color_cycle[min(seg_group, 2)], width=1.5,
                          dash="dot" if dashed else "solid"),
                showlegend=False, hoverinfo="skip",
            ))
            count_in_group += 1
            if seg_group < 2 and count_in_group >= (3 if di < 0 else 2):
                seg_group += 1
                count_in_group = 0

        fig.update_layout(
            title=dict(text=f"{'Converging' if f > 0 else 'Diverging'} lens, |f| = {f_mag:.1f}", pad=dict(b=15)),
            xaxis=dict(title="Distance along axis", range=[x_min, x_max]),
            yaxis=dict(title="Height", range=[-y_extent, y_extent], scaleanchor="x", scaleratio=1),
            height=560,
            margin=dict(l=10, r=10, t=50, b=10),
        )

        # object icon - always upright, sitting on the axis
        fig.add_layout_image(dict(
            source=icon_upright, x=-do, y=ho, xref="x", yref="y",
            sizex=ho * icon_aspect, sizey=ho, xanchor="center", yanchor="top",
            sizing="stretch", layer="above",
        ))

        # image icon - correct pre-rotated version, faded if virtual
        img_top = max(hi, 0)
        img_size = abs(hi)
        fig.add_layout_image(dict(
            source=icon_inverted if hi < 0 else icon_upright,
            x=di, y=img_top, xref="x", yref="y",
            sizex=img_size * icon_aspect, sizey=img_size,
            xanchor="center", yanchor="top", sizing="stretch",
            layer="above", opacity=1.0 if di > 0 else 0.5,
        ))

        st.plotly_chart(fig, width='stretch')

        if di > 0:
            st.caption(
                "🖼️ The image is projected onto the gray **screen** — a real image, exactly what a "
                "camera sensor or a piece of paper held at that spot would show."
            )
        else:
            st.caption(
                "👻 The faded icon is a **virtual image** — light never actually converges there, so no "
                "screen placed at that spot would show anything. It's only what your eye perceives "
                "looking back *through* the lens toward the object."
            )
        st.caption(
            "🔴🟣 Colored rays: the two 'bending' principal rays. ⚫ Gray ray: straight through the lens "
            "center. Dotted segments are backward constructions locating a virtual image."
        )

    # ----------------------------
    # Bonus: animated sweep of the object distance
    # ----------------------------
    st.subheader("Watch the Image Change as the Object Moves")
    st.caption(
        "Press Play to sweep the object distance and watch the image flip between real/virtual and "
        "inverted/upright as the object crosses the focal point."
    )

    @st.cache_data(show_spinner=False)
    def sweep(f, ho, do_start, do_end, n_steps=120):
        do_vals = np.linspace(do_start, do_end, n_steps)
        # Exclude a margin around the do=f singularity that SCALES WITH f -
        # a fixed margin (e.g. 0.05) still lets di = f*do/(do-f) blow up to
        # huge values for larger f, which is what was dragging the axis
        # range out to +/-100. A margin of ~15% of |f| keeps di bounded to
        # a few multiples of f, which is what the "max axis range" slider
        # below then safely displays.
        margin = max(0.15 * abs(f), 0.05)
        do_vals = do_vals[np.abs(do_vals - abs(f)) > margin]
        results = []
        for d in do_vals:
            di = f * d / (d - f)
            m = -di / d
            hi = m * ho
            results.append((d, di, hi))
        return results

    sweep_data = sweep(f, ho, do_start, do_end)

    x_max_sw = min(
        max(do_start, do_end, abs(f_mag), *(abs(r[1]) for r in sweep_data)) * 1.3 + 1,
        max_axis_range,
    )
    x_min_sw = -x_max_sw
    y_extent_sw = min(max(ho, *(abs(r[2]) for r in sweep_data), f_mag * 0.6) * 1.6 + 0.5, max_axis_range)

    frames = []
    for d, di, hi in sweep_data:
        data = [
            go.Scatter(x=[x_min_sw, x_max_sw], y=[0, 0], mode="lines",
                       line=dict(color="gray", width=1), showlegend=False, hoverinfo="skip"),
            go.Scatter(x=[-d, -d], y=[0, ho], mode="lines+markers",
                       line=dict(color="#1B8A3A", width=4),
                       marker=dict(size=[0, 12], symbol="triangle-up", color="#1B8A3A"),
                       showlegend=False, hoverinfo="skip"),
            go.Scatter(x=[di, di], y=[0, hi], mode="lines+markers",
                       line=dict(color="#D2691E", width=4, dash="solid" if di > 0 else "dash"),
                       marker=dict(size=[0, 12],
                                   symbol="triangle-up" if hi > 0 else "triangle-down", color="#D2691E"),
                       showlegend=False, hoverinfo="skip"),
        ]
        frames.append(go.Frame(data=data, name=f"{d:.2f}"))

    lens_h_sw = y_extent_sw * 0.85
    lx_sw, ly_sw = lens_polygon(f, lens_h_sw)
    base_lens = [
        go.Scatter(x=lx_sw, y=ly_sw, fill="toself", fillcolor="rgba(150,200,255,0.55)",
                   line=dict(color="#2E86FF", width=2), showlegend=False, hoverinfo="skip"),
        go.Scatter(x=[-f, f], y=[0, 0], mode="markers+text",
                   marker=dict(size=7, color="black"), text=["F", "F'"],
                   textposition="top center", showlegend=False, hoverinfo="skip"),
    ]

    n_base = len(base_lens)
    for fr in frames:
        fr.traces = list(range(n_base, n_base + len(fr.data)))

    fig_sw = go.Figure(data=base_lens + list(frames[0].data), frames=frames)
    fig_sw.update_layout(
        xaxis=dict(title="Distance along axis", range=[x_min_sw, x_max_sw]),
        yaxis=dict(title="Height", range=[-y_extent_sw, y_extent_sw], scaleanchor="x", scaleratio=1),
        height=480,
        margin=dict(l=10, r=10, t=30, b=10),
        updatemenus=[dict(
            type="buttons",
            showactive=False,
            y=1.1, x=0.02,
            buttons=[
                dict(label="▶ Play", method="animate",
                     args=[None, dict(frame=dict(duration=1000 / fps, redraw=True),
                                       fromcurrent=True, transition=dict(duration=0))]),
                dict(label="⏸ Pause", method="animate",
                     args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate")]),
            ],
        )],
    )
    st.plotly_chart(fig_sw, width='stretch')

    with st.expander("How this works"):
        st.markdown(
            r"""
The thin lens equation relates object distance $d_o$, image distance $d_i$, and focal length $f$:

$$
\frac{1}{d_i} = \frac{1}{f} - \frac{1}{d_o}, \qquad m = -\frac{d_i}{d_o}, \qquad h_i = m\, h_o
$$

**Sign convention used here**: light travels left to right; $d_o$ and $d_i$ are measured from the lens;
$f>0$ for a converging lens, $f<0$ for diverging. A **positive** $d_i$ means a **real** image (light
actually converges there — a screen placed at that point really would show it). A **negative** $d_i$
means a **virtual** image (the light only *appears* to diverge from that point — no screen there would
show anything, but your eye, looking backward through the lens, perceives an image there).

**Why the picture is rotated a full 180°, not just flipped vertically**: a real optical image is
point-inverted through the lens's optical center — both up/down AND left/right are swapped, exactly
like a photograph rotated upside down. That's why, for an asymmetric object like the letter "F", the
image doesn't just appear as "Ⅎ" (mirrored) or "F" flipped vertically — it appears fully upside-down,
which you can check directly by selecting it above.

**Why three rays are enough**: every ray leaving a single point on the object and passing through the
lens converges to (or appears to diverge from) the *same* image point. Once two rays are traced, the
image is located — a third is just a convenient check. The three traditionally chosen are easy to trace
by eye: parallel to the axis (bends through the far focal point F′), through the lens's center
(undeviated), and through the near focal point F (emerges parallel to the axis).

For a **converging** lens: object beyond $2f$ gives a real, inverted, reduced image; between $f$ and
$2f$ gives real, inverted, magnified; inside $f$ gives virtual, upright, magnified — exactly what the
sweep animation above shows as the object crosses each of those thresholds. A **diverging** lens always
produces a virtual, upright, reduced image, no matter the object distance.
"""
        )