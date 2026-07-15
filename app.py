import streamlit as st

# ---------------------------------------------------
# Page Configuration
# ---------------------------------------------------

st.set_page_config(
    page_title="Physics Studio",
    page_icon="⚛️",
    layout="wide"
)

# ---------------------------------------------------
# Custom CSS
# ---------------------------------------------------

st.markdown("""
<style>

.hero{
    padding-top:40px;
    padding-bottom:40px;
}

.big-title{
    font-size:60px;
    font-weight:700;
    color:#1565C0;
}

.subtitle{
    font-size:28px;
    color:#555;
}

.section-title{
    font-size:32px;
    font-weight:600;
    margin-top:30px;
}

.feature-box{
    padding:20px;
    border-radius:12px;
    background:#F7F9FC;
    border:1px solid #E0E0E0;
    height:210px;
}

.footer{
    text-align:center;
    color:gray;
    margin-top:40px;
}

</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------
# Hero
# ---------------------------------------------------

st.markdown('<div class="hero">', unsafe_allow_html=True)

col1, col2 = st.columns([3,1])

with col1:

    st.markdown(
        '<div class="big-title">⚛️ Physics Studio</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">Visualize. Explore. Discover.</div>',
        unsafe_allow_html=True
    )

    st.write("")

    st.write("""
Interactive simulations that bring Physics to life.

Explore the laws of nature through real numerical simulations,
interactive graphs and beautiful visualisations.
""")

    st.button("🚀 Launch Simulation")

with col2:

    st.image(
        "https://upload.wikimedia.org/wikipedia/commons/5/58/Atom_symbol.svg",
        use_container_width=True
    )

st.divider()

# ---------------------------------------------------
# Why Physics Studio
# ---------------------------------------------------

st.markdown(
    '<div class="section-title">Why Physics Studio?</div>',
    unsafe_allow_html=True
)

st.write("""

Physics is best learned by experimentation.

Instead of memorising equations,
change parameters and immediately observe
their effect on the physical system.

Every simulation is built from the governing equations
and solved numerically using Python.

""")

# ---------------------------------------------------
# Topics
# ---------------------------------------------------

st.markdown(
    '<div class="section-title">Explore Physics</div>',
    unsafe_allow_html=True
)

c1,c2,c3 = st.columns(3)

with c1:

    st.markdown("""
<div class="feature-box">

### ⚙️ Mechanics

- Projectile Motion
- Pendulum
- Double Pendulum
- Banana Kick
- Coriolis Force
- Circular Motion

</div>
""",unsafe_allow_html=True)

with c2:

    st.markdown("""
<div class="feature-box">

### 🌊 Waves & Optics

- Standing Waves
- Interference
- Diffraction
- Refraction
- Doppler Effect
- Polarisation

</div>
""",unsafe_allow_html=True)

with c3:

    st.markdown("""
<div class="feature-box">

### ⚡ Modern Physics

- Electric Fields
- Magnetism
- Quantum Physics
- Thermodynamics
- Astronomy
- Relativity

</div>
""",unsafe_allow_html=True)

# ---------------------------------------------------
# Features
# ---------------------------------------------------

st.markdown(
    '<div class="section-title">Features</div>',
    unsafe_allow_html=True
)

col1,col2,col3=st.columns(3)

with col1:
    st.success("Interactive Controls")
    st.write("Adjust parameters using sliders and instantly observe the results.")

with col2:
    st.success("Real-time Graphs")
    st.write("Powered by Plotly for smooth zooming, panning and interaction.")

with col3:
    st.success("Scientific Computing")
    st.write("Built using NumPy and SciPy numerical solvers.")

# ---------------------------------------------------
# Featured Simulation
# ---------------------------------------------------

st.markdown(
    '<div class="section-title">Featured Simulation</div>',
    unsafe_allow_html=True
)

st.info("🕰 **Simple Pendulum**")

st.write("""

Explore the motion of damped and undamped pendulums.

Adjust

- Gravity
- Length
- Damping
- Initial Angle
- Initial Angular Velocity

Observe

- Pendulum Motion
- Angle vs Time
- Angular Velocity
- Phase Space

""")

if st.button("Open Simple Pendulum"):

    st.success("Simulation module coming soon...")

# ---------------------------------------------------
# Technologies
# ---------------------------------------------------

st.markdown(
    '<div class="section-title">Powered By</div>',
    unsafe_allow_html=True
)

st.write("""
🐍 Python

📊 Plotly

⚡ Streamlit

🔢 NumPy

📐 SciPy

""")

st.divider()

# ---------------------------------------------------
# Footer
# ---------------------------------------------------

st.markdown(
"""
<div class="footer">

<h3>Physics Studio</h3>

Visualize • Explore • Discover

Interactive Physics Simulations built with Python.

</div>
""",
unsafe_allow_html=True
)