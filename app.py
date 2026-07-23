import streamlit as st

from simulations import bananakick, lorenz

# -------------------------------------------------------
# Page Configuration
# -------------------------------------------------------

st.set_page_config(
    page_title="Physics Studio",
    page_icon="⚛️",
    layout="wide"
)

# -------------------------------------------------------
# Sidebar
# -------------------------------------------------------

#st.sidebar.title("⚛️ Physics Studio")

#page = st.sidebar.radio(
#    "Navigate",
#    [
#        "Home",
#        "Banana Kick"
#    ],
#    key="page"
#)

# -------------------------------------------
# Navigation state
# -------------------------------------------

if "simulation" not in st.session_state:
    st.session_state.simulation = "home"

# -------------------------------------------------------
# HOME PAGE
# -------------------------------------------------------

if st.session_state.simulation == "home":

    st.title("⚛️ Physics Studio")

    st.subheader("Visualize • Explore • Discover")

    st.write(
        """
        Welcome to **Physics Studio**, an interactive collection of
        physics simulations built using Python, Streamlit and Plotly.

        Explore classical mechanics, waves, optics,
        electromagnetism, thermodynamics, quantum physics,
        astronomy and much more.
        """
    )

    st.divider()

    st.header("Available Simulations")

    col1, col2, col3 = st.columns(3)

    with col1:

        st.markdown("### ⚽ Banana Kick")

        st.write(
            """
            Explore the Magnus effect and understand how
            spin bends the trajectory of a football.
            """
        )

        if st.button("Launch Banana Kick",
                use_container_width=True):
            st.session_state.simulation = "bananakick"
            st.rerun()

    with col2:

        st.markdown("### 🕰 Pendulum")

        st.write("Coming Soon")

    with col3:

        st.markdown("### 🦋 Butterfly Effect / 🌀 Lorenz Attractor")

        st.write(
            """
            Discover chaos theory through the Lorenz system —
            deterministic equations that never repeat.
            """
        )

        if st.button("Launch Butterfly Effect",
                use_container_width=True):
            st.session_state.simulation = "lorenz"
            st.rerun()

# -------------------------------------------------------
# BANANA KICK
# -------------------------------------------------------

elif st.session_state.simulation == "bananakick":

    bananakick.run()

# -------------------------------------------------------
# LORENZ ATTRACTOR
# -------------------------------------------------------

elif st.session_state.simulation == "lorenz":

    lorenz.run()

st.divider()

st.markdown(
"""
<div style="text-align:center; padding:20px; color:#666;">
<br>
Developed and Maintained by <b>Dr. Noble P. Abraham</b><br>
Department of Physics, 
Mar Thoma College, Tiruvalla, Kerala, India
<br>
<!--© 2026 Physics Studio. All Rights Reserved.-->
</div>
""",
unsafe_allow_html=True
)