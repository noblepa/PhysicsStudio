import streamlit as st

from simulations import bananakick

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

st.sidebar.title("⚛️ Physics Studio")

page = st.sidebar.radio(
    "Navigate",
    [
        "Home",
        "Banana Kick"
    ]
)

# -------------------------------------------------------
# HOME PAGE
# -------------------------------------------------------

if page == "Home":

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

        if st.button("Launch Banana Kick"):
            st.session_state.page = "Banana Kick"

    with col2:

        st.markdown("### 🕰 Pendulum")

        st.write("Coming Soon")

    with col3:

        st.markdown("### 🚀 Projectile Motion")

        st.write("Coming Soon")

# -------------------------------------------------------
# BANANA KICK
# -------------------------------------------------------

elif page == "Banana Kick":

    bananakick.run()