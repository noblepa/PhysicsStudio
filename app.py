import streamlit as st

st.set_page_config(
    page_title="Physics Studio",
    layout="wide"
)

st.title("Interactive Physics Studio")

st.sidebar.title("Simulation Controls")

length = st.sidebar.slider(
    "Pendulum Length (m)",
    0.5,
    5.0,
    1.0
)

gravity = st.sidebar.slider(
    "Gravity (m/s²)",
    1.0,
    20.0,
    9.81
)

st.write(f"Length = {length:.2f} m")
st.write(f"Gravity = {gravity:.2f} m/s²")