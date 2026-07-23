import streamlit as st
import numpy as np
import plotly.graph_objects as go
from scipy.integrate import solve_ivp

st.set_page_config(
    page_title="Interactive Physics Studio",
    layout="wide"
)

st.title("Simple Pendulum")

# =====================================
# Sidebar Controls
# =====================================

st.sidebar.header("Simulation Parameters")

g = st.sidebar.slider(
    "Gravity (m/s²)",
    1.0,
    20.0,
    9.81
)

L = st.sidebar.slider(
    "Length (m)",
    0.2,
    5.0,
    1.0
)

b = st.sidebar.slider(
    "Damping",
    0.0,
    1.0,
    0.15
)

theta_deg = st.sidebar.slider(
    "Initial Angle (°)",
    0,
    180,
    10
)

omega0 = st.sidebar.slider(
    "Initial Angular Velocity",
    -10.0,
    10.0,
    0.0
)

duration = st.sidebar.slider(
    "Simulation Time (s)",
    5,
    60,
    20
)

theta0 = np.radians(theta_deg)

# =====================================
# Pendulum ODE
# =====================================

def pendulum_ode(t, y):
    theta, omega = y

    dtheta = omega

    domega = -(g/L)*np.sin(theta) - b*omega

    return [dtheta, domega]

# =====================================
# Solve
# =====================================

t = np.linspace(0, duration, 1000)

solution = solve_ivp(
    pendulum_ode,
    (0, duration),
    [theta0, omega0],
    t_eval=t,
    method="RK45"
)

theta = solution.y[0]
omega = solution.y[1]
x = L*np.sin(theta)
y = -L*np.cos(theta)

# =====================================
# Display
# =====================================

st.success("Simulation completed successfully!")

st.write(f"Number of time steps : {len(t)}")

st.write(f"Final angle : {np.degrees(theta[-1]):.2f}°")

st.write(f"Final angular velocity : {omega[-1]:.3f} rad/s")

# Initial figure
fig = go.Figure(
    data=[
        go.Scatter(
            x=[0, x[0]],
            y=[0, y[0]],
            mode="lines+markers",
            line=dict(width=4),
            marker=dict(size=[10, 20]),
        )
    ]
)

#fig.update_yaxes(scaleanchor="x", scaleratio=1)

#st.plotly_chart(fig)

# Create animation frames
frames = []

for i in range(len(x)):
    frames.append(
        go.Frame(
            data=[
                go.Scatter(
                    x=[0, x[i]],
                    y=[0, y[i]],
                    mode="lines+markers",
                    line=dict(width=4),
                    marker=dict(size=[10, 20]),
                )
            ],
            name=str(i),
        )
    )

fig.frames = frames

fig.update_layout(
    updatemenus=[
        dict(
            type="buttons",
            buttons=[
                dict(
                    label="▶ Play",
                    method="animate",
                    args=[
                        None,
                        {
                            "frame": {"duration": 20, "redraw": True},
                            "fromcurrent": True,
                        },
                    ],
                ),
                dict(
                    label="⏸ Pause",
                    method="animate",
                    args=[
                        [None],
                        {
                            "frame": {"duration": 0, "redraw": False},
                            "mode": "immediate",
                        },
                    ],
                ),
            ],
        )
    ]
)

fig.update_xaxes(
    range=[-L - 0.2, L + 0.2]
)

fig.update_yaxes(
    range=[-L - 0.2, 0.2],
    scaleanchor="x",
    scaleratio=1,
)

st.plotly_chart(fig, use_container_width=True)


tab1, tab2, tab3 = st.tabs([
    "Angle",
    "Angular Velocity",
    "Phase Space"
])

with tab1:

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            line=dict(width=3),
            x=solution.t,
            y=np.degrees(theta),
            mode="lines",
            name="Angle"
        )
    )
    fig.add_trace(
        go.Scatter(
        x=solution.t,
        y=np.zeros_like(solution.t),
        mode="lines",
        name="Equilibrium",
        line=dict(dash="dash")
        )
    )

    fig.update_layout(
        title="Angular Displacement",
        xaxis_title="Time (s)",
        yaxis_title="Angle (°)",
        template="plotly_white"
    )

    st.plotly_chart(fig, use_container_width=True)

with tab2:

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            line=dict(width=3),
            x=solution.t,
            y=omega,
            mode="lines",
            name="Angular Velocity"
        )
    )

    fig.update_layout(
        title="Angular Velocity",
        xaxis_title="Time (s)",
        yaxis_title="ω (rad/s)",
        template="plotly_white"
    )

    st.plotly_chart(fig, use_container_width=True)

with tab3:

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            line=dict(width=3),
            x=theta,
            y=omega,
            mode="lines",
            name="Phase Space"
        )
    )

    fig.update_layout(
        title="Phase Space",
        xaxis_title="θ (rad)",
        yaxis_title="ω (rad/s)",
        template="plotly_white"
    )

    st.plotly_chart(fig, use_container_width=True)