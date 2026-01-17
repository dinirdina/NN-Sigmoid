import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Sigmoid Visualizer", layout="wide")

st.title("Activation Function Visualiser: Sigmoid")
st.write(r"Sigmoid is defined as:  **f(x) = 1 / (1 + e^{-x})**")

st.sidebar.header("Settings")
x_min = st.sidebar.slider("x min", -20.0, 0.0, -10.0, step=1.0)
x_max = st.sidebar.slider("x max", 0.0, 20.0, 10.0, step=1.0)
n_points = st.sidebar.slider("Number of points", 50, 2000, 400, step=50)

x = np.linspace(x_min, x_max, n_points)
y = 1 / (1 + np.exp(-x))

col1, col2 = st.columns([2, 1])

with col1:
    fig, ax = plt.subplots()
    ax.plot(x, y)
    ax.axhline(0, linewidth=1)
    ax.axvline(0, linewidth=1)
    ax.set_title("Sigmoid Function")
    ax.set_xlabel("x")
    ax.set_ylabel("Sigmoid(x)")
    st.pyplot(fig)

with col2:
    st.subheader("Formula")
    st.latex(r"f(x) = \frac{1}{1 + e^{-x}}")
    st.subheader("Sample values")
    idx = np.linspace(0, len(x) - 1, 10, dtype=int)
    st.dataframe({"x": x[idx], "Sigmoid(x)": y[idx]})
