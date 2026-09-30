"""Neural Network Playground - Activity 2 (Artificial Neural Network)
A web-based interactive demo built with Streamlit + TensorFlow/Keras.
Pick a 2-D dataset, design a network, train it, and watch the decision
boundary, loss curve and accuracy change in real time."""

import time
import numpy as np
import streamlit as st
import tensorflow as tf
import matplotlib.pyplot as plt

st.set_page_config(page_title="Neural Network Playground", page_icon="🧠", layout="wide")

DATASETS = ["XOR", "Circle", "Spiral", "Moons"]
ACTIVATIONS = ["relu", "tanh", "sigmoid"]
LEARNING_RATES = [0.001, 0.01, 0.03, 0.1, 0.5]


# ---------------------------------------------------------------- data
def generate_data(name, n, noise, seed):
    rng = np.random.default_rng(seed)
    if name == "XOR":
        X = rng.uniform(-1, 1, (n, 2))
        y = (X[:, 0] * X[:, 1] > 0).astype(int)
        X += rng.normal(0, noise, X.shape)
    elif name == "Circle":
        X = rng.uniform(-1, 1, (n, 2))
        y = ((X ** 2).sum(axis=1) < 0.45).astype(int)
        X += rng.normal(0, noise, X.shape)
    elif name == "Spiral":
        y = np.arange(n) % 2
        t = rng.uniform(0.3, 3.3, n)
        ang = t * 1.5 + y * np.pi
        r = t / 3.4
        X = np.c_[r * np.cos(ang), r * np.sin(ang)] + rng.normal(0, noise, (n, 2))
    else:  # Moons
        half = n // 2
        a = rng.uniform(0, np.pi, half)
        b = rng.uniform(0, np.pi, n - half)
        top = np.c_[np.cos(a), np.sin(a)]
        bot = np.c_[1 - np.cos(b), 0.5 - np.sin(b)]
        X = np.vstack([top, bot])
        X = (X - X.mean(axis=0)) / 1.4
        y = np.r_[np.zeros(half), np.ones(n - half)].astype(int)
        X += rng.normal(0, noise, X.shape)
    idx = rng.permutation(n)
    X, y = X[idx].astype("float32"), y[idx].astype("float32")
    split = int(0.8 * n)  # 80% train / 20% test
    return X[:split], y[:split], X[split:], y[split:]


# --------------------------------------------------------------- model
def build_model(hidden_layers, neurons, activation, lr):
    tf.keras.utils.set_random_seed(1)
    model = tf.keras.Sequential([tf.keras.layers.Input(shape=(2,))])
    for _ in range(hidden_layers):
        model.add(tf.keras.layers.Dense(neurons, activation=activation))
    model.add(tf.keras.layers.Dense(1, activation="sigmoid"))
    model.compile(optimizer=tf.keras.optimizers.Adam(lr),
                  loss="binary_crossentropy", metrics=["accuracy"])
    return model


def new_run(cfg):
    Xtr, ytr, Xte, yte = generate_data(cfg["dataset"], cfg["samples"], cfg["noise"], 7)
    st.session_state.update(
        cfg=cfg, Xtr=Xtr, ytr=ytr, Xte=Xte, yte=yte,
        model=build_model(cfg["layers"], cfg["neurons"], cfg["activation"], cfg["lr"]),
        epoch=0, train_loss=[], test_loss=[], train_acc=[], test_acc=[],
    )


def train(epochs):
    s = st.session_state
    for _ in range(epochs):
        h = s.model.fit(s.Xtr, s.ytr, epochs=1, batch_size=16, shuffle=True, verbose=0,
                        validation_data=(s.Xte, s.yte))
        s.epoch += 1
        s.train_loss.append(h.history["loss"][0])
        s.test_loss.append(h.history["val_loss"][0])
        s.train_acc.append(h.history["accuracy"][0])
        s.test_acc.append(h.history["val_accuracy"][0])


# ------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("⚙️ Network settings")
    cfg = dict(
        dataset=st.selectbox("Dataset", DATASETS),
        samples=st.slider("Number of samples", 100, 600, 300, 50),
        noise=st.slider("Noise", 0.0, 0.3, 0.05, 0.01),
        layers=st.slider("Hidden layers", 1, 4, 2),
        neurons=st.slider("Neurons per layer", 1, 16, 8),
        activation=st.selectbox("Activation function", ACTIVATIONS),
        lr=st.select_slider("Learning rate", LEARNING_RATES, value=0.03),
    )
    st.caption("Changing any setting builds a fresh, untrained network.")
    epochs_per_click = st.slider("Epochs per 'Train' click", 1, 100, 20)
    c1, c2 = st.columns(2)
    do_train = c1.button("▶ Train", use_container_width=True, type="primary")
    do_step = c2.button("⏭ 1 epoch", use_container_width=True)
    do_reset = st.button("🔄 Reset network", use_container_width=True)

# Auto-rebuild whenever a setting changes (no separate "apply" button needed)
if "cfg" not in st.session_state or st.session_state.cfg != cfg or do_reset:
    new_run(cfg)

if do_train:
    with st.spinner("Training..."):
        train(epochs_per_click)
elif do_step:
    train(1)

s = st.session_state
model = s.model

# ---------------------------------------------------------------- main
st.title("🧠 Neural Network Playground")
st.caption("Design a small neural network, train it with TensorFlow, and see how it "
           "learns to separate two classes of points.")

m1, m2, m3, m4 = st.columns(4)
m1.metric("Epoch", s.epoch)
m2.metric("Train loss", f"{s.train_loss[-1]:.4f}" if s.train_loss else "–")
m3.metric("Train accuracy", f"{s.train_acc[-1]*100:.1f}%" if s.train_acc else "–")
m4.metric("Test accuracy", f"{s.test_acc[-1]*100:.1f}%" if s.test_acc else "–")

left, right = st.columns(2)

with left:
    st.subheader("Decision boundary")
    g = np.linspace(-1.3, 1.3, 120)
    gx, gy = np.meshgrid(g, g)
    grid = np.c_[gx.ravel(), gy.ravel()].astype("float32")
    zz = model(grid, training=False).numpy().reshape(gx.shape)
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    cs = ax.contourf(gx, gy, zz, levels=20, cmap="coolwarm", alpha=0.6, vmin=0, vmax=1)
    ax.contour(gx, gy, zz, levels=[0.5], colors="k", linewidths=1.5)
    ax.scatter(s.Xtr[:, 0], s.Xtr[:, 1], c=s.ytr, cmap="coolwarm", edgecolors="white",
               s=35, label="train")
    ax.scatter(s.Xte[:, 0], s.Xte[:, 1], c=s.yte, cmap="coolwarm", edgecolors="k",
               marker="s", s=35, label="test")
    ax.set_xlim(-1.3, 1.3); ax.set_ylim(-1.3, 1.3)
    ax.set_xlabel("x1"); ax.set_ylabel("x2"); ax.legend(loc="upper right")
    fig.colorbar(cs, ax=ax, label="P(class 1)")
    st.pyplot(fig); plt.close(fig)

with right:
    st.subheader("Learning curves")
    if s.epoch:
        fig, (a1, a2) = plt.subplots(2, 1, figsize=(6, 5.5), sharex=True)
        a1.plot(s.train_loss, label="train"); a1.plot(s.test_loss, label="test")
        a1.set_ylabel("Loss"); a1.legend(); a1.grid(alpha=.3)
        a2.plot(s.train_acc, label="train"); a2.plot(s.test_acc, label="test")
        a2.set_ylabel("Accuracy"); a2.set_xlabel("Epoch"); a2.set_ylim(0, 1.02)
        a2.legend(); a2.grid(alpha=.3)
        st.pyplot(fig); plt.close(fig)
    else:
        st.info("Press **Train** in the sidebar to start learning.")

    st.subheader("Model summary")
    st.write(f"**{s.cfg['layers']}** hidden layer(s) × **{s.cfg['neurons']}** neurons, "
             f"**{s.cfg['activation']}** activation, sigmoid output, Adam optimizer "
             f"(lr = {s.cfg['lr']}). Trainable parameters: **{model.count_params()}**.")

with st.expander("💡 What to try"):
    st.markdown("""
- **XOR** needs at least one hidden layer – try 1 neuron vs 8 neurons.
- **Spiral** is hard: use 3–4 layers, 12–16 neurons, `tanh`, and 100+ epochs.
- A learning rate of **0.5** often makes the loss jump around; **0.001** learns very slowly.
- If train accuracy is high but test accuracy is low, the network is **overfitting** (raise the noise).
""")
