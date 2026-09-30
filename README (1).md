# 🧠 Neural Network Playground

An interactive, web-based neural network learning demo built with **Streamlit** and **TensorFlow/Keras**
(Artificial Neural Network – Activity 2).

## Features
- 4 datasets: XOR, Circle, Spiral, Moons (adjustable size and noise)
- 80/20 train–test split with separate train/test metrics
- 1–4 hidden layers, 1–16 neurons per layer
- ReLU / tanh / sigmoid activations, 5 learning rates (Adam optimizer)
- Train N epochs, single-epoch step, and reset
- Live decision boundary, loss and accuracy curves, parameter count

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy
Push this folder to GitHub, then create an app on https://share.streamlit.io pointing to `app.py`.
