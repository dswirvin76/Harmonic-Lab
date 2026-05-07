# 🌊 Harmonic-Lab: Continuous-Time Resonant Latent Spaces

Welcome to Harmonic-Lab. As a hobbyist and AI enthusiast, I have been toying with the idea that **frequencies, phase, and frequency modulation** are the keys to creating the ultimate latent space for AI. 

Instead of treating a neural network as a series of static algebraic transformations, this codebase treats the neural network as a **dynamic physical medium**—a vibrating membrane, a set of coupled oscillators, or a mass-spring system. 

I have independently implemented and visualized concepts adjacent to cutting-edge research in *Liquid Time-Constant Networks (LTCs)*, *Continuous RNNs*, and *Predictive Coding*.

## 🧠 The Core Philosophy

This project rejects standard feedforward Deep Learning in favor of **Dynamical Systems**. The latent spaces herein possess *inertia, momentum, frequency, and phase*. 

Key architectural choices include:
*   **Time as a First-Class Citizen:** Networks integrate over time using continuous physics (Euler integration) with explicit `dt` timesteps.
*   **Skew-Symmetric Matrices:** Rotational dynamics ensure stable, long-term memory without vanishing gradients or complex LSTM gates.
*   **Internal Metronomes:** The networks are restless, driven by oscillatory drivers rather than relying purely on data input.
*   **Autoregressive Dreaming:** The models can hallucinate based on physical inertia, passing momentum forward when sensory input is removed.

---

## 🔬 The Lab Experiments

This repository contains interactive, real-time visualizers built with `matplotlib`. 

### 1. The Toddler Brain (`Toddler_Brain_V2Dream.py`)
A 3D organically structured mass-spring-damper graph. It learns associations between sights and sounds. You can prompt it with audio to "imagine" the corresponding visual, or prompt it with a visual to predict the audio. Features a continuous "Dream State" fueled by spontaneous neural static.

### 2. The Maze Solver / Probability Antenna (`HarmonicLab_Mazesolver5.7.py`)
An agent that navigates a maze by projecting "thoughts" forward before physical movement. It utilizes phase and overtones (2nd harmonics) to plan trajectories, casting a visual "antenna" that turns cyan when confident and red when uncertain.

### 3. Spatio-Temporal Spiral Solver (`Harmonic_Lab_Spiral.py`)
Translates space into time. It maps (x,y) coordinates of a complex spiral into physical parameters (Amplitude, Phase, Frequency). By allowing the points to "spin" in latent space, the spiral eventually untangles itself linearly over time.

### 4. Bouncing Ball Predictive Coder (`Harmonic_Lab_BallPredictor.py`)
Predicts a bouncing ball's trajectory using a resonant hidden state and manual Hebbian learning/Predictive coding gradients. Minimizes "surprise" instead of standard backpropagation.

---

## ⚙️ Installation & Usage

To run these experiments, you will need Python installed. 

1. Clone or download this repository.
2. Install the required dependencies using pip:
   ```bash
   pip install -r requirements.txt
